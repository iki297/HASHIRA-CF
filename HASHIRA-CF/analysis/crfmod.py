# CRF_MQTT_Humidity_Forecasting.py
#
# Standalone (non-Streamlit) service that:
#   1. Trains a Causal Random Forest (CRF) pipeline ONCE at startup, using a
#      fixed/hardcoded historical CSV dataset (no manual upload).
#   2. Polls the ThingsBoard REST API for the device's latest telemetry
#      (no manual input, no MQTT subscribe). ThingsBoard does NOT allow
#      devices to *subscribe* to "v1/devices/me/telemetry" over MQTT — that
#      topic is publish-only from the device's perspective (confirmed in
#      ThingsBoard issues #814 and #5742). REST polling is the officially
#      supported way to read a device's live telemetry from an external app.
#   3. Continuously builds live lag/rolling features from the buffered
#      readings and forecasts Humidity 15 minutes ahead.
#   4. Publishes the forecast back to ThingsBoard as a new key, "hum_pred",
#      via the stateless HTTP telemetry API (NOT MQTT). This avoids a second
#      failure mode: the ESP32 keeps its own MQTT session open on this same
#      device access token continuously, and ThingsBoard only allows ONE
#      active MQTT session per device token — a second MQTT client using the
#      same token repeatedly fights the ESP32 for that single slot, which is
#      what caused the connect/disconnect loop even with subscribing removed.
#      HTTP POSTs have no persistent session, so there's nothing to collide.
#
# Run:
#   pip install pandas numpy scikit-learn econml requests
#   python CRF_MQTT_Humidity_Forecasting.py

import os
import csv
import time
import logging
import threading
import collections

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# EconML / CRF
# ============================================================
HAS_ECONML, ECONML_ERR = True, ""
try:
    from econml.dml import CausalForestDML, LinearDML
    from econml.dr import DRLearner
    from econml.metalearners import TLearner
except Exception as e:
    HAS_ECONML, ECONML_ERR = False, str(e)


# ============================================================
# Fixed config — model
# ============================================================
SEED = 42
TEST_FRAC = 0.20
TAU_Q = 0.70
ALPHA_HIGH = 0.02
ALPHA_MED = 0.10
GUARDRAIL = 0.5
COLLINEAR_THR = 0.93

# ============================================================
# Mode keluaran prediksi
# ============================================================
# BLEND_WITH_NAIVE = False  -> keluarkan prediksi CRF MURNI, tanpa dicampur
#   nilai sekarang. Campuran naive (alpha 0.02/0.10) memang menstabilkan
#   MAE, tetapi juga menarik hasil mendekati "nilai sekarang".
#
# CALIBRATE_AMPLITUDE = False -> kalikan perubahan yang diprediksi dengan
#   faktor yang menyamakan variansnya dengan varians perubahan nyata di data
#   latih. Model yang dioptimasi untuk MAE selalu menyusutkan tebakannya ke
#   arah nol (shrinkage-to-mean); pada data ini amplitudonya hanya ~20-50%
#   dari kenyataan, sehingga hasilnya tampak "menyalin" nilai sebelumnya.
#   Kalibrasi mengembalikan amplitudo penuh.
#
#   HASIL UJI 80/20 (n_test=25, dataset 21 Agustus): kalibrasi 2.97x
#   menghasilkan MASE 1.870 dan secara statistik LEBIH BURUK dari naive
#   (Diebold-Mariano p=0.013). CRF murni tanpa kalibrasi jauh lebih baik
#   (MASE 1.067, R2 0.810 vs naive 0.797) dan tetap unggul di lonjakan
#   >=2 poin (MAE 3.55 vs naive 4.80). Karena itu kalibrasi DIMATIKAN.
#
#   PENTING: kalibrasi MEMPERBURUK MAE secara sengaja. Ini pertukaran sadar
#   antara akurasi rata-rata dan responsivitas terhadap lonjakan. Dari uji
#   sebelumnya, faktor ~2 menaikkan deteksi lonjakan >3 poin dari 47% ke 95%,
#   dengan MASE naik dari 1.03 ke 1.40.
BLEND_WITH_NAIVE = False
CALIBRATE_AMPLITUDE = False
CALIBRATION_CAP = 4.0        # batas atas faktor, mencegah penguatan ekstrem

# ============================================================
# Mode deployment
# ============================================================
# DEPLOY_MODE = True -> pakai hampir SELURUH data historis untuk melatih
#   model, dan jadikan prediksi live sebagai evaluasi sesungguhnya. Cocok
#   ketika data historis terlalu sedikit untuk menyisihkan test set yang
#   bermakna (mis. hanya belasan sampel; test set 20% dari itu tidak dapat
#   menyimpulkan apa pun).
#
#   Sebagian kecil data tetap disisihkan sebagai VALIDASI, karena dua
#   mekanisme membutuhkannya: penentuan guardrail dan perhitungan faktor
#   kalibrasi amplitudo. Tanpa validasi keduanya tidak dapat dihitung.
#
#   Konsekuensi yang harus disadari: tidak ada lagi angka akurasi offline
#   yang dapat dilaporkan sebelum deployment. Kualitas model baru diketahui
#   setelah prediksi live dibandingkan dengan nilai aktual yang menyusul —
#   gunakan hum_pred_1h_log.csv untuk evaluasi retrospektif itu.
DEPLOY_MODE = True
DEPLOY_VALIDATION_FRAC = 0.20   # porsi ekor data untuk validasi internal

# Data historis yang terpecah-pecah (banyak celah pengumpulan) kehilangan
# banyak sampel karena fitur lag menuntut riwayat tak terputus: setiap blok
# kehilangan `lag_maksimum + horizon` baris pertamanya. SPARSE_LAGS memangkas
# lag terpanjang sehingga blok-blok pendek tetap terpakai — menaikkan jumlah
# sampel dengan mengorbankan kemampuan melihat pola setengah harian.
SPARSE_LAGS = True

# ============================================================
# Baseline pembanding live
# ============================================================
# Selain CRF, latih juga beberapa model pembanding pada data dan fitur yang
# SAMA PERSIS, lalu terbitkan prediksinya berdampingan ke ThingsBoard.
# Tujuannya membandingkan secara adil di kondisi nyata, bukan hanya lewat
# uji offline pada test set yang kecil.
#
# Latar belakang: uji 80/20 pada dataset 21 Agustus (n_test=25) menunjukkan
# Linear Regression justru terbaik (MASE 0.847, R2 0.916), mengungguli CRF
# (MASE 1.058). Dengan hanya ~98 sampel latih, model sederhana lebih tahan
# overfitting daripada causal forest. Perbandingan live ini menguji apakah
# temuan itu bertahan seiring data bertambah.
PUBLISH_BASELINES = True
BASELINE_KEYS = {
    # --- pembanding korelasional ---
    "naive": "hum_pred_naive",
    "drift": "hum_pred_drift",
    "linear": "hum_pred_lr",
    "ridge": "hum_pred_ridge",
    "rf": "hum_pred_rf",
    "gb": "hum_pred_gb",
    # --- pembanding KAUSAL (ablasi metode) ---
    "lineardml": "hum_pred_lineardml",
    "drlearner": "hum_pred_drlearner",
    "tlearner": "hum_pred_tlearner",
}

# Tiga pembanding kausal ditambahkan agar klaim "causal forest berharga"
# dapat diuji terhadap metode kausal LAIN, bukan hanya terhadap model
# korelasional. Ketiganya membentuk ablasi yang memisahkan sumbangan tiap
# komponen dalam CausalForestDML:
#
#   TLearner    — meta-learner paling sederhana: latih dua model terpisah
#                 untuk kelompok treatment tinggi dan rendah, selisihnya
#                 dipakai sebagai efek. TANPA ortogonalisasi DML, TANPA
#                 causal forest. Menjawab: apakah kerangka DML memang perlu?
#
#   LinearDML   — kerangka DML yang sama persis (ortogonalisasi nuisance),
#                 tetapi tahap akhirnya linear, bukan forest. Menjawab:
#                 apakah komponen FOREST menambah nilai di atas DML biasa?
#
#   DRLearner   — pendekatan doubly robust: tetap konsisten bila salah satu
#                 dari model outcome atau model treatment benar. Menjawab:
#                 apakah properti robustness ganda membantu pada data ini?
#
# Bila CRF mengungguli ketiganya, kontribusi spesifik causal forest
# terbukti. Bila LinearDML setara, forest-nya tidak memberi nilai tambah.

# Dua baseline ditambahkan setelah evaluasi live 116 titik (21-30 Agustus):
#
# "ridge" — Linear Regression biasa runtuh total pada evaluasi itu
#   (MAE 16.13, R2 -1.559). Sebabnya ekstrapolasi: ia hanya memprediksi
#   rentang 17.5-59.0 sementara kelembapan aktual mencapai 86.5. Regresi
#   linear tanpa regularisasi memberi bobot besar pada fitur berkorelasi
#   tinggi, lalu meleset jauh ketika kondisi keluar dari rentang latih.
#   Ridge menghukum bobot besar, sehingga jauh lebih stabil menghadapi
#   pergeseran distribusi — pembanding yang lebih adil bagi model linear.
#
# "drift" — persistence yang diberi kecenderungan: nilai sekarang ditambah
#   perubahan satu langkah terakhir. Ini baseline klasik di literatur
#   forecasting (random walk with drift) dan lebih ketat daripada naive
#   murni, karena sudah menangkap arah pergerakan. Berguna untuk menguji
#   apakah keunggulan CRF saat lonjakan benar-benar berasal dari pemodelan
#   kausal, atau sekadar dari mengikuti momentum.

DATE_COL = "timestamp"

# Historical CSV used to train the model at startup. No file upload / manual
# selection — this path is called directly.
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "20260902_telemetry_training.csv",
)

# Fixed prediction horizon: 15 minutes ahead. Matched to the ESP32's actual
# reporting interval (also 15 minutes) — freq="15min", horizon_steps=1 means
# "predict the value at the NEXT reading", since readings now arrive every
# 15 minutes. (Previously this was freq="1min"/steps=15, which assumed
# ~1-per-minute data — that no longer matches the ESP32's 15-minute
# publish interval, so the live lag/rolling buffer would rarely fill.)
# ============================================================
# Profil horizon — memungkinkan dua instance berjalan PARALEL
# ============================================================
# Pilih lewat argumen atau environment variable:
#   python CRF_MQTT_Humidity_Forecasting.py --profile 15min
#   python CRF_MQTT_Humidity_Forecasting.py --profile 1h
#
# Kedua profil menulis ke KEY dan FILE LOG yang berbeda, sehingga bisa
# dijalankan bersamaan tanpa saling menimpa. Ini memungkinkan perbandingan
# adil: keduanya melihat aliran data sensor yang sama persis, sehingga
# setelah beberapa minggu bisa dinilai apakah keunggulan horizon 1 jam
# (MASE 0.85 pada n=31) benar-benar bertahan pada sampel lebih besar.
PROFILES = {
    "15min": {
        "freq": "15min",
        "horizon_steps": 1,
        "horizon_minutes": 15,
        "label": "15 minutes",
        "pred_key": "hum_pred",
        "log_name": "hum_pred_log.csv",
        # lag12 pada bin 15 menit butuh >= 180 menit riwayat; dilebihkan
        # untuk mentoleransi transmisi yang terlewat.
        "buffer_minutes": 300,
        "min_readings": 3,
    },
    "1h": {
        # Bin 1 jam, 1 langkah ke depan. Disesuaikan dengan resolusi dataset
        # training (Open-Meteo Kanazawa, per jam): memakai bin 30 menit di
        # atas data per jam hanya menghasilkan baris interpolasi tanpa
        # informasi baru, sekaligus menggandakan beban training.
        "freq": "1h",
        "horizon_steps": 1,
        "horizon_minutes": 60,
        "label": "1 hour",
        "pred_key": "hum_pred_1h",
        "log_name": "hum_pred_1h_log.csv",
        # lag12 pada bin 1 jam butuh >= 720 menit riwayat; dilebihkan.
        "buffer_minutes": 1440,
        "min_readings": 3,
    },
}


def _select_profile() -> str:
    import sys
    # Default diubah ke "1h": dataset training (Open-Meteo Kanazawa) beresolusi
    # per jam, sehingga horizon 15 menit tidak dapat dipelajari darinya.
    if "--profile" in sys.argv:
        i = sys.argv.index("--profile")
        if i + 1 < len(sys.argv):
            name = sys.argv[i + 1]
            if name not in PROFILES:
                raise SystemExit(
                    f"Profil '{name}' tidak dikenal. Pilihan: {', '.join(PROFILES)}"
                )
            return name
    return os.environ.get("CRF_PROFILE", "1h")


PROFILE_NAME = _select_profile()
_P = PROFILES[PROFILE_NAME]

PREDICT_FREQ = _P["freq"]
PREDICT_HORIZON_STEPS = _P["horizon_steps"]
PREDICT_HORIZON_LABEL = _P["label"]
HORIZON_MINUTES = _P["horizon_minutes"]

# Diambil dari profil aktif — horizon lebih panjang memakai bin lebih lebar,
# sehingga butuh riwayat lebih panjang untuk mengisi fitur lag.
BUFFER_MINUTES = _P["buffer_minutes"]
MIN_RAW_READINGS_TO_PREDICT = _P["min_readings"]

# PM1_0 sengaja TIDAK disertakan. ESP32 tetap mengirimnya, tetapi analisis
# korelasi menunjukkan PM1_0 (seperti PM2_5/PM10) hampir tidak berhubungan
# dengan PERUBAHAN kelembapan (r kira-kira 0.01), sehingga hanya menambah
# dimensi fitur tanpa daya prediktif. Kunci yang tidak ada di daftar ini
# akan diabaikan begitu saja saat telemetry dibaca.
WANTED = [
    "UV",
    "Temperature",
    "PM2_5",
    "PM10",
    "Humidity",
    "BarPressure",
]

TARGET_COL = "Humidity"
TREAT_COL = "Temperature"

# Dua prediktor eksternal teratas, dipilih dengan memeringkat setiap sensor
# terhadap PERUBAHAN Humidity 1 jam ke depan memakai tiga ukuran sekaligus
# (Pearson, Spearman, dan mutual information), lalu merata-ratakan
# peringkatnya. Hasil pada dataset 21 Agustus:
#
#   UV (ΔUV)            Pearson 0.422  Spearman 0.485  MI 0.233  -> skor 1.67
#   Temperature (ΔTemp) Pearson 0.286  Spearman 0.510  MI 0.236  -> skor 1.67
#   BarPressure         Pearson 0.102  Spearman 0.207  MI 0.208  -> skor 5.00
#   PM10                Pearson 0.101  Spearman 0.109  MI 0.062  -> skor 8.00
#   PM2_5               Pearson 0.104  Spearman 0.109  MI 0.018  -> skor 8.33
#   PM1_0               Pearson 0.145  Spearman 0.013  MI 0.036  -> skor 8.67
#
# Jurang antara dua teratas dan sisanya lebar, jadi pemilihan ini kokoh.
# Uji 80/20 membandingkan 2 vs 3 prediktor: menambah BarPressure sedikit
# memperbaiki MAE keseluruhan (2.343 vs 2.476) tetapi MEMPERBURUK akurasi
# saat lonjakan >=2 poin (3.716 vs 3.454) — regime yang justru paling
# relevan untuk kontrol iklim. Dengan hanya ~123 sampel, menambah fitur
# (23 -> 29 kolom) mulai menimbulkan overfitting. Karena itu dipilih dua.
KEY_EXTERNAL_FORECAST_COLS = ["Temperature", "UV"]
FORECAST_STATE_COLS = ["Humidity"]
KEY_FORECAST_COLS = KEY_EXTERNAL_FORECAST_COLS + FORECAST_STATE_COLS


# ============================================================
# Fixed config — ThingsBoard REST API (for reading live telemetry)
# ============================================================
TB_BASE_URL = "http://<THINGSBOARD_HOST>:8080"
TB_USERNAME = "<TB_USERNAME>"
TB_PASSWORD = "<TB_PASSWORD>"
TB_DEVICE_NAME = "Hydroponic"
# 30 detik memadai untuk ESP32 yang mengirim tiap 15 menit. Polling hanya
# mengambil nilai TERBARU, jadi interval polling yang jauh lebih rapat dari
# interval kirim hanya menghasilkan pembacaan berulang yang sama.
POLL_INTERVAL_SECONDS = 30

# ============================================================
# Pelatihan ulang berkala
# ============================================================
# DINONAKTIFKAN atas permintaan: model dilatih SEKALI saat startup dari
# CSV_PATH, lalu dipakai apa adanya sepanjang program berjalan. Untuk
# memperbarui model, unduh data terbaru dan jalankan ulang program.
#
# Bila suatu saat ingin model menyerap data baru secara otomatis, cukup
# set RETRAIN_ENABLED = True. Mekanismenya sudah lengkap dan teruji:
# mengunduh seluruh riwayat dari ThingsBoard, melatih ulang, lalu menukar
# model secara atomik. Bila pelatihan ulang gagal (jaringan putus, data
# kosong), model lama tetap dipakai dan layanan terus berjalan.
#
# Kenapa 24 jam, bukan tiap jam: pada bin 1 jam, satu jam hanya menambah
# SATU sampel dari beberapa ratus yang sudah ada (<0,5%) — estimasi model
# nyaris tidak berubah, sementara pelatihan ulang memakan waktu dan CPU.
# Sehari memberi hingga 24 sampel baru sekaligus, sekaligus memastikan
# model selalu mencakup siklus harian penuh (siang-malam) yang menjadi
# pola dominan pada kelembapan.
#
# CausalForestDML tidak mendukung pembelajaran inkremental (partial_fit),
# jadi pelatihan ulang selalu dari awal memakai seluruh riwayat.
RETRAIN_ENABLED = False
RETRAIN_INTERVAL_HOURS = 24

# ThingsBoard REST API always returns "ts" as UTC epoch milliseconds.
# Everything else in this script (pd.Timestamp.now(), the historical CSV's
# timestamps, hour-of-day/day-of-week features) is in LOCAL wall-clock time.
# Without converting, live readings would be off by the local UTC offset
# (e.g. 7 hours for WIB) from everything else — this constant fixes that.
LOCAL_TIMEZONE = "Asia/Tokyo"  # JST (UTC+9) — change if deployed elsewhere


def now_local() -> pd.Timestamp:
    """Current time as a naive Timestamp whose VALUE is LOCAL_TIMEZONE
    wall-clock — regardless of what timezone the machine running this
    script is actually set to. Use this instead of pd.Timestamp.now()
    anywhere a "current time" is needed, to avoid silently mixing in the
    system's own local timezone.
    """
    return pd.Timestamp.now(tz=LOCAL_TIMEZONE).tz_localize(None)

# ============================================================
# Fixed config — ThingsBoard HTTP device API (for publishing hum_pred back)
# ============================================================
# NOTE: We publish via HTTP instead of MQTT. The ESP32 keeps an MQTT session
# open on this same device access token continuously; ThingsBoard only
# allows ONE active MQTT session per device token, so a second MQTT client
# using the same token fights the ESP32 for that single slot (this is what
# caused the earlier connect/disconnect loop even after removing the
# subscribe call). HTTP telemetry POSTs are stateless — no persistent
# session, so no conflict with the ESP32's MQTT connection.
TB_DEVICE_TOKEN = "<DEVICE_ACCESS_TOKEN>"
TB_TELEMETRY_HTTP_URL = f"{TB_BASE_URL}/api/v1/{TB_DEVICE_TOKEN}/telemetry"
PRED_KEY = _P["pred_key"]

# Every prediction is also appended (full float precision, not truncated)
# to this local CSV for research/analysis — separate from what's published
# to ThingsBoard (which is truncated to a whole number).
PRED_LOG_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    _P["log_name"],
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("crf_mqtt")


# ============================================================
# ThingsBoard REST client (used to READ latest telemetry; MQTT is only
# used to WRITE the forecast back — see header note on why)
# ============================================================
class ThingsBoardRestClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token = None
        self.device_id = None

    def login(self):
        r = requests.post(
            f"{self.base_url}/api/auth/login",
            json={"username": self.username, "password": self.password},
            timeout=10,
        )
        r.raise_for_status()
        self.token = r.json()["token"]
        log.info("Logged into ThingsBoard REST API as %s", self.username)

    def _headers(self):
        return {"X-Authorization": f"Bearer {self.token}"}

    def _get(self, path, params=None):
        if self.token is None:
            self.login()
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, headers=self._headers(), timeout=10)
        if r.status_code == 401:
            # Token expired — re-login once and retry.
            self.login()
            r = requests.get(url, params=params, headers=self._headers(), timeout=10)
        r.raise_for_status()
        return r.json()

    def list_device_names(self, page_size: int = 100):
        """List device names/tokens for this tenant — used as a diagnostic
        when get_device_id() can't find the configured TB_DEVICE_NAME.
        """
        data = self._get(
            "/api/tenant/devices",
            params={"pageSize": page_size, "page": 0},
        )
        results = []
        for d in data.get("data", []):
            device_id = d["id"]["id"]
            name = d.get("name")
            token = None
            try:
                cred = self._get(f"/api/device/{device_id}/credentials")
                token = cred.get("credentialsId")
            except Exception:
                pass
            results.append((name, token))
        return results

    def get_device_id(self, device_name: str) -> str:
        try:
            data = self._get("/api/tenant/devices", params={"deviceName": device_name})
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.error(
                    "Device named '%s' not found. TB_DEVICE_NAME must match the "
                    "device's NAME in ThingsBoard (Devices page), NOT its access "
                    "token — these are often different fields.",
                    device_name,
                )
                try:
                    devices = self.list_device_names()
                    if devices:
                        log.error("Devices found in this tenant (name -> access token):")
                        for name, token in devices:
                            log.error("  '%s' -> token='%s'", name, token)
                    else:
                        log.error("No devices found for this tenant at all.")
                except Exception as e2:
                    log.error("Could not list devices for diagnostics: %s", e2)
            raise
        self.device_id = data["id"]["id"]
        log.info("Resolved ThingsBoard device '%s' -> id=%s", device_name, self.device_id)
        return self.device_id

    def get_history_telemetry(self, keys, limit=1000000):
        """Ambil SELURUH riwayat telemetry untuk dipakai melatih ulang model.

        Berbeda dari get_latest_telemetry() yang hanya mengambil satu titik
        terbaru per key, metode ini menarik rentang penuh sejak 2020 hingga
        sekarang, lalu menyusunnya menjadi DataFrame bergaya CSV training.
        """
        if self.device_id is None:
            self.get_device_id(TB_DEVICE_NAME)

        start_ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
        end_ts = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        data = self._get(
            f"/api/plugins/telemetry/DEVICE/{self.device_id}/values/timeseries",
            params={
                "keys": ",".join(keys),
                "startTs": start_ts,
                "endTs": end_ts,
                "limit": limit,
                "agg": "NONE",
                "orderBy": "ASC",
            },
        )

        rows = {}
        for k, arr in data.items():
            for item in arr:
                ts = int(item["ts"])
                rows.setdefault(ts, {})[k] = item["value"]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [{"ts": ts, **vals} for ts, vals in rows.items()]
        )
        df[DATE_COL] = (
            pd.to_datetime(df["ts"], unit="ms", utc=True)
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )
        df = df.drop(columns=["ts"])
        for c in keys:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values(DATE_COL).reset_index(drop=True)

    def get_latest_telemetry(self, keys):
        if self.device_id is None:
            self.get_device_id(TB_DEVICE_NAME)
        data = self._get(
            f"/api/plugins/telemetry/DEVICE/{self.device_id}/values/timeseries",
            params={"keys": ",".join(keys)},
        )
        values, ts_ms = {}, None
        for k, arr in data.items():
            if arr:
                try:
                    values[k] = float(arr[0]["value"])
                except (TypeError, ValueError):
                    continue
                ts_ms = max(ts_ms or 0, int(arr[0]["ts"]))
        return values, ts_ms


# ============================================================
# Utilities
# ============================================================
def robust_parse_date(s: pd.Series) -> pd.Series:
    """Parse datetime with several fallback formats."""
    dt = pd.to_datetime(s, errors="coerce")

    if dt.isna().mean() > 0.2:
        for fmt in (
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
        ):
            tmp = pd.to_datetime(s, format=fmt, errors="coerce")
            if tmp.notna().sum() > dt.notna().sum():
                dt = tmp

    return dt


def smape(y, yhat, eps=1e-6):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    den = (np.abs(y) + np.abs(yhat)) / 2.0
    return float(np.mean(np.abs(y - yhat) / np.maximum(den, eps)) * 100)


def add_lags(df, cols, lags=(1, 2, 3, 6, 12)):
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            for L in lags:
                df[f"{c}_lag{L}"] = x.shift(L)
    return df


def add_rolls(df, cols, windows=(3, 6, 12)):
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            for w in windows:
                df[f"{c}_roll{w}"] = x.rolling(w, min_periods=1).mean()
    return df


def get_lags_and_rolls(freq: str):
    """Select lag/rolling windows so valid data is not exhausted after shift/dropna."""
    if SPARSE_LAGS:
        # Lag maksimum 3 agar blok data pendek tidak habis terbuang.
        return (1, 2, 3), (2, 3)
    if freq in ["5s", "1min"]:
        return (1, 2, 3, 6, 12), (3, 6, 12)
    if freq in ["5min", "10min", "15min"]:
        return (1, 2, 3, 6, 12), (3, 6, 12)
    if freq in ["30min", "1H", "1h"]:
        return (1, 2, 3, 6), (3, 6)
    return (1, 2, 3), (3,)


def build_treatment_cv(Tb_tr: np.ndarray, max_splits: int = 5):
    """Pick a CV splitter for CausalForestDML whose every training fold
    contains both treatment classes (required by discrete_treatment=True).

    Prefers a chronological TimeSeriesSplit (fewer splits first, since a
    small number of blocks is less likely to isolate a single-class
    stretch). Falls back to a shuffled StratifiedKFold, which sacrifices
    strict time-ordering in the nuisance cross-fitting step but guarantees
    a valid fit — this only affects the nuisance models used internally by
    DML, not the final chronological train/val/test split used for
    evaluation.
    """
    n = len(Tb_tr)

    for n_splits in range(2, min(max_splits, n // 20) + 1):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ok = True
        for tr_idx, te_idx in tscv.split(Tb_tr):
            if len(np.unique(Tb_tr[tr_idx])) < 2 or len(te_idx) == 0:
                ok = False
                break
        if ok:
            return tscv

    n_splits = max(2, min(max_splits, n // 20))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def clean_iot_data(df_raw: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Clean the ThingsBoard environmental telemetry dataset."""
    df = df_raw.copy()
    df[date_col] = robust_parse_date(df[date_col])

    for c in WANTED:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rules = pd.Series(True, index=df.index)
    rules &= df[date_col].notna()

    if "UV" in df.columns:
        rules &= df["UV"] >= 0

    if "Temperature" in df.columns:
        rules &= df["Temperature"].between(0, 60)

    if "PM2_5" in df.columns:
        rules &= df["PM2_5"] >= 0

    if "PM10" in df.columns:
        rules &= df["PM10"] >= 0

    if "Humidity" in df.columns:
        rules &= df["Humidity"].between(0, 100)

    if "BarPressure" in df.columns:
        rules &= df["BarPressure"] > 0

    df = df.loc[rules].copy()
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col]).reset_index(drop=True)

    return df


def resample_iot_data(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    """Resample IoT data into a fixed interval.

    Does NOT drop rows with missing TARGET_COL/TREAT_COL here — real data
    gaps must stay as NaN through the continuous grid so lag/rolling
    features computed downstream correctly propagate NaN across them,
    instead of silently bridging a gap (e.g. a lag1 value quietly coming
    from a reading taken days earlier). Callers that need only the latest
    complete row (e.g. build_live_feature_row) already guard against a NaN
    last row explicitly.
    """
    numeric_cols = [c for c in WANTED if c in df.columns]

    out = (
        df.set_index(date_col)[numeric_cols]
        .resample(freq)
        .mean(numeric_only=True)
        .reset_index()
    )

    # Interpolate small gaps only.
    # limit=3 prevents long gaps from being filled too aggressively.
    for c in numeric_cols:
        out[c] = out[c].interpolate(
            method="linear",
            limit=3,
            limit_direction="both",
        )

    return out


# ============================================================
# CRF training for environmental telemetry IoT
# ============================================================
def train_iot_crf(
    df_raw: pd.DataFrame,
    date_col: str,
    freq: str = "1min",
    horizon_steps: int = 15,
):
    if not HAS_ECONML:
        raise RuntimeError(f"EconML is not available: {ECONML_ERR}")

    missing_base = [c for c in [date_col, TARGET_COL, TREAT_COL] if c not in df_raw.columns]
    if missing_base:
        raise ValueError(f"Required columns are missing: {', '.join(missing_base)}")

    df_clean = clean_iot_data(df_raw, date_col)
    df = resample_iot_data(df_clean, date_col, freq)

    if len(df) < (20 if DEPLOY_MODE else 80):
        raise ValueError(
            f"Valid data after resampling is too small: n={len(df)}. "
            "Use a longer/denser historical CSV or a coarser frequency."
        )

    # Time features
    df["hour"] = df[date_col].dt.hour
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["dayofyear"] = df[date_col].dt.dayofyear

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)

    # Lag and rolling features
    lags, rolls = get_lags_and_rolls(freq)

    # Forecasting uses the top-3 external predictors identified from the
    # lag-to-lead correlation analysis plus current humidity as a temporal state.
    lag_cols = [c for c in KEY_FORECAST_COLS if c in df.columns]

    add_lags(df, lag_cols, lags=lags)
    add_rolls(df, lag_cols, windows=rolls)

    # Target: humidity change at the selected future horizon.
    hum_lvl = pd.to_numeric(df[TARGET_COL], errors="coerce")
    Y_lvl = hum_lvl.shift(-int(horizon_steps))
    Y_delta = Y_lvl - hum_lvl

    T_ser = pd.to_numeric(df[TREAT_COL], errors="coerce")

    # Confounders/features for CRF
    conf_cols = [
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    for c in KEY_FORECAST_COLS:
        if c in df.columns:
            conf_cols.append(c)

    for c in df.columns:
        if c.endswith(tuple([f"lag{L}" for L in lags])):
            conf_cols.append(c)
        if c.endswith(tuple([f"roll{w}" for w in rolls])):
            conf_cols.append(c)

    conf_cols = list(dict.fromkeys([c for c in conf_cols if c in df.columns]))

    X_df = (
        df[conf_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    mask = (
        (~X_df.isna().any(axis=1)).to_numpy()
        & T_ser.notna().to_numpy()
        & Y_delta.notna().to_numpy()
        & hum_lvl.notna().to_numpy()
    )

    X_all = X_df.loc[mask].to_numpy(float)
    T_all = T_ser.loc[mask].to_numpy(float)
    Y_all = Y_delta.loc[mask].to_numpy(float)
    H_t_all = hum_lvl.loc[mask].to_numpy(float)
    time_all = df.loc[mask, date_col].to_numpy()

    n = len(X_all)
    if n < (20 if DEPLOY_MODE else 80):
        raise ValueError(
            f"Valid data is too small after creating lag/target features: n={n}. "
            "Use a longer/denser historical CSV."
        )

    # Dalam DEPLOY_MODE tidak ada test set terpisah: seluruh data dipakai
    # melatih, dan "test set" dilaporkan ulang dari partisi validasi hanya
    # sebagai diagnostik internal, bukan klaim akurasi out-of-sample.
    split = n if DEPLOY_MODE else int((1 - TEST_FRAC) * n)

    X_tr_full, X_te = X_all[:split], X_all[split:]
    T_tr_full, T_te = T_all[:split], T_all[split:]
    Y_tr_full, Y_te = Y_all[:split], Y_all[split:]
    H_t_te, time_te = H_t_all[split:], time_all[split:]

    # Drop features that are too collinear with the Temperature treatment
    drop_cols = []
    dfX_tr = pd.DataFrame(X_tr_full, columns=conf_cols)

    for col in conf_cols:
        try:
            c = np.corrcoef(dfX_tr[col], T_tr_full)[0, 1]
            if np.isfinite(c) and abs(float(c)) >= COLLINEAR_THR:
                drop_cols.append(col)
        except Exception:
            pass

    if drop_cols:
        conf_cols = [c for c in conf_cols if c not in drop_cols]

        X_df = (
            df[conf_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        mask = (
            (~X_df.isna().any(axis=1)).to_numpy()
            & T_ser.notna().to_numpy()
            & Y_delta.notna().to_numpy()
            & hum_lvl.notna().to_numpy()
        )

        X_all = X_df.loc[mask].to_numpy(float)
        T_all = T_ser.loc[mask].to_numpy(float)
        Y_all = Y_delta.loc[mask].to_numpy(float)
        H_t_all = hum_lvl.loc[mask].to_numpy(float)
        time_all = df.loc[mask, date_col].to_numpy()

        n = len(X_all)
        if n < (20 if DEPLOY_MODE else 80):
            raise ValueError(
                f"Valid data is too small after dropping collinear features: n={n}. "
                "Use a longer/denser historical CSV."
            )

        split = n if DEPLOY_MODE else int((1 - TEST_FRAC) * n)

        X_tr_full, X_te = X_all[:split], X_all[split:]
        T_tr_full, T_te = T_all[:split], T_all[split:]
        Y_tr_full, Y_te = Y_all[:split], Y_all[split:]
        H_t_te, time_te = H_t_all[split:], time_all[split:]

    # Tail validation split from the training data
    frac_val = DEPLOY_VALIDATION_FRAC if DEPLOY_MODE else 0.2
    split_val = int((1 - frac_val) * len(X_tr_full))

    X_tr, X_val = X_tr_full[:split_val], X_tr_full[split_val:]
    T_tr, T_val = T_tr_full[:split_val], T_tr_full[split_val:]
    Y_tr, Y_val = Y_tr_full[:split_val], Y_tr_full[split_val:]
    H_t_val = H_t_all[:split][split_val:]

    min_tr, min_val, min_te = (12, 4, 0) if DEPLOY_MODE else (50, 10, 10)
    if len(X_tr) < min_tr or len(X_val) < min_val or len(X_te) < min_te:
        raise ValueError(
            f"The train/validation/test split is too small. "
            f"train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}. "
            "Use a longer/denser historical CSV."
        )

    # DEPLOY_MODE: tidak ada test set terpisah. Partisi validasi dipakai
    # ulang sebagai test set hanya untuk diagnostik internal (guardrail dan
    # metrik), BUKAN sebagai klaim akurasi out-of-sample. Evaluasi yang sah
    # datang dari prediksi live yang tercatat di file log CSV.
    if DEPLOY_MODE and len(X_te) == 0:
        X_te, T_te, Y_te = X_val, T_val, Y_val
        H_t_te = H_t_val
        time_te = time_all[:split][split_val:]

    # Discrete treatment: high vs low temperature based on the training median
    thr = float(np.percentile(T_tr, 50))
    Tb_tr = (T_tr >= thr).astype(int)
    Tb_val = (T_val >= thr).astype(int)
    Tb_te = (T_te >= thr).astype(int)

    n_tr = len(X_tr)
    n_splits = min(5, max(2, n_tr // 40))

    # Scale forest size down for large training sets so training finishes in
    # reasonable time without exhausting memory/CPU. Smaller datasets keep
    # the original, more thorough sizing.
    if n_tr < 500:
        min_leaf = 10
        n_estimators_cf = 800
        base_rf_estimators = 600
        max_depth = 12
    elif n_tr < 3000:
        min_leaf = 20
        n_estimators_cf = 1500
        base_rf_estimators = 600
        max_depth = 12
    else:
        min_leaf = max(30, n_tr // 300)
        n_estimators_cf = 300
        base_rf_estimators = 200
        max_depth = 10
        n_splits = min(n_splits, 3)

    tscv = build_treatment_cv(Tb_tr, max_splits=n_splits)

    cf = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=base_rf_estimators,
            min_samples_leaf=min_leaf,
            random_state=SEED,
            n_jobs=-1,
            max_depth=max_depth,
        ),
        model_t=RandomForestClassifier(
            n_estimators=base_rf_estimators,
            min_samples_leaf=min_leaf,
            random_state=SEED + 1,
            n_jobs=-1,
            max_depth=max_depth,
        ),
        n_estimators=n_estimators_cf,
        min_samples_leaf=min_leaf,
        cv=tscv,
        random_state=SEED,
        n_jobs=-1,
        honest=True,
        discrete_treatment=True,
    )

    cf.fit(Y_tr, Tb_tr, X=X_tr, W=X_tr)

    tau_tr = cf.effect(X_tr)
    tau_val = cf.effect(X_val)
    tau_te = cf.effect(X_te)

    p_ref = float(np.mean(Tb_tr))
    Y_star_tr = Y_tr - tau_tr * (Tb_tr - p_ref)

    # Baseline model for predicted humidity change
    g = HistGradientBoostingRegressor(
        max_iter=1200,
        learning_rate=0.04,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )

    g.fit(X_tr, Y_star_tr)

    y_val_star = g.predict(X_val) + tau_val * (Tb_val - p_ref)
    y_te_star = g.predict(X_te) + tau_te * (Tb_te - p_ref)

    Y_crf_val = H_t_val + y_val_star
    Y_crf_te = H_t_te + y_te_star
    Y_true_val = H_t_val + Y_val
    Y_true_te = H_t_te + Y_te

    Y_crf_val = np.clip(Y_crf_val, 0, 100)
    Y_crf_te = np.clip(Y_crf_te, 0, 100)
    Y_true_val = np.clip(Y_true_val, 0, 100)
    Y_true_te = np.clip(Y_true_te, 0, 100)

    Y_naive_val = H_t_val
    Y_naive_te = H_t_te

    tau_thr = float(np.quantile(np.abs(tau_val), TAU_Q))
    alpha_te = np.where(np.abs(tau_te) >= tau_thr, ALPHA_HIGH, ALPHA_MED)

    Y_blend_te = alpha_te * Y_naive_te + (1 - alpha_te) * Y_crf_te

    tol = GUARDRAIL / 100.0
    mae_blend = mean_absolute_error(Y_true_te, Y_blend_te)
    mae_naive = mean_absolute_error(Y_true_te, Y_naive_te)

    if mae_blend > (1.0 + tol) * mae_naive:
        grid = np.linspace(0.0, 1.0, 51)
        best_alpha, best_mae = 0.0, 1e9

        for a in grid:
            yb = a * Y_naive_val + (1 - a) * Y_crf_val
            m = mean_absolute_error(Y_true_val, yb)
            if m < best_mae:
                best_mae, best_alpha = m, float(a)

        yb2 = best_alpha * Y_naive_te + (1 - best_alpha) * Y_crf_te

        if mean_absolute_error(Y_true_te, yb2) <= mae_naive:
            Y_blend_te = yb2
        else:
            Y_blend_te = Y_naive_te

    Y_blend_te = np.clip(Y_blend_te, 0, 100)

    actual_col = f"Humidity_actual_t+{horizon_steps}"

    out = pd.DataFrame(
        {
            "timestamp_t": time_te,
            actual_col: Y_true_te,
            "Humidity_t_naive": Y_naive_te,
            "Humidity_pred_CRFonly": Y_crf_te,
            "Humidity_pred_final": Y_blend_te,
            "tau_effect_Temperature": tau_te,
            "Temperature_t": T_te,
            "Temperature_high_threshold": thr,
        }
    )

    mets = {
        "MAE_final": float(mean_absolute_error(out[actual_col], out["Humidity_pred_final"])),
        "SMAPE_final": smape(out[actual_col], out["Humidity_pred_final"]),
        "MAE_naive": float(mean_absolute_error(out[actual_col], out["Humidity_t_naive"])),
        "SMAPE_naive": smape(out[actual_col], out["Humidity_t_naive"]),
        "MAE_CRFonly": float(mean_absolute_error(out[actual_col], out["Humidity_pred_CRFonly"])),
        "SMAPE_CRFonly": smape(out[actual_col], out["Humidity_pred_CRFonly"]),
    }

    # Causal diagnostics
    kf = KFold(n_splits=5, shuffle=False)
    oof_T = np.zeros_like(T_tr_full)
    diag_estimators = 350 if n_tr < 3000 else 150
    diag_min_leaf = 10 if n_tr < 3000 else max(20, n_tr // 300)

    for tr, va in kf.split(X_tr_full):
        mt = RandomForestRegressor(
            n_estimators=diag_estimators,
            min_samples_leaf=diag_min_leaf,
            random_state=SEED,
            n_jobs=-1,
        )
        mt.fit(X_tr_full[tr], T_tr_full[tr])
        oof_T[va] = mt.predict(X_tr_full[va])

    R2_TX = float(r2_score(T_tr_full, oof_T))

    cate = out["tau_effect_Temperature"].to_numpy()
    ate = float(np.mean(cate))

    rng = np.random.default_rng(SEED)
    boot = [np.mean(rng.choice(cate, size=len(cate), replace=True)) for _ in range(600)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    meta = {
        "R2_T|X_OOF": R2_TX,
        "ATE_mean_test": ate,
        "ATE_CI95": (float(lo), float(hi)),
        "tau_thr": float(tau_thr),
        "treatment_threshold_Temperature": float(thr),
        "n_clean_raw": int(len(df_clean)),
        "n_resampled": int(len(df)),
        "n_model": int(n),
        "dropped_collinear_features": drop_cols,
        "confounder_count": int(len(conf_cols)),
        "frequency": freq,
        "horizon_steps": int(horizon_steps),
        "lags_used": lags,
        "rolls_used": rolls,
        "actual_col": actual_col,
    }

    X_complete = X_df.loc[mask].copy()
    # Faktor kalibrasi amplitudo: dihitung dari partisi VALIDASI (bukan test),
    # agar tidak membocorkan informasi test set. Rasio simpangan baku
    # perubahan nyata terhadap perubahan yang diprediksi menunjukkan seberapa
    # besar model menyusutkan tebakannya.
    delta_pred_val = y_val_star
    delta_true_val = Y_val
    if delta_pred_val.std() > 1e-9:
        calib = float(delta_true_val.std() / delta_pred_val.std())
        calib = float(np.clip(calib, 1.0, CALIBRATION_CAP))
    else:
        calib = 1.0

    # --- Baseline pembanding: dilatih pada X dan target LEVEL yang sama ---
    # Catatan: CRF memprediksi PERUBAHAN humidity lalu ditambahkan ke nilai
    # sekarang, sedangkan baseline di bawah memprediksi LEVEL humidity
    # berikutnya secara langsung. Keduanya melihat matriks fitur yang sama,
    # sehingga perbandingannya tetap adil.
    baselines = {}
    causal_models = {}
    if PUBLISH_BASELINES:
        Y_lvl_tr = H_t_all[:split][:split_val] + Y_tr
        try:
            for nm, mdl in [
                ("linear", LinearRegression()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                ("rf", RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=3,
                    random_state=SEED, n_jobs=-1)),
                ("gb", HistGradientBoostingRegressor(
                    max_iter=500, learning_rate=0.05, random_state=SEED)),
            ]:
                mdl.fit(X_tr, Y_lvl_tr)
                baselines[nm] = mdl
        except Exception as e:
            log.warning("Gagal melatih baseline pembanding: %s", e)

        # --- Pembanding KAUSAL ---
        # Ketiganya memprediksi PERUBAHAN humidity (seperti CRF), lalu
        # ditambahkan ke nilai sekarang saat inferensi. Disimpan bersama
        # p_ref dan threshold agar rekonstruksinya identik dengan CRF.
        causal_specs = [
            ("lineardml", lambda: LinearDML(
                model_y=RandomForestRegressor(
                    n_estimators=base_rf_estimators, min_samples_leaf=min_leaf,
                    random_state=SEED, n_jobs=-1, max_depth=max_depth),
                model_t=RandomForestClassifier(
                    n_estimators=base_rf_estimators, min_samples_leaf=min_leaf,
                    random_state=SEED + 1, n_jobs=-1, max_depth=max_depth),
                discrete_treatment=True, cv=tscv, random_state=SEED)),
            ("drlearner", lambda: DRLearner(
                model_propensity=RandomForestClassifier(
                    n_estimators=base_rf_estimators, min_samples_leaf=min_leaf,
                    random_state=SEED + 2, n_jobs=-1, max_depth=max_depth),
                model_regression=RandomForestRegressor(
                    n_estimators=base_rf_estimators, min_samples_leaf=min_leaf,
                    random_state=SEED + 3, n_jobs=-1, max_depth=max_depth),
                cv=tscv, random_state=SEED)),
            ("tlearner", lambda: TLearner(
                models=RandomForestRegressor(
                    n_estimators=base_rf_estimators, min_samples_leaf=min_leaf,
                    random_state=SEED + 4, n_jobs=-1, max_depth=max_depth))),
        ]
        for nm, ctor in causal_specs:
            try:
                est = ctor()
                est.fit(Y_tr, Tb_tr, X=X_tr)
                # Model outcome pendamping: memetakan fitur -> perubahan
                # setelah efek treatment diresidualkan, sama seperti g pada CRF.
                tau_c = est.effect(X_tr)
                g_c = HistGradientBoostingRegressor(
                    max_iter=600, learning_rate=0.05, random_state=SEED)
                g_c.fit(X_tr, Y_tr - tau_c * (Tb_tr - p_ref))
                causal_models[nm] = (est, g_c)
            except Exception as e:
                log.warning("Gagal melatih pembanding kausal %s: %s", nm, str(e)[:80])

    crf_bundle = {
        "cf": cf,
        "baselines": baselines,
        "causal_models": causal_models,
        "g": g,
        "p_ref": p_ref,
        "thr": thr,
        "tau_thr": tau_thr,
        "calibration_factor": calib,
        "conf_cols": conf_cols,
        "feature_medians": X_complete.median(numeric_only=True).to_dict(),
        "last_feature_row": X_complete.iloc[-1].to_dict(),
        "date_col": date_col,
        "freq": freq,
        "horizon_steps": int(horizon_steps),
        "lags_used": lags,
        "rolls_used": rolls,
    }

    return out, mets, meta, df, crf_bundle


# ============================================================
# Live telemetry buffer (fed by MQTT) + feature engineering
# ============================================================
class TelemetryBuffer:
    """Rolling in-memory buffer of raw MQTT telemetry readings."""

    def __init__(self, buffer_minutes: int = BUFFER_MINUTES):
        self.max_age = pd.Timedelta(minutes=buffer_minutes)
        self.records = collections.deque()
        self.lock = threading.Lock()

    def add(self, payload: dict, ts=None):
        # Fallback only (poll_and_predict always passes an explicit,
        # already-JST-converted reading_time) — kept JST-consistent too.
        ts = now_local() if ts is None else ts
        rec = {DATE_COL: ts}
        for c in WANTED:
            if c in payload:
                rec[c] = payload[c]

        with self.lock:
            self.records.append(rec)
            cutoff = ts - self.max_age
            while self.records and self.records[0][DATE_COL] < cutoff:
                self.records.popleft()

    def snapshot(self):
        with self.lock:
            return list(self.records)

    def __len__(self):
        with self.lock:
            return len(self.records)


def build_live_feature_row(buffer: TelemetryBuffer, crf_bundle: dict):
    """Build the latest feature row from buffered raw MQTT readings, using
    the SAME cleaning/resampling/lag/rolling pipeline as training. Returns
    None if there isn't enough buffered history yet.
    """
    records = buffer.snapshot()
    if len(records) < MIN_RAW_READINGS_TO_PREDICT:
        return None

    date_col = crf_bundle["date_col"]
    freq = crf_bundle["freq"]
    lags = crf_bundle["lags_used"]
    rolls = crf_bundle["rolls_used"]
    conf_cols = crf_bundle["conf_cols"]

    df_raw = pd.DataFrame(records)
    df_clean = clean_iot_data(df_raw, date_col)
    if df_clean.empty:
        return None

    df_rs = resample_iot_data(df_clean, date_col, freq)
    if df_rs.empty or TARGET_COL not in df_rs.columns or TREAT_COL not in df_rs.columns:
        return None

    df_rs["hour"] = df_rs[date_col].dt.hour
    df_rs["dayofweek"] = df_rs[date_col].dt.dayofweek
    df_rs["month"] = df_rs[date_col].dt.month
    df_rs["dayofyear"] = df_rs[date_col].dt.dayofyear
    df_rs["hour_sin"] = np.sin(2 * np.pi * df_rs["hour"] / 24.0)
    df_rs["hour_cos"] = np.cos(2 * np.pi * df_rs["hour"] / 24.0)
    df_rs["doy_sin"] = np.sin(2 * np.pi * df_rs["dayofyear"] / 365.0)
    df_rs["doy_cos"] = np.cos(2 * np.pi * df_rs["dayofyear"] / 365.0)

    lag_cols = [c for c in KEY_FORECAST_COLS if c in df_rs.columns]
    add_lags(df_rs, lag_cols, lags=lags)
    add_rolls(df_rs, lag_cols, windows=rolls)

    last_row = df_rs.iloc[-1]
    if pd.isna(last_row[TARGET_COL]) or pd.isna(last_row[TREAT_COL]):
        return None

    current_hum = float(last_row[TARGET_COL])
    current_temp = float(last_row[TREAT_COL])
    medians = crf_bundle["feature_medians"]

    values = []
    for c in conf_cols:
        v = last_row[c] if c in df_rs.columns else np.nan
        if pd.isna(v):
            v = medians.get(c, 0.0)
        values.append(float(v))

    # Humidity pada bin SEBELUMNYA — dipakai baseline "drift" (random walk
    # with drift) untuk memperkirakan kecenderungan arah pergerakan.
    prev_hum = np.nan
    if len(df_rs) >= 2:
        pv = df_rs.iloc[-2][TARGET_COL]
        if not pd.isna(pv):
            prev_hum = float(pv)

    x_row = np.array([values], dtype=float)
    return x_row, current_hum, current_temp, last_row[date_col], prev_hum


def predict_live_humidity(crf_bundle: dict, x_row: np.ndarray, current_hum: float,
                          current_temp: float, prev_hum: float = np.nan):
    """Forecast Humidity PREDICT_HORIZON_STEPS ahead from one live feature row.

    Dua saklar mengatur seberapa "berani" keluarannya (lihat blok konfigurasi
    Mode keluaran prediksi di atas): CALIBRATE_AMPLITUDE mengembalikan besaran
    perubahan ke skala nyata, BLEND_WITH_NAIVE menentukan apakah hasil masih
    dijangkarkan ke nilai sekarang.
    """
    cf = crf_bundle["cf"]
    g = crf_bundle["g"]
    p_ref = crf_bundle["p_ref"]
    thr = crf_bundle["thr"]
    tau_thr = crf_bundle["tau_thr"]
    calib = crf_bundle.get("calibration_factor", 1.0)

    tau = float(cf.effect(x_row)[0])
    treatment_bin = int(current_temp >= thr)

    y_star = float(g.predict(x_row)[0])
    delta_raw = y_star + tau * (treatment_bin - p_ref)
    delta_pred = delta_raw * calib if CALIBRATE_AMPLITUDE else delta_raw

    crf_pred = float(np.clip(current_hum + delta_pred, 0, 100))
    naive_pred = current_hum

    if BLEND_WITH_NAIVE:
        alpha = ALPHA_HIGH if abs(tau) >= tau_thr else ALPHA_MED
        final = float(np.clip(alpha * naive_pred + (1 - alpha) * crf_pred, 0, 100))
    else:
        final = crf_pred

    # Prediksi baseline pembanding pada baris fitur yang sama.
    # Naive tidak butuh model: menurut definisinya, nilai sekarang.
    baseline_preds = {"naive": float(np.clip(current_hum, 0, 100))}
    if np.isfinite(prev_hum):
        baseline_preds["drift"] = float(
            np.clip(current_hum + (current_hum - prev_hum), 0, 100))
    for nm, mdl in (crf_bundle.get("baselines") or {}).items():
        try:
            baseline_preds[nm] = float(np.clip(mdl.predict(x_row)[0], 0, 100))
        except Exception:
            pass

    # Pembanding kausal: direkonstruksi persis seperti CRF, yaitu
    # nilai_sekarang + g(x) + tau(x) * (treatment - p_ref).
    for nm, pair in (crf_bundle.get("causal_models") or {}).items():
        try:
            est, g_c = pair
            tau_c = float(est.effect(x_row)[0])
            delta_c = float(g_c.predict(x_row)[0]) + tau_c * (treatment_bin - p_ref)
            baseline_preds[nm] = float(np.clip(current_hum + delta_c, 0, 100))
        except Exception:
            pass

    return {
        "hum_pred": final,
        "baseline_preds": baseline_preds,
        "hum_pred_crf_only": crf_pred,
        "current_hum": current_hum,
        "current_temp": current_temp,
        "tau": tau,
        "delta_pred": delta_pred,
        "delta_raw": delta_raw,
        "calibration_factor": calib,
        "treatment_bin": treatment_bin,
        "threshold_temp": thr,
    }


# ============================================================
# Polling loop (reads AND writes via ThingsBoard HTTP/REST — no MQTT at all,
# so there's no session to fight the ESP32's MQTT connection for)
# ============================================================
buffer = TelemetryBuffer(BUFFER_MINUTES)
crf_bundle = None  # populated in main() before the polling loop starts


_pred_log_lock = threading.Lock()
_PRED_LOG_FIELDS = [
    "logged_at",
    "reading_time",     # waktu pembacaan sensor mentah
    "feature_time",     # bin 15-menit hasil resample yang jadi dasar fitur
    "UV", "Temperature", "PM2_5", "PM10", "Humidity", "BarPressure",
    "target_time",      # waktu target prediksi (feature_time + 15 menit)
    "hum_pred", "hum_pred_crf_only",
    "pred_naive", "pred_drift", "pred_linear", "pred_ridge", "pred_rf", "pred_gb",
    "pred_lineardml", "pred_drlearner", "pred_tlearner",
    "current_hum", "current_temp",
    "tau", "delta_raw", "delta_pred", "calibration_factor",
    "treatment_bin", "threshold_temp", "published",
]


def log_prediction_to_csv(reading_time, feature_time, values: dict, target_time,
                          result: dict, published: bool):
    """Append one prediction event to a local CSV — includes the raw sensor
    values, the raw reading timestamp, the resampled feature bin the model
    actually used, the forecast target timestamp (feature bin + horizon),
    and the prediction at full float precision.
    """
    row = {
        "logged_at": now_local().isoformat(),
        "reading_time": reading_time,
        "feature_time": feature_time,
        "target_time": target_time,
        "hum_pred": result["hum_pred"],
        "hum_pred_crf_only": result["hum_pred_crf_only"],
        "pred_naive": (result.get("baseline_preds") or {}).get("naive"),
        "pred_drift": (result.get("baseline_preds") or {}).get("drift"),
        "pred_ridge": (result.get("baseline_preds") or {}).get("ridge"),
        "pred_lineardml": (result.get("baseline_preds") or {}).get("lineardml"),
        "pred_drlearner": (result.get("baseline_preds") or {}).get("drlearner"),
        "pred_tlearner": (result.get("baseline_preds") or {}).get("tlearner"),
        "pred_linear": (result.get("baseline_preds") or {}).get("linear"),
        "pred_rf": (result.get("baseline_preds") or {}).get("rf"),
        "pred_gb": (result.get("baseline_preds") or {}).get("gb"),
        "current_hum": result["current_hum"],
        "current_temp": result["current_temp"],
        "tau": result["tau"],
        "delta_raw": result.get("delta_raw"),
        "delta_pred": result["delta_pred"],
        "calibration_factor": result.get("calibration_factor"),
        "treatment_bin": result["treatment_bin"],
        "threshold_temp": result["threshold_temp"],
        "published": published,
    }
    for c in WANTED:
        row[c] = values.get(c)

    with _pred_log_lock:
        file_exists = os.path.isfile(PRED_LOG_CSV_PATH)
        with open(PRED_LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_PRED_LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def publish_prediction_http(hum_pred: float, target_time: pd.Timestamp,
                            baseline_preds: dict = None):
    """POST the forecast to ThingsBoard's HTTP device telemetry API.
    Stateless — no persistent connection, so it can't collide with the
    ESP32's MQTT session on the same device token.

    target_time (reading_time + PREDICT_HORIZON minutes) is used as-is, so
    the published timestamp is anchored to the sensor reading the forecast
    is based on — not to whenever the script happens to get around to
    processing/publishing it. This keeps the published ts, the CSV log, and
    the console log all consistent with each other.

    hum_pred is published rounded to 2 decimal places.
    """
    # target_time is naive but its VALUE represents LOCAL_TIMEZONE wall-clock
    # (JST). Explicitly localize before computing epoch millis — otherwise
    # .timestamp() on a naive Timestamp silently uses the machine's own
    # system timezone (which may not be JST), shifting the epoch by whatever
    # offset separates the two zones.
    target_ms = int(target_time.tz_localize(LOCAL_TIMEZONE).timestamp() * 1000)

    # Semua model dikirim dalam SATU permintaan dengan timestamp yang sama,
    # sehingga di dashboard semuanya sejajar tepat pada waktu target dan
    # dapat dibandingkan langsung dengan nilai aktual yang menyusul.
    payload = {PRED_KEY: round(hum_pred, 2)}
    if PUBLISH_BASELINES and baseline_preds:
        for nm, val in baseline_preds.items():
            key = BASELINE_KEYS.get(nm)
            if key:
                payload[key] = round(float(val), 2)
    try:
        r = requests.post(
            TB_TELEMETRY_HTTP_URL,
            json={"ts": target_ms, "values": payload},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("Failed to publish hum_pred via HTTP: %s", e)
        return False


def process_new_reading(values: dict, reading_time: pd.Timestamp):
    buffer.add(values, ts=reading_time)
    log.info("New reading @ %s: %s", reading_time, values)

    try:
        feats = build_live_feature_row(buffer, crf_bundle)
    except Exception as e:
        log.exception("Failed to build live features: %s", e)
        return

    if feats is None:
        log.info(
            "Buffered %d raw readings, not enough resampled history yet to forecast.",
            len(buffer),
        )
        return

    # feature_time is the resampled bin the features were computed from; it
    # defines the forecast target (see below). reading_time is the raw sensor
    # timestamp, kept for logging so the CSV records both.
    x_row, current_hum, current_temp, feature_time, prev_hum = feats

    try:
        result = predict_live_humidity(crf_bundle, x_row, current_hum, current_temp, prev_hum)
    except Exception as e:
        log.exception("Prediction failed: %s", e)
        return

    # The model's target is defined relative to the RESAMPLED BIN the features
    # came from, not the raw reading time. Features at bin i predict humidity
    # at bin i+1, so the forecast target is feature_time + horizon. Anchoring
    # to reading_time instead would push the published timestamp up to a full
    # bin-width too far into the future (e.g. a 13:29 reading falls in the
    # 13:15 bin, whose target is 13:30 — not 13:44), which shows up on the
    # dashboard as the forecast lagging the actual by roughly two horizons
    # instead of one.
    target_time = feature_time + pd.Timedelta(minutes=HORIZON_MINUTES)
    ok = publish_prediction_http(result["hum_pred"], target_time,
                                 result.get("baseline_preds"))

    try:
        log_prediction_to_csv(reading_time, feature_time, values, target_time, result, ok)
    except Exception as e:
        log.warning("Failed to write prediction to CSV log: %s", e)

    log.info(
        "reading %s (bin %s) -> hum_pred for %s (+%s) = %.2f%% (current=%.2f%%, tau=%.3f, temp=%.2f) %s",
        reading_time, feature_time, target_time, PREDICT_HORIZON_LABEL, result["hum_pred"],
        current_hum, result["tau"], current_temp,
        "[published]" if ok else "[publish FAILED]",
    )
    bp = result.get("baseline_preds") or {}
    if bp:
        log.info(
            "    pembanding -> naive=%.2f  drift=%s  linear=%s  ridge=%s  rf=%s  gb=%s",
            bp.get("naive", float("nan")),
            f"{bp['drift']:.2f}" if "drift" in bp else "-",
            f"{bp['linear']:.2f}" if "linear" in bp else "-",
            f"{bp['ridge']:.2f}" if "ridge" in bp else "-",
            f"{bp['rf']:.2f}" if "rf" in bp else "-",
            f"{bp['gb']:.2f}" if "gb" in bp else "-",
        )
        log.info(
            "    kausal     -> lineardml=%s  drlearner=%s  tlearner=%s",
            f"{bp['lineardml']:.2f}" if "lineardml" in bp else "-",
            f"{bp['drlearner']:.2f}" if "drlearner" in bp else "-",
            f"{bp['tlearner']:.2f}" if "tlearner" in bp else "-",
        )


def retrain_from_thingsboard(tb_client: ThingsBoardRestClient) -> bool:
    """Latih ulang model dari SELURUH riwayat telemetry di ThingsBoard.

    Mengembalikan True bila berhasil. Bila gagal karena apa pun (jaringan,
    data belum cukup, kesalahan pelatihan), model LAMA tetap dipakai dan
    layanan terus berjalan — pelatihan ulang tidak boleh menjatuhkan sistem
    yang sudah berfungsi.
    """
    global crf_bundle
    try:
        log.info("Pelatihan ulang: mengunduh riwayat telemetry...")
        df_hist = tb_client.get_history_telemetry(WANTED)
        if df_hist.empty:
            log.warning("Pelatihan ulang dibatalkan: riwayat kosong.")
            return False

        log.info("Riwayat terunduh: %d baris. Melatih ulang...", len(df_hist))
        _, metrics, meta, _, new_bundle = train_iot_crf(
            df_hist, DATE_COL,
            freq=PREDICT_FREQ, horizon_steps=PREDICT_HORIZON_STEPS,
        )
        # Penukaran atomik: bundle lama baru dilepas setelah yang baru siap,
        # sehingga prediksi tidak pernah memakai model setengah jadi.
        crf_bundle = new_bundle
        log.info(
            "Pelatihan ulang SELESAI: n_model=%d, MAE=%.3f, ATE=%.3f",
            meta["n_model"], metrics["MAE_final"], meta["ATE_mean_test"],
        )
        return True
    except Exception as e:
        log.warning("Pelatihan ulang GAGAL (%s). Model lama tetap dipakai.",
                    str(e)[:120])
        return False


def poll_and_predict(tb_client: ThingsBoardRestClient):
    last_ts_seen = None
    last_retrain = now_local()
    if RETRAIN_ENABLED:
        log.info("Pelatihan ulang berkala aktif: tiap %d jam.",
                 RETRAIN_INTERVAL_HOURS)
    log.info(
        "Polling ThingsBoard REST API every %ds for device '%s'...",
        POLL_INTERVAL_SECONDS, TB_DEVICE_NAME,
    )

    while True:
        try:
            values, ts_ms = tb_client.get_latest_telemetry(WANTED)
        except Exception as e:
            log.warning("Failed to poll ThingsBoard REST telemetry: %s", e)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if RETRAIN_ENABLED:
            jam_berlalu = (now_local() - last_retrain).total_seconds() / 3600.0
            if jam_berlalu >= RETRAIN_INTERVAL_HOURS:
                retrain_from_thingsboard(tb_client)
                last_retrain = now_local()

        if ts_ms is not None and ts_ms != last_ts_seen and values:
            last_ts_seen = ts_ms
            reading_time = (
                pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                .tz_convert(LOCAL_TIMEZONE)
                .tz_localize(None)
            )
            process_new_reading(values, reading_time)

        time.sleep(POLL_INTERVAL_SECONDS)


# ============================================================
# Startup: train the model once from the fixed historical CSV
# ============================================================
def train_model_from_csv():
    log.info("Loading historical dataset: %s", CSV_PATH)
    df_raw = pd.read_csv(CSV_PATH)

    log.info(
        "Training CRF pipeline (horizon=%s, freq=%s, steps=%d)...",
        PREDICT_HORIZON_LABEL, PREDICT_FREQ, PREDICT_HORIZON_STEPS,
    )
    out_df, metrics, meta, df_model, crf_bundle = train_iot_crf(
        df_raw, DATE_COL, freq=PREDICT_FREQ, horizon_steps=PREDICT_HORIZON_STEPS
    )

    log.info(
        "Model trained: n_model=%d MAE_final=%.3f MAE_naive=%.3f SMAPE_final=%.2f%%",
        meta["n_model"], metrics["MAE_final"], metrics["MAE_naive"], metrics["SMAPE_final"],
    )
    log.info(
        "ATE(Temperature high vs low)=%.4f  95%% CI=(%.4f, %.4f)  R2(T|X)_OOF=%.3f",
        meta["ATE_mean_test"], meta["ATE_CI95"][0], meta["ATE_CI95"][1], meta["R2_T|X_OOF"],
    )

    return crf_bundle


def main():
    global crf_bundle
    log.info(
        "=== PROFIL AKTIF: %s === horizon=%s, bin=%s x %d langkah, key='%s'",
        PROFILE_NAME, PREDICT_HORIZON_LABEL, PREDICT_FREQ,
        PREDICT_HORIZON_STEPS, PRED_KEY,
    )
    log.info(
        "Mode keluaran: blend_naive=%s, kalibrasi_amplitudo=%s",
        BLEND_WITH_NAIVE, CALIBRATE_AMPLITUDE,
    )
    crf_bundle = train_model_from_csv()

    tb_client = ThingsBoardRestClient(TB_BASE_URL, TB_USERNAME, TB_PASSWORD)
    tb_client.login()
    tb_client.get_device_id(TB_DEVICE_NAME)

    log.info("Publishing forecasts via HTTP to: %s", TB_TELEMETRY_HTTP_URL)
    log.info("Logging every prediction (full precision) to: %s", PRED_LOG_CSV_PATH)

    try:
        poll_and_predict(tb_client)
    except KeyboardInterrupt:
        log.info("Stopping (KeyboardInterrupt)...")


if __name__ == "__main__":
    main()
