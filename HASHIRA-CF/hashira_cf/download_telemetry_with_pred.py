import os
import requests
import pandas as pd
from datetime import datetime, timezone

TB_URL = "http://<THINGSBOARD_HOST>:8080"
USERNAME = "<TB_USERNAME>"
PASSWORD = "<TB_PASSWORD>"
DEVICE_ID = "<DEVICE_ID>"

# Sesuaikan dengan telemetry key milikmu.
SENSOR_KEYS = [
    "UV",
    "Temperature",
    "PM2_5",
    "PM10",
    "PM1_0",
    "Humidity",
    "BarPressure",
]

# Key hasil prediksi (ditambahkan oleh CRF_MQTT_Humidity_Forecasting.py).
PRED_KEYS = [
    "hum_pred_1h",      # CRF (model utama, horizon 1 jam)
    "hum_pred_naive",   # baseline persistence
    "hum_pred_lr",      # baseline Linear Regression
    "hum_pred_rf",      # baseline Random Forest
    "hum_pred_gb",      # baseline Gradient Boosting
    "hum_pred_drift",   # baseline random walk with drift
    "hum_pred_ridge",   # baseline Ridge (linear teregularisasi)
    "hum_pred_lineardml",   # pembanding kausal: DML tahap akhir linear
    "hum_pred_drlearner",   # pembanding kausal: doubly robust
    "hum_pred_tlearner",    # pembanding kausal: meta-learner dua model
    "hum_pred",         # key lama horizon 15 menit (data historis)
]

TELEMETRY_KEYS = SENSOR_KEYS + PRED_KEYS

# Path diselesaikan relatif terhadap lokasi FILE SKRIP INI, bukan direktori
# kerja Python. Tanpa ini, menjalankan lewat IDLE menyimpan file ke folder
# lain sehingga terkesan "tidak tersimpan".
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Nama file mengikuti tanggal unduhan, sehingga setiap pengambilan tersimpan
# terpisah dan unduhan sebelumnya tidak tertimpa.
OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    f"{datetime.now():%Y%m%d}_telemetry_with_all_predictions.csv",
)


def login():
    response = requests.post(
        f"{TB_URL}/api/auth/login",
        json={
            "username": USERNAME,
            "password": PASSWORD,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["token"]


def download_telemetry(token):
    # Ambil semua data dari 1 Januari 2020 sampai waktu sekarang.
    start_ts = int(
        datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    )
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    params = {
        "keys": ",".join(TELEMETRY_KEYS),
        "startTs": start_ts,
        "endTs": end_ts,
        "limit": 1000000,
        "agg": "NONE",
        "orderBy": "ASC",
    }

    response = requests.get(
        (
            f"{TB_URL}/api/plugins/telemetry/"
            f"DEVICE/{DEVICE_ID}/values/timeseries"
        ),
        headers={
            "X-Authorization": f"Bearer {token}",
        },
        params=params,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def convert_to_csv(data):
    rows = {}

    for key, values in data.items():
        for item in values:
            timestamp = int(item["ts"])
            if timestamp not in rows:
                rows[timestamp] = {"timestamp_ms": timestamp}
            rows[timestamp][key] = item["value"]

    dataframe = pd.DataFrame(rows.values())

    if dataframe.empty:
        print("Tidak ada telemetry yang ditemukan.")
        return

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp_ms"],
        unit="ms",
        utc=True,
    ).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

    dataframe = dataframe.sort_values("timestamp")

    columns = [
        "timestamp",
        "timestamp_ms",
        *TELEMETRY_KEYS,
    ]
    columns = [
        column for column in columns
        if column in dataframe.columns
    ]
    dataframe = dataframe[columns]

    dataframe.to_csv(OUTPUT_FILE, index=False)

    print(f"Berhasil menyimpan {len(dataframe)} baris.")
    print(f"File tersimpan di: {OUTPUT_FILE}")
    print(f"Ukuran file      : {os.path.getsize(OUTPUT_FILE):,} byte")
    print(f"Rentang waktu    : {dataframe['timestamp'].min()} s/d {dataframe['timestamp'].max()}")

    # hum_pred punya timestamp TARGET (waktu yang diprediksi, +15 menit dari
    # saat pembacaan sensor dipakai) — bukan timestamp sensor mentahnya.
    # Karena itu, baris berisi Humidity aktual dan baris berisi hum_pred
    # untuk waktu yang "sama" biasanya TIDAK persis satu baris (beda
    # milidetik), melainkan berdekatan di baris yang berlainan. Kalau mau
    # membandingkan hum_pred vs Humidity aktual secara langsung per baris,
    # perlu langkah pencocokan waktu terdekat (nearest-time match)
    # terpisah — beri tahu saya kalau mau dibuatkan skripnya.
    print("\nJumlah baris per key prediksi:")
    for k in PRED_KEYS:
        if k in dataframe.columns:
            print(f"  {k:18s}: {dataframe[k].notna().sum()}")


def main():
    print(f"Menghubungi ThingsBoard di {TB_URL} ...")
    token = login()
    print("Login berhasil. Mengunduh telemetry...")

    telemetry = download_telemetry(token)

    # Diagnostik: tunjukkan key mana yang benar-benar dikembalikan server.
    # Kalau sebuah key tampil KOSONG, berarti memang belum pernah ada datanya
    # di ThingsBoard — bukan kegagalan penyimpanan file.
    print("\nKey yang dikembalikan ThingsBoard:")
    for k in TELEMETRY_KEYS:
        n = len(telemetry.get(k, []))
        tanda = "" if n else "   <-- KOSONG"
        print(f"  {k:18s}: {n:6d} titik{tanda}")

    convert_to_csv(telemetry)


if __name__ == "__main__":
    main()
