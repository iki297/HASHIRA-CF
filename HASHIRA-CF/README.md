# HASHIRA-CF

Code and data for **"HASHIRA-CF: A Hybrid Causal Forest Framework for
Short-Horizon Humidity Forecasting in Hydroponic Systems, from Offline Data to
Real-Time IoT Deployment"** (submitted to *Smart Agricultural Technology*).

HASHIRA-CF uses a causal forest as a forecasting engine. Ambient temperature is
treated as a binary treatment on the one-hour-ahead change in relative
humidity; a generalized-random-forest final stage (`CausalForestDML`) estimates
its heterogeneous effect, and a gradient-boosting component models the
remaining variation. The forecast is

    H(t+1) = H(t) + g(X) + tau(X) * (T - p)

## Repository layout

| Folder | Content |
|---|---|
| `hashira_cf/` | `CRF_MQTT_Humidity_Forecasting.py`: the deployed implementation (training, live feature construction, live prediction, ThingsBoard integration). `download_telemetry_with_pred.py`: telemetry export. |
| `analysis/` | Scripts that reproduce the numbers in the paper (see below). `crfmod*.py` are copies of the deployed module with the variant noted. |
| `figures/` | Scripts for the observed-versus-forecast figures and the MASE/scarcity plots. |
| `firmware/` | ESP32 telemetry sketches (1-min, 15-min and default intervals). |
| `data/` | Datasets (see below). |
| `results/` | Reference outputs: offline predictions, scarcity results, collected metrics. |

## Data

| File | Description |
|---|---|
| `kanazawa_outdoor_training.csv` | Hourly outdoor archive for Kanazawa, Japan (offline experiment) |
| `kanazawa_outdoor_full.csv` | Same archive before preprocessing |
| `20260902_telemetry_training.csv` | Indoor hydroponic testbed telemetry used to fit the deployed model |
| `20260913_telemetry_with_all_predictions.csv` | Raw indoor telemetry through 11 Sep 2026 with live predictions |
| `live_eval_10sep.csv` | Live forecasts vs observations; evaluation window 4–10 Sep 2026 (138 hours) |

The outdoor archive was obtained from the Open-Meteo Historical Weather API
(ERA5 reanalysis). Please cite Open-Meteo and ERA5 when reusing it.

## Reproducing the paper

Run each script from inside `analysis/`.

| Script | Reproduces |
|---|---|
| `fair.py` | Offline comparison (Table 3) with baselines given X, X+T and X+Temperature |
| `replay.py`, `dep_tests.py` | Calendar-index replay of the deployment, Diebold–Mariano and Wilcoxon tests (Table 4, Section 4.3) |
| `scarcity.py` | Data-scarcity rolling-origin experiment (Table S2) |
| `thr.py`, `thr_noise.py`, `refit_noise.py` | Overlap-screening ablation (Table S3) and refit ranges |
| `allnums.py` | Collects the reported metrics |

| Module | Variant |
|---|---|
| `crfmod.py` | Deployed code (indoor) |
| `crfmod_neg.py` | Outdoor archive; temperature bound −40 to 60 °C |
| `crfmod_nocal.py` | Month and day-of-year features removed |

Forest fitting uses parallel jobs and is not bit-for-bit reproducible; the paper
reports ranges over refits.

## Environment

- Deployment: Raspberry Pi, Python 3.9
- Offline analysis: Python 3.13
- `pip install -r requirements.txt` (EconML 0.16.0, scikit-learn 1.5.2)

## Running the live service

Set the ThingsBoard host, credentials and device token in
`hashira_cf/CRF_MQTT_Humidity_Forecasting.py` (placeholders such as
`<THINGSBOARD_HOST>`), and the Wi-Fi and MQTT settings in the ESP32 sketch.

## License

Code: MIT (see `LICENSE`).
