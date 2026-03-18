# Grisounet — Project Guide

> Methane time-series forecasting for underground mining environments.
> Consolidates: SETUP_GUIDE, QUERY_GUIDE, TA_WALKTHROUGH, MM256_TEAMMATE_GUIDE, GUIDE_THIRD_PARTY.
> Last updated: 2026-03-18

---

## 1. What is Grisounet?

Grisounet is a **methane concentration forecasting system** for underground mines. Given 180 seconds of sensor readings (methane, airflow, temperature, humidity, pressure, motor currents), it predicts methane concentration for the next **120 seconds** using an LSTM encoder-decoder neural network.

### Why it matters

- **Safety**: early warning of dangerous methane buildups before they reach critical thresholds
- **Conservative by design**: uses **pinball loss (quantile 0.9)** — the model intentionally over-predicts, because a false alarm is far less costly than a missed spike
- **Two pipelines**: a 3-sensor model (MM256 + MM263 + MM264 jointly) and a single-sensor model (MM256 only) for simpler deployment

### Key concepts

| Concept | Detail |
|---------|--------|
| **Input window** | 180 seconds of sensor data (1 reading/second) |
| **Forecast horizon** | 120 seconds into the future |
| **Sensors** | MM256, MM263, MM264 (methane), plus airflow, temperature, humidity, pressure, motor current |
| **Alert trigger** | Sequences built around moments when any sensor exceeds the alert threshold |
| **Model type** | LSTM encoder-decoder with quantile regression |
| **Loss function** | Pinball loss (quantile=0.9) — biases predictions upward for safety |
| **Scaling** | MinMaxScaler fitted on training data only (no data leakage) |
| **Versioning** | Models and results are timestamped (`YYYYMMDD_HHMMSS`) — no overwrites, full history preserved |

---

## 2. Infrastructure Overview

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                        │
│                  Data project: spheric-voyager-484810-k0         │
│                  Compute project: grisounet                      │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────┐   ┌──────────────┐ │
│  │   BigQuery (EU)  │    │  Cloud Storage   │   │Secret Manager│ │
│  │   Dataset:       │    │  grisou_bucket   │   │              │ │
│  │   grisou_eu      │    │                  │   │  GCP_PROJECT │ │
│  │                  │    │  /models/         │   │  BQ_DATASET  │ │
│  │  Table_grisou    │    │    model_*.keras  │   │  BQ_TABLE    │ │
│  │  preprocess_*    │    │  /preprocessing/  │   │  BUCKET_NAME │ │
│  │  history_*       │    │    preprocess_*.npz│  │  BQ_REGION   │ │
│  │  predictions_*   │    │                  │   │              │ │
│  └────────┬─────────┘    └────────┬─────────┘   └──────┬───────┘ │
│           │                       │                     │         │
│           ▼                       ▼                     ▼         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Compute Engine VM                        │  │
│  │              grisou-instance (europe-west1-b)              │  │
│  │                   n2-highcpu-32                             │  │
│  │              32 vCPU — 32 GB RAM — Intel Cascade Lake      │  │
│  │         Auto-start 9 AM / Auto-stop 6 PM (Mon–Fri)        │  │
│  │                   IP: 34.77.27.64                          │  │
│  │                                                            │  │
│  │  - Runs training pipelines                                 │  │
│  │  - Writes models → GCS, metrics → BigQuery                │  │
│  │  - Uses Secret Manager (no .env needed)                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      Cloud Run                              │  │
│  │           FastAPI container (api/fast.py)                   │  │
│  │                                                            │  │
│  │  - Serves predictions via REST API                         │  │
│  │  - Loads models from GCS on demand                         │  │
│  │  - Caches data and models in memory                        │  │
│  │  - Scales to zero when idle                                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### What is stored where and why

| Component | What it stores | Why there |
|-----------|---------------|-----------|
| **BigQuery** (`grisou_eu`) | Raw sensor data (`Table_grisou`), preprocessed arrays (`preprocess_*`), training history (`history_*`), predictions (`predictions_*`) | Queryable archive — any team member can SQL into any past run. Central source of truth. |
| **Cloud Storage** (`grisou_bucket`) | Trained Keras models (`/models/`), preprocessing artifacts (`/preprocessing/`) | Binary files that BigQuery can't store efficiently. Versioned by timestamp. |
| **Secret Manager** (`grisounet` project) | GCP_PROJECT, BQ_DATASET, BQ_TABLE, BUCKET_NAME, BQ_REGION | Keeps credentials out of code and `.env` files. VM and Cloud Run read secrets at runtime. |
| **Compute Engine VM** | Nothing persistent — it's a compute worker | Runs training jobs. Ephemeral by design. |
| **Cloud Run** | Nothing persistent — stateless container | Serves the API. Loads models from GCS into memory on first request. Scales to zero. |

### How run history is tracked

Every pipeline run creates **timestamped tables** in BigQuery:

```
preprocess_20260318_120000   ← data after preprocessing
history_20260318_120000      ← loss/val_loss per epoch
predictions_20260318_120000  ← actual vs predicted per sensor
```

Compare any two runs with the `bq_query.py` helper (see section 6).

---

## 3. Setup

### Prerequisites

- Python 3.12+
- GCP credentials with access to the project (ask Quentin for IAM permissions)

### Clone / update the repo

```bash
# First time
git clone git@github.com:ArthurVigier/grisounet.git
cd grisounet

# Already cloned
git checkout master && git pull origin master
```

### Create your local .env

```bash
cat > .env << 'EOF'
GCP_PROJECT=spheric-voyager-484810-k0
GCP_REGION=europe-west1
BUCKET_NAME=grisou_bucket
BQ_REGION=EU
BQ_DATASET=grisou_eu
BQ_TABLE=Table_grisou
EOF
```

This file is gitignored — each person creates their own. **Do not commit it.**

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate

# Pick the right requirements file:
pip install -r requirements/app.txt       # production / runtime only
pip install -r requirements/dev.txt       # contributor tooling
pip install -r requirements/research.txt  # notebooks and exploratory analysis
```

### GCP authentication

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project spheric-voyager-484810-k0
```

---

## 4. Running the Pipelines

### MM256 single-sensor pipeline (current focus)

Run from the repository root:

```bash
# Full workflow (preprocessing + CV + final train + evaluation)
python interface/workflow_mm256.py --mode full --source cache

# CV only
python interface/workflow_mm256.py --mode cv --source cache

# Single training run
python interface/workflow_mm256.py --mode single --source cache
```

**MM256 CLI options:**

| Flag | Effect |
|------|--------|
| `--mode full` | Preprocessing → CV → final train → evaluation |
| `--mode cv` | Cross-validation only |
| `--mode single` | Single training run |
| `--source bq` | Pull fresh data from BigQuery |
| `--source cache` | Use last cached local data |
| `--use-catch22` | Compute catch22 statistical features |
| `--include-secondary-diagnostics` | Extra diagnostic metrics |
| `--save-cv-plots` | Save cross-validation plots |
| `--save-final-analysis` | Save final evaluation plots |
| `--push-bq` | Upload predictions to BigQuery |
| `--n-splits N` | Number of CV folds (default: 5) |
| `--frozen-sensor-window N` | Frozen sensor detection window in seconds (default: 3600) |
| `--sensor-disagreement-z-threshold N` | Co-located sensor disagreement threshold in std devs (default: 6.0) |

**Example — full run with everything enabled:**

```bash
python interface/workflow_mm256.py --mode full --source cache \
  --use-catch22 --save-cv-plots --save-final-analysis --push-bq
```

### 3-sensor pipeline

```bash
# Full pipeline
python -m interface.workflow --source bq     # fresh data from BigQuery
python -m interface.workflow --source cache   # cached data
```

| Flag | Effect |
|------|--------|
| `--source bq` | Pull fresh data from BigQuery |
| `--source cache` | Use last cached local CSV |
| `--source local` | Use a specific local file |
| `--start N` / `--stop N` | Array slicing indices |
| `--cache-raw` | Save raw pull to `results/raw_pulls/` |
| `--upload-preprocess` | Upload preprocessing to GCS |
| `--save-preprocess-bq` | Save preprocessing to BigQuery |

### Pull data only (no training)

```bash
make pull_bq
# or: python -c "from ml_logic.data import pull_data_from_bq; pull_data_from_bq()"
```

### How the MM256 workflow works (step by step)

1. Load the modeling table from BigQuery or cache
2. Run anomaly cleaning (frozen sensor detection: 3600 s, co-located sensor disagreement: 6 σ)
3. Keep only days where MM256 is active
4. Build MM256 windows (180 s input → 120 s forecast)
5. Optionally compute catch22 features from each window
6. Run time-series CV (5-fold by default)
7. Train a final model on the train split
8. Evaluate on the holdout split
9. Save metrics, plots, model artifacts; optionally upload predictions to BigQuery

---

## 5. Using the API

### Start locally

```bash
fastapi dev api/fast.py
# Available at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check → `{"greeting": "Hello"}` |
| `GET` | `/preprocess` | Get preprocessed data (3-sensor) |
| `POST` | `/predict` | Predict methane (3-sensor) |
| `POST` | `/predict_mm256` | Predict methane (MM256 only) |
| `GET` | `/preprocess_mm256` | MM256 preprocessing metadata |
| `GET` | `/predict_mm256/info` | MM256 input/output shapes |
| `POST` | `/reload` | Clear all caches |

### Example: call with Python

```python
import requests, numpy as np

BASE = "http://127.0.0.1:8000"  # or Cloud Run URL

# Health check
requests.get(f"{BASE}/").json()

# Get preprocessed test data
data = requests.get(f"{BASE}/preprocess", params={"start_index": 0, "stop_index": 10}).json()
X_test = np.array(data["X_test"])

# Run a prediction
r = requests.post(f"{BASE}/predict", json={
    "timestamp": "20260318_120000",   # model version to load from GCS
    "X_pred": X_test.tolist()
})
predictions = np.array(r.json()["prediction"])  # shape: (n_samples, 120, 3)
```

### Example: call with cURL

```bash
curl http://127.0.0.1:8000/
curl -X POST http://127.0.0.1:8000/predict_mm256 \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "20260318_120000", "X_pred": [[[0.1, 0.2, 0.3]]]}'
```

---

## 6. Querying BigQuery

Use the `bq_query.py` CLI to inspect any past run without downloading data:

```bash
# List all tables
python scripts/bq_query.py --list

# Show schema and row count
python scripts/bq_query.py --info Table_grisou

# Summary statistics (runs on BQ, not locally)
python scripts/bq_query.py --describe Table_grisou

# Latest results from the most recent run
python scripts/bq_query.py --latest preprocess
python scripts/bq_query.py --latest history
python scripts/bq_query.py --latest predictions

# Custom SQL
python scripts/bq_query.py "SELECT AVG(ABS(residual)) as mae, sensor \
  FROM predictions_20260318_120000 GROUP BY sensor"

# Export to CSV
python scripts/bq_query.py "SELECT * FROM predictions_20260318_120000" \
  --save results/my_export.csv
```

### When to use BigQuery vs local

| Use BigQuery when… | Use local download when… |
|---------------------|--------------------------|
| Quick stats (counts, averages, distributions) | Custom Python/pandas analysis |
| Checking pipeline results | Creating plots or visualizations |
| Exploring data before committing to analysis | Feature engineering experiments |
| Sharing results with teammates | Offline access |

### MM256 prediction query pattern

```sql
SELECT *
FROM `spheric-voyager-484810-k0.grisou_eu.predictions_mm256_final_<timestamp>`
WHERE sensor = 'MM256'
  AND target_date = DATE '2014-05-24'
ORDER BY target_time, forecast_step;
```

Prediction columns include: `forecast_origin_time`, `target_start_time`, `target_time`, `target_end_time`, `target_date`.

---

## 7. Working on the VM

```bash
# SSH in
gcloud compute ssh grisou-instance --project=grisounet --zone=europe-west1-b

# Once connected
cd /home/grisounet
git pull origin master
source venv/bin/activate
pip install -r requirements/app.txt
python interface/workflow_mm256.py --mode full --source cache
```

### VM details

| Property | Value |
|----------|-------|
| Machine type | n2-highcpu-32 (32 vCPU, 32 GB RAM) |
| Zone | europe-west1-b |
| External IP | 34.77.27.64 |
| Schedule | Auto-start 9 AM, auto-stop 6 PM Mon–Fri (Paris time) |
| API port | 8000 (firewall open) |
| Config source | Secret Manager (no .env needed) |
| VM management | Only Quentin can start/stop/resize |

### Start the API on the VM

```bash
uvicorn api.fast:app --host 0.0.0.0 --port 8000
# Accessible at http://34.77.27.64:8000
```

### Secret Manager

```bash
# Read a secret
gcloud secrets versions access latest --secret=GCP_PROJECT --project=grisounet

# Create or update a secret
echo -n "new_value" | gcloud secrets create SECRET_NAME --project=grisounet --data-file=-
```

Available secrets: `GCP_PROJECT`, `BUCKET_NAME`, `BQ_DATASET`, `BQ_TABLE`, `BQ_REGION`.

---

## 8. Deploying to Cloud Run

```bash
# Build
docker build -t grisounet .

# Tag for Artifact Registry
docker tag grisounet europe-west1-docker.pkg.dev/spheric-voyager-484810-k0/grisounet/api:latest

# Push
docker push europe-west1-docker.pkg.dev/spheric-voyager-484810-k0/grisounet/api:latest

# Deploy
gcloud run deploy grisounet-api \
  --image europe-west1-docker.pkg.dev/spheric-voyager-484810-k0/grisounet/api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080
```

The container reads all config from Secret Manager at runtime — no environment variables to set manually.

---

## 9. Repository Structure

```
grisounet/
├── api/
│   └── fast.py                  # FastAPI endpoints (predict, preprocess, reload)
├── interface/
│   ├── workflow.py              # 3-sensor end-to-end pipeline
│   └── workflow_mm256.py        # MM256 single-sensor pipeline (current focus)
├── ml_logic/
│   ├── data.py                  # BigQuery pull, local caching, artifact persistence
│   ├── data_cleaning.py         # Anomaly cleaning utilities
│   ├── preprocessor.py          # Feature engineering, scaling, sequence building
│   ├── model.py                 # 3-sensor LSTM architectures
│   ├── model_mm256.py           # MM256-only LSTM architectures
│   ├── model_save.py            # GCS upload/download for Keras models
│   ├── results_bq_save.py       # Save history/predictions/preprocessing to BigQuery
│   ├── analysis.py              # Plotting helpers (loss curves, predictions vs actual)
│   └── secrets.py               # Secret Manager with .env fallback
├── scripts/
│   ├── bq_query.py              # CLI for querying BigQuery
│   ├── fetch_tables.py          # Download BQ tables as CSV
│   ├── list_resources.py        # Inventory of GCS + BQ resources
│   ├── cv_time_series.py        # Time-series cross-validation
│   ├── preprocessor_MM256.py    # MM256-specific preprocessing
│   ├── train_final_mm256.py     # Final train/test run + plots + optional BQ upload
│   ├── visualize_mm256_day.py   # Day-level visual inspection
│   └── qn_methane_analysis.py   # Methane analysis scripts
├── research/
│   ├── notebooks/               # Exploratory notebooks (gitignored)
│   ├── analysis/                # Analysis scripts and processed artifacts
│   └── references/              # Reference PDFs and background reading
├── model_eval/                  # Model evaluation artifacts
├── results/                     # Runtime outputs (gitignored)
│   ├── cv_metrics/
│   ├── final_metrics/
│   ├── graphs/
│   ├── logs/
│   ├── predictions/
│   └── model_history/
├── requirements/
│   ├── app.txt                  # Production runtime deps
│   ├── dev.txt                  # Development tools
│   ├── research.txt             # Notebook/research deps
│   └── full.txt                 # Legacy all-in-one snapshot
├── Dockerfile                   # Container config for Cloud Run
├── makefile                     # Task automation
├── setup.py
├── requirements.txt             # Top-level requirements (points to app.txt)
└── PROJECT_GUIDE.md             # This file
```

---

## 10. What to Inspect After a Run

| What to check | Where |
|----------------|-------|
| Console logs | `results/logs/` |
| Fold-level CV metrics | `results/cv_metrics/` |
| Final holdout metrics | `results/final_metrics/` |
| Forecast plots, loss curves | `results/graphs/` |
| Predictions data | `results/predictions/` |

### Where graphs are produced

| Graph type | Produced by |
|------------|-------------|
| Final MM256 plots | `scripts/train_final_mm256.py` |
| CV plots | `scripts/cv_time_series.py` |
| Shared plotting helpers | `ml_logic/analysis.py` |
| Day-level inspection | `scripts/visualize_mm256_day.py` |

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| Preprocessing looks wrong | Inspect `scripts/preprocessor_MM256.py` |
| CV metrics look wrong | Inspect `scripts/cv_time_series.py` |
| Final outputs / BQ export look wrong | Inspect `scripts/train_final_mm256.py` |
| Model behavior looks wrong | Inspect `ml_logic/model_mm256.py` |
| Plots are missing | Inspect `ml_logic/analysis.py` and `results/graphs/` |
| BQ queries fail | Check `.env` has `BQ_REGION=EU` and `BQ_DATASET=grisou_eu` |

---

## 12. Git Workflow

- Feature branches for each piece of work
- PRs into master with review
- `.env` is gitignored — each person creates their own locally
- VM synced via `git pull origin master`
- Always pull before pushing: `git pull origin master`

---

## 13. Contacts & Links

| Resource | Link |
|----------|------|
| GitHub repo | https://github.com/ArthurVigier/grisounet |
| BigQuery console | https://console.cloud.google.com/bigquery?project=spheric-voyager-484810-k0&d=grisou_eu |
| Cloud Storage bucket | https://console.cloud.google.com/storage/browser/grisou_bucket |
| VM instance | https://console.cloud.google.com/compute/instancesDetail/zones/europe-west1-b/instances/grisou-instance?project=grisounet |

For GCP access or questions, contact Quentin.
