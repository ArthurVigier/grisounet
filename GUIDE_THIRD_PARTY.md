# Grisounet — Third-Party Guide

> Methane time-series forecasting: run the model, call the API, understand the infrastructure.
> Last updated: 2026-03-17

---

## 1. What is Grisounet?

Grisounet is a **methane concentration forecasting system** for underground mining environments. Given 180 seconds of sensor readings (methane levels, airflow, temperature, humidity, pressure, motor currents), it predicts methane concentration for the next **120 seconds** using an LSTM encoder-decoder neural network.

### Why it matters

- **Safety**: early warning of dangerous methane buildups before they reach critical thresholds
- **Conservative by design**: uses **pinball loss (quantile 0.9)** — the model intentionally over-predicts rather than under-predicts, because a false alarm is far less costly than a missed spike
- **Two pipelines**: a 3-sensor model (MM256 + MM263 + MM264 jointly) and a single-sensor model (MM256 only) for simpler deployment scenarios

---

## 2. Infrastructure Overview

### Architecture diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                        │
│                  Project: spheric-voyager-484810-k0               │
│                  Secrets Project: grisounet                       │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────┐   ┌──────────────┐ │
│  │   BigQuery (EU)  │    │  Cloud Storage   │   │Secret Manager│ │
│  │   Dataset:       │    │  grisou_bucket   │   │              │ │
│  │   grisou_eu      │    │                  │   │  GCP_PROJECT │ │
│  │                  │    │  /models/         │   │  BQ_DATASET  │ │
│  │  Table_grisou    │    │    model_*.keras  │   │  BQ_TABLE    │ │
│  │  preprocess_*    │    │  /preprocessing/  │   │  BUCKET_NAME │ │
│  │  history_*       │    │    preprocess_*.npz│  │  BQ_REGION   │ │
│  │  predictions_*   │    │                  │   │  GCP_REGION  │ │
│  └────────┬─────────┘    └────────┬─────────┘   └──────┬───────┘ │
│           │                       │                     │         │
│           ▼                       ▼                     ▼         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Compute Engine VM                        │  │
│  │              grisou-instance (europe-west1-b)              │  │
│  │         Auto-start 9AM / Auto-stop 6PM (Mon-Fri)          │  │
│  │                                                            │  │
│  │  - Runs training pipelines (interface/workflow.py)         │  │
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
| **BigQuery** (`grisou_eu`) | Raw sensor data (`Table_grisou`), preprocessed arrays (`preprocess_*`), training history (`history_*`), predictions (`predictions_*`) | Queryable archive — any team member can SQL into any past run without downloading files. Central source of truth for data and experiment tracking. |
| **Cloud Storage** (`grisou_bucket`) | Trained Keras models (`/models/`), preprocessing artifacts (`/preprocessing/`) | Binary files (model weights, numpy arrays) that BigQuery can't store efficiently. Versioned by timestamp. |
| **Secret Manager** (`grisounet`) | Connection strings, project IDs, bucket names | Keeps credentials out of code and `.env` files. VM and Cloud Run read secrets at runtime. |
| **Compute Engine VM** | Nothing persistent — it's a compute worker | Runs training jobs that read from BigQuery and write results back to BigQuery + GCS. Ephemeral by design. |
| **Cloud Run** | Nothing persistent — stateless container | Serves the API. Loads models from GCS into memory on first request, caches them. Scales to zero. |

### How history is tracked

Every pipeline run creates **timestamped tables** in BigQuery:

```
preprocess_20260317_120000   ← what data looked like after preprocessing
history_20260317_120000      ← loss/val_loss per epoch (did the model converge?)
predictions_20260317_120000  ← actual vs predicted per sensor (how good is it?)
```

This means you can always go back and compare any two runs:
```bash
python scripts/bq_query.py --latest predictions   # most recent run
python scripts/bq_query.py --latest history        # training curve
python scripts/bq_query.py --list                  # all tables ever created
```

### Why this architecture makes sense

1. **Separation of compute and storage**: the VM is a training worker, not a data warehouse. If the VM goes down, nothing is lost — everything persists in BigQuery and GCS.
2. **BigQuery as experiment tracker**: instead of a separate MLflow server, every run's inputs, metrics, and outputs are queryable SQL tables. Simple, auditable, no extra infrastructure.
3. **Cloud Run for serving**: scales to zero when nobody is calling the API, scales up automatically under load. No VM cost when idle.
4. **Secret Manager**: one place for all config. No `.env` files to sync across machines, no secrets in git.

---

## 3. Running the Model Locally

### Prerequisites

- Python 3.12+
- GCP credentials with access to the project (ask Quentin for IAM permissions)

### Setup

```bash
# Clone the repo
git clone https://github.com/ArthurVigier/grisounet.git
cd grisounet

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements/app.txt

# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Create local config (.env)
cat > .env << 'EOF'
GCP_PROJECT=spheric-voyager-484810-k0
GCP_REGION=europe-west1
BUCKET_NAME=grisou_bucket
BQ_REGION=EU
BQ_DATASET=grisou_eu
BQ_TABLE=Table_grisou
EOF
```

### Run the 3-sensor pipeline

```bash
# Pull data from BigQuery, preprocess, train, save model + results
python -m interface.workflow --source bq

# Or use cached data from a previous pull
python -m interface.workflow --source cache
```

**CLI options:**
| Flag | Effect |
|------|--------|
| `--source bq` | Pull fresh data from BigQuery |
| `--source cache` | Use last cached local CSV |
| `--source local` | Use a specific local file |
| `--start N` | Start index for array slicing |
| `--stop N` | Stop index for array slicing |
| `--cache-raw` | Save raw pull to `results/raw_pulls/` |
| `--upload-preprocess` | Upload preprocessing to GCS |
| `--save-preprocess-bq` | Save preprocessing to BigQuery |
| `--skip-preprocess-save` | Skip local preprocessing save |

### Run the MM256 single-sensor pipeline

```bash
# Single training run
python interface/workflow_mm256.py --mode single --source cache

# Cross-validation (5-fold time series split)
python interface/workflow_mm256.py --mode cv --n-splits 5 --source cache
```

### Pull data only (no training)

```bash
make data
# or equivalently:
python -c "from ml_logic.data import pull_data_from_bq; pull_data_from_bq()"
```

---

## 4. Using the API

### Start locally

```bash
fastapi dev api/fast.py
# API available at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

### Endpoints

#### Health check

```
GET /
→ {"greeting": "Hello"}
```

#### Get preprocessed data (3-sensor)

```
GET /preprocess?start_index=0&stop_index=100
→ {
    "X_train": [...],   # shape: (n, 180, n_features)
    "y_train": [...],   # shape: (n, 120, 3)
    "X_test":  [...],
    "y_test":  [...]
  }
```

#### Predict methane (3-sensor)

```
POST /predict
Content-Type: application/json

{
  "timestamp": "20260317_120000",
  "X_pred": [...]   // shape: (n_samples, 180, n_features)
}

→ {
    "prediction": [...]   // shape: (n_samples, 120, 3) — 3 sensors × 120s
  }
```

The `timestamp` identifies which model version to load from GCS (e.g. `"20260317_120000"` loads `gs://grisou_bucket/models/model_20260317_120000.keras`).

#### Predict methane (MM256 only)

```
POST /predict_mm256
Content-Type: application/json

{
  "timestamp": "20260317_120000",
  "X_pred": [...]   // shape: (n_samples, 180, n_features)
}

→ {
    "sensor": "MM256",
    "prediction": [...]   // shape: (n_samples, 120, 1)
  }
```

#### Get MM256 preprocessing metadata

```
GET /preprocess_mm256
→ {
    "target_sensor": "MM256",
    "concentration_threshold": ...,
    "n_active_days": ...,
    "n_active_rows": ...,
    "n_alert_rows": ...,
    "feature_columns": [...]
  }
```

#### Get MM256 input/output shapes

```
GET /predict_mm256/info
→ {
    "sensor": "MM256",
    "input_shape": "(n_samples, 180, n_features)",
    "output_shape": "(n_samples, 120, 1)",
    "note": "Input length = window_length - forecast_horizon = 300 - 120 = 180 seconds"
  }
```

#### Clear all caches

```
POST /reload
→ {"status": "all caches cleared"}
```

### Example: call the API with Python

```python
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000"   # or your Cloud Run URL

# 1. Check the API is alive
r = requests.get(f"{BASE_URL}/")
print(r.json())  # {"greeting": "Hello"}

# 2. Get preprocessed test data
r = requests.get(f"{BASE_URL}/preprocess", params={"start_index": 0, "stop_index": 10})
data = r.json()
X_test = np.array(data["X_test"])

# 3. Run a prediction
r = requests.post(f"{BASE_URL}/predict", json={
    "timestamp": "20260317_120000",   # replace with an actual model timestamp
    "X_pred": X_test.tolist()
})
predictions = np.array(r.json()["prediction"])
print(f"Predicted shape: {predictions.shape}")  # (n_samples, 120, 3)
```

### Example: call the API with cURL

```bash
# Health check
curl http://127.0.0.1:8000/

# Predict (MM256)
curl -X POST http://127.0.0.1:8000/predict_mm256 \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "20260317_120000", "X_pred": [[[0.1, 0.2, 0.3]]]}'
```

---

## 5. Querying Past Results

Use the `bq_query.py` helper to inspect any past pipeline run without downloading data:

```bash
# List all tables
python scripts/bq_query.py --list

# View latest training metrics
python scripts/bq_query.py --latest history

# View latest predictions
python scripts/bq_query.py --latest predictions

# Custom SQL
python scripts/bq_query.py "SELECT AVG(ABS(residual)) as mae, sensor \
  FROM predictions_20260317_120000 GROUP BY sensor"

# Export to CSV
python scripts/bq_query.py "SELECT * FROM predictions_20260317_120000" \
  --save results/my_export.csv
```

---

## 6. Deploying to Cloud Run

```bash
# Build the Docker image
docker build -t grisounet .

# Tag for Artifact Registry (adjust region/project as needed)
docker tag grisounet europe-west1-docker.pkg.dev/spheric-voyager-484810-k0/grisounet/api:latest

# Push
docker push europe-west1-docker.pkg.dev/spheric-voyager-484810-k0/grisounet/api:latest

# Deploy to Cloud Run
gcloud run deploy grisounet-api \
  --image europe-west1-docker.pkg.dev/spheric-voyager-484810-k0/grisounet/api:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080
```

The container reads all configuration from Secret Manager at runtime — no environment variables to configure manually.

---

## 7. Repository Structure Reference

```
grisounet/
├── api/
│   └── fast.py                  # FastAPI endpoints (predict, preprocess, reload)
├── interface/
│   ├── workflow.py              # 3-sensor end-to-end pipeline
│   └── workflow_mm256.py        # MM256 single-sensor pipeline
├── ml_logic/
│   ├── data.py                  # BigQuery pull, local caching, artifact persistence
│   ├── data_cleaning.py         # Data cleaning utilities
│   ├── preprocessor.py          # Feature engineering, scaling, sequence building
│   ├── model.py                 # 3-sensor LSTM architectures
│   ├── model_mm256.py           # MM256-only LSTM architectures
│   ├── model_save.py            # GCS upload/download for Keras models
│   ├── results_bq_save.py       # Save history/predictions/preprocessing to BigQuery
│   ├── analysis.py              # Plotting, metrics (MAE, RMSE, MAPE)
│   └── secrets.py               # Secret Manager with .env fallback
├── scripts/
│   ├── bq_query.py              # CLI for querying BigQuery
│   ├── fetch_tables.py          # Download BQ tables as CSV
│   ├── list_resources.py        # Inventory of GCS + BQ resources
│   ├── cv_time_series.py        # Time series cross-validation
│   ├── preprocessor_MM256.py    # MM256-specific preprocessing
│   └── qn_methane_analysis.py   # Methane analysis scripts
├── research/
│   ├── notebooks/               # Exploratory notebooks (gitignored)
│   ├── analysis/                # Analysis scripts and processed artifacts
│   └── references/              # Reference PDFs and background reading
├── requirements/
│   ├── app.txt                  # Production runtime deps
│   ├── dev.txt                  # Development tools
│   ├── research.txt             # Notebook/research deps
│   └── full.txt                 # Legacy all-in-one snapshot
├── Dockerfile                   # Container config for Cloud Run
├── makefile                     # Task automation (make data, etc.)
├── QUERY_GUIDE.txt              # BigQuery operations guide
└── README.md                    # Project overview
```

---

## 8. Key Concepts

| Concept | Detail |
|---------|--------|
| **Input window** | 180 seconds of sensor data (1 reading/second) |
| **Forecast horizon** | 120 seconds into the future |
| **Sensors** | MM256, MM263, MM264 (methane), plus airflow, temperature, humidity, pressure, motor current |
| **Alert trigger** | Sequences are built around moments when any sensor exceeds the alert threshold |
| **Model type** | LSTM encoder-decoder with quantile regression |
| **Loss function** | Pinball loss (quantile=0.9) — biases predictions upward for safety |
| **Scaling** | MinMaxScaler fitted on training data only (no data leakage) |
| **Versioning** | Models and results are timestamped (`YYYYMMDD_HHMMSS`) — no overwrites, full history preserved |

---

## 9. Contacts & Links

| Resource | Link |
|----------|------|
| GitHub repo | https://github.com/ArthurVigier/grisounet |
| BigQuery console | https://console.cloud.google.com/bigquery?project=spheric-voyager-484810-k0&d=grisou_eu |
| Cloud Storage bucket | https://console.cloud.google.com/storage/browser/grisou_bucket |
| VM instance | https://console.cloud.google.com/compute/instancesDetail/zones/europe-west1-b/instances/grisou-instance?project=grisounet |

For GCP access or questions, contact the team
