import json
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.model_save import load_model_from_gcs
from ml_logic.data import load_modeling_dataframe
from ml_logic.preprocessor import preprocess_split, slice_arrays
import numpy as np
from pydantic import BaseModel
from typing import List, Any, Optional

app = FastAPI()
MODELS_DIR = Path("results/models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Caches — 3-sensor pipeline
# ---------------------------------------------------------------------------
_cached_data = {"df": None, "train": None, "test": None, "scalers": None}
_cached_models = {}


def _get_preprocessed_data():
    """Load and preprocess data once, cache for subsequent requests."""
    if _cached_data["df"] is None:
        df = load_modeling_dataframe(source="cache")
        train_data, test_data, scalers = preprocess_split(df)
        _cached_data["df"] = df
        _cached_data["train"] = train_data
        _cached_data["test"] = test_data
        _cached_data["scalers"] = scalers
    return _cached_data["train"], _cached_data["test"]


def _get_model(timestamp: str):
    """Load model from GCS once per timestamp, cache in memory."""
    if timestamp not in _cached_models:
        _cached_models[timestamp] = load_model_from_gcs(timestamp)
    return _cached_models[timestamp]


# ---------------------------------------------------------------------------
# Caches — MM256 single-sensor pipeline
# ---------------------------------------------------------------------------
_cached_mm256 = {"data": None, "scalers": None, "meta": None}
_cached_mm256_models = {}


def _get_mm256_preprocessed():
    """Load and preprocess MM256-only data, cache for subsequent requests."""
    if _cached_mm256["data"] is None:
        from scripts.preprocessor_MM256 import preprocess_mm256
        data, scalers, meta = preprocess_mm256(source="cache")
        _cached_mm256["data"] = data
        _cached_mm256["scalers"] = scalers
        _cached_mm256["meta"] = meta
    return _cached_mm256["data"], _cached_mm256["scalers"], _cached_mm256["meta"]


def _get_mm256_model(timestamp: str):
    """Load MM256 model and local inference artifacts for one saved timestamp."""
    key = f"mm256_{timestamp}"
    if key not in _cached_mm256_models:
        model_timestamp = f"mm256_{timestamp}"
        metadata_path = MODELS_DIR / f"model_{model_timestamp}_metadata.json"
        scalers_path = MODELS_DIR / f"model_{model_timestamp}_scalers.pkl"
        model = load_model_from_gcs(model_timestamp)

        metadata = {}
        scalers = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        if scalers_path.exists():
            with open(scalers_path, "rb") as stream:
                scalers = pickle.load(stream)

        _cached_mm256_models[key] = {
            "model": model,
            "metadata": metadata,
            "scalers": scalers,
        }
    return _cached_mm256_models[key]


# ---------------------------------------------------------------------------
# Existing endpoints — 3-sensor pipeline
# ---------------------------------------------------------------------------
@app.get("/preprocess")
def preprocess(start_index: int, stop_index: int):
    """
    Preprocess data and return sliced train/test arrays.
    Uses cached data to avoid pulling from BQ on every call.
    """
    train_data, test_data = _get_preprocessed_data()
    X_train, y_train = slice_arrays(train_data, start_index, stop_index)
    X_test, y_test = slice_arrays(test_data, start_index, stop_index)
    return {
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
    }


class PredictRequest(BaseModel):
    timestamp: str
    X_pred: List[Any]


@app.post("/predict")
def predict(data: PredictRequest):
    """Predict using the 3-sensor model."""
    model = _get_model(data.timestamp)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    X_pred = np.array(data.X_pred)
    prediction = model.predict(X_pred)

    return {"prediction": prediction.tolist()}


# ---------------------------------------------------------------------------
# New endpoints — MM256 single-sensor pipeline
# ---------------------------------------------------------------------------
@app.get("/preprocess_mm256")
def preprocess_mm256_endpoint():
    """Return MM256 preprocessing metadata (active days, row counts)."""
    _, _, meta = _get_mm256_preprocessed()
    return {
        "target_sensor": meta["target_sensor"],
        "concentration_threshold": meta["concentration_threshold"],
        "n_active_days": meta["n_active_days"],
        "n_active_rows": meta["n_active_rows"],
        "n_alert_rows": meta["n_alert_rows"],
        "feature_columns": meta["feature_columns"],
    }


class PredictMM256Request(BaseModel):
    timestamp: str
    X_pred: List[Any]


@app.post("/predict_mm256")
def predict_mm256(data: PredictMM256Request):
    """Predict MM256 methane concentration using the single-sensor model.

    Request body:
        timestamp : str — model version to load (e.g. "20260317_120000")
        X_pred : list — input tensor, shape (n_samples, input_length, n_features)

    Response:
        prediction : list — predicted MM256 values, shape (n_samples, horizon, 1)
        sensor : "MM256"
    """
    bundle = _get_mm256_model(data.timestamp)
    model = bundle["model"]

    if model is None:
        raise HTTPException(status_code=404, detail="MM256 model not found")

    X_pred = np.array(data.X_pred, dtype=np.float32)
    metadata = bundle.get("metadata", {})
    scalers = bundle.get("scalers", {})
    catch22_meta = metadata.get("catch22", {})
    expects_catch22 = bool(catch22_meta.get("enabled")) or len(getattr(model, "inputs", [])) > 1

    if expects_catch22:
        from scripts.preprocessor_MM256 import build_mm256_model_inputs, transform_catch22_windows

        catch22_scalers = scalers.get("catch22_feature_scalers")
        if catch22_scalers is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model expects catch22 inputs but the saved catch22 scalers "
                    "are missing locally"
                ),
            )
        X_pred_c22 = transform_catch22_windows(X_pred, catch22_scalers)
        prediction = model.predict(build_mm256_model_inputs(X_pred, X_pred_c22))
    else:
        prediction = model.predict(X_pred)

    return {
        "sensor": "MM256",
        "prediction": prediction.tolist(),
    }


@app.get("/predict_mm256/info")
def predict_mm256_info():
    """Return expected input/output shapes for the MM256 prediction endpoint."""
    return {
        "sensor": "MM256",
        "input_shape": "(n_samples, 180, n_features)",
        "output_shape": "(n_samples, 120, 1)",
        "note": (
            "Input length = window_length - forecast_horizon = 300 - 120 = 180 seconds. "
            "If the saved model was trained with catch22, the API derives the static catch22 "
            "branch internally from the provided sequence windows."
        ),
    }


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
@app.post("/reload")
def reload_cache():
    """Force reload data and clear model cache (both pipelines)."""
    _cached_data["df"] = None
    _cached_data["train"] = None
    _cached_data["test"] = None
    _cached_data["scalers"] = None
    _cached_models.clear()
    _cached_mm256["data"] = None
    _cached_mm256["scalers"] = None
    _cached_mm256["meta"] = None
    _cached_mm256_models.clear()
    return {"status": "all caches cleared"}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
