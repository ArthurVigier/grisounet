from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.model_save import load_model_from_gcs
from ml_logic.data import load_modeling_dataframe
from ml_logic.preprocessor import preprocess_split, slice_arrays
import numpy as np
from pydantic import BaseModel
from typing import List, Any, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory caches to avoid reloading on every request
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
    model = _get_model(data.timestamp)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    X_pred = np.array(data.X_pred)
    prediction = model.predict(X_pred)

    return {"prediction": prediction.tolist()}


@app.post("/reload")
def reload_cache():
    """Force reload data and clear model cache."""
    _cached_data["df"] = None
    _cached_data["train"] = None
    _cached_data["test"] = None
    _cached_data["scalers"] = None
    _cached_models.clear()
    return {"status": "cache cleared"}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
