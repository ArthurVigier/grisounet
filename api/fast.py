from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.model_save import load_model_from_gcs
from ml_logic.data import load_modeling_dataframe
from ml_logic.preprocessor import preprocess_split, slice_arrays
import numpy as np
from pydantic import BaseModel
from typing import List, Any
from fastapi import HTTPException
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/preprocess")
def preprocess(start_index: int, stop_index: int):
    """
    Preprocess data pulled from BigQuery and return sliced train/test arrays.
    start_index and stop_index are positional row offsets, not datetime strings.
    """
    df = load_modeling_dataframe(source="bq")
    train_data, test_data, _ = preprocess_split(df)
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
    model = load_model_from_gcs(data.timestamp)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    X_pred = np.array(data.X_pred)
    prediction = model.predict(X_pred)

    return {"prediction": prediction.tolist()}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
