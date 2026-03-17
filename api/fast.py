import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.model import more_advanced_lstm
from ml_logic.model_save import save_model_to_gcs,load_model_from_gcs
from ml_logic.preprocessor import load_data_local , preprocess_split,slice_arrays,feature_target
import numpy as np
from pydantic import BaseModel
from typing import List, Any
from fastapi import FastAPI, HTTPException
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
def preprocess(start_index,stop_index):
    """
    Preprocess step , please keep in mind that you must precise which index you want to use
    index work in datetime format only between 2014-03-02 00:00:00 and 2014-05-15 13:52:30
    """
    df = load_data_local()
    train_data, test_data, scalers = preprocess_split(df)
    X_train , y_train = slice_arrays(train_data,start_index,stop_index)
    X_test, y_test = slice_arrays(test_data,start_index,stop_index)
    return X_train,y_train,X_test,y_test

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
