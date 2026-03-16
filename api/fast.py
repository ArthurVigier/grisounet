import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.model import more_advanced_lstm
from ml_logic.preprocessor import load_data_local , preprocess_split,slice_arrays,feature_target
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
def preprocess():
    df = load_data_local()
    train_data, test_data, scalers = preprocess_split(df)
    X_train , y_train = slice_arrays(train_data)
    X_test, y_test = slice_arrays(test_data)
    return X_train,y_train


# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/train_and_test_model")
def train_and_test_model(X_train,y_train,X_test,y_test):      # 1
    """
    Train the model from the preprocessing function data , then make a predict
    """
    model , history,y_pred = more_advanced_lstm(X_train,y_train,X_test,y_test)
    assert model is not None
    return model,history,y_pred

@app.get("/")
def root():
    return {'greeting': 'Hello'}
