"""Save training history and model predictions to BigQuery"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

SENSORS = ["MM256", "MM263", "MM264"]


def save_history_to_bq(history, timestamp=None):
    """Save training history (loss per epoch) to indexed BQ table."""
    project = os.environ.get("GCP_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df["run_timestamp"] = timestamp

    region = os.environ.get("BQ_REGION")
    table_ref = f"{project}.{dataset}.history_{timestamp}"
    client = bigquery.Client(project=project, location=region)
    client.load_table_from_dataframe(history_df, table_ref).result()

    # Also save locally
    os.makedirs("results/model_history", exist_ok=True)
    history_df.to_csv(f"results/model_history/history_{timestamp}.csv", index=False)

    print(f"History saved -> BQ: {table_ref}")
    return table_ref


def save_predictions_to_bq(y_test, y_pred, timestamp=None):
    """
    Saves predictions vs actuals for each sensor to a timestamped BQ table.

    y_test: shape (n_samples, horizon, 3) -- actual values
    y_pred: shape (n_samples, horizon, 3) -- predicted values
    """
    project = os.environ.get("GCP_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for sample_idx in range(y_test.shape[0]):
        for step in range(y_test.shape[1]):
            for sensor_idx, sensor_name in enumerate(SENSORS):
                rows.append({
                    "sample_id": sample_idx,
                    "forecast_step": step,
                    "sensor": sensor_name,
                    "actual": float(y_test[sample_idx, step, sensor_idx]),
                    "predicted": float(y_pred[sample_idx, step, sensor_idx]),
                    "residual": float(y_test[sample_idx, step, sensor_idx]
                                      - y_pred[sample_idx, step, sensor_idx]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df["run_timestamp"] = timestamp

    region = os.environ.get("BQ_REGION")
    table_ref = f"{project}.{dataset}.predictions_{timestamp}"
    client = bigquery.Client(project=project, location=region)
    client.load_table_from_dataframe(pred_df, table_ref).result()

    # Also save locally
    os.makedirs("results/predictions", exist_ok=True)
    pred_df.to_csv(f"results/predictions/predictions_{timestamp}.csv", index=False)

    print(f"Predictions saved -> BQ: {table_ref}")
    return pred_df
