"""Save training history and model predictions to BigQuery"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery
from ml_logic.secrets import get_secret

SENSORS = ["MM256", "MM263", "MM264"]


def save_history_to_bq(history, timestamp=None):
    """Save training history (loss per epoch) to indexed BQ table."""
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df["run_timestamp"] = timestamp

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
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    columns = ["sample_id", "forecast_step", "sensor", "actual", "predicted", "residual", "run_timestamp"]
    if y_test.size == 0 or y_pred.size == 0:
        print("Predictions skipped: no test windows available.")
        return pd.DataFrame(columns=columns)

    sample_count, horizon, sensor_count = y_test.shape
    actual = y_test.reshape(-1)
    predicted = y_pred.reshape(-1)
    pred_df = pd.DataFrame({
        "sample_id": np.repeat(np.arange(sample_count), horizon * sensor_count),
        "forecast_step": np.tile(np.repeat(np.arange(horizon), sensor_count), sample_count),
        "sensor": np.tile(np.array(SENSORS), sample_count * horizon),
        "actual": actual.astype(float),
        "predicted": predicted.astype(float),
        "residual": (actual - predicted).astype(float),
    })
    pred_df["run_timestamp"] = timestamp

    table_ref = f"{project}.{dataset}.predictions_{timestamp}"
    client = bigquery.Client(project=project, location=region)
    client.load_table_from_dataframe(pred_df, table_ref).result()

    # Also save locally
    os.makedirs("results/predictions", exist_ok=True)
    pred_df.to_csv(f"results/predictions/predictions_{timestamp}.csv", index=False)

    print(f"Predictions saved -> BQ: {table_ref}")
    return pred_df
