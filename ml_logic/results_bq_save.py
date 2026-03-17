"""Save training history and model predictions to BigQuery"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery
from ml_logic.secrets import get_secret

SENSORS_DEFAULT = ["MM256", "MM263", "MM264"]


def save_history_to_bq(history, timestamp=None, table_suffix=None):
    """Save training history (loss per epoch) to indexed BQ table.

    Parameters
    ----------
    history : keras History object
    timestamp : str, optional
    table_suffix : str, optional
        Extra suffix appended to the table name (e.g. ``"mm256"`` produces
        ``history_mm256_{timestamp}``).  When None the table is named
        ``history_{timestamp}`` for backward compatibility.
    """
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df["run_timestamp"] = timestamp

    suffix = f"_{table_suffix}" if table_suffix else ""
    table_ref = f"{project}.{dataset}.history{suffix}_{timestamp}"
    client = bigquery.Client(project=project, location=region)
    client.load_table_from_dataframe(history_df, table_ref).result()

    # Also save locally
    os.makedirs("results/model_history", exist_ok=True)
    history_df.to_csv(f"results/model_history/history{suffix}_{timestamp}.csv", index=False)

    print(f"History saved -> BQ: {table_ref}")
    return table_ref


def save_predictions_to_bq(y_test, y_pred, timestamp=None, sensors=None, table_suffix=None):
    """Save predictions vs actuals for each sensor to a timestamped BQ table.

    Parameters
    ----------
    y_test : np.ndarray
        Shape ``(n_samples, horizon, n_sensors)`` — actual values.
    y_pred : np.ndarray
        Shape ``(n_samples, horizon, n_sensors)`` — predicted values.
    timestamp : str, optional
    sensors : list[str], optional
        Sensor names matching the last axis of y_test / y_pred.
        Defaults to ``["MM256", "MM263", "MM264"]`` (3-sensor pipeline).
        Pass ``["MM256"]`` for the single-sensor MM256 pipeline.
    table_suffix : str, optional
        Extra suffix for the BQ table name.
    """
    if sensors is None:
        sensors = SENSORS_DEFAULT

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
        "sensor": np.tile(np.array(sensors[:sensor_count]), sample_count * horizon),
        "actual": actual.astype(float),
        "predicted": predicted.astype(float),
        "residual": (actual - predicted).astype(float),
    })
    pred_df["run_timestamp"] = timestamp

    suffix = f"_{table_suffix}" if table_suffix else ""
    table_ref = f"{project}.{dataset}.predictions{suffix}_{timestamp}"
    client = bigquery.Client(project=project, location=region)
    client.load_table_from_dataframe(pred_df, table_ref).result()

    # Also save locally
    os.makedirs("results/predictions", exist_ok=True)
    pred_df.to_csv(f"results/predictions/predictions{suffix}_{timestamp}.csv", index=False)

    print(f"Predictions saved -> BQ: {table_ref}")
    return pred_df
