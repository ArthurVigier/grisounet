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


def _attach_window_timestamps(pred_df: pd.DataFrame, window_index: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row forecast timestamps using one metadata row per sample_id."""
    required_cols = {
        "sample_id",
        "input_start_time",
        "input_end_time",
        "target_start_time",
        "target_end_time",
    }
    missing = required_cols.difference(window_index.columns)
    if missing:
        raise ValueError(
            f"window_index is missing required columns: {sorted(missing)}"
        )

    sample_count = int(pred_df["sample_id"].max()) + 1 if not pred_df.empty else 0
    ordered_index = (
        window_index.loc[:, sorted(required_cols)]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")
        .sort_index()
    )
    expected_sample_ids = pd.Index(np.arange(sample_count), name="sample_id")
    ordered_index = ordered_index.reindex(expected_sample_ids)
    if ordered_index.isnull().any(axis=None):
        raise ValueError("window_index does not align with prediction sample_id values")

    sample_meta = pred_df["sample_id"].to_numpy(dtype=np.int64, copy=False)
    forecast_steps = pred_df["forecast_step"].to_numpy(dtype=np.int64, copy=False)

    input_start = pd.to_datetime(
        ordered_index.loc[sample_meta, "input_start_time"].to_numpy()
    )
    input_end = pd.to_datetime(
        ordered_index.loc[sample_meta, "input_end_time"].to_numpy()
    )
    target_start = pd.to_datetime(
        ordered_index.loc[sample_meta, "target_start_time"].to_numpy()
    )
    target_end = pd.to_datetime(
        ordered_index.loc[sample_meta, "target_end_time"].to_numpy()
    )
    target_time = target_start + pd.to_timedelta(forecast_steps, unit="s")

    pred_df = pred_df.copy()
    pred_df["input_start_time"] = input_start
    pred_df["forecast_origin_time"] = input_end
    pred_df["target_start_time"] = target_start
    pred_df["target_time"] = target_time
    pred_df["target_end_time"] = target_end
    pred_df["target_date"] = pd.to_datetime(target_time).date
    return pred_df


def build_prediction_frame(
    y_test,
    y_pred,
    timestamp=None,
    sensors=None,
    window_index: pd.DataFrame | None = None,
):
    """Build the long prediction dataframe used for CSV export and BigQuery."""
    if sensors is None:
        sensors = SENSORS_DEFAULT
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    columns = ["sample_id", "forecast_step", "sensor", "actual", "predicted", "residual", "run_timestamp"]
    if y_test.size == 0 or y_pred.size == 0:
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

    if window_index is not None:
        pred_df = _attach_window_timestamps(pred_df, window_index)

    return pred_df


def save_predictions_to_bq(
    y_test,
    y_pred,
    timestamp=None,
    sensors=None,
    table_suffix=None,
    window_index: pd.DataFrame | None = None,
):
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
    window_index : pd.DataFrame, optional
        One metadata row per sample_id. When provided, forecast timestamps are
        added to the exported table so downstream queries can filter by date.
    """
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_df = build_prediction_frame(
        y_test,
        y_pred,
        timestamp=timestamp,
        sensors=sensors,
        window_index=window_index,
    )
    if pred_df.empty:
        print("Predictions skipped: no test windows available.")
        return pred_df

    suffix = f"_{table_suffix}" if table_suffix else ""
    table_ref = f"{project}.{dataset}.predictions{suffix}_{timestamp}"
    client = bigquery.Client(project=project, location=region)
    client.load_table_from_dataframe(pred_df, table_ref).result()

    # Also save locally
    os.makedirs("results/predictions", exist_ok=True)
    pred_df.to_csv(f"results/predictions/predictions{suffix}_{timestamp}.csv", index=False)

    print(f"Predictions saved -> BQ: {table_ref}")
    return pred_df
