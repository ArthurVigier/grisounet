"""
Preprocessing pipeline focused on sensor MM256 only.

Loads the full dataset, filters to days where MM256 reaches >= 1% methane
concentration, engineers features for a single-sensor target, and prepares
data ready for TimeSeriesSplit cross-validation.

This module is designed to be imported by cv_time_series.py (which handles
the actual fold loop and model training), or run standalone for inspection.

Usage:
    # As a module
    from scripts.preprocessor_MM256 import preprocess_mm256, slice_windows_mm256
    data, scalers, meta = preprocess_mm256(source="cache", alert_rate=1.0)
    X, y = slice_windows_mm256(data, start=0, stop=len(data))

    # Standalone (inspect + save artifacts)
    python scripts/preprocessor_MM256.py [--source cache] [--alert-rate 1.0] [--push-bq]
"""

import argparse
import gc
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ml_logic imports work when
# running the script directly (python scripts/preprocessor_MM256.py).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.data import load_modeling_dataframe
from ml_logic.secrets import get_secret

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SENSOR = "MM256"

# Feature columns relevant to the MM256 sensor location.
# We keep co-located environmental sensors + infrastructure readings.
# Individual motor currents are replaced by AMP_AVG (see below).
FEATURE_COLS_TO_KEEP = [
    # Methanometer (target)
    "MM256",
    # Airflow
    "AN422", "AN423",
    # Temperature (paired probes)
    "TP1711", "TP1721",
    # Humidity
    "RH1712", "RH1722",
    # Barometric pressure
    "BA1713", "BA1723",
    # Pipeline / drainage
    "CM861", "CR863", "P_864", "TC862", "WM868",
    # Motor current (will be aggregated)
    "AMP1_IR", "AMP2_IR", "DMP3_IR", "DMP4_IR", "AMP5_IR",
    # Discrete
    "F_SIDE", "V",
]

# Sensors that are NOT relevant when focusing on MM256 alone.
# We exclude the other two critical methanometers from features to avoid
# information leakage (they are highly correlated and would mask the signal
# we want MM256 to learn from environmental predictors).
SENSORS_TO_DROP = ["MM263", "MM264", "MM252", "MM261", "MM262", "MM211"]

# Motor current columns that will be replaced by a single average.
AMP_COLS = ["AMP1_IR", "AMP2_IR", "DMP3_IR", "DMP4_IR", "AMP5_IR"]


# ---------------------------------------------------------------------------
# 1. Identify "active days" — days where MM256 >= concentration_threshold
# ---------------------------------------------------------------------------
def identify_active_days(
    df: pd.DataFrame,
    concentration_threshold: float = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame of unique dates where MM256 peaked above the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with a DatetimeIndex named 'time'.
    concentration_threshold : float
        Minimum peak MM256 value (% vol) for a day to be considered "active".

    Returns
    -------
    pd.DataFrame
        Columns: [date, day_peak_mm256, n_seconds_above]
    """
    daily = df.groupby(df.index.date).agg(
        day_peak_mm256=(TARGET_SENSOR, "max"),
        n_seconds_above=(TARGET_SENSOR, lambda s: (s >= concentration_threshold).sum()),
    )
    active = daily[daily["day_peak_mm256"] >= concentration_threshold].copy()
    active.index.name = "date"
    active = active.reset_index()
    return active


# ---------------------------------------------------------------------------
# 2. Filter the full dataset to active days only
# ---------------------------------------------------------------------------
def filter_to_active_days(
    df: pd.DataFrame,
    active_days: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only rows whose date appears in *active_days*.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex.
    active_days : pd.DataFrame
        Output of ``identify_active_days`` — must contain a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    active_set = set(active_days["date"])
    mask = df.index.date  # array of datetime.date objects
    keep = pd.Series([d in active_set for d in mask], index=df.index)
    filtered = df.loc[keep].copy()
    return filtered


# ---------------------------------------------------------------------------
# 3. Main preprocessing entry point
# ---------------------------------------------------------------------------
def preprocess_mm256(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
) -> tuple:
    """Load data, filter to active days, engineer features, and scale.

    This function does NOT split into train/test — that is handled by the
    TimeSeriesSplit cross-validation harness in ``cv_time_series.py``.

    Parameters
    ----------
    source : str
        Data source ("bq", "cache", or "local").
    cache_raw : bool
        Whether to cache a local CSV snapshot when pulling from BQ.
    alert_rate : float
        Methane concentration (% vol) above which the ALERT binary flag = 1.
        This governs which timestamps become window "trigger points" during
        slicing.  Should equal *concentration_threshold* for consistency.
    concentration_threshold : float
        Minimum daily peak MM256 value to include a day.  Days where MM256
        never reaches this value are entirely excluded.

    Returns
    -------
    (processed_df, scalers, metadata)
        processed_df : pd.DataFrame
            Scaled DataFrame with DatetimeIndex, ready for slicing into
            (X, y) windows.  Contains only active-day rows.
        scalers : dict
            {column_name: fitted MinMaxScaler} — needed for inverse transform.
        metadata : dict
            Diagnostic information (active_days DataFrame, row counts, etc.).
    """
    # ---- Load raw data ----
    print(f"\n{'='*60}")
    print(f"  Preprocessor MM256 — threshold >= {concentration_threshold}%")
    print(f"{'='*60}\n")

    raw_df = load_modeling_dataframe(source=source, cache_raw=cache_raw)

    # ---- Build DatetimeIndex ----
    df = raw_df.copy()
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    df.set_index("time", inplace=True)
    df.drop(columns=["year", "month", "day", "hour", "minute", "second"], inplace=True)
    df["CR863"] = df["CR863"].astype(np.float32)

    total_rows = len(df)
    print(f"Total rows loaded: {total_rows:,}")

    # ---- Identify and filter to active days ----
    active_days = identify_active_days(df, concentration_threshold)
    print(f"Active days (MM256 peak >= {concentration_threshold}%): {len(active_days)}")
    print(f"  Date range: {active_days['date'].min()} to {active_days['date'].max()}")

    df = filter_to_active_days(df, active_days)
    active_rows = len(df)
    print(f"Rows after day filter: {active_rows:,} ({active_rows/total_rows*100:.1f}%)")

    # ---- Drop other methanometers (prevent information leakage) ----
    cols_to_drop = [c for c in SENSORS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # ---- Aggregate motor currents ----
    amp_present = [c for c in AMP_COLS if c in df.columns]
    if amp_present:
        df["AMP_AVG"] = df[amp_present].mean(axis=1)
        df.drop(columns=amp_present, inplace=True)

    # ---- Create ALERT flag: MM256 >= alert_rate ----
    df["ALERT"] = (df[TARGET_SENSOR] >= alert_rate).astype(np.int8)
    n_alert = df["ALERT"].sum()
    print(f"Alert rows (MM256 >= {alert_rate}%): {n_alert:,} "
          f"({n_alert/active_rows*100:.2f}%)")

    # ---- Cast to float32 ----
    float_cols = df.select_dtypes(include=["floating"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    # ---- Scale with MinMaxScaler (fit on FULL filtered data) ----
    # NOTE: When used with TimeSeriesSplit, we refit the scaler on each
    # fold's training portion only.  The scaler returned here is fitted on
    # the entire filtered dataset and is useful for quick inspection / plots.
    # The CV harness in cv_time_series.py handles per-fold scaling properly.
    scalers = {}
    features_to_scale = df.select_dtypes(include=["floating"]).columns
    for col in features_to_scale:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]]).astype(np.float32)
        scalers[col] = scaler

    feature_cols = [c for c in df.columns if c not in ["ALERT"]]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {TARGET_SENSOR}")

    metadata = {
        "target_sensor": TARGET_SENSOR,
        "concentration_threshold": concentration_threshold,
        "alert_rate": alert_rate,
        "total_rows_raw": total_rows,
        "active_days": active_days,
        "n_active_days": len(active_days),
        "n_active_rows": active_rows,
        "n_alert_rows": int(n_alert),
        "feature_columns": feature_cols,
    }

    return df, scalers, metadata


# ---------------------------------------------------------------------------
# 4. Windowing / slicing for MM256 (single-sensor target)
# ---------------------------------------------------------------------------
def slice_windows_mm256(
    df: pd.DataFrame,
    start_index: int = 0,
    stop_index: int | None = None,
    window_length_in_sec: int = 300,
    forecast_horizon_in_sec: int = 120,
) -> tuple:
    """Create (X, y) window pairs centered on ALERT triggers.

    Identical logic to ``ml_logic.preprocessor.slice_arrays`` but with a
    single-sensor target (MM256 only), producing y of shape
    ``(n_samples, forecast_horizon, 1)``.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with DatetimeIndex and an ALERT column.
    start_index, stop_index : int
        Row range to consider.
    window_length_in_sec : int
        Total window size (input + forecast).
    forecast_horizon_in_sec : int
        Number of seconds to forecast ahead.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        X: (n_samples, input_length, n_features)  float32
        y: (n_samples, forecast_horizon, 1)        float32
    """
    if stop_index is None:
        stop_index = len(df)

    input_length = window_length_in_sec - forecast_horizon_in_sec
    if input_length <= 0:
        raise ValueError("window_length_in_sec must be > forecast_horizon_in_sec")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex")

    subset = df.iloc[start_index:stop_index]

    excluded = {"ALERT"}
    feature_cols = [c for c in subset.columns if c not in excluded]
    target_cols = [TARGET_SENSOR]

    trigger_mask = subset["ALERT"] == 1
    trigger_times = subset.index[trigger_mask]

    if trigger_times.empty:
        return (
            np.empty((0, input_length, len(feature_cols)), dtype=np.float32),
            np.empty((0, forecast_horizon_in_sec, 1), dtype=np.float32),
        )

    subset_index = subset.index
    feat_vals = subset[feature_cols].to_numpy(dtype=np.float32, copy=False)
    tgt_vals = subset[target_cols].to_numpy(dtype=np.float32, copy=False)
    one_sec = pd.Timedelta(seconds=1)

    X_list, y_list = [], []
    for t0 in trigger_times:
        x_times = pd.date_range(end=t0, periods=input_length, freq="s")
        y_times = pd.date_range(start=t0 + one_sec, periods=forecast_horizon_in_sec, freq="s")

        x_idx = subset_index.get_indexer(x_times)
        y_idx = subset_index.get_indexer(y_times)

        if (x_idx < 0).any() or (y_idx < 0).any():
            continue

        X_list.append(feat_vals[x_idx])
        y_list.append(tgt_vals[y_idx])

    del feat_vals, tgt_vals, trigger_mask, trigger_times, subset
    gc.collect()

    if not X_list:
        return (
            np.empty((0, input_length, len(feature_cols)), dtype=np.float32),
            np.empty((0, forecast_horizon_in_sec, 1), dtype=np.float32),
        )

    X_arr = np.stack(X_list).astype(np.float32)
    y_arr = np.stack(y_list).astype(np.float32)
    del X_list, y_list
    gc.collect()

    return X_arr, y_arr


# ---------------------------------------------------------------------------
# 5. Per-fold scaler: refit on training portion only (no data leakage)
# ---------------------------------------------------------------------------
def scale_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple:
    """Fit MinMaxScaler on *train_df*, transform both train & val.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Unscaled DataFrames sharing the same columns.

    Returns
    -------
    (scaled_train, scaled_val, scalers)
    """
    train_out = train_df.copy()
    val_out = val_df.copy()
    scalers = {}

    float_cols = train_df.select_dtypes(include=["floating"]).columns
    for col in float_cols:
        scaler = MinMaxScaler()
        train_out[col] = scaler.fit_transform(train_out[[col]]).astype(np.float32)
        val_out[col] = scaler.transform(val_out[[col]]).astype(np.float32)
        scalers[col] = scaler

    return train_out, val_out, scalers


# ---------------------------------------------------------------------------
# 6. Push active-day summary to BigQuery (optional)
# ---------------------------------------------------------------------------
def push_active_days_to_bq(active_days: pd.DataFrame, metadata: dict):
    """Push the active-day summary table to BigQuery."""
    from google.cloud import bigquery

    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")

    table_id = f"{project}.{dataset}.mm256_active_days"
    client = bigquery.Client(project=project, location=region)

    upload_df = active_days.copy()
    upload_df["date"] = pd.to_datetime(upload_df["date"])
    upload_df["concentration_threshold"] = metadata["concentration_threshold"]

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(upload_df, table_id, job_config=job_config).result()
    print(f"Active-day summary pushed -> {table_id}")
    return table_id


# ---------------------------------------------------------------------------
# CLI entry point — inspect preprocessing, generate diagnostic plots
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess methane data for MM256 single-sensor pipeline"
    )
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0,
                        help="Methane %% threshold for ALERT flag")
    parser.add_argument("--concentration-threshold", type=float, default=1.0,
                        help="Min daily peak MM256 %% to include a day")
    parser.add_argument("--push-bq", action="store_true",
                        help="Push active-day summary to BigQuery")
    args = parser.parse_args()

    data, scalers, meta = preprocess_mm256(
        source=args.source,
        cache_raw=args.cache_raw,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
    )

    # ---- Diagnostic output ----
    print(f"\n--- Preprocessing summary ---")
    print(f"Shape: {data.shape}")
    print(f"Active days: {meta['n_active_days']}")
    print(f"Active rows: {meta['n_active_rows']:,}")
    print(f"Alert rows:  {meta['n_alert_rows']:,}")
    print(f"Columns: {list(data.columns)}")

    active_days = meta["active_days"]
    print(f"\nActive days table:")
    print(active_days.to_string(index=False))

    # ---- Quick diagnostic plots ----
    plots_dir = os.path.join(PROJECT_ROOT, "results", "graphs", "mm256_preprocessing")
    os.makedirs(plots_dir, exist_ok=True)

    # (a) Daily peak concentration
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(active_days)), active_days["day_peak_mm256"], color="tomato", alpha=0.8)
    ax.set_xlabel("Active day index")
    ax.set_ylabel("Peak MM256 (%)")
    ax.set_title(f"Daily peak MM256 on active days (>= {args.concentration_threshold}%)")
    ax.axhline(args.concentration_threshold, color="grey", ls="--", label="threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "daily_peak_mm256.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {plots_dir}/daily_peak_mm256.png")

    # (b) Seconds above threshold per day
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(active_days)), active_days["n_seconds_above"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Active day index")
    ax.set_ylabel("Seconds above threshold")
    ax.set_title("Duration of elevated MM256 per active day")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "seconds_above_per_day.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {plots_dir}/seconds_above_per_day.png")

    # (c) Full time series with active-day shading
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(data.index, data[TARGET_SENSOR], linewidth=0.3, color="navy", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("MM256 (scaled)")
    ax.set_title("MM256 concentration on active days (scaled)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mm256_active_timeseries.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {plots_dir}/mm256_active_timeseries.png")

    # ---- Quick window test ----
    X, y = slice_windows_mm256(data)
    print(f"\nWindow test (full data): X={X.shape}, y={y.shape}")

    # ---- Optionally push to BQ ----
    if args.push_bq:
        push_active_days_to_bq(active_days, meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
