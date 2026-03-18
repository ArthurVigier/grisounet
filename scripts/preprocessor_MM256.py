"""
Preprocessing pipeline focused on sensor MM256 only.

Loads the full dataset, filters to days where MM256 reaches >= 1% methane
concentration, engineers features for a single-sensor target, and prepares
data ready for TimeSeriesSplit cross-validation.

This module is designed to be imported by the MM256 workflow, the CV harness,
and the final-training module, or run standalone for inspection.

Usage:
    # As a module
    from scripts.preprocessor_MM256 import preprocess_mm256, slice_windows_mm256
    data, scalers, meta = preprocess_mm256(source="cache", alert_rate=1.0, scale=False)
    X, y = slice_windows_mm256(data, start_index=0, stop_index=len(data))

    # Standalone (inspect + save artifacts)
    python scripts/preprocessor_MM256.py [--source cache] [--alert-rate 1.0] [--push-bq]
"""

import argparse
import gc
import os
import sys

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
from ml_logic.data_cleaning import clean_dataframe
from ml_logic.secrets import get_secret

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SENSOR = "MM256"

# Sensors that are NOT relevant when focusing on MM256 alone.
# We exclude the other methanometers from features to avoid
# information leakage (they are highly correlated and would mask the signal
# we want MM256 to learn from environmental predictors).
SENSORS_TO_DROP = ["MM263", "MM264", "MM252", "MM261", "MM262", "MM211"]

# Motor current columns that will be replaced by a single average.
AMP_COLS = ["AMP1_IR", "AMP2_IR", "DMP3_IR", "DMP4_IR", "AMP5_IR"]

SEQUENCE_INPUT_NAME = "sequence_input"
CATCH22_INPUT_NAME = "catch22_input"


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
    active_dates = pd.to_datetime(active_days["date"])
    keep = df.index.normalize().isin(active_dates)
    return df.loc[keep].copy()


# ---------------------------------------------------------------------------
# 3. Main preprocessing entry point
# ---------------------------------------------------------------------------
def preprocess_mm256(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    scale: bool = True,
    clean_abnormal_values: bool = True,
    frozen_sensor_window: int = 3600,
    sensor_disagreement_z_threshold: float = 6.0,
) -> tuple:
    """Load data, filter to active days, engineer features, and optionally scale.

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
    scale : bool
        If True, fit MinMaxScaler on the full filtered dataset.
        If False, return the unscaled dataframe so a downstream caller can
        fit scalers on train folds / train split only.
    clean_abnormal_values : bool
        If True, run anomaly cleaning before MM256-specific preprocessing and
        drop rows flagged as abnormal.
    frozen_sensor_window : int
        Frozen-sensor detection horizon in seconds. Default is 3600 (60 min).
    sensor_disagreement_z_threshold : float
        Accepted z-score gap between co-located sensors before a disagreement
        row is flagged and removed.

    Returns
    -------
    (processed_df, scalers, metadata)
        processed_df : pd.DataFrame
            DataFrame with DatetimeIndex, ready for slicing into (X, y)
            windows. Scaled if ``scale=True``, otherwise left unscaled.
        scalers : dict
            {column_name: fitted MinMaxScaler}. Empty when ``scale=False``.
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
    if not df.index.is_monotonic_increasing:
        # Temporal splits and contiguous-window extraction both assume sorted rows.
        df.sort_index(inplace=True)
        print("Sorted rows by timestamp to restore chronological order.")
    df["CR863"] = df["CR863"].astype(np.float32)

    total_rows = len(df)
    print(f"Total rows loaded: {total_rows:,}")

    cleaning_meta = {
        "applied": bool(clean_abnormal_values),
        "frozen_sensor_window": int(frozen_sensor_window),
        "sensor_disagreement_z_threshold": float(sensor_disagreement_z_threshold),
        "rows_before_cleaning": int(total_rows),
        "rows_after_cleaning": int(total_rows),
        "rows_removed": 0,
    }
    if clean_abnormal_values:
        print(
            "Running anomaly cleaning before MM256 preprocessing"
            f" (frozen_window={frozen_sensor_window}s,"
            f" disagreement_z={sensor_disagreement_z_threshold})..."
        )
        df = clean_dataframe(
            df,
            drop=True,
            frozen_window=frozen_sensor_window,
            z_threshold=sensor_disagreement_z_threshold,
            verbose=True,
        )
        cleaned_rows = len(df)
        cleaning_meta["rows_after_cleaning"] = int(cleaned_rows)
        cleaning_meta["rows_removed"] = int(total_rows - cleaned_rows)
        print(
            f"Rows after cleaning: {cleaned_rows:,}"
            f" ({cleaned_rows / total_rows * 100:.1f}% kept)"
        )

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

    # ---- Optional full-data scaling ----
    scalers = {}
    if scale:
        features_to_scale = df.select_dtypes(include=["floating"]).columns
        for col in features_to_scale:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]]).astype(np.float32)
            scalers[col] = scaler

    feature_cols = [c for c in df.columns if c not in ["ALERT"]]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {TARGET_SENSOR}")
    print(f"Scaled: {scale}")

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
        "cleaning": cleaning_meta,
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

    if subset.empty:
        return (
            np.empty((0, input_length, len(feature_cols)), dtype=np.float32),
            np.empty((0, forecast_horizon_in_sec, 1), dtype=np.float32),
        )

    feat_vals = subset[feature_cols].to_numpy(dtype=np.float32, copy=False)
    tgt_vals = subset[target_cols].to_numpy(dtype=np.float32, copy=False)
    alert_vals = subset["ALERT"].to_numpy(dtype=bool, copy=False)
    timestamp_ns = subset.index.asi8
    total_window = input_length + forecast_horizon_in_sec

    if not alert_vals.any():
        return (
            np.empty((0, input_length, len(feature_cols)), dtype=np.float32),
            np.empty((0, forecast_horizon_in_sec, 1), dtype=np.float32),
        )

    segment_breaks = np.flatnonzero(np.diff(timestamp_ns) != 1_000_000_000) + 1
    segment_starts = np.concatenate(([0], segment_breaks))
    segment_stops = np.concatenate((segment_breaks, [len(subset)]))

    X_chunks, y_chunks = [], []
    for seg_start, seg_stop in zip(segment_starts, segment_stops):
        seg_len = seg_stop - seg_start
        if seg_len < total_window:
            continue

        n_candidates = seg_len - total_window + 1
        trigger_candidates = alert_vals[
            seg_start + input_length - 1 : seg_start + input_length - 1 + n_candidates
        ]
        if not trigger_candidates.any():
            continue

        feat_segment = feat_vals[seg_start:seg_stop]
        tgt_segment = tgt_vals[seg_start:seg_stop]

        X_windows = np.lib.stride_tricks.sliding_window_view(
            feat_segment,
            window_shape=input_length,
            axis=0,
        )[:n_candidates]
        y_windows = np.lib.stride_tricks.sliding_window_view(
            tgt_segment[input_length:],
            window_shape=forecast_horizon_in_sec,
            axis=0,
        )

        valid_offsets = np.flatnonzero(trigger_candidates)
        X_chunks.append(np.swapaxes(X_windows[valid_offsets], 1, 2))
        y_chunks.append(np.swapaxes(y_windows[valid_offsets], 1, 2))

    del feat_vals, tgt_vals, alert_vals, subset
    gc.collect()

    if not X_chunks:
        return (
            np.empty((0, input_length, len(feature_cols)), dtype=np.float32),
            np.empty((0, forecast_horizon_in_sec, 1), dtype=np.float32),
        )

    X_arr = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
    y_arr = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False)
    del X_chunks, y_chunks
    gc.collect()

    return X_arr, y_arr


def _get_catch22_transformer():
    """Instantiate aeon's Catch22 transformer with a clear dependency error."""
    try:
        from aeon.transformations.collection.feature_based import Catch22
    except ImportError as exc:
        raise ImportError(
            "Catch22 feature engineering requires aeon. "
            "Install aeon==1.3.0 before running the MM256 catch22 pipeline."
        ) from exc

    return Catch22(replace_nans=True)


def _to_catch22_collection(X: np.ndarray) -> np.ndarray:
    """Convert (samples, timesteps, features) to aeon's collection axis order."""
    if X.ndim != 3:
        raise ValueError("Expected a 3D window tensor shaped (samples, timesteps, features)")
    return np.swapaxes(np.asarray(X, dtype=np.float32), 1, 2)


def _fit_feature_scalers(feature_matrix: np.ndarray) -> tuple[np.ndarray, dict[int, MinMaxScaler]]:
    """Fit one MinMaxScaler per engineered feature column."""
    if feature_matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")

    scaled = np.empty_like(feature_matrix, dtype=np.float32)
    scalers: dict[int, MinMaxScaler] = {}

    for feature_idx in range(feature_matrix.shape[1]):
        scaler = MinMaxScaler()
        scaled[:, [feature_idx]] = scaler.fit_transform(
            feature_matrix[:, [feature_idx]]
        ).astype(np.float32)
        scalers[feature_idx] = scaler

    return scaled, scalers


def _apply_feature_scalers(feature_matrix: np.ndarray, scalers: dict[int, MinMaxScaler]) -> np.ndarray:
    """Apply per-column scalers to a 2D engineered feature matrix."""
    if feature_matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")

    scaled = np.empty_like(feature_matrix, dtype=np.float32)
    for feature_idx in range(feature_matrix.shape[1]):
        scaler = scalers[feature_idx]
        scaled[:, [feature_idx]] = scaler.transform(
            feature_matrix[:, [feature_idx]]
        ).astype(np.float32)
    return scaled


def fit_transform_catch22_windows(
    X_train: np.ndarray,
    *extra_sets: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Fit catch22 on train windows and transform train plus any extra window sets.

    The transformer expects collections shaped (samples, channels, timepoints),
    while our pipeline stores windows as (samples, timesteps, features). The
    axis swap happens here to avoid the orientation mistake present in the
    original Charles-Henri sketch.
    """
    if X_train.shape[0] == 0:
        raise ValueError("Cannot fit catch22 features on an empty training window set")

    transformer = _get_catch22_transformer()
    train_collection = _to_catch22_collection(X_train)
    train_features = np.asarray(
        transformer.fit_transform(train_collection),
        dtype=np.float32,
    )
    train_features_scaled, catch22_scalers = _fit_feature_scalers(train_features)

    outputs: list[np.ndarray] = [train_features_scaled]
    for X_other in extra_sets:
        if X_other.shape[0] == 0:
            outputs.append(
                np.empty((0, train_features_scaled.shape[1]), dtype=np.float32)
            )
            continue
        other_collection = _to_catch22_collection(X_other)
        other_features = np.asarray(
            transformer.transform(other_collection),
            dtype=np.float32,
        )
        outputs.append(_apply_feature_scalers(other_features, catch22_scalers))

    metadata = {
        "enabled": True,
        "transformer": "aeon.Catch22",
        "replace_nans": True,
        "n_base_sequence_features": int(X_train.shape[2]),
        "n_catch22_features": int(train_features_scaled.shape[1]),
    }
    return (*outputs, catch22_scalers, metadata)


def transform_catch22_windows(
    X: np.ndarray,
    catch22_scalers: dict[int, MinMaxScaler],
) -> np.ndarray:
    """Transform new windows into scaled catch22 descriptors using saved scalers."""
    if X.shape[0] == 0:
        return np.empty((0, len(catch22_scalers)), dtype=np.float32)

    transformer = _get_catch22_transformer()
    collection = _to_catch22_collection(X)
    features = np.asarray(transformer.fit_transform(collection), dtype=np.float32)
    return _apply_feature_scalers(features, catch22_scalers)


def build_mm256_model_inputs(
    X_sequence: np.ndarray,
    X_catch22: np.ndarray | None = None,
):
    """Return the model input payload expected by the MM256 architectures."""
    if X_catch22 is None:
        return X_sequence
    return {
        SEQUENCE_INPUT_NAME: X_sequence,
        CATCH22_INPUT_NAME: X_catch22,
    }


def build_window_index_mm256(
    df: pd.DataFrame,
    start_index: int = 0,
    stop_index: int | None = None,
    window_length_in_sec: int = 300,
    forecast_horizon_in_sec: int = 120,
) -> pd.DataFrame:
    """Return one row of metadata per valid MM256 forecast window.

    The returned rows match the sample order produced by ``slice_windows_mm256``
    for the same dataframe and arguments.
    """
    if stop_index is None:
        stop_index = len(df)

    input_length = window_length_in_sec - forecast_horizon_in_sec
    if input_length <= 0:
        raise ValueError("window_length_in_sec must be > forecast_horizon_in_sec")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex")

    subset = df.iloc[start_index:stop_index]
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "input_start_time",
                "input_end_time",
                "target_start_time",
                "target_end_time",
            ]
        )

    alert_vals = subset["ALERT"].to_numpy(dtype=bool, copy=False)
    timestamp_index = subset.index
    timestamp_ns = timestamp_index.asi8
    total_window = input_length + forecast_horizon_in_sec

    if not alert_vals.any():
        return pd.DataFrame(
            columns=[
                "sample_id",
                "input_start_time",
                "input_end_time",
                "target_start_time",
                "target_end_time",
            ]
        )

    segment_breaks = np.flatnonzero(np.diff(timestamp_ns) != 1_000_000_000) + 1
    segment_starts = np.concatenate(([0], segment_breaks))
    segment_stops = np.concatenate((segment_breaks, [len(subset)]))

    rows = []
    sample_id = 0
    for seg_start, seg_stop in zip(segment_starts, segment_stops):
        seg_len = seg_stop - seg_start
        if seg_len < total_window:
            continue

        n_candidates = seg_len - total_window + 1
        trigger_candidates = alert_vals[
            seg_start + input_length - 1 : seg_start + input_length - 1 + n_candidates
        ]
        valid_offsets = np.flatnonzero(trigger_candidates)
        if valid_offsets.size == 0:
            continue

        global_starts = seg_start + valid_offsets
        input_starts = timestamp_index[global_starts]
        input_ends = timestamp_index[global_starts + input_length - 1]
        target_starts = timestamp_index[global_starts + input_length]
        target_ends = timestamp_index[
            global_starts + input_length + forecast_horizon_in_sec - 1
        ]

        for input_start, input_end, target_start, target_end in zip(
            input_starts,
            input_ends,
            target_starts,
            target_ends,
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "input_start_time": input_start,
                    "input_end_time": input_end,
                    "target_start_time": target_start,
                    "target_end_time": target_end,
                }
            )
            sample_id += 1

    return pd.DataFrame(rows)


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Preprocess methane data for MM256 single-sensor pipeline"
    )
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0,
                        help="Methane %% threshold for ALERT flag")
    parser.add_argument("--concentration-threshold", type=float, default=1.0,
                        help="Min daily peak MM256 %% to include a day")
    parser.add_argument("--skip-cleaning", dest="clean_abnormal_values", action="store_false",
                        help="Disable anomaly cleaning before MM256 preprocessing")
    parser.add_argument("--frozen-sensor-window", type=int, default=3600,
                        help="Frozen-sensor detection window in seconds (default: 3600)")
    parser.add_argument("--sensor-disagreement-z-threshold", type=float, default=6.0,
                        help="Accepted z-score gap between co-located sensors (default: 6.0)")
    parser.set_defaults(clean_abnormal_values=True)
    parser.add_argument("--push-bq", action="store_true",
                        help="Push active-day summary to BigQuery")
    args = parser.parse_args()

    data, scalers, meta = preprocess_mm256(
        source=args.source,
        cache_raw=args.cache_raw,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        scale=True,
        clean_abnormal_values=args.clean_abnormal_values,
        frozen_sensor_window=args.frozen_sensor_window,
        sensor_disagreement_z_threshold=args.sensor_disagreement_z_threshold,
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
