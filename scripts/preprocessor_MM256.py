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
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.data import load_modeling_dataframe
from ml_logic.data_cleaning import SENSOR_BOUNDS, clean_dataframe
from ml_logic.secrets import get_secret

TARGET_SENSOR = "MM256"

# Iteration 1 feature selection — physically motivated for MM256 mid-longwall.
# See research/analysis/qn_analysis/sensor_column_mapping.json for full mapping.
#   MM256  = methane(t)           autoregressive target
#   AN422  = airflow_headgate_ms  airflow power — dilution & transport
#   AMP1_IR = shearer_left_head_A machine power — emission rate proxy
#   AN423  = airflow_tailgate_ms  airflow direction
#   F_SIDE = shearer_direction    machine direction (reserve, kept for iter 2)
FEATURES_KEEP = ["MM256", "AN422", "AMP1_IR", "AN423", "F_SIDE"]

SEQUENCE_INPUT_NAME = "sequence_input"
CATCH22_INPUT_NAME = "catch22_input"

MAX_DAILY_PEAK_MM256 = float(SENSOR_BOUNDS[TARGET_SENSOR][1])


def identify_active_days(
    df: pd.DataFrame,
    concentration_threshold: float = 1.0,
    excluded_dates: list[str] | None = None,
    max_daily_peak_mm256: float | None = MAX_DAILY_PEAK_MM256,
) -> pd.DataFrame:
    """Return active MM256 days while excluding only clearly saturated days."""
    daily = df.groupby(df.index.date).agg(
        day_peak_mm256=(TARGET_SENSOR, "max"),
        n_seconds_above=(TARGET_SENSOR, lambda s: (s >= concentration_threshold).sum()),
    )
    active = daily[daily["day_peak_mm256"] >= concentration_threshold].copy()
    active.index.name = "date"
    active = active.reset_index()

    if max_daily_peak_mm256 is not None:
        saturated_mask = active["day_peak_mm256"] > max_daily_peak_mm256
        n_excluded = int(saturated_mask.sum())
        if n_excluded > 0:
            excluded = (
                active.loc[saturated_mask, ["date", "day_peak_mm256"]]
                .sort_values("date")
                .reset_index(drop=True)
            )
            excluded_dates_str = excluded["date"].astype(str).tolist()
            print(
                "Excluded"
                f" {n_excluded} saturated MM256 days with daily peak > {max_daily_peak_mm256}%:"
                f" {excluded_dates_str}"
            )
        active = active.loc[~saturated_mask].reset_index(drop=True)

    if excluded_dates:
        exclude_set = set(pd.to_datetime(excluded_dates).date)
        n_before = len(active)
        active = active[~active["date"].isin(exclude_set)].reset_index(drop=True)
        n_excluded = n_before - len(active)
        if n_excluded > 0:
            print(f"Excluded {n_excluded} saturated days: {sorted(exclude_set)}")

    return active


def filter_to_active_days(
    df: pd.DataFrame,
    active_days: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only rows whose date appears in active_days."""
    active_dates = pd.to_datetime(active_days["date"])
    keep = df.index.normalize().isin(active_dates)
    return df.loc[keep].copy()


def preprocess_mm256(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    scale: bool = True,
    clean_abnormal_values: bool = False,
    frozen_sensor_window: int = 3600,
    sensor_disagreement_z_threshold: float = 6.0,
    enable_sensor_disagreement: bool = False,
    max_daily_peak_mm256: float | None = MAX_DAILY_PEAK_MM256,
) -> tuple:
    """Load data, filter to active days, engineer features, and optionally scale."""
    print(f"\n{'='*60}")
    print(f"  Preprocessor MM256 — threshold >= {concentration_threshold}%")
    print(f"{'='*60}\n")

    raw_df = load_modeling_dataframe(source=source, cache_raw=cache_raw)

    df = raw_df.copy()
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    df.set_index("time", inplace=True)
    df.drop(columns=["year", "month", "day", "hour", "minute", "second"], inplace=True)

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
        print("Sorted rows by timestamp to restore chronological order.")

    if "CR863" in df.columns:
        df["CR863"] = df["CR863"].astype(np.float32)

    total_rows = len(df)
    print(f"Total rows loaded: {total_rows:,}")

    cleaning_meta = {
        "applied": bool(clean_abnormal_values),
        "frozen_sensor_window": int(frozen_sensor_window),
        "sensor_disagreement_z_threshold": float(sensor_disagreement_z_threshold),
        "sensor_disagreement_enabled": bool(enable_sensor_disagreement),
        "rows_before_cleaning": int(total_rows),
        "rows_after_cleaning": int(total_rows),
        "rows_removed": 0,
    }

    if clean_abnormal_values:
        print(
            "Running anomaly cleaning before MM256 preprocessing"
            f" (frozen_window={frozen_sensor_window}s,"
            f" disagreement={'on' if enable_sensor_disagreement else 'off'},"
            f" disagreement_z={sensor_disagreement_z_threshold})..."
        )
        df = clean_dataframe(
            df,
            drop=True,
            frozen_window=frozen_sensor_window,
            z_threshold=sensor_disagreement_z_threshold,
            use_sensor_disagreement=enable_sensor_disagreement,
            verbose=True,
        )
        cleaned_rows = len(df)
        cleaning_meta["rows_after_cleaning"] = int(cleaned_rows)
        cleaning_meta["rows_removed"] = int(total_rows - cleaned_rows)
        print(
            f"Rows after cleaning: {cleaned_rows:,}"
            f" ({cleaned_rows / total_rows * 100:.1f}% kept)"
        )

    active_days = identify_active_days(
        df,
        concentration_threshold=concentration_threshold,
        max_daily_peak_mm256=max_daily_peak_mm256,
    )
    print(f"Active days (MM256 peak >= {concentration_threshold}%): {len(active_days)}")
    if len(active_days) > 0:
        print(f"  Date range: {active_days['date'].min()} to {active_days['date'].max()}")

    df = filter_to_active_days(df, active_days)
    active_rows = len(df)
    print(f"Rows after day filter: {active_rows:,} ({active_rows / total_rows * 100:.1f}%)")

    cols_to_keep = [c for c in FEATURES_KEEP if c in df.columns]
    dropped = sorted(set(df.columns) - set(cols_to_keep))
    df = df[cols_to_keep]
    print(f"Kept {len(cols_to_keep)} feature columns: {cols_to_keep}")
    print(f"Dropped {len(dropped)} columns: {dropped}")

    df["ALERT"] = (df[TARGET_SENSOR] >= alert_rate).astype(np.int8)
    n_alert = int(df["ALERT"].sum())
    print(f"Alert rows (MM256 >= {alert_rate}%): {n_alert:,} ({n_alert / active_rows * 100:.2f}%)")

    float_cols = df.select_dtypes(include=["floating"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

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

    if len(df) > 1:
        deltas = df.index.to_series().diff()
        print(f"Median timestep: {deltas.median()}")
        print(f"Max timestep:    {deltas.max()}")
        print(f"Duplicate timestamps: {int(df.index.duplicated().sum()):,}")

    metadata = {
        "target_sensor": TARGET_SENSOR,
        "concentration_threshold": concentration_threshold,
        "alert_rate": alert_rate,
        "max_daily_peak_mm256": max_daily_peak_mm256,
        "total_rows_raw": total_rows,
        "active_days": active_days,
        "n_active_days": len(active_days),
        "n_active_rows": active_rows,
        "n_alert_rows": int(n_alert),
        "feature_columns": feature_cols,
        "cleaning": cleaning_meta,
    }

    return df, scalers, metadata


def slice_windows_mm256(
    df: pd.DataFrame,
    start_index: int = 0,
    stop_index: int | None = None,
    window_length_in_sec: int = 300,
    forecast_horizon_in_sec: int = 120,
    require_alert_trigger: bool = True,
    debug: bool = False,
) -> tuple:
    """
    Create (X, y) window pairs for MM256 forecasting.

    By default, keep only windows whose trigger timestamp (last timestep of the
    input sequence) has ALERT==1. For debugging, this can be disabled with
    require_alert_trigger=False to keep every valid contiguous window.
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

    empty_X = np.empty((0, input_length, len(feature_cols)), dtype=np.float32)
    empty_y = np.empty((0, forecast_horizon_in_sec, 1), dtype=np.float32)

    if subset.empty:
        if debug:
            print("[slice_windows_mm256] subset is empty")
        return empty_X, empty_y

    feat_vals = subset[feature_cols].to_numpy(dtype=np.float32, copy=False)
    tgt_vals = subset[target_cols].to_numpy(dtype=np.float32, copy=False)
    alert_vals = subset["ALERT"].to_numpy(dtype=bool, copy=False)
    total_window = input_length + forecast_horizon_in_sec

    deltas = subset.index.to_series().diff()
    one_second = pd.Timedelta(seconds=1)

    break_mask = deltas.ne(one_second).to_numpy().copy()
    break_mask[0] = True
    segment_starts = np.flatnonzero(break_mask)
    segment_stops = np.concatenate((segment_starts[1:], [len(subset)]))
    segment_lengths = segment_stops - segment_starts

    if debug:
        print("\n[slice_windows_mm256] DEBUG")
        print(f"  subset rows:           {len(subset):,}")
        print(f"  feature columns:       {len(feature_cols)}")
        print(f"  input_length:          {input_length}")
        print(f"  forecast_horizon:      {forecast_horizon_in_sec}")
        print(f"  total_window:          {total_window}")
        print(f"  alert rows in subset:  {int(alert_vals.sum()):,}")
        print(f"  require_alert_trigger: {require_alert_trigger}")
        print(f"  n_segments:            {len(segment_starts):,}")
        print(f"  longest_segment:       {int(segment_lengths.max()):,}")
        print(f"  shortest_segment:      {int(segment_lengths.min()):,}")
        print(f"  segments >= window:    {int((segment_lengths >= total_window).sum()):,}")

        delta_counts = deltas.value_counts(dropna=False).head(10)
        print("  top timestep counts:")
        for delta_val, count in delta_counts.items():
            print(f"    {delta_val}: {int(count):,}")

    if require_alert_trigger and not alert_vals.any():
        if debug:
            print("  no ALERT rows in subset -> returning empty")
        return empty_X, empty_y

    X_chunks, y_chunks = [], []

    total_segments_considered = 0
    total_segments_too_short = 0
    total_candidates = 0
    total_trigger_candidates = 0
    total_valid_offsets = 0

    for seg_start, seg_stop in zip(segment_starts, segment_stops):
        seg_len = seg_stop - seg_start
        if seg_len < total_window:
            total_segments_too_short += 1
            continue

        total_segments_considered += 1
        n_candidates = seg_len - total_window + 1
        total_candidates += n_candidates

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

        if require_alert_trigger:
            trigger_candidates = alert_vals[
                seg_start + input_length - 1 : seg_start + input_length - 1 + n_candidates
            ]
            total_trigger_candidates += int(trigger_candidates.sum())
            valid_offsets = np.flatnonzero(trigger_candidates)
        else:
            valid_offsets = np.arange(n_candidates, dtype=np.int64)

        if valid_offsets.size == 0:
            continue

        total_valid_offsets += int(valid_offsets.size)
        X_chunks.append(np.swapaxes(X_windows[valid_offsets], 1, 2))
        y_chunks.append(np.swapaxes(y_windows[valid_offsets], 1, 2))

    del feat_vals, tgt_vals, alert_vals, subset
    gc.collect()

    if debug:
        print(f"  segments_considered:   {total_segments_considered:,}")
        print(f"  segments_too_short:    {total_segments_too_short:,}")
        print(f"  total_candidates:      {total_candidates:,}")
        if require_alert_trigger:
            print(f"  trigger_candidates=1:  {total_trigger_candidates:,}")
        print(f"  valid_offsets_kept:    {total_valid_offsets:,}")

    if not X_chunks:
        if debug:
            print("  no valid windows kept -> returning empty\n")
        return empty_X, empty_y

    X_arr = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
    y_arr = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False)

    del X_chunks, y_chunks
    gc.collect()

    if debug:
        print(f"  final X shape:         {X_arr.shape}")
        print(f"  final y shape:         {y_arr.shape}\n")

    return X_arr, y_arr


def _get_catch22_transformer():
    """Instantiate aeon's Catch22 transformer with a clear dependency error."""
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(tempfile.gettempdir(), "numba-cache")
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    try:
        from numba.core import config as numba_config

        if not getattr(numba_config, "CACHE_DIR", ""):
            numba_config.CACHE_DIR = cache_dir
    except Exception:
        pass

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
    """Fit catch22 on train windows and transform train plus any extra window sets."""
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
    require_alert_trigger: bool = True,
) -> pd.DataFrame:
    """Return one row of metadata per valid MM256 forecast window."""
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
    total_window = input_length + forecast_horizon_in_sec

    deltas = timestamp_index.to_series().diff()
    one_second = pd.Timedelta(seconds=1)
    break_mask = deltas.ne(one_second).to_numpy().copy()
    break_mask[0] = True
    segment_starts = np.flatnonzero(break_mask)
    segment_stops = np.concatenate((segment_starts[1:], [len(subset)]))

    rows = []
    sample_id = 0
    for seg_start, seg_stop in zip(segment_starts, segment_stops):
        seg_len = seg_stop - seg_start
        if seg_len < total_window:
            continue

        n_candidates = seg_len - total_window + 1

        if require_alert_trigger:
            trigger_candidates = alert_vals[
                seg_start + input_length - 1 : seg_start + input_length - 1 + n_candidates
            ]
            valid_offsets = np.flatnonzero(trigger_candidates)
        else:
            valid_offsets = np.arange(n_candidates, dtype=np.int64)

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


def scale_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple:
    """Fit MinMaxScaler on train_df, transform both train and val."""
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


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Preprocess methane data for MM256 single-sensor pipeline"
    )
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0, help="Methane %% threshold for ALERT flag")
    parser.add_argument("--concentration-threshold", type=float, default=1.0, help="Min daily peak MM256 %% to include a day")
    parser.add_argument("--enable-cleaning", dest="clean_abnormal_values", action="store_true", help="Enable anomaly cleaning before MM256 preprocessing")
    parser.add_argument("--skip-cleaning", dest="clean_abnormal_values", action="store_false", help="Disable anomaly cleaning before MM256 preprocessing")
    parser.add_argument("--frozen-sensor-window", type=int, default=3600, help="Frozen-sensor detection window in seconds (default: 3600)")
    parser.add_argument("--enable-sensor-disagreement", action="store_true", help="Re-enable the standard-deviation disagreement filter between co-located sensors")
    parser.add_argument("--sensor-disagreement-z-threshold", type=float, default=6.0, help="Accepted z-score gap between co-located sensors (default: 6.0)")
    parser.add_argument("--max-daily-peak-mm256", type=float, default=MAX_DAILY_PEAK_MM256, help="Exclude days whose daily MM256 peak exceeds this value (default: 10.0)")
    parser.set_defaults(clean_abnormal_values=False)

    parser.add_argument("--window-length", type=int, default=300, help="Total window length in seconds (input + forecast)")
    parser.add_argument("--forecast-horizon", type=int, default=120, help="Forecast horizon in seconds")
    parser.add_argument("--no-require-alert-trigger", dest="require_alert_trigger", action="store_false", help="Keep all contiguous windows, not only windows triggered by ALERT")
    parser.add_argument("--debug-windows", action="store_true", help="Print detailed diagnostics from slice_windows_mm256")
    parser.set_defaults(require_alert_trigger=True)

    parser.add_argument("--push-bq", action="store_true", help="Push active-day summary to BigQuery")
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
        enable_sensor_disagreement=args.enable_sensor_disagreement,
        max_daily_peak_mm256=args.max_daily_peak_mm256,
    )

    print(f"\n--- Preprocessing summary ---")
    print(f"Shape: {data.shape}")
    print(f"Active days: {meta['n_active_days']}")
    print(f"Active rows: {meta['n_active_rows']:,}")
    print(f"Alert rows:  {meta['n_alert_rows']:,}")
    print(f"Columns: {list(data.columns)}")

    active_days = meta["active_days"]
    print(f"\nActive days table:")
    print(active_days.to_string(index=False))

    plots_dir = os.path.join(PROJECT_ROOT, "results", "graphs", "mm256_preprocessing")
    os.makedirs(plots_dir, exist_ok=True)

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

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(active_days)), active_days["n_seconds_above"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Active day index")
    ax.set_ylabel("Seconds above threshold")
    ax.set_title("Duration of elevated MM256 per active day")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "seconds_above_per_day.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {plots_dir}/seconds_above_per_day.png")

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(data.index, data[TARGET_SENSOR], linewidth=0.3, color="navy", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("MM256 (scaled)")
    ax.set_title("MM256 concentration on active days (scaled)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mm256_active_timeseries.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {plots_dir}/mm256_active_timeseries.png")

    X, y = slice_windows_mm256(
        data,
        window_length_in_sec=args.window_length,
        forecast_horizon_in_sec=args.forecast_horizon,
        require_alert_trigger=args.require_alert_trigger,
        debug=args.debug_windows,
    )
    print(
        f"\nWindow test (full data): "
        f"X={X.shape}, y={y.shape}, "
        f"require_alert_trigger={args.require_alert_trigger}"
    )

    if args.push_bq:
        push_active_days_to_bq(active_days, meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
