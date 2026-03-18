"""
Preprocessing pipeline V2 for sensor MM256 — with feature engineering.

Extends the V1 preprocessor by computing derived features (lags, deltas,
rolling statistics, slopes, interaction features) on the raw sensor columns
before windowing.  The result is a wider DataFrame (~37-40 columns) that
feeds the same downstream pipeline (CV, training, analysis) unchanged.

Feature engineering is inspired by IJCRS'15 competition insights:
  - Differential features (deltas, slopes) are robust to distribution drift
  - Rolling MAX outperforms rolling MEAN for threshold prediction tasks
  - Interaction features (dilution ratio) capture physical relationships

Usage:
    from scripts.preprocessor_MM256_v2 import preprocess_mm256_v2
    data, scalers, meta = preprocess_mm256_v2(source="cache", iteration=1)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.data import load_modeling_dataframe
from ml_logic.data_cleaning import clean_dataframe
from ml_logic.secrets import get_secret
from scripts.preprocessor_MM256 import (
    EXCLUDED_DATES,
    TARGET_SENSOR,
    identify_active_days,
    filter_to_active_days,
)


# ─── Column constants ────────────────────────────────────────────────────────

# Iteration 1 columns (same 5 as V1)
FEATURES_KEEP_V2_ITER1 = ["MM256", "AN422", "AMP1_IR", "AN423", "F_SIDE"]

# Iteration 2 adds barometric pressure
FEATURES_KEEP_V2_ITER2 = ["MM256", "AN422", "AMP1_IR", "AN423", "F_SIDE", "BA1713"]


# ─── Feature engineering functions ────────────────────────────────────────────

def _add_methane_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Autoregressive methane features.

    Competition insight: rolling MAX beats rolling MEAN for threshold prediction.
    Differential features are robust to distribution drift across time periods.
    """
    m = df[TARGET_SENSOR]

    # ── Raw lags ──────────────────────────────────────────────────────────────
    df["MM256_lag_1"] = m.shift(1)
    df["MM256_lag_5"] = m.shift(5)
    df["MM256_lag_10"] = m.shift(10)

    # ── Deltas (drift-robust) ─────────────────────────────────────────────────
    df["MM256_delta_1"] = m.diff(1)    # second-over-second change
    df["MM256_delta_5"] = m.diff(5)    # 5-step trend
    df["MM256_delta_10"] = m.diff(10)  # 10-step trend

    # ── Slope (linear trend over 10 steps) ────────────────────────────────────
    df["MM256_slope_10s"] = (
        m.rolling(10)
         .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )

    # ── Rolling windows — SHORT (capture immediate spikes) ────────────────────
    for w in [10, 30]:
        df[f"MM256_max_{w}s"] = m.rolling(w).max()    # key feature per competition
        df[f"MM256_mean_{w}s"] = m.rolling(w).mean()
        df[f"MM256_std_{w}s"] = m.rolling(w).std()
        df[f"MM256_min_{w}s"] = m.rolling(w).min()

    # ── Rolling windows — LONG (capture sustained trends for t+2) ─────────────
    for w in [60, 180]:
        df[f"MM256_max_{w}s"] = m.rolling(w).max()
        df[f"MM256_mean_{w}s"] = m.rolling(w).mean()
        df[f"MM256_std_{w}s"] = m.rolling(w).std()

    # ── Range within window (max - min = volatility proxy) ────────────────────
    df["MM256_range_30s"] = df["MM256_max_30s"] - df["MM256_min_30s"]
    df["MM256_range_10s"] = df["MM256_max_10s"] - df["MM256_min_10s"]

    # ── Distance from recent max (are we heading toward a peak?) ──────────────
    df["MM256_dist_from_max_30s"] = df["MM256_max_30s"] - m

    # ── Acceleration (second derivative) ──────────────────────────────────────
    df["MM256_accel"] = df["MM256_delta_1"].diff(1)

    return df


def _add_airflow_features(
    df: pd.DataFrame,
    iteration: int = 1,
) -> pd.DataFrame:
    """
    Airflow features.

    Physical rationale: airflow dilutes methane.  Low airflow + rising methane
    is the highest-risk combination (methane_dilution_ratio captures this).
    """
    col = "AN422"
    if col not in df.columns:
        return df

    a = df[col]

    # ── Rolling stats ─────────────────────────────────────────────────────────
    df["AN422_mean_30s"] = a.rolling(30).mean()
    df["AN422_std_30s"] = a.rolling(30).std()
    df["AN422_delta_5"] = a.diff(5)

    # ── Interaction feature: methane / airflow (physical dilution ratio) ──────
    df["methane_dilution_ratio"] = (
        df[TARGET_SENSOR] / df["AN422_mean_30s"].clip(lower=0.01)
    )

    # ── Direction change (iteration 2 only) ───────────────────────────────────
    if iteration >= 2 and "AN423" in df.columns:
        df["AN423_dir_change"] = df["AN423"].diff(1).abs()

    return df


def _add_machine_features(
    df: pd.DataFrame,
    iteration: int = 1,
) -> pd.DataFrame:
    """
    Machine power features.

    Physical rationale: higher shearer power -> more coal cut -> more CH4 emitted.
    Machine startup (power delta positive and large) is a leading indicator.
    """
    col = "AMP1_IR"
    if col not in df.columns:
        return df

    mp = df[col]

    # ── Rolling stats ─────────────────────────────────────────────────────────
    df["AMP1_IR_mean_30s"] = mp.rolling(30).mean()
    df["AMP1_IR_delta_5"] = mp.diff(5)

    # ── Machine ON/OFF flag ───────────────────────────────────────────────────
    df["machine_is_on"] = (mp > mp.quantile(0.1)).astype(np.int8)

    # ── Direction change (iteration 2 only) ───────────────────────────────────
    if iteration >= 2 and "F_SIDE" in df.columns:
        df["F_SIDE_dir_change"] = df["F_SIDE"].diff(1).abs()

    return df


def _add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended features for iteration 2+.
    Barometric pressure affects methane release from coal seams.
    """
    col = "BA1713"
    if col not in df.columns:
        return df

    p = df[col]
    df["BA1713_mean_60s"] = p.rolling(60).mean()
    df["BA1713_delta_10"] = p.diff(10)

    return df


def engineer_features_v2(
    df: pd.DataFrame,
    iteration: int = 1,
) -> pd.DataFrame:
    """
    Apply feature engineering to the preprocessed MM256 DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sensor data with DatetimeIndex. Must contain MM256 and ALERT columns.
    iteration : int
        1 = core features (~37 columns)
        2 = adds direction changes + pressure features (~40 columns)

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame, NaN warm-up rows dropped.
    """
    df = df.copy()

    # Group A: Autoregressive methane (priority 1)
    df = _add_methane_features(df)

    # Group B: Airflow / dilution (priority 2)
    df = _add_airflow_features(df, iteration=iteration)

    # Group C: Machine power (priority 3)
    df = _add_machine_features(df, iteration=iteration)

    # Group D: Pressure (iteration 2 only)
    if iteration >= 2:
        df = _add_pressure_features(df)

    # Drop NaN warm-up rows from rolling windows
    df = df.dropna().reset_index(drop=True) if not isinstance(df.index, pd.DatetimeIndex) else df.dropna()

    return df


# ─── Main preprocessor entry point ───────────────────────────────────────────

def preprocess_mm256_v2(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    scale: bool = False,
    clean_abnormal_values: bool = False,
    frozen_sensor_window: int = 3600,
    sensor_disagreement_z_threshold: float = 6.0,
    iteration: int = 1,
) -> tuple:
    """
    Load data, filter to active days, engineer features, and optionally scale.

    This is the V2 preprocessor: same loading/filtering as V1, plus feature
    engineering from the competition-inspired features.py approach.

    Parameters
    ----------
    iteration : int
        1 = core features (~37 columns), 2 = extended (+pressure, +directions)
    """
    print(f"\n{'='*60}")
    print(f"  Preprocessor MM256 V2 — iteration {iteration}")
    print(f"  threshold >= {concentration_threshold}%")
    print(f"{'='*60}\n")

    # ── Load raw data ─────────────────────────────────────────────────────────
    raw_df = load_modeling_dataframe(source=source, cache_raw=cache_raw)

    df = raw_df.copy()
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    df.set_index("time", inplace=True)
    df.drop(columns=["year", "month", "day", "hour", "minute", "second"], inplace=True)

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
        print("Sorted rows by timestamp to restore chronological order.")

    total_rows = len(df)
    print(f"Total rows loaded: {total_rows:,}")

    # ── Anomaly cleaning (optional) ───────────────────────────────────────────
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
            f"Running anomaly cleaning (frozen_window={frozen_sensor_window}s,"
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
        print(f"Rows after cleaning: {cleaned_rows:,} ({cleaned_rows / total_rows * 100:.1f}% kept)")

    # ── Filter to active days ─────────────────────────────────────────────────
    active_days = identify_active_days(df, concentration_threshold)
    print(f"Active days (MM256 peak >= {concentration_threshold}%): {len(active_days)}")
    if len(active_days) > 0:
        print(f"  Date range: {active_days['date'].min()} to {active_days['date'].max()}")

    df = filter_to_active_days(df, active_days)
    active_rows = len(df)
    print(f"Rows after day filter: {active_rows:,} ({active_rows / total_rows * 100:.1f}%)")

    # ── Keep feature columns (iteration-dependent) ────────────────────────────
    features_keep = FEATURES_KEEP_V2_ITER2 if iteration >= 2 else FEATURES_KEEP_V2_ITER1
    cols_to_keep = [c for c in features_keep if c in df.columns]
    dropped = sorted(set(df.columns) - set(cols_to_keep))
    df = df[cols_to_keep]
    print(f"Kept {len(cols_to_keep)} raw columns: {cols_to_keep}")
    print(f"Dropped {len(dropped)} columns")

    # ── ALERT flag (computed on raw MM256 before feature engineering) ──────────
    df["ALERT"] = (df[TARGET_SENSOR] >= alert_rate).astype(np.int8)
    n_alert = int(df["ALERT"].sum())
    print(f"Alert rows (MM256 >= {alert_rate}%): {n_alert:,} ({n_alert / active_rows * 100:.2f}%)")

    # ── Cast to float32 ──────────────────────────────────────────────────────
    float_cols = df.select_dtypes(include=["floating"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    # ── Feature engineering ───────────────────────────────────────────────────
    rows_before_fe = len(df)
    df = engineer_features_v2(df, iteration=iteration)
    rows_after_fe = len(df)
    print(f"Feature engineering: {rows_before_fe:,} -> {rows_after_fe:,} rows ({rows_before_fe - rows_after_fe} NaN warm-up rows dropped)")

    # ── Optional scaling ──────────────────────────────────────────────────────
    scalers = {}
    if scale:
        features_to_scale = df.select_dtypes(include=["floating"]).columns
        for col in features_to_scale:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]]).astype(np.float32)
            scalers[col] = scaler

    feature_cols = [c for c in df.columns if c not in ["ALERT"]]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {TARGET_SENSOR}")
    print(f"Scaled: {scale}")
    print(f"Iteration: {iteration}")

    if len(df) > 1:
        deltas = df.index.to_series().diff()
        print(f"Median timestep: {deltas.median()}")
        print(f"Max timestep:    {deltas.max()}")
        print(f"Duplicate timestamps: {int(df.index.duplicated().sum()):,}")

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
        "version": "v2",
        "iteration": iteration,
        "n_features": len(feature_cols),
        "rows_dropped_by_feature_engineering": rows_before_fe - rows_after_fe,
    }

    return df, scalers, metadata


# ─── Standalone CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V2 Preprocessor for MM256 with feature engineering"
    )
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--iteration", type=int, choices=[1, 2], default=1)
    parser.add_argument("--skip-cleaning", dest="clean_abnormal_values", action="store_false")
    parser.add_argument("--frozen-sensor-window", type=int, default=3600)
    parser.add_argument("--sensor-disagreement-z-threshold", type=float, default=6.0)
    parser.set_defaults(clean_abnormal_values=True)
    args = parser.parse_args()

    data, scalers, meta = preprocess_mm256_v2(
        source=args.source,
        cache_raw=args.cache_raw,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        scale=False,
        clean_abnormal_values=args.clean_abnormal_values,
        frozen_sensor_window=args.frozen_sensor_window,
        sensor_disagreement_z_threshold=args.sensor_disagreement_z_threshold,
        iteration=args.iteration,
    )

    print(f"\n--- V2 Preprocessing summary ---")
    print(f"Shape: {data.shape}")
    print(f"Active days: {meta['n_active_days']}")
    print(f"Active rows: {meta['n_active_rows']:,}")
    print(f"Alert rows:  {meta['n_alert_rows']:,}")
    print(f"Iteration:   {meta['iteration']}")
    print(f"Features:    {meta['n_features']}")
    print(f"Columns: {list(data.columns)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
