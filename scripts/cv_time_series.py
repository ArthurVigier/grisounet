"""
TimeSeriesSplit cross-validation harness for the MM256 pipeline.

Performs k-fold temporal cross-validation using sklearn's TimeSeriesSplit,
trains the LSTM encoder-decoder on each fold, aggregates metrics, and
optionally pushes fold results to BigQuery.

Key design decisions
--------------------
* **Per-fold scaling**: the MinMaxScaler is fit ONLY on the training portion
  of each fold, preventing data leakage.
* **Expanding window**: TimeSeriesSplit produces expanding training sets
  (fold 1 has the least data, fold k the most).  This mirrors how the model
  would be retrained in production as new data arrives.
* **Gap between train and validation**: an optional ``gap`` parameter inserts
  a buffer between the training and validation sets to avoid temporal
  autocorrelation bleeding across the boundary.

Usage:
    python scripts/cv_time_series.py \\
        --n-splits 5 \\
        --gap 300 \\
        --source cache \\
        --push-bq

    # Or import the harness:
    from scripts.cv_time_series import run_cv
    results = run_cv(n_splits=5, gap=300)
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocessor_MM256 import (
    preprocess_mm256,
    scale_fold,
    slice_windows_mm256,
    TARGET_SENSOR,
)
from ml_logic.secrets import get_secret


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute error metrics for a single fold.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape (n_samples, horizon, 1).

    Returns
    -------
    dict with MAE, RMSE, MAPE, pinball_90
    """
    residual = y_true.ravel() - y_pred.ravel()
    actual = y_true.ravel()
    predicted = y_pred.ravel()

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    # MAPE — guard against division by zero
    denom = np.abs(actual) + 1e-8
    mape = float(np.mean(np.abs(residual) / denom) * 100)
    # Pinball loss at quantile 0.9 (matches the training loss)
    error = actual - predicted
    q = 0.9
    pinball = float(np.mean(np.maximum(q * error, (q - 1) * error)))

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MAPE_%": round(mape, 4),
        "pinball_90": round(pinball, 6),
    }


# ---------------------------------------------------------------------------
# Build the model (single-sensor variant)
# ---------------------------------------------------------------------------
def _build_model(input_shape: tuple, horizon: int):
    """Build an encoder-decoder LSTM for single-sensor forecasting.

    Parameters
    ----------
    input_shape : (input_length, n_features)
    horizon : int (forecast steps)

    Returns
    -------
    Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

    def pinball_loss(y_true, y_pred, quantile=0.9):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        RepeatVector(horizon),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(1)),
    ])
    model.compile(
        optimizer="adam",
        loss=lambda yt, yp: pinball_loss(yt, yp, quantile=0.9),
    )
    return model


# ---------------------------------------------------------------------------
# Main CV harness
# ---------------------------------------------------------------------------
def run_cv(
    n_splits: int = 5,
    gap: int = 300,
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    window_length: int = 300,
    forecast_horizon: int = 120,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 5,
    push_bq: bool = False,
) -> dict:
    """Run k-fold TimeSeriesSplit cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of CV folds (default 5).
    gap : int
        Number of rows to skip between train and validation sets.
        Prevents temporal leakage from autocorrelation.  300 rows ~ 5 min
        at 1-second resolution.
    source, cache_raw : data loading options.
    alert_rate : float
        ALERT threshold for window triggering.
    concentration_threshold : float
        Minimum daily peak MM256 to include a day.
    window_length, forecast_horizon : int
        Window parameters (seconds).
    epochs, batch_size, patience : training hyperparameters.
    push_bq : bool
        Push per-fold metrics to BigQuery.

    Returns
    -------
    dict with keys:
        fold_metrics : list of dicts (per-fold metrics)
        aggregate_metrics : dict (mean +/- std across folds)
        fold_histories : list of History objects
        metadata : preprocessing metadata dict
    """
    import tensorflow as tf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_started = perf_counter()

    print(f"\n{'='*60}")
    print(f"  TimeSeriesSplit CV — MM256 — {n_splits} folds")
    print(f"  gap={gap}  window={window_length}s  horizon={forecast_horizon}s")
    print(f"  timestamp: {timestamp}")
    print(f"{'='*60}\n")

    # ---- Step 1: Preprocess (unscaled — we scale per fold) ----
    # We need the UNSCALED data to fit scalers per fold.
    # preprocess_mm256 returns scaled data, so we load raw and do minimal
    # preprocessing here.  Alternatively, we could add a scale=False param.
    # For clarity, we call the full preprocessor but keep a reference to
    # the unscaled version.

    print("Step 1 — Loading and preprocessing data...")
    step_t = perf_counter()

    raw_df = _load_and_prepare_unscaled(
        source=source,
        cache_raw=cache_raw,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
    )
    print(f"  Preprocessed (unscaled): {raw_df.shape}")
    print(f"  Done in {perf_counter() - step_t:.1f}s\n")

    # ---- Step 2: TimeSeriesSplit ----
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    n_rows = len(raw_df)
    indices = np.arange(n_rows)

    fold_metrics = []
    fold_histories = []
    fold_details = []

    plots_dir = os.path.join(PROJECT_ROOT, "results", "graphs", "mm256_cv")
    os.makedirs(plots_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(indices), start=1):
        fold_t = perf_counter()
        print(f"{'─'*50}")
        print(f"Fold {fold_idx}/{n_splits}")
        print(f"  Train: rows {train_idx[0]:,}–{train_idx[-1]:,} ({len(train_idx):,} rows)")
        print(f"  Val:   rows {val_idx[0]:,}–{val_idx[-1]:,} ({len(val_idx):,} rows)")

        train_df = raw_df.iloc[train_idx]
        val_df = raw_df.iloc[val_idx]

        # Date ranges for reporting
        train_start = train_df.index.min()
        train_end = train_df.index.max()
        val_start = val_df.index.min()
        val_end = val_df.index.max()
        print(f"  Train period: {train_start} -> {train_end}")
        print(f"  Val period:   {val_start} -> {val_end}")

        # ---- Per-fold scaling (no data leakage) ----
        scaled_train, scaled_val, fold_scalers = scale_fold(train_df, val_df)

        # ---- Create windows ----
        X_train, y_train = slice_windows_mm256(
            scaled_train, 0, len(scaled_train),
            window_length, forecast_horizon,
        )
        X_val, y_val = slice_windows_mm256(
            scaled_val, 0, len(scaled_val),
            window_length, forecast_horizon,
        )

        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")

        if X_train.shape[0] == 0:
            print("  WARNING: No training windows — skipping fold.")
            fold_metrics.append({"fold": fold_idx, "status": "skipped", "reason": "no_train_windows"})
            continue
        if X_val.shape[0] == 0:
            print("  WARNING: No validation windows — skipping fold.")
            fold_metrics.append({"fold": fold_idx, "status": "skipped", "reason": "no_val_windows"})
            continue

        # ---- Build & train model ----
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = _build_model(input_shape, y_train.shape[1])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1,
        )
        fold_histories.append(history)

        # ---- Predict & evaluate ----
        y_pred = model.predict(X_val, batch_size=batch_size)
        metrics = _compute_fold_metrics(y_val, y_pred)
        metrics["fold"] = fold_idx
        metrics["status"] = "ok"
        metrics["n_train_windows"] = int(X_train.shape[0])
        metrics["n_val_windows"] = int(X_val.shape[0])
        metrics["n_epochs_trained"] = len(history.history["loss"])
        metrics["best_val_loss"] = float(min(history.history["val_loss"]))
        metrics["train_period"] = f"{train_start} -> {train_end}"
        metrics["val_period"] = f"{val_start} -> {val_end}"

        fold_metrics.append(metrics)
        print(f"  Metrics: MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  "
              f"Pinball90={metrics['pinball_90']:.5f}")

        # ---- Fold plots ----
        # (a) Loss curve
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.history["loss"], label="Train loss")
        ax.plot(history.history["val_loss"], label="Val loss")
        ax.set_title(f"Fold {fold_idx} — Training & Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"fold{fold_idx}_loss_{timestamp}.png"), dpi=150)
        plt.close(fig)

        # (b) Sample prediction
        if X_val.shape[0] > 0:
            sample_idx = min(len(y_val) - 1, len(y_val) // 2)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(y_val[sample_idx, :, 0], label="Actual", linewidth=2)
            ax.plot(y_pred[sample_idx, :, 0], label="Predicted", linestyle=":", linewidth=2)
            ax.set_title(f"Fold {fold_idx} — MM256 Forecast (sample {sample_idx})")
            ax.set_xlabel("Forecast step (seconds)")
            ax.set_ylabel("MM256 (scaled)")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"fold{fold_idx}_sample_{timestamp}.png"), dpi=150)
            plt.close(fig)

        elapsed = perf_counter() - fold_t
        print(f"  Fold {fold_idx} done in {elapsed:.1f}s\n")

        # Clean up to free GPU/CPU memory
        del model, X_train, y_train, X_val, y_val, y_pred
        del scaled_train, scaled_val, fold_scalers
        tf.keras.backend.clear_session()
        gc.collect()

    # ---- Aggregate metrics across folds ----
    ok_folds = [m for m in fold_metrics if m.get("status") == "ok"]
    if ok_folds:
        agg = {}
        for key in ["MAE", "RMSE", "MAPE_%", "pinball_90"]:
            vals = [m[key] for m in ok_folds]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 6)
            agg[f"{key}_std"] = round(float(np.std(vals)), 6)
        agg["n_folds_ok"] = len(ok_folds)
        agg["n_folds_skipped"] = n_splits - len(ok_folds)
    else:
        agg = {"error": "All folds skipped"}

    # ---- Summary ----
    print(f"{'='*60}")
    print(f"  CV Summary — {len(ok_folds)}/{n_splits} folds completed")
    print(f"{'='*60}")
    if ok_folds:
        print(f"  MAE:       {agg['MAE_mean']:.5f} +/- {agg['MAE_std']:.5f}")
        print(f"  RMSE:      {agg['RMSE_mean']:.5f} +/- {agg['RMSE_std']:.5f}")
        print(f"  MAPE:      {agg['MAPE_%_mean']:.2f}% +/- {agg['MAPE_%_std']:.2f}%")
        print(f"  Pinball90: {agg['pinball_90_mean']:.5f} +/- {agg['pinball_90_std']:.5f}")

    # ---- Per-fold summary table ----
    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\nPer-fold details:")
    print(metrics_df.to_string(index=False))

    # ---- Save metrics locally ----
    results_dir = os.path.join(PROJECT_ROOT, "results", "cv_metrics")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"cv_mm256_{timestamp}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved -> {metrics_path}")

    agg_path = os.path.join(results_dir, f"cv_mm256_aggregate_{timestamp}.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate saved -> {agg_path}")

    # ---- Aggregate plot: metrics across folds ----
    if len(ok_folds) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fold_nums = [m["fold"] for m in ok_folds]

        for ax, key, color in zip(
            axes,
            ["MAE", "RMSE", "pinball_90"],
            ["steelblue", "tomato", "seagreen"],
        ):
            vals = [m[key] for m in ok_folds]
            ax.bar(fold_nums, vals, color=color, alpha=0.8)
            ax.axhline(np.mean(vals), color="grey", ls="--", label=f"mean={np.mean(vals):.4f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"cv_metrics_summary_{timestamp}.png"), dpi=150)
        plt.close(fig)
        print(f"Summary plot -> {plots_dir}/cv_metrics_summary_{timestamp}.png")

    # ---- Optionally push to BQ ----
    if push_bq and ok_folds:
        _push_cv_results_to_bq(metrics_df, agg, timestamp)

    total_time = perf_counter() - cv_started
    print(f"\nTotal CV time: {total_time:.1f}s")

    return {
        "timestamp": timestamp,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": agg,
        "fold_histories": fold_histories,
        "n_splits": n_splits,
    }


# ---------------------------------------------------------------------------
# Internal: load & prepare UNSCALED data (scaling happens per fold)
# ---------------------------------------------------------------------------
def _load_and_prepare_unscaled(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
) -> pd.DataFrame:
    """Load, filter to active days, engineer features — but do NOT scale.

    This produces the same DataFrame as ``preprocess_mm256`` but without
    MinMaxScaler applied, so that we can fit the scaler per CV fold.
    """
    from ml_logic.data import load_modeling_dataframe
    from scripts.preprocessor_MM256 import (
        identify_active_days,
        filter_to_active_days,
        SENSORS_TO_DROP,
        AMP_COLS,
        TARGET_SENSOR,
    )

    raw_df = load_modeling_dataframe(source=source, cache_raw=cache_raw)
    df = raw_df.copy()
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    df.set_index("time", inplace=True)
    df.drop(columns=["year", "month", "day", "hour", "minute", "second"], inplace=True)
    df["CR863"] = df["CR863"].astype(np.float32)

    # Identify active days
    active_days = identify_active_days(df, concentration_threshold)
    print(f"  Active days (peak >= {concentration_threshold}%): {len(active_days)}")

    # Filter
    df = filter_to_active_days(df, active_days)

    # Drop other methanometers
    cols_to_drop = [c for c in SENSORS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Aggregate motor currents
    amp_present = [c for c in AMP_COLS if c in df.columns]
    if amp_present:
        df["AMP_AVG"] = df[amp_present].mean(axis=1)
        df.drop(columns=amp_present, inplace=True)

    # ALERT flag
    df["ALERT"] = (df[TARGET_SENSOR] >= alert_rate).astype(np.int8)

    # Cast to float32
    float_cols = df.select_dtypes(include=["floating"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Push CV results to BigQuery
# ---------------------------------------------------------------------------
def _push_cv_results_to_bq(metrics_df: pd.DataFrame, agg: dict, timestamp: str):
    """Push per-fold metrics and aggregate summary to BigQuery."""
    from google.cloud import bigquery

    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    client = bigquery.Client(project=project, location=region)

    # Per-fold table
    fold_table = f"{project}.{dataset}.cv_mm256_folds_{timestamp}"
    upload_df = metrics_df.copy()
    upload_df["run_timestamp"] = timestamp
    client.load_table_from_dataframe(upload_df, fold_table).result()
    print(f"Fold metrics -> BQ: {fold_table}")

    # Aggregate table
    agg_table = f"{project}.{dataset}.cv_mm256_aggregate_{timestamp}"
    agg_df = pd.DataFrame([agg])
    agg_df["run_timestamp"] = timestamp
    client.load_table_from_dataframe(agg_df, agg_table).result()
    print(f"Aggregate metrics -> BQ: {agg_table}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TimeSeriesSplit cross-validation for MM256 LSTM pipeline"
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--gap", type=int, default=300,
                        help="Gap (rows) between train and val sets")
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--forecast-horizon", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--push-bq", action="store_true",
                        help="Push fold metrics and aggregates to BigQuery")
    args = parser.parse_args()

    run_cv(
        n_splits=args.n_splits,
        gap=args.gap,
        source=args.source,
        cache_raw=args.cache_raw,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        push_bq=args.push_bq,
    )


if __name__ == "__main__":
    main()
