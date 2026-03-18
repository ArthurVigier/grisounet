"""TimeSeriesSplit cross-validation harness for the MM256 pipeline."""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.model_mm256 import build_mm256_model
from ml_logic.secrets import get_secret
from scripts.preprocessor_MM256 import (
    TARGET_SENSOR,
    build_mm256_model_inputs,
    fit_transform_catch22_windows,
    preprocess_mm256,
    scale_fold,
    slice_windows_mm256,
)


PRIMARY_METRIC_KEYS = ("pinball_90",)
SECONDARY_METRIC_KEYS = ("MAE", "RMSE")


def get_metric_keys(include_secondary_diagnostics: bool = False) -> tuple[str, ...]:
    """Return the ordered metric keys for the MM256 reporting layer."""
    if include_secondary_diagnostics:
        return PRIMARY_METRIC_KEYS + SECONDARY_METRIC_KEYS
    return PRIMARY_METRIC_KEYS


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def compute_mm256_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_secondary_diagnostics: bool = True,
) -> dict:
    """Compute regression metrics for MM256 forecasts."""
    residual = y_true.ravel() - y_pred.ravel()
    actual = y_true.ravel()
    predicted = y_pred.ravel()

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    error = actual - predicted
    q = 0.9
    pinball = float(np.mean(np.maximum(q * error, (q - 1) * error)))

    metrics = {
        "pinball_90": round(pinball, 6),
    }
    if include_secondary_diagnostics:
        metrics["MAE"] = round(mae, 6)
        metrics["RMSE"] = round(rmse, 6)
    return metrics


def compute_mm256_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    include_secondary_diagnostics: bool = True,
) -> pd.DataFrame:
    """Compute per-step forecast metrics for one prediction source."""
    records = []
    horizon = y_true.shape[1]
    for forecast_step in range(horizon):
        step_true = y_true[:, forecast_step:forecast_step + 1, :]
        step_pred = y_pred[:, forecast_step:forecast_step + 1, :]
        step_metrics = compute_mm256_metrics(
            step_true,
            step_pred,
            include_secondary_diagnostics=include_secondary_diagnostics,
        )
        bias = float(np.mean(step_pred.ravel() - step_true.ravel()))
        records.append(
            {
                "forecast_step": forecast_step + 1,
                "label": label,
                **step_metrics,
                "bias": round(bias, 6),
            }
    )
    return pd.DataFrame(records)


def select_validation_monitor_subset(
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the most recent validation windows for early stopping."""
    if max_windows is None or max_windows <= 0 or X_val.shape[0] <= max_windows:
        return X_val, y_val
    return X_val[-max_windows:], y_val[-max_windows:]


def inference_batch_size(batch_size: int) -> int:
    """Use a larger batch for validation / inference than for training."""
    return max(int(batch_size), 512)


def build_last_input_baseline(X: np.ndarray, target_feature_idx: int, horizon: int) -> np.ndarray:
    """Repeat the last MM256 value seen in the input window across the horizon."""
    if X.size == 0:
        return np.empty((0, horizon, 1), dtype=np.float32)

    last_seen = X[:, -1, target_feature_idx].astype(np.float32)
    return np.repeat(last_seen[:, None, None], horizon, axis=1)


def _merge_model_and_baseline_metrics(
    model_metrics: dict,
    baseline_metrics: dict,
    metric_keys: tuple[str, ...],
) -> dict:
    merged = dict(model_metrics)
    merged.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
    for key in metric_keys:
        merged[f"improvement_vs_baseline_{key}"] = round(
            baseline_metrics[key] - model_metrics[key],
            6,
        )
    return merged


def _aggregate_cv_metrics(
    ok_folds: list[dict],
    n_splits: int,
    metric_keys: tuple[str, ...],
) -> dict:
    if not ok_folds:
        return {"error": "All folds skipped"}

    agg = {}
    for key in metric_keys:
        vals = [m[key] for m in ok_folds]
        baseline_vals = [m[f"baseline_{key}"] for m in ok_folds]
        gain_vals = [m[f"improvement_vs_baseline_{key}"] for m in ok_folds]

        agg[f"{key}_mean"] = round(float(np.mean(vals)), 6)
        agg[f"{key}_std"] = round(float(np.std(vals)), 6)
        agg[f"baseline_{key}_mean"] = round(float(np.mean(baseline_vals)), 6)
        agg[f"baseline_{key}_std"] = round(float(np.std(baseline_vals)), 6)
        agg[f"improvement_vs_baseline_{key}_mean"] = round(float(np.mean(gain_vals)), 6)
        agg[f"improvement_vs_baseline_{key}_std"] = round(float(np.std(gain_vals)), 6)

    best_epochs = [m["best_epoch"] for m in ok_folds]
    agg["recommended_epochs"] = int(round(float(np.median(best_epochs))))
    agg["n_folds_ok"] = len(ok_folds)
    agg["n_folds_skipped"] = n_splits - len(ok_folds)
    return agg


def _aggregate_horizon_metrics(horizon_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-step metrics across folds."""
    if not horizon_frames:
        return pd.DataFrame()

    all_horizon = pd.concat(horizon_frames, ignore_index=True)
    agg_spec = {
        "pinball_mean": ("pinball_90", "mean"),
        "pinball_std": ("pinball_90", "std"),
        "bias_mean": ("bias", "mean"),
        "bias_std": ("bias", "std"),
        "n_folds": ("forecast_step", "size"),
    }
    if "MAE" in all_horizon.columns:
        agg_spec["MAE_mean"] = ("MAE", "mean")
        agg_spec["MAE_std"] = ("MAE", "std")
    if "RMSE" in all_horizon.columns:
        agg_spec["RMSE_mean"] = ("RMSE", "mean")
        agg_spec["RMSE_std"] = ("RMSE", "std")
    summary = (
        all_horizon.groupby(["label", "forecast_step"])
        .agg(**agg_spec)
        .reset_index()
        .fillna(0.0)
    )
    return summary


def _plot_horizon_summary(horizon_summary: pd.DataFrame, output_path: str):
    """Plot per-step error profiles for model vs baseline."""
    if horizon_summary.empty:
        return
    plt = _get_pyplot()

    plot_specs = [("Pinball q=0.9", "pinball_mean", "pinball_std")]
    if {"MAE_mean", "MAE_std"}.issubset(horizon_summary.columns):
        plot_specs.append(("MAE", "MAE_mean", "MAE_std"))
    if {"RMSE_mean", "RMSE_std"}.issubset(horizon_summary.columns):
        plot_specs.append(("RMSE", "RMSE_mean", "RMSE_std"))
    plot_specs.append(("Bias", "bias_mean", "bias_std"))

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(6 * len(plot_specs), 4.5), sharex=True)
    if len(plot_specs) == 1:
        axes = [axes]
    color_map = {"model": "#1565c0", "baseline": "#ef6c00"}

    for ax, (title, mean_col, std_col) in zip(axes, plot_specs):
        for label in ("model", "baseline"):
            subset = horizon_summary[horizon_summary["label"] == label].sort_values("forecast_step")
            if subset.empty:
                continue
            x = subset["forecast_step"].to_numpy()
            mean_vals = subset[mean_col].to_numpy()
            std_vals = subset[std_col].to_numpy()
            color = color_map[label]
            ax.plot(x, mean_vals, label=label.title(), color=color, linewidth=2)
            ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Forecast step (s)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    axes[0].legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_cv_mm256(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 300,
    window_length: int = 300,
    forecast_horizon: int = 120,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 5,
    model_variant: str = "advanced",
    push_bq: bool = False,
    validation_monitor_max_windows: int | None = 8192,
    save_plots: bool = False,
    use_catch22: bool = True,
    include_secondary_diagnostics: bool = False,
) -> dict:
    """Run cross-validation on a pre-split MM256 training dataframe."""
    import tensorflow as tf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_started = perf_counter()
    plots_dir = os.path.join(PROJECT_ROOT, "results", "graphs", "mm256_cv")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TimeSeriesSplit CV — MM256 — {n_splits} folds")
    print(f"  gap={gap}  window={window_length}s  horizon={forecast_horizon}s")
    print(f"  model={model_variant}  timestamp={timestamp}")
    print(f"{'='*60}\n")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    indices = np.arange(len(train_df))
    metric_keys = get_metric_keys(include_secondary_diagnostics=include_secondary_diagnostics)
    fold_metrics = []
    fold_histories = []
    fold_horizon_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(indices), start=1):
        fold_t = perf_counter()
        print(f"{'─'*50}")
        print(f"Fold {fold_idx}/{n_splits}")

        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]
        print(f"  Train rows: {len(fold_train_df):,}  |  Val rows: {len(fold_val_df):,}")
        print(f"  Train period: {fold_train_df.index.min()} -> {fold_train_df.index.max()}")
        print(f"  Val period:   {fold_val_df.index.min()} -> {fold_val_df.index.max()}")

        scaled_train, scaled_val, _ = scale_fold(fold_train_df, fold_val_df)
        X_train, y_train = slice_windows_mm256(
            scaled_train,
            0,
            len(scaled_train),
            window_length,
            forecast_horizon,
        )
        X_val, y_val = slice_windows_mm256(
            scaled_val,
            0,
            len(scaled_val),
            window_length,
            forecast_horizon,
        )
        X_train_c22 = None
        X_val_c22 = None

        feature_cols = [col for col in scaled_val.columns if col != "ALERT"]
        target_feature_idx = feature_cols.index(TARGET_SENSOR)
        X_val_monitor_seq, y_val_monitor = select_validation_monitor_subset(
            X_val,
            y_val,
            validation_monitor_max_windows,
        )
        predict_batch_size = inference_batch_size(batch_size)

        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        if X_val_monitor_seq.shape[0] != X_val.shape[0]:
            print(
                f"  Val monitor subset: {X_val_monitor_seq.shape[0]:,}"
                f" / {X_val.shape[0]:,} windows"
            )

        if X_train.shape[0] == 0:
            print("  WARNING: No training windows — skipping fold.")
            fold_metrics.append({"fold": fold_idx, "status": "skipped", "reason": "no_train_windows"})
            continue
        if X_val.shape[0] == 0:
            print("  WARNING: No validation windows — skipping fold.")
            fold_metrics.append({"fold": fold_idx, "status": "skipped", "reason": "no_val_windows"})
            continue

        if use_catch22:
            X_train_c22, X_val_c22, _, catch22_meta = fit_transform_catch22_windows(
                X_train,
                X_val,
            )
            X_val_monitor_c22 = X_val_c22[-X_val_monitor_seq.shape[0]:]
            print(
                f"  Catch22 static features: {X_train_c22.shape[1]}"
                f" (base sequence features: {catch22_meta['n_base_sequence_features']})"
            )
        else:
            X_val_monitor_c22 = None

        model = build_mm256_model(
            variant=model_variant,
            input_length=X_train.shape[1],
            n_features=X_train.shape[2],
            forecast_horizon=y_train.shape[1],
            n_targets=y_train.shape[2],
            n_static_features=0 if X_train_c22 is None else X_train_c22.shape[1],
        )
        train_inputs = build_mm256_model_inputs(X_train, X_train_c22)
        val_inputs = build_mm256_model_inputs(X_val, X_val_c22)
        val_monitor_inputs = build_mm256_model_inputs(X_val_monitor_seq, X_val_monitor_c22)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
        history = model.fit(
            train_inputs,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_monitor_inputs, y_val_monitor),
            validation_batch_size=predict_batch_size,
            callbacks=[early_stop],
            verbose=2,
        )
        fold_histories.append(history)

        y_pred = model.predict(val_inputs, batch_size=predict_batch_size, verbose=0)
        y_baseline = build_last_input_baseline(X_val, target_feature_idx, y_val.shape[1])
        fold_horizon = pd.concat(
            [
                compute_mm256_horizon_metrics(
                    y_val,
                    y_pred,
                    label="model",
                    include_secondary_diagnostics=include_secondary_diagnostics,
                ),
                compute_mm256_horizon_metrics(
                    y_val,
                    y_baseline,
                    label="baseline",
                    include_secondary_diagnostics=include_secondary_diagnostics,
                ),
            ],
            ignore_index=True,
        )
        fold_horizon["fold"] = fold_idx
        fold_horizon_metrics.append(fold_horizon)

        metrics = _merge_model_and_baseline_metrics(
            compute_mm256_metrics(
                y_val,
                y_pred,
                include_secondary_diagnostics=include_secondary_diagnostics,
            ),
            compute_mm256_metrics(
                y_val,
                y_baseline,
                include_secondary_diagnostics=include_secondary_diagnostics,
            ),
            metric_keys=metric_keys,
        )
        metrics["fold"] = fold_idx
        metrics["status"] = "ok"
        metrics["n_train_windows"] = int(X_train.shape[0])
        metrics["n_val_windows"] = int(X_val.shape[0])
        metrics["n_val_monitor_windows"] = int(X_val_monitor_seq.shape[0])
        metrics["use_catch22"] = bool(use_catch22)
        metrics["n_catch22_features"] = int(0 if X_train_c22 is None else X_train_c22.shape[1])
        metrics["n_epochs_trained"] = len(history.history["loss"])
        metrics["best_epoch"] = int(np.argmin(history.history["val_loss"]) + 1)
        metrics["best_val_loss"] = float(np.min(history.history["val_loss"]))
        metrics["train_period"] = f"{fold_train_df.index.min()} -> {fold_train_df.index.max()}"
        metrics["val_period"] = f"{fold_val_df.index.min()} -> {fold_val_df.index.max()}"
        fold_metrics.append(metrics)

        print(
            f"  Pinball q=0.9: model={metrics['pinball_90']:.5f}"
            f" | baseline={metrics['baseline_pinball_90']:.5f}"
            f" | gain={metrics['improvement_vs_baseline_pinball_90']:.5f}"
        )
        if include_secondary_diagnostics:
            print(
                f"  MAE: model={metrics['MAE']:.5f}"
                f" | baseline={metrics['baseline_MAE']:.5f}"
                f" | gain={metrics['improvement_vs_baseline_MAE']:.5f}"
            )
            print(
                f"  RMSE: model={metrics['RMSE']:.5f}"
                f" | baseline={metrics['baseline_RMSE']:.5f}"
                f" | gain={metrics['improvement_vs_baseline_RMSE']:.5f}"
            )

        if save_plots:
            plt = _get_pyplot()

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

            sample_idx = min(len(y_val) - 1, len(y_val) // 2)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(y_val[sample_idx, :, 0], label="Actual", linewidth=2)
            ax.plot(y_pred[sample_idx, :, 0], label=f"{model_variant.title()} LSTM", linestyle=":", linewidth=2)
            ax.plot(y_baseline[sample_idx, :, 0], label="Last value seen", linestyle="--", linewidth=2)
            ax.set_title(f"Fold {fold_idx} — MM256 Forecast (sample {sample_idx})")
            ax.set_xlabel("Forecast step (seconds)")
            ax.set_ylabel("MM256 (scaled)")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"fold{fold_idx}_sample_{timestamp}.png"), dpi=150)
            plt.close(fig)

        print(f"  Fold {fold_idx} done in {perf_counter() - fold_t:.1f}s\n")
        del model, X_train, y_train, X_val, y_val, y_pred, y_baseline, scaled_train, scaled_val
        del X_train_c22, X_val_c22, X_val_monitor_seq, X_val_monitor_c22
        del train_inputs, val_inputs, val_monitor_inputs
        tf.keras.backend.clear_session()
        gc.collect()

    ok_folds = [metric for metric in fold_metrics if metric.get("status") == "ok"]
    agg = _aggregate_cv_metrics(ok_folds, n_splits, metric_keys=metric_keys)
    agg["use_catch22"] = bool(use_catch22)
    agg["include_secondary_diagnostics"] = bool(include_secondary_diagnostics)
    agg["n_catch22_features"] = int(
        max((metric.get("n_catch22_features", 0) for metric in ok_folds), default=0)
    )

    print(f"{'='*60}")
    print(f"  CV Summary — {len(ok_folds)}/{n_splits} folds completed")
    print(f"{'='*60}")
    if ok_folds:
        print(
            f"  Recommended epochs: {agg['recommended_epochs']}"
            f" (median best epoch across folds)"
        )
        print(
            f"  Pinball q=0.9: model={agg['pinball_90_mean']:.5f}"
            f" | baseline={agg['baseline_pinball_90_mean']:.5f}"
            f" | gain={agg['improvement_vs_baseline_pinball_90_mean']:.5f}"
        )
        if include_secondary_diagnostics:
            print(
                f"  MAE:  model={agg['MAE_mean']:.5f} | baseline={agg['baseline_MAE_mean']:.5f}"
                f" | gain={agg['improvement_vs_baseline_MAE_mean']:.5f}"
            )
            print(
                f"  RMSE: model={agg['RMSE_mean']:.5f} | baseline={agg['baseline_RMSE_mean']:.5f}"
                f" | gain={agg['improvement_vs_baseline_RMSE_mean']:.5f}"
            )

    metrics_df = pd.DataFrame(fold_metrics)
    results_dir = os.path.join(PROJECT_ROOT, "results", "cv_metrics")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"cv_mm256_{timestamp}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    horizon_metrics_path = os.path.join(results_dir, f"cv_mm256_horizon_{timestamp}.csv")
    horizon_summary_path = os.path.join(results_dir, f"cv_mm256_horizon_summary_{timestamp}.csv")
    horizon_plot_path = os.path.join(plots_dir, f"cv_horizon_summary_{timestamp}.png")

    horizon_df = pd.concat(fold_horizon_metrics, ignore_index=True) if fold_horizon_metrics else pd.DataFrame()
    horizon_df.to_csv(horizon_metrics_path, index=False)
    horizon_summary = _aggregate_horizon_metrics(fold_horizon_metrics)
    horizon_summary.to_csv(horizon_summary_path, index=False)
    if save_plots:
        _plot_horizon_summary(horizon_summary, horizon_plot_path)

    agg_path = os.path.join(results_dir, f"cv_mm256_aggregate_{timestamp}.json")
    with open(agg_path, "w", encoding="utf-8") as stream:
        json.dump(agg, stream, indent=2)

    print(f"\nMetrics saved -> {metrics_path}")
    print(f"Aggregate saved -> {agg_path}")
    if not horizon_df.empty:
        print(f"Horizon metrics saved -> {horizon_metrics_path}")
        print(f"Horizon summary saved -> {horizon_summary_path}")
        if save_plots:
            print(f"Horizon plot -> {horizon_plot_path}")

    if save_plots and len(ok_folds) > 1:
        plt = _get_pyplot()
        metric_specs = [("pinball_90", "seagreen")]
        if include_secondary_diagnostics:
            metric_specs = [
                ("pinball_90", "seagreen"),
                ("MAE", "steelblue"),
                ("RMSE", "tomato"),
            ]
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 4))
        if len(metric_specs) == 1:
            axes = [axes]
        fold_nums = [m["fold"] for m in ok_folds]
        for ax, (key, color) in zip(axes, metric_specs):
            vals = [m[key] for m in ok_folds]
            baseline_vals = [m[f"baseline_{key}"] for m in ok_folds]
            ax.bar(fold_nums, vals, color=color, alpha=0.8)
            ax.axhline(np.mean(vals), color="grey", ls="--", label=f"model={np.mean(vals):.4f}")
            ax.axhline(np.mean(baseline_vals), color="black", ls=":", label=f"baseline={np.mean(baseline_vals):.4f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"cv_metrics_summary_{timestamp}.png"), dpi=150)
        plt.close(fig)

    if push_bq and ok_folds:
        _push_cv_results_to_bq(metrics_df, agg, timestamp)

    total_time = perf_counter() - cv_started
    print(f"\nTotal CV time: {total_time:.1f}s")

    return {
        "timestamp": timestamp,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": agg,
        "fold_histories": fold_histories,
        "recommended_epochs": agg.get("recommended_epochs", epochs),
        "n_splits": n_splits,
        "model_variant": model_variant,
        "use_catch22": bool(use_catch22),
        "include_secondary_diagnostics": bool(include_secondary_diagnostics),
        "metrics_path": metrics_path,
        "aggregate_path": agg_path,
        "horizon_metrics_path": horizon_metrics_path,
        "horizon_summary_path": horizon_summary_path,
        "horizon_plot_path": horizon_plot_path if save_plots else None,
    }


def _push_cv_results_to_bq(metrics_df: pd.DataFrame, agg: dict, timestamp: str):
    """Push per-fold metrics and aggregate summary to BigQuery."""
    from google.cloud import bigquery

    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_OUTPUT_DATASET") or get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    client = bigquery.Client(project=project, location=region)

    fold_table = f"{project}.{dataset}.cv_mm256_folds_{timestamp}"
    upload_df = metrics_df.copy()
    upload_df["run_timestamp"] = timestamp
    client.load_table_from_dataframe(upload_df, fold_table).result()
    print(f"Fold metrics -> BQ: {fold_table}")

    agg_table = f"{project}.{dataset}.cv_mm256_aggregate_{timestamp}"
    agg_df = pd.DataFrame([agg])
    agg_df["run_timestamp"] = timestamp
    client.load_table_from_dataframe(agg_df, agg_table).result()
    print(f"Aggregate metrics -> BQ: {agg_table}")


def main():
    parser = argparse.ArgumentParser(description="TimeSeriesSplit cross-validation for MM256")
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=300)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--forecast-horizon", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-monitor-max-windows", type=int, default=8192)
    parser.add_argument("--model-variant", choices=["simple", "advanced"], default="advanced")
    parser.add_argument("--push-bq", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--use-catch22", dest="use_catch22", action="store_true")
    parser.add_argument("--disable-catch22", dest="use_catch22", action="store_false")
    parser.add_argument("--include-secondary-diagnostics", action="store_true")
    parser.set_defaults(use_catch22=True)
    args = parser.parse_args()

    data, _, _ = preprocess_mm256(
        source=args.source,
        cache_raw=args.cache_raw,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        scale=False,
    )
    split_idx = int(len(data) * args.train_ratio)
    train_df = data.iloc[:split_idx].copy()

    run_cv_mm256(
        train_df=train_df,
        n_splits=args.n_splits,
        gap=args.gap,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        model_variant=args.model_variant,
        push_bq=args.push_bq,
        validation_monitor_max_windows=args.validation_monitor_max_windows,
        save_plots=args.save_plots,
        use_catch22=args.use_catch22,
        include_secondary_diagnostics=args.include_secondary_diagnostics,
    )


if __name__ == "__main__":
    main()
