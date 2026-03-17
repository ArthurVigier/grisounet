"""Learning-curve diagnostics for MM256 underfitting / overfitting analysis.

The script keeps the final holdout test untouched. It takes the train split of
the MM256 workflow, carves out a fixed validation block at the end of that
train split, then retrains the chosen architecture on progressively larger
prefixes of the remaining train pool.
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.model_mm256 import build_mm256_model
from scripts.cv_time_series import build_last_input_baseline, compute_mm256_metrics
from scripts.preprocessor_MM256 import TARGET_SENSOR, preprocess_mm256, scale_fold, slice_windows_mm256


def _parse_train_fractions(raw: str) -> list[float]:
    fractions = sorted({float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()})
    if not fractions:
        raise ValueError("No train fractions were provided")
    for value in fractions:
        if value <= 0 or value > 1:
            raise ValueError("Train fractions must be in the interval (0, 1]")
    return fractions


def _split_temporal_holdout(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(data) * train_ratio)
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("train_ratio produces an empty train or test split")
    return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()


def _split_train_validation(train_df: pd.DataFrame, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(train_df) * (1 - validation_ratio))
    if split_idx <= 0 or split_idx >= len(train_df):
        raise ValueError("validation_ratio produces an empty train-pool or validation split")
    return train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()


def _plot_learning_curve_summary(results_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].plot(results_df["train_fraction"], results_df["min_train_loss"], marker="o", label="Min train loss")
    axes[0].plot(results_df["train_fraction"], results_df["min_val_loss"], marker="o", label="Min val loss")
    axes[0].set_title("Loss vs train size")
    axes[0].set_xlabel("Train fraction")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(results_df["train_fraction"], results_df["generalization_gap"], marker="o", color="#c62828")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Generalization gap")
    axes[1].set_xlabel("Train fraction")
    axes[1].set_ylabel("Min val loss - min train loss")
    axes[1].grid(alpha=0.25)

    axes[2].plot(results_df["train_fraction"], results_df["MAE"], marker="o", label="Model MAE", color="#1565c0")
    axes[2].plot(results_df["train_fraction"], results_df["baseline_MAE"], marker="o", label="Baseline MAE", color="#ef6c00")
    axes[2].set_title("Validation MAE vs train size")
    axes[2].set_xlabel("Train fraction")
    axes[2].set_ylabel("MAE")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_learning_curve_histories(history_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharex=False)

    for train_fraction, subset in history_df.groupby("train_fraction"):
        label = f"{train_fraction:.0%}"
        axes[0].plot(subset["epoch"], subset["train_loss"], label=label)
        axes[1].plot(subset["epoch"], subset["val_loss"], label=label)

    axes[0].set_title("Train loss history")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Validation loss history")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Train fraction")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_mm256_learning_curves(
    source: str = "local",
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    train_fractions: list[float] | None = None,
    window_length: int = 300,
    forecast_horizon: int = 120,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 5,
    model_variant: str = "advanced",
) -> dict:
    import tensorflow as tf

    if train_fractions is None:
        train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "results", "learning_curves")
    os.makedirs(output_dir, exist_ok=True)

    data, _, _ = preprocess_mm256(
        source=source,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
        scale=False,
    )
    train_df, _ = _split_temporal_holdout(data, train_ratio=train_ratio)
    train_pool_df, val_df = _split_train_validation(train_df, validation_ratio=validation_ratio)

    results = []
    histories = []

    print(f"\n{'='*60}")
    print("  MM256 Learning Curves")
    print(f"  model={model_variant}  fractions={train_fractions}")
    print(f"  train period: {train_pool_df.index.min()} -> {train_pool_df.index.max()}")
    print(f"  val period:   {val_df.index.min()} -> {val_df.index.max()}")
    print(f"{'='*60}\n")

    for train_fraction in train_fractions:
        subtrain_rows = max(1, int(len(train_pool_df) * train_fraction))
        subtrain_df = train_pool_df.iloc[:subtrain_rows].copy()
        print(f"Train fraction {train_fraction:.0%} -> {len(subtrain_df):,} rows")

        scaled_train, scaled_val, _ = scale_fold(subtrain_df, val_df)
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
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")

        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            print("  WARNING: skipping fraction because no windows are available.\n")
            continue

        feature_cols = [col for col in scaled_val.columns if col != "ALERT"]
        target_feature_idx = feature_cols.index(TARGET_SENSOR)

        model = build_mm256_model(
            variant=model_variant,
            input_length=X_train.shape[1],
            n_features=X_train.shape[2],
            forecast_horizon=y_train.shape[1],
            n_targets=y_train.shape[2],
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1,
        )
        y_pred = model.predict(X_val, batch_size=batch_size)
        y_baseline = build_last_input_baseline(X_val, target_feature_idx, y_val.shape[1])

        model_metrics = compute_mm256_metrics(y_val, y_pred)
        baseline_metrics = compute_mm256_metrics(y_val, y_baseline)
        record = {
            "train_fraction": float(train_fraction),
            "n_train_rows": int(len(subtrain_df)),
            "n_val_rows": int(len(val_df)),
            "n_train_windows": int(X_train.shape[0]),
            "n_val_windows": int(X_val.shape[0]),
            "trained_epochs": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "min_train_loss": float(np.min(history.history["loss"])),
            "min_val_loss": float(np.min(history.history["val_loss"])),
            "generalization_gap": float(np.min(history.history["val_loss"]) - np.min(history.history["loss"])),
            **model_metrics,
            **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            "improvement_vs_baseline_MAE": round(baseline_metrics["MAE"] - model_metrics["MAE"], 6),
            "improvement_vs_baseline_RMSE": round(baseline_metrics["RMSE"] - model_metrics["RMSE"], 6),
        }
        results.append(record)

        history_df = pd.DataFrame(
            {
                "epoch": np.arange(1, len(history.history["loss"]) + 1),
                "train_loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "train_fraction": float(train_fraction),
            }
        )
        histories.append(history_df)
        print(
            f"  min train loss={record['min_train_loss']:.5f}"
            f" | min val loss={record['min_val_loss']:.5f}"
            f" | MAE={record['MAE']:.5f}"
            f" | baseline MAE={record['baseline_MAE']:.5f}\n"
        )

        del model, X_train, y_train, X_val, y_val, y_pred, y_baseline, scaled_train, scaled_val
        tf.keras.backend.clear_session()
        gc.collect()

    if not results:
        raise ValueError("No valid learning-curve runs were produced")

    results_df = pd.DataFrame(results).sort_values("train_fraction")
    history_df = pd.concat(histories, ignore_index=True)

    csv_path = os.path.join(output_dir, f"mm256_learning_curve_{timestamp}.csv")
    history_path = os.path.join(output_dir, f"mm256_learning_curve_history_{timestamp}.csv")
    summary_plot_path = os.path.join(output_dir, f"mm256_learning_curve_summary_{timestamp}.png")
    history_plot_path = os.path.join(output_dir, f"mm256_learning_curve_histories_{timestamp}.png")
    summary_json_path = os.path.join(output_dir, f"mm256_learning_curve_{timestamp}.json")

    results_df.to_csv(csv_path, index=False)
    history_df.to_csv(history_path, index=False)
    _plot_learning_curve_summary(results_df, summary_plot_path)
    _plot_learning_curve_histories(history_df, history_plot_path)

    summary = {
        "timestamp": timestamp,
        "model_variant": model_variant,
        "train_fractions": train_fractions,
        "train_period": f"{train_pool_df.index.min()} -> {train_pool_df.index.max()}",
        "validation_period": f"{val_df.index.min()} -> {val_df.index.max()}",
        "best_fraction_by_mae": float(results_df.loc[results_df["MAE"].idxmin(), "train_fraction"]),
        "max_generalization_gap": float(results_df["generalization_gap"].max()),
    }
    with open(summary_json_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(f"Learning-curve metrics -> {csv_path}")
    print(f"Learning-curve histories -> {history_path}")
    print(f"Learning-curve summary plot -> {summary_plot_path}")
    print(f"Learning-curve history plot -> {history_plot_path}")
    print(f"Learning-curve summary JSON -> {summary_json_path}")

    return {
        "timestamp": timestamp,
        "metrics_path": csv_path,
        "history_path": history_path,
        "summary_plot_path": summary_plot_path,
        "history_plot_path": history_plot_path,
        "summary_json_path": summary_json_path,
    }


def main():
    parser = argparse.ArgumentParser(description="MM256 learning curves for under/overfitting analysis")
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="local")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--train-fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--forecast-horizon", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model-variant", choices=["simple", "advanced"], default="advanced")
    args = parser.parse_args()

    run_mm256_learning_curves(
        source=args.source,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        train_fractions=_parse_train_fractions(args.train_fractions),
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        model_variant=args.model_variant,
    )


if __name__ == "__main__":
    main()
