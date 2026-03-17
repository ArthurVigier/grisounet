"""Generate analysis graphs from predictions and training history"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SENSORS_DEFAULT = ["MM256", "MM263", "MM264"]


def plot_loss_curves(history, timestamp, label_prefix=""):
    """Plot training vs validation loss.

    Parameters
    ----------
    label_prefix : str
        Optional prefix for the saved filename (e.g. ``"mm256_"``).
    """
    os.makedirs("results/graphs", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/graphs/{label_prefix}loss_{timestamp}.png", dpi=150)
    plt.close()
    print(f"Loss curve -> results/graphs/{label_prefix}loss_{timestamp}.png")


def plot_predictions_vs_actual(pred_df, timestamp, sensors=None, label_prefix=""):
    """Plot predicted vs actual for each sensor.

    Parameters
    ----------
    sensors : list[str], optional
        Sensor names to plot.  Defaults to ``["MM256", "MM263", "MM264"]``.
        Pass ``["MM256"]`` for the single-sensor pipeline.
    label_prefix : str
        Optional prefix for saved filenames.
    """
    if sensors is None:
        sensors = SENSORS_DEFAULT

    os.makedirs("results/graphs", exist_ok=True)

    for sensor in sensors:
        sensor_df = pred_df[pred_df["sensor"] == sensor]
        if sensor_df.empty:
            print(f"  No data for sensor {sensor}, skipping plot.")
            continue

        # Average across forecast steps per sample
        avg = sensor_df.groupby("sample_id")[["actual", "predicted"]].mean()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Time series overlay
        axes[0].plot(avg["actual"].values, label="Actual", alpha=0.8)
        axes[0].plot(avg["predicted"].values, label="Predicted", alpha=0.8)
        axes[0].set_title(f"{sensor} — Predicted vs Actual")
        axes[0].legend()

        # Residual distribution
        residuals = sensor_df["residual"]
        axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1].set_title(f"{sensor} — Residual Distribution (mean={residuals.mean():.4f})")
        axes[1].axvline(0, color="red", linestyle="--")

        plt.tight_layout()
        plt.savefig(f"results/graphs/{label_prefix}{sensor}_{timestamp}.png", dpi=150)
        plt.close()

    print(f"Sensor graphs -> results/graphs/")


def compute_metrics(pred_df, timestamp, sensors=None, label_prefix=""):
    """Compute error metrics per sensor.

    Parameters
    ----------
    sensors : list[str], optional
        Sensor names to evaluate.  Defaults to ``["MM256", "MM263", "MM264"]``.
        Pass ``["MM256"]`` for the single-sensor pipeline.
    label_prefix : str
        Optional prefix for saved filenames.

    Returns
    -------
    pd.DataFrame with columns [sensor, MAE, RMSE, MAPE_%].
    """
    if sensors is None:
        sensors = SENSORS_DEFAULT

    os.makedirs("results/graphs", exist_ok=True)

    metrics = []
    for sensor in sensors:
        s = pred_df[pred_df["sensor"] == sensor]
        if s.empty:
            continue
        mae = np.mean(np.abs(s["residual"]))
        rmse = np.sqrt(np.mean(s["residual"] ** 2))
        mape = np.mean(np.abs(s["residual"] / (s["actual"] + 1e-8))) * 100
        metrics.append({"sensor": sensor, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"results/graphs/{label_prefix}metrics_{timestamp}.csv", index=False)

    print(f"\nModel Performance:")
    print(metrics_df.to_string(index=False))
    return metrics_df
