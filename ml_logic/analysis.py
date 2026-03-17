"""Generate analysis graphs from predictions and training history"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SENSORS = ["MM256", "MM263", "MM264"]


def plot_loss_curves(history, timestamp):
    """Plot training vs validation loss."""
    os.makedirs("results/graphs", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/graphs/loss_{timestamp}.png", dpi=150)
    plt.close()
    print(f"Loss curve -> results/graphs/loss_{timestamp}.png")


def plot_predictions_vs_actual(pred_df, timestamp):
    """Plot predicted vs actual for each sensor."""
    os.makedirs("results/graphs", exist_ok=True)

    for sensor in SENSORS:
        sensor_df = pred_df[pred_df["sensor"] == sensor]

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
        plt.savefig(f"results/graphs/{sensor}_{timestamp}.png", dpi=150)
        plt.close()

    print(f"Sensor graphs -> results/graphs/")


def compute_metrics(pred_df, timestamp):
    """Compute error metrics per sensor."""
    os.makedirs("results/graphs", exist_ok=True)

    metrics = []
    for sensor in SENSORS:
        s = pred_df[pred_df["sensor"] == sensor]
        mae = np.mean(np.abs(s["residual"]))
        rmse = np.sqrt(np.mean(s["residual"] ** 2))
        mape = np.mean(np.abs(s["residual"] / (s["actual"] + 1e-8))) * 100
        metrics.append({"sensor": sensor, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"results/graphs/metrics_{timestamp}.csv", index=False)

    print(f"\nModel Performance:")
    print(metrics_df.to_string(index=False))
    return metrics_df
