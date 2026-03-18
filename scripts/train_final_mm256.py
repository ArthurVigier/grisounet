"""Final MM256 training on the full train split, followed by holdout test evaluation."""

import json
import os
import sys
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_logic.analysis import compute_metrics, plot_loss_curves, plot_predictions_vs_actual
from ml_logic.data import save_preprocessing_artifact
from ml_logic.model_mm256 import build_mm256_model
from ml_logic.model_save import save_model_artifacts_to_gcs, save_model_to_gcs
from ml_logic.results_bq_save import save_history_to_bq, save_predictions_to_bq
from scripts.cv_time_series import build_last_input_baseline, compute_mm256_metrics, inference_batch_size
from scripts.preprocessor_MM256 import TARGET_SENSOR, scale_fold, slice_windows_mm256


def _build_prediction_df(y_true: np.ndarray, y_pred: np.ndarray, timestamp: str, table_suffix: str) -> pd.DataFrame:
    sample_count, horizon, sensor_count = y_true.shape
    actual = y_true.reshape(-1)
    predicted = y_pred.reshape(-1)
    pred_df = pd.DataFrame({
        "sample_id": np.repeat(np.arange(sample_count), horizon * sensor_count),
        "forecast_step": np.tile(np.repeat(np.arange(horizon), sensor_count), sample_count),
        "sensor": np.tile(np.array([TARGET_SENSOR]), sample_count * horizon),
        "actual": actual.astype(float),
        "predicted": predicted.astype(float),
        "residual": (actual - predicted).astype(float),
        "run_timestamp": timestamp,
    })

    os.makedirs("results/predictions", exist_ok=True)
    pred_path = f"results/predictions/predictions_{table_suffix}_{timestamp}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved locally -> {pred_path}")
    return pred_df


def _save_history_local(history, timestamp: str, table_suffix: str) -> str:
    os.makedirs("results/model_history", exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df["run_timestamp"] = timestamp
    history_path = f"results/model_history/history_{table_suffix}_{timestamp}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"History saved locally -> {history_path}")
    return history_path


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def train_final_mm256(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    recommended_epochs: int,
    batch_size: int = 128,
    window_length: int = 300,
    forecast_horizon: int = 120,
    model_variant: str = "advanced",
    push_bq: bool = False,
    save_preprocess: bool = False,
    upload_preprocess: bool = False,
    save_analysis_outputs: bool = False,
) -> dict:
    """Train on the full train split and evaluate once on the holdout test split."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = perf_counter()

    print(f"\n{'='*60}")
    print("  Final MM256 Training")
    print(f"  model={model_variant}  epochs={recommended_epochs}  timestamp={timestamp}")
    print(f"{'='*60}\n")

    scaled_train, scaled_test, scalers = scale_fold(train_df, test_df)
    X_train, y_train = slice_windows_mm256(
        scaled_train,
        0,
        len(scaled_train),
        window_length,
        forecast_horizon,
    )
    X_test, y_test = slice_windows_mm256(
        scaled_test,
        0,
        len(scaled_test),
        window_length,
        forecast_horizon,
    )

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    if X_train.shape[0] == 0:
        return {"error": "no_training_windows"}
    if X_test.shape[0] == 0:
        return {"error": "no_test_windows"}

    if save_preprocess:
        save_preprocessing_artifact(
            X_train,
            X_test,
            y_train,
            y_test,
            timestamp=f"mm256_{timestamp}",
            upload_to_gcs=upload_preprocess,
        )

    feature_cols = [col for col in scaled_test.columns if col != "ALERT"]
    target_feature_idx = feature_cols.index(TARGET_SENSOR)

    model = build_mm256_model(
        variant=model_variant,
        input_length=X_train.shape[1],
        n_features=X_train.shape[2],
        forecast_horizon=y_train.shape[1],
        n_targets=y_train.shape[2],
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=recommended_epochs,
        batch_size=batch_size,
        verbose=2,
    )
    predict_batch_size = inference_batch_size(batch_size)
    y_pred = model.predict(X_test, batch_size=predict_batch_size, verbose=0)
    y_baseline = build_last_input_baseline(X_test, target_feature_idx, y_test.shape[1])

    model_metrics = compute_mm256_metrics(y_test, y_pred)
    baseline_metrics = compute_mm256_metrics(y_test, y_baseline)
    improvement_metrics = {
        f"improvement_vs_baseline_{key}": round(
            baseline_metrics[key] - model_metrics[key],
            6 if key != "MAPE_%" else 4,
        )
        for key in model_metrics
    }

    model_timestamp = save_model_to_gcs(model, f"mm256_{timestamp}")
    metadata = {
        "target_sensor": TARGET_SENSOR,
        "model_variant": model_variant,
        "recommended_epochs": int(recommended_epochs),
        "window_length": int(window_length),
        "forecast_horizon": int(forecast_horizon),
        "train_period": f"{train_df.index.min()} -> {train_df.index.max()}",
        "test_period": f"{test_df.index.min()} -> {test_df.index.max()}",
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "feature_columns": feature_cols,
    }
    artifact_paths = save_model_artifacts_to_gcs(model_timestamp, scalers=scalers, metadata=metadata)

    pred_df = None
    if push_bq:
        save_history_to_bq(history, timestamp, table_suffix="mm256_final")
        pred_df = save_predictions_to_bq(
            y_test,
            y_pred,
            timestamp,
            sensors=[TARGET_SENSOR],
            table_suffix="mm256_final",
        )
    elif save_analysis_outputs:
        _save_history_local(history, timestamp, "mm256_final")
        pred_df = _build_prediction_df(y_test, y_pred, timestamp, "mm256_final")

    sample_plot_path = None
    metrics_df = pd.DataFrame([{"sensor": TARGET_SENSOR, **model_metrics}])
    if save_analysis_outputs or push_bq:
        plot_loss_curves(history, timestamp, label_prefix="mm256_final_")

        plt = _get_pyplot()
        sample_idx = min(100, y_test.shape[0] - 1)
        plt.figure(figsize=(12, 5))
        plt.plot(y_test[sample_idx, :, 0], label="Actual", linewidth=2)
        plt.plot(y_pred[sample_idx, :, 0], label=f"{model_variant.title()} LSTM", linestyle=":")
        plt.plot(y_baseline[sample_idx, :, 0], label="Last value seen", linestyle="--")
        plt.title(f"MM256 — holdout test sample {sample_idx}")
        plt.xlabel("Forecast step (s)")
        plt.ylabel("MM256 (scaled)")
        plt.legend()
        os.makedirs("results/graphs", exist_ok=True)
        sample_plot_path = f"results/graphs/mm256_final_forecast_{timestamp}.png"
        plt.savefig(sample_plot_path, dpi=150)
        plt.close()
        print(f"Sample forecast -> {sample_plot_path}")

        if pred_df is not None:
            metrics_df = compute_metrics(pred_df, timestamp, sensors=[TARGET_SENSOR], label_prefix="mm256_final_")
            plot_predictions_vs_actual(pred_df, timestamp, sensors=[TARGET_SENSOR], label_prefix="mm256_final_")

    final_metrics = {
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        **improvement_metrics,
        "model_variant": model_variant,
        "recommended_epochs": int(recommended_epochs),
        "model_timestamp": model_timestamp,
    }
    os.makedirs("results/final_metrics", exist_ok=True)
    metrics_path = f"results/final_metrics/mm256_final_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(final_metrics, stream, indent=2)
    print(f"Final metrics -> {metrics_path}")

    print(f"\nTotal final training time: {perf_counter() - started:.1f}s")

    return {
        "timestamp": timestamp,
        "model_timestamp": model_timestamp,
        "history": history,
        "predictions": pred_df,
        "metrics_df": metrics_df,
        "final_metrics": final_metrics,
        "artifact_paths": artifact_paths,
        "sample_plot_path": sample_plot_path,
    }
