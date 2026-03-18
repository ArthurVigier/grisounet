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
from ml_logic.results_bq_save import (
    build_prediction_frame,
    save_history_to_bq,
    save_predictions_to_bq,
)
from scripts.cv_time_series import build_last_input_baseline, compute_mm256_metrics, inference_batch_size
from scripts.preprocessor_MM256 import (
    TARGET_SENSOR,
    build_mm256_model_inputs,
    build_window_index_mm256,
    fit_transform_catch22_windows,
    scale_fold,
    slice_windows_mm256,
)


def _build_prediction_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamp: str,
    table_suffix: str,
    window_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred_df = build_prediction_frame(
        y_true,
        y_pred,
        timestamp=timestamp,
        sensors=[TARGET_SENSOR],
        window_index=window_index,
    )
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
    use_catch22: bool = True,
    include_secondary_diagnostics: bool = False,
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
    test_window_index = build_window_index_mm256(
        test_df,
        0,
        len(test_df),
        window_length,
        forecast_horizon,
    )
    X_train_c22 = None
    X_test_c22 = None

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print(f"  Window index rows: {len(test_window_index)}")

    if X_train.shape[0] == 0:
        return {"error": "no_training_windows"}
    if X_test.shape[0] == 0:
        return {"error": "no_test_windows"}
    if len(test_window_index) != X_test.shape[0]:
        raise ValueError(
            "MM256 test window index does not match the number of generated test windows"
        )

    catch22_scalers = None
    catch22_meta = {"enabled": False, "n_catch22_features": 0}
    if use_catch22:
        X_train_c22, X_test_c22, catch22_scalers, catch22_meta = fit_transform_catch22_windows(
            X_train,
            X_test,
        )
        print(
            f"  Catch22 static features: {X_train_c22.shape[1]}"
            f" (base sequence features: {catch22_meta['n_base_sequence_features']})"
        )

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
        n_static_features=0 if X_train_c22 is None else X_train_c22.shape[1],
    )
    train_inputs = build_mm256_model_inputs(X_train, X_train_c22)
    test_inputs = build_mm256_model_inputs(X_test, X_test_c22)
    history = model.fit(
        train_inputs,
        y_train,
        epochs=recommended_epochs,
        batch_size=batch_size,
        verbose=2,
    )
    predict_batch_size = inference_batch_size(batch_size)
    y_pred = model.predict(test_inputs, batch_size=predict_batch_size, verbose=0)
    y_baseline = build_last_input_baseline(X_test, target_feature_idx, y_test.shape[1])

    model_metrics = compute_mm256_metrics(
        y_test,
        y_pred,
        include_secondary_diagnostics=include_secondary_diagnostics,
    )
    baseline_metrics = compute_mm256_metrics(
        y_test,
        y_baseline,
        include_secondary_diagnostics=include_secondary_diagnostics,
    )
    improvement_metrics = {
        f"improvement_vs_baseline_{key}": round(
            baseline_metrics[key] - model_metrics[key],
            6,
        )
        for key in model_metrics
    }

    print(
        f"  Pinball q=0.9: model={model_metrics['pinball_90']:.5f}"
        f" | baseline={baseline_metrics['pinball_90']:.5f}"
        f" | gain={improvement_metrics['improvement_vs_baseline_pinball_90']:.5f}"
    )
    if include_secondary_diagnostics:
        print(
            f"  MAE: model={model_metrics['MAE']:.5f}"
            f" | baseline={baseline_metrics['MAE']:.5f}"
            f" | gain={improvement_metrics['improvement_vs_baseline_MAE']:.5f}"
        )
        print(
            f"  RMSE: model={model_metrics['RMSE']:.5f}"
            f" | baseline={baseline_metrics['RMSE']:.5f}"
            f" | gain={improvement_metrics['improvement_vs_baseline_RMSE']:.5f}"
        )

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
        "catch22": catch22_meta,
    }
    scaler_bundle = {
        "raw_feature_scalers": scalers,
        "catch22_feature_scalers": catch22_scalers,
    }
    artifact_paths = save_model_artifacts_to_gcs(
        model_timestamp,
        scalers=scaler_bundle,
        metadata=metadata,
    )

    pred_df = None
    if push_bq:
        save_history_to_bq(history, timestamp, table_suffix="mm256_final")
        pred_df = save_predictions_to_bq(
            y_test,
            y_pred,
            timestamp,
            sensors=[TARGET_SENSOR],
            table_suffix="mm256_final",
            window_index=test_window_index,
        )
    elif save_analysis_outputs:
        _save_history_local(history, timestamp, "mm256_final")
        pred_df = _build_prediction_df(
            y_test,
            y_pred,
            timestamp,
            "mm256_final",
            window_index=test_window_index,
        )

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
        "use_catch22": bool(catch22_meta["enabled"]),
        "include_secondary_diagnostics": bool(include_secondary_diagnostics),
        "n_catch22_features": int(catch22_meta["n_catch22_features"]),
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
