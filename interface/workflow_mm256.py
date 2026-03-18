"""
MM256 single-sensor workflow orchestrator.

Default flow:
  1. Load and preprocess the full dataset without scaling.
  2. Split once into chronological train / holdout test blocks.
  3. Run TimeSeriesSplit cross-validation on the train block only.
  4. Retrain once on the full train block and evaluate once on the untouched test block.

The holdout test remains untouched during cross-validation, so the final metrics
are a proper benchmark instead of a model-selection estimate.
"""

import argparse
import os
import sys
from time import perf_counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

from scripts.cv_time_series import run_cv_mm256
from scripts.preprocessor_MM256 import preprocess_mm256
from scripts.train_final_mm256 import train_final_mm256

load_dotenv()


def _fmt(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining:.1f}s" if minutes >= 1 else f"{remaining:.1f}s"


def split_temporal_holdout(data, train_ratio: float = 0.7):
    """Split a time-ordered dataframe into chronological train and holdout test."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1")

    split_idx = int(len(data) * train_ratio)
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("train_ratio produces an empty train or test split")

    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    return train_df, test_df


def run_pipeline_mm256(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    clean_abnormal_values: bool = False,
    frozen_sensor_window: int = 3600,
    sensor_disagreement_z_threshold: float = 6.0,
    train_ratio: float = 0.7,
    n_splits: int = 5,
    gap: int = 300,
    window_length: int = 300,
    forecast_horizon: int = 120,
    epochs: int = 5,
    batch_size: int = 128,
    patience: int = 5,
    model_variant: str = "advanced",
    pinball_quantile: float = 0.8,
    skip_cv: bool = True,
    push_bq: bool = False,
    save_preprocess: bool = False,
    upload_preprocess: bool = False,
    validation_monitor_max_windows: int | None = 8192,
    save_cv_plots: bool = False,
    save_final_analysis: bool = False,
    use_catch22: bool = True,
    include_secondary_diagnostics: bool = False,
) -> dict:
    """Run the full MM256 benchmark workflow."""
    started = perf_counter()

    print(f"\n{'='*60}")
    print("  MM256 Workflow")
    print("  holdout split -> CV on train only -> final train -> holdout test")
    print(f"{'='*60}\n")

    step_t = perf_counter()
    print("Step 1/4 — Loading & preprocessing (unscaled)...")
    data, _, preprocessing_meta = preprocess_mm256(
        source=source,
        cache_raw=cache_raw,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
        scale=False,
        clean_abnormal_values=clean_abnormal_values,
        frozen_sensor_window=frozen_sensor_window,
        sensor_disagreement_z_threshold=sensor_disagreement_z_threshold,
    )
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    step_t = perf_counter()
    print("\nStep 2/4 — Temporal holdout split...")
    train_df, test_df = split_temporal_holdout(data, train_ratio=train_ratio)
    print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")
    print(f"  Train period: {train_df.index.min()} -> {train_df.index.max()}")
    print(f"  Test period:  {test_df.index.min()} -> {test_df.index.max()}")
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    step_t = perf_counter()
    print("\nStep 3/4 — Cross-validation on train split...")
    if skip_cv:
        cv_results = {
            "timestamp": None,
            "skipped": True,
            "recommended_epochs": int(epochs),
            "aggregate_metrics": {},
            "fold_metrics": [],
            "n_splits": 0,
            "model_variant": model_variant,
        }
        print(f"  CV skipped. Using epochs={epochs} for final training.")
    else:
        cv_results = run_cv_mm256(
            train_df=train_df,
            n_splits=n_splits,
            gap=gap,
            window_length=window_length,
            forecast_horizon=forecast_horizon,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            model_variant=model_variant,
            pinball_quantile=pinball_quantile,
            push_bq=push_bq,
            validation_monitor_max_windows=validation_monitor_max_windows,
            save_plots=save_cv_plots,
            use_catch22=use_catch22,
            include_secondary_diagnostics=include_secondary_diagnostics,
        )
    recommended_epochs = int(cv_results.get("recommended_epochs", epochs))
    print(f"  Recommended epochs for final fit: {recommended_epochs}")
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    step_t = perf_counter()
    print("\nStep 4/4 — Final training on full train split + holdout test...")
    final_results = train_final_mm256(
        train_df=train_df,
        test_df=test_df,
        recommended_epochs=recommended_epochs,
        batch_size=batch_size,
        window_length=window_length,
        forecast_horizon=forecast_horizon,
        model_variant=model_variant,
        pinball_quantile=pinball_quantile,
        push_bq=push_bq,
        save_preprocess=save_preprocess,
        upload_preprocess=upload_preprocess,
        save_analysis_outputs=save_final_analysis,
        use_catch22=use_catch22,
        include_secondary_diagnostics=include_secondary_diagnostics,
    )
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    total = perf_counter() - started
    print(f"\n{'='*60}")
    print(f"  MM256 workflow complete in {_fmt(total)}")
    print(f"{'='*60}")

    return {
        "preprocessing": preprocessing_meta,
        "split": {
            "train_ratio": float(train_ratio),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "train_period": f"{train_df.index.min()} -> {train_df.index.max()}",
            "test_period": f"{test_df.index.min()} -> {test_df.index.max()}",
        },
        "cv": cv_results,
        "final": final_results,
    }


def run_cv_pipeline_mm256(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    clean_abnormal_values: bool = False,
    frozen_sensor_window: int = 3600,
    sensor_disagreement_z_threshold: float = 6.0,
    train_ratio: float = 0.7,
    n_splits: int = 5,
    gap: int = 300,
    window_length: int = 300,
    forecast_horizon: int = 120,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 5,
    model_variant: str = "advanced",
    pinball_quantile: float = 0.8,
    push_bq: bool = False,
    validation_monitor_max_windows: int | None = 8192,
    save_cv_plots: bool = False,
    use_catch22: bool = True,
    include_secondary_diagnostics: bool = False,
) -> dict:
    """Run the CV stage only, using the train portion of a holdout split."""
    data, _, _ = preprocess_mm256(
        source=source,
        cache_raw=cache_raw,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
        scale=False,
        clean_abnormal_values=clean_abnormal_values,
        frozen_sensor_window=frozen_sensor_window,
        sensor_disagreement_z_threshold=sensor_disagreement_z_threshold,
    )
    train_df, _ = split_temporal_holdout(data, train_ratio=train_ratio)
    return run_cv_mm256(
        train_df=train_df,
        n_splits=n_splits,
        gap=gap,
        window_length=window_length,
        forecast_horizon=forecast_horizon,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        model_variant=model_variant,
        pinball_quantile=pinball_quantile,
        push_bq=push_bq,
        validation_monitor_max_windows=validation_monitor_max_windows,
        save_plots=save_cv_plots,
        use_catch22=use_catch22,
        include_secondary_diagnostics=include_secondary_diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description="MM256 single-sensor workflow")
    parser.add_argument("--mode", choices=["full", "cv", "single"], default="full")
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--enable-cleaning", dest="clean_abnormal_values", action="store_true")
    parser.add_argument("--frozen-sensor-window", type=int, default=3600)
    parser.add_argument("--sensor-disagreement-z-threshold", type=float, default=6.0)
    parser.set_defaults(clean_abnormal_values=False)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=300)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--forecast-horizon", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pinball-quantile", type=float, default=0.8)
    parser.add_argument("--validation-monitor-max-windows", type=int, default=8192)
    parser.add_argument("--model-variant", choices=["simple", "advanced"], default="advanced")
    parser.add_argument("--skip-cv", action="store_true", default=True)
    parser.add_argument("--run-cv", dest="skip_cv", action="store_false")
    parser.add_argument("--push-bq", action="store_true")
    parser.add_argument("--save-cv-plots", action="store_true")
    parser.add_argument("--save-final-analysis", action="store_true")
    parser.add_argument("--save-preprocess", dest="save_preprocess", action="store_true")
    parser.add_argument("--skip-preprocess-save", dest="save_preprocess", action="store_false")
    parser.set_defaults(save_preprocess=False)
    parser.add_argument("--upload-preprocess", action="store_true")
    parser.add_argument("--use-catch22", dest="use_catch22", action="store_true")
    parser.add_argument("--disable-catch22", dest="use_catch22", action="store_false")
    parser.add_argument("--include-secondary-diagnostics", action="store_true")
    parser.set_defaults(use_catch22=True)
    args = parser.parse_args()

    if args.mode == "cv":
        run_cv_pipeline_mm256(
            source=args.source,
            cache_raw=args.cache_raw,
            alert_rate=args.alert_rate,
            concentration_threshold=args.concentration_threshold,
            clean_abnormal_values=args.clean_abnormal_values,
            frozen_sensor_window=args.frozen_sensor_window,
            sensor_disagreement_z_threshold=args.sensor_disagreement_z_threshold,
            train_ratio=args.train_ratio,
            n_splits=args.n_splits,
            gap=args.gap,
            window_length=args.window_length,
            forecast_horizon=args.forecast_horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            model_variant=args.model_variant,
            pinball_quantile=args.pinball_quantile,
            push_bq=args.push_bq,
            validation_monitor_max_windows=args.validation_monitor_max_windows,
            save_cv_plots=args.save_cv_plots,
            use_catch22=args.use_catch22,
            include_secondary_diagnostics=args.include_secondary_diagnostics,
        )
        return

    run_pipeline_mm256(
        source=args.source,
        cache_raw=args.cache_raw,
        alert_rate=args.alert_rate,
        concentration_threshold=args.concentration_threshold,
        clean_abnormal_values=args.clean_abnormal_values,
        frozen_sensor_window=args.frozen_sensor_window,
        sensor_disagreement_z_threshold=args.sensor_disagreement_z_threshold,
        train_ratio=args.train_ratio,
        n_splits=args.n_splits,
        gap=args.gap,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        model_variant=args.model_variant,
        pinball_quantile=args.pinball_quantile,
        skip_cv=args.skip_cv,
        push_bq=args.push_bq,
        save_preprocess=args.save_preprocess,
        upload_preprocess=args.upload_preprocess,
        validation_monitor_max_windows=args.validation_monitor_max_windows,
        save_cv_plots=args.save_cv_plots,
        save_final_analysis=args.save_final_analysis,
        use_catch22=args.use_catch22,
        include_secondary_diagnostics=args.include_secondary_diagnostics,
    )


if __name__ == "__main__":
    main()
