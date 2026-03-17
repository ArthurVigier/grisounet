"""
Pipeline orchestrator for MM256 single-sensor methane forecasting.

Two execution modes:
  1. **Single run** (``run_pipeline_mm256``): simple train/test split, trains
     the simple LSTM, saves model + predictions.  Good for a quick sanity check.
  2. **Cross-validated run** (``run_cv_pipeline_mm256``): delegates to
     ``cv_time_series.run_cv`` for proper k-fold TimeSeriesSplit evaluation.

Both modes use the MM256-specific preprocessor, the single-sensor model,
and the parameterised results_bq_save / analysis modules.

Usage:
    # Single run
    python interface/workflow_mm256.py --mode single --source cache

    # Cross-validated run
    python interface/workflow_mm256.py --mode cv --n-splits 5 --source cache --push-bq
"""

import argparse
import os
import sys
from datetime import datetime
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

SENSOR = "MM256"
SENSORS_LIST = ["MM256"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining:.1f}s" if minutes >= 1 else f"{remaining:.1f}s"


# ---------------------------------------------------------------------------
# Mode 1: Single train/test run (simple LSTM)
# ---------------------------------------------------------------------------
def run_pipeline_mm256(
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    test_size: float = 0.3,
    save_preprocess: bool = True,
    upload_preprocess: bool = False,
    save_preprocess_bq: bool = False,
    push_bq: bool = False,
) -> dict:
    """Run the full single-run MM256 pipeline: load -> preprocess -> train -> predict -> analyse.

    Trains the **simple LSTM** (one encoder layer).  For the advanced model,
    swap ``simple_lstm_mm256`` for ``advanced_lstm_mm256`` in this function.
    """
    from scripts.preprocessor_MM256 import preprocess_mm256, slice_windows_mm256
    from ml_logic.model_mm256 import simple_lstm_mm256
    from ml_logic.model_save import save_model_to_gcs
    from ml_logic.results_bq_save import save_history_to_bq, save_predictions_to_bq
    from ml_logic.analysis import plot_loss_curves, plot_predictions_vs_actual, compute_metrics
    from ml_logic.data import save_preprocessing_artifact
    from sklearn.model_selection import train_test_split

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = perf_counter()

    print(f"\n{'='*60}")
    print(f"  MM256 Pipeline — single run — {timestamp}")
    print(f"{'='*60}\n")

    # ---- Step 1: Load & preprocess ----
    step_t = perf_counter()
    print("Step 1/5 — Loading & preprocessing (MM256 only)...")
    data, scalers, meta = preprocess_mm256(
        source=source,
        cache_raw=cache_raw,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
    )
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    # ---- Step 2: Train / test split + windowing ----
    step_t = perf_counter()
    print("\nStep 2/5 — Splitting & windowing...")

    # Temporal split (no shuffle)
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"  Train: {len(train_data):,} rows  |  Test: {len(test_data):,} rows")

    X_train, y_train = slice_windows_mm256(train_data, 0, len(train_data))
    X_test, y_test = slice_windows_mm256(test_data, 0, len(test_data))

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    if save_preprocess:
        save_preprocessing_artifact(
            X_train, X_test, y_train, y_test,
            timestamp=f"mm256_{timestamp}",
            upload_to_gcs=upload_preprocess,
        )
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    # ---- Step 3: Train simple LSTM ----
    step_t = perf_counter()
    print("\nStep 3/5 — Training simple LSTM (MM256)...")

    if X_train.shape[0] == 0:
        print("  ERROR: No training windows. Aborting.")
        return {"error": "no_training_windows"}

    model, history, y_pred = simple_lstm_mm256(
        X_train, y_train, X_test, y_test,
    )

    save_model_to_gcs(model, f"mm256_{timestamp}")
    save_history_to_bq(history, timestamp, table_suffix="mm256")
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    # ---- Step 4: Save predictions ----
    step_t = perf_counter()
    print("\nStep 4/5 — Saving predictions...")

    if y_test.shape[0] == 0 or y_pred is None:
        print("  No test windows; skipping prediction export.")
        pred_df = pd.DataFrame()
    else:
        pred_df = save_predictions_to_bq(
            y_test, y_pred, timestamp,
            sensors=SENSORS_LIST,
            table_suffix="mm256",
        )
    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    # ---- Step 5: Analysis ----
    step_t = perf_counter()
    print("\nStep 5/5 — Generating analysis...")

    plot_loss_curves(history, timestamp, label_prefix="mm256_")

    if y_test.shape[0] > 0 and y_pred is not None:
        # Sample forecast plot
        sample_idx = min(100, y_test.shape[0] - 1)
        plt.figure(figsize=(12, 5))
        plt.plot(y_test[sample_idx, :, 0], label="Actual", linewidth=2)
        plt.plot(y_pred[sample_idx, :, 0], label="Simple LSTM", linestyle=":")
        plt.title(f"MM256 — sample {sample_idx}")
        plt.xlabel("Forecast step (s)")
        plt.ylabel("MM256 (scaled)")
        plt.legend()
        os.makedirs("results/graphs", exist_ok=True)
        plt.savefig(f"results/graphs/mm256_forecast_{timestamp}.png", dpi=150)
        plt.close()
        print(f"  Saved results/graphs/mm256_forecast_{timestamp}.png")

        if not pred_df.empty:
            compute_metrics(pred_df, timestamp, sensors=SENSORS_LIST, label_prefix="mm256_")
            plot_predictions_vs_actual(pred_df, timestamp, sensors=SENSORS_LIST, label_prefix="mm256_")

    print(f"  Done in {_fmt(perf_counter() - step_t)}")

    total = perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  MM256 pipeline complete: {timestamp} ({_fmt(total)})")
    print(f"{'='*60}")

    return {
        "timestamp": timestamp,
        "model": model,
        "history": history,
        "predictions": pred_df,
        "scalers": scalers,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Mode 2: Cross-validated run (delegates to cv_time_series)
# ---------------------------------------------------------------------------
def run_cv_pipeline_mm256(
    n_splits: int = 5,
    gap: int = 300,
    source: str = "cache",
    cache_raw: bool = False,
    alert_rate: float = 1.0,
    concentration_threshold: float = 1.0,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 5,
    push_bq: bool = False,
) -> dict:
    """Run the MM256 pipeline with TimeSeriesSplit cross-validation.

    This is a thin wrapper around ``cv_time_series.run_cv`` that makes it
    callable from the same CLI as the single-run mode.
    """
    from scripts.cv_time_series import run_cv

    return run_cv(
        n_splits=n_splits,
        gap=gap,
        source=source,
        cache_raw=cache_raw,
        alert_rate=alert_rate,
        concentration_threshold=concentration_threshold,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        push_bq=push_bq,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MM256 single-sensor pipeline orchestrator"
    )
    parser.add_argument(
        "--mode", choices=["single", "cv"], default="single",
        help="'single' = one train/test run; 'cv' = k-fold TimeSeriesSplit"
    )
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="cache")
    parser.add_argument("--cache-raw", action="store_true")
    parser.add_argument("--alert-rate", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=1.0)
    parser.add_argument("--push-bq", action="store_true")

    # Single-run options
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--skip-preprocess-save", action="store_true")
    parser.add_argument("--upload-preprocess", action="store_true")

    # CV options
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "single":
        run_pipeline_mm256(
            source=args.source,
            cache_raw=args.cache_raw,
            alert_rate=args.alert_rate,
            concentration_threshold=args.concentration_threshold,
            test_size=args.test_size,
            save_preprocess=not args.skip_preprocess_save,
            upload_preprocess=args.upload_preprocess,
            push_bq=args.push_bq,
        )
    else:
        run_cv_pipeline_mm256(
            n_splits=args.n_splits,
            gap=args.gap,
            source=args.source,
            cache_raw=args.cache_raw,
            alert_rate=args.alert_rate,
            concentration_threshold=args.concentration_threshold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            push_bq=args.push_bq,
        )
