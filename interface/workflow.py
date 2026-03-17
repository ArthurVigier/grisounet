"""Full pipeline orchestrator"""
import os
from datetime import datetime
from time import perf_counter

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()


def _format_duration(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {remaining:.1f}s"
    return f"{remaining:.1f}s"


def run_pipeline(
    start_index=0,
    stop_index=1000000,
    source="bq",
    cache_raw=False,
    save_preprocess=True,
    upload_preprocess=False,
    save_preprocess_bq=False,
):
    """
    Run the full pipeline: load -> preprocess -> train -> predict -> analyse

    Args:
        start_index: start row for slice_arrays (subset of data to train on)
        stop_index: stop row for slice_arrays
        source: one of "bq", "cache", or "local"
        cache_raw: when source="bq", also save a local raw CSV snapshot
        save_preprocess: save preprocessed tensors locally as a compressed artifact
        upload_preprocess: upload the compressed preprocessing artifact to GCS
        save_preprocess_bq: also save flattened preprocessing outputs to BigQuery
    """
    from ml_logic.data import (
        load_modeling_dataframe,
        save_preprocessing_artifact,
        save_preprocessing_to_bq,
    )
    from ml_logic.preprocessor import preprocess_split, slice_arrays
    from ml_logic.model import more_advanced_lstm
    from ml_logic.model_save import save_model_to_gcs
    from ml_logic.results_bq_save import save_history_to_bq, save_predictions_to_bq
    # rajouter un py avec des analyses

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_started_at = perf_counter()
    print(f"\n{'='*50}")
    print(f"Pipeline run: {timestamp}")
    print(f"{'='*50}\n")

    # Step 1: Load data
    step_started_at = perf_counter()
    print("Step 1/5 -- Loading data...")
    df = load_modeling_dataframe(source=source, cache_raw=cache_raw)
    print(f"  Completed in {_format_duration(perf_counter() - step_started_at)}")

    # Step 2: Preprocess
    # preprocess_split returns: (train_data, test_data, scalers)
    # slice_arrays returns: (X_array, y_array)
    step_started_at = perf_counter()
    print("\nStep 2/5 -- Preprocessing...")
    train_data, test_data, scalers = preprocess_split(df)

    X_train, y_train = slice_arrays(train_data, start_index, stop_index)
    X_test, y_test = slice_arrays(test_data, start_index, stop_index)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    preprocess_artifact = None
    if save_preprocess:
        preprocess_artifact = save_preprocessing_artifact(
            X_train,
            X_test,
            y_train,
            y_test,
            timestamp=timestamp,
            upload_to_gcs=upload_preprocess,
        )
    if save_preprocess_bq:
        save_preprocessing_to_bq(X_train, X_test, y_train, y_test)
    print(f"  Completed in {_format_duration(perf_counter() - step_started_at)}")

    # Step 3: Train model
    # more_advanced_lstm returns: (model, history, y_pred)
    step_started_at = perf_counter()
    print("\nStep 3/5 -- Training LSTM...")
    model, history, y_pred = more_advanced_lstm(X_train, y_train, X_test, y_test)

    save_model_to_gcs(model, timestamp)
    save_history_to_bq(history, timestamp)
    print(f"  Completed in {_format_duration(perf_counter() - step_started_at)}")

    # Step 4: Save predictions
    step_started_at = perf_counter()
    print("\nStep 4/5 -- Saving predictions...")
    if y_test.shape[0] == 0 or y_pred.shape[0] == 0:
        print("  No test windows available; skipping prediction export.")
        pred_df = pd.DataFrame()
    else:
        pred_df = save_predictions_to_bq(y_test, y_pred, timestamp)
    print(f"  Completed in {_format_duration(perf_counter() - step_started_at)}")

    # Step 5: Analyses
    step_started_at = perf_counter()
    print("\nStep 5/5 -- Generating analysis...")
    # todo: add more analysis functions here and save results to BQ or GCS as needed
    os.makedirs("results/graphs", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/graphs/loss_{timestamp}.png", dpi=150)
    plt.close()
    print(f"  Saved results/graphs/loss_{timestamp}.png")
    # plot_loss_curves
    # plot_predictions_vs_actual
    if y_test.shape[0] == 0 or y_pred.shape[0] == 0:
        print("  Skipped sample forecast plots: no test windows were generated.")
    else:
        sample_idx = min(4000, y_test.shape[0] - 1)
        target_cols = ["MM256", "MM263", "MM264"]

        for target_idx, target_name in enumerate(target_cols):
            plt.figure(figsize=(12, 5))
            plt.plot(y_test[sample_idx, :, target_idx], label="Actual", linewidth=2)
            plt.plot(y_pred[sample_idx, :, target_idx], label="LSTM", linestyle=":")
            plt.title(f"{target_name} - sample {sample_idx}")
            plt.xlabel("Forecast step")
            plt.ylabel("Methane rate")
            plt.legend()
            os.makedirs("results/graphs", exist_ok=True)
            plt.savefig(f"results/graphs/{target_name}_{timestamp}.png", dpi=150)
            plt.close()
            print(f"  Saved results/graphs/{target_name}_{timestamp}.png")
    print(f"  Completed in {_format_duration(perf_counter() - step_started_at)}")

    # compute_metrics(pred_df, timestamp)

    total_duration = perf_counter() - pipeline_started_at
    print(f"\n{'='*50}")
    print(f"Pipeline complete: {timestamp} ({_format_duration(total_duration)})")
    print(f"{'='*50}")

    return {
        "timestamp": timestamp,
        "model": model,
        "history": history,
        "predictions": pred_df,
        "preprocess_artifact": preprocess_artifact,
        "scalers": scalers,
    }
# rajouter les metrics dans le return aussi ... (à definir et computer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start row index")
    parser.add_argument("--stop", type=int, default=1000000, help="Stop row index")
    parser.add_argument("--source", choices=["bq", "cache", "local"], default="bq", help="Input source")
    parser.add_argument("--cache-raw", action="store_true", help="Save a raw CSV snapshot when pulling from BigQuery")
    parser.add_argument("--skip-preprocess-save", action="store_true", help="Skip saving preprocessing outputs locally")
    parser.add_argument("--upload-preprocess", action="store_true", help="Upload preprocessing artifact to GCS")
    parser.add_argument("--save-preprocess-bq", action="store_true", help="Also save flattened preprocessing outputs to BigQuery")
    args = parser.parse_args()
    run_pipeline(
        start_index=args.start,
        stop_index=args.stop,
        source=args.source,
        cache_raw=args.cache_raw,
        save_preprocess=not args.skip_preprocess_save,
        upload_preprocess=args.upload_preprocess,
        save_preprocess_bq=args.save_preprocess_bq,
    )
