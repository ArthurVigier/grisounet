"""Full pipeline orchestrator"""
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def run_pipeline(start_index=0, stop_index=100000):
    """
    Run the full pipeline: pull -> preprocess -> train -> predict -> analyse

    Args:
        start_index: start row for slice_arrays (subset of data to train on)
        stop_index: stop row for slice_arrays
    """
    from ml_logic.data import pull_data_from_bq, save_preprocessing_to_bq
    from ml_logic.preprocessor import preprocess_split, slice_arrays
    from ml_logic.model import more_advanced_lstm
    from ml_logic.model_save import save_model_to_gcs
    from ml_logic.results_bq_save import save_history_to_bq, save_predictions_to_bq
    # rajouter un py avec des analyses

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*50}")
    print(f"Pipeline run: {timestamp}")
    print(f"{'='*50}\n")

    # Step 1: Pull data from BigQuery
    print("Step 1/5 -- Pulling data from BigQuery...")
    df = pull_data_from_bq()

    # Step 2: Preprocess
    # preprocess_split returns: (train_data, test_data, scalers)
    # slice_arrays returns: (X_array, y_array)
    print("\nStep 2/5 -- Preprocessing...")
    train_data, test_data, scalers = preprocess_split(df)

    X_train, y_train = slice_arrays(train_data, start_index, stop_index)
    X_test, y_test = slice_arrays(test_data, start_index, stop_index)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    save_preprocessing_to_bq(X_train, X_test, y_train, y_test)

    # Step 3: Train model
    # more_advanced_lstm returns: (model, history, y_pred)
    print("\nStep 3/5 -- Training LSTM...")
    model, history, y_pred = more_advanced_lstm(X_train, y_train, X_test, y_test)

    save_model_to_gcs(model, timestamp)
    save_history_to_bq(history, timestamp)

    # Step 4: Save predictions
    print("\nStep 4/5 -- Saving predictions...")
    pred_df = save_predictions_to_bq(y_test, y_pred, timestamp)

    # Step 5: Analyses
    print("\nStep 5/5 -- Generating analysis...")
    # todo: add more analysis functions here and save results to BQ or GCS as needed
    # plot_loss_curves
    # plot_predictions_vs_actual
    # compute_metrics(pred_df, timestamp)

    print(f"\n{'='*50}")
    print(f"Pipeline complete: {timestamp}")
    print(f"{'='*50}")

    return {
        "timestamp": timestamp,
        "model": model,
        "history": history,
        "predictions": pred_df,
        "scalers": scalers,
    }
# rajouter les metrics dans le return aussi ... (à definir et computer)

if __name__ == "__main__":
    run_pipeline()
