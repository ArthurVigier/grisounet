"""Pull data from BigQuery and save locally"""
import os
from datetime import datetime
from google.cloud import bigquery
import pandas as pd
from ml_logic.secrets import get_secret


def pull_data_from_bq():
    """
    Pulls full dataset from BigQuery using Secret Manager / .env variables.
    Returns DataFrame and saves timestamped CSV locally.
    """
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    table = get_secret("BQ_TABLE")
    region = get_secret("BQ_REGION")

    query = f"SELECT * FROM `{project}.{dataset}.{table}`"

    client = bigquery.Client(project=project, location=region)
    df = client.query(query).result().to_dataframe()

    # Save timestamped copy locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/raw_pulls", exist_ok=True)
    path = f"results/raw_pulls/data_{timestamp}.csv"
    df.to_csv(path, index=False)

    print(f"Pulled {df.shape[0]} rows, {df.shape[1]} cols -> saved to {path}")
    return df


def load_modeling_dataframe(source: str = "bq") -> pd.DataFrame:
    """Load the modeling dataframe from the requested source.

    The VM path should use BigQuery and Secret Manager by default.
    Local CSV loading remains available for research workflows only.
    """
    if source == "bq":
        return pull_data_from_bq()
    if source == "local":
        from ml_logic.preprocessor import load_data_local

        return load_data_local()
    raise ValueError("source must be either 'bq' or 'local'")


def save_preprocessing_to_bq(X_train, X_test, y_train, y_test):
    """
    Saves preprocessing results to a timestamped BigQuery table.
    """
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_name = f"preprocess_{timestamp}"

    # Combine into a single DataFrame with a split column
    train_df = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    train_df["split"] = "train"
    test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
    test_df["split"] = "test"
    result_df = pd.concat([train_df, test_df], ignore_index=True)

    client = bigquery.Client(project=project, location=region)
    table_ref = f"{project}.{dataset}.{table_name}"
    client.load_table_from_dataframe(result_df, table_ref).result()

    # Also save locally
    os.makedirs("results/preprocessing", exist_ok=True)
    result_df.to_csv(f"results/preprocessing/{table_name}.csv", index=False)

    print(f"Preprocessing saved -> BQ: {table_ref}")
    return table_name
