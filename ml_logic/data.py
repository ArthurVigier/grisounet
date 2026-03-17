"""Data loading and persistence helpers for local, BigQuery, and GCS workflows."""
import glob
import os
from datetime import datetime
from urllib.parse import urlparse

from google.cloud import bigquery
from google.cloud import storage
import numpy as np
import pandas as pd

from ml_logic.secrets import get_secret

CACHE_DIR = "results/raw_pulls"
PREPROCESS_DIR = "results/preprocessing"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_cached_csv():
    """Return the path to the most recent cached CSV, or None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "data_*.csv")))
    return files[-1] if files else None


def _flatten_time_series(array: np.ndarray) -> np.ndarray:
    """Flatten a 3D tensor to 2D while keeping empty tensors valid."""
    if array.ndim < 2:
        raise ValueError("Expected at least 2 dimensions to flatten time-series data")
    if array.shape[0] == 0:
        width = int(np.prod(array.shape[1:])) if array.ndim > 1 else 0
        return np.empty((0, width), dtype=np.float32)
    return array.reshape(array.shape[0], -1)


def _build_preprocessing_split_df(X_array, y_array, split_name: str) -> pd.DataFrame:
    X_df = pd.DataFrame(_flatten_time_series(X_array)).add_prefix("x_")
    y_df = pd.DataFrame(_flatten_time_series(y_array)).add_prefix("y_")
    split_df = pd.concat([X_df, y_df], axis=1)
    split_df["split"] = split_name
    return split_df


def pull_data_from_bq(save_local: bool = False):
    """
    Pulls full dataset from BigQuery using Secret Manager / .env variables.
    Returns a DataFrame. Optionally persists a local CSV snapshot.
    """
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    table = get_secret("BQ_TABLE")
    region = get_secret("BQ_REGION")

    query = f"SELECT * FROM `{project}.{dataset}.{table}`"

    client = bigquery.Client(project=project, location=region)
    df = client.query(query).result().to_dataframe(create_bqstorage_client=True)

    if save_local:
        timestamp = _timestamp()
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = f"{CACHE_DIR}/data_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"Pulled {df.shape[0]} rows, {df.shape[1]} cols -> saved to {path}")
    else:
        print(f"Pulled {df.shape[0]} rows, {df.shape[1]} cols from BigQuery")
    return df


def load_modeling_dataframe(source: str = "cache", cache_raw: bool = False) -> pd.DataFrame:
    """Load the modeling dataframe from the requested source.

    Sources:
        - "cache" (default): use local CSV cache if available, else pull from BQ
        - "bq": always pull fresh data from BigQuery
        - "local": use the legacy local CSV loader
    """
    if source == "cache":
        cached = _latest_cached_csv()
        if cached:
            df = pd.read_csv(cached)
            print(f"Loaded {df.shape[0]} rows from cache: {cached}")
            return df
        print("No cached data found, pulling from BigQuery...")
        return pull_data_from_bq(save_local=True)
    if source == "bq":
        return pull_data_from_bq(save_local=cache_raw)
    if source == "local":
        from ml_logic.preprocessor import load_data_local

        return load_data_local()
    raise ValueError("source must be 'cache', 'bq', or 'local'")


def save_preprocessing_artifact(X_train, X_test, y_train, y_test, timestamp=None, upload_to_gcs: bool = False):
    """Persist preprocessing outputs as a compressed artifact for reuse."""
    if not timestamp:
        timestamp = _timestamp()

    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    local_path = os.path.join(PREPROCESS_DIR, f"preprocess_{timestamp}.npz")
    np.savez_compressed(
        local_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    artifact = {"timestamp": timestamp, "local_path": local_path}

    if upload_to_gcs:
        bucket_name = get_secret("BUCKET_NAME")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_path = f"preprocessing/preprocess_{timestamp}.npz"
        bucket.blob(blob_path).upload_from_filename(local_path)
        artifact["gcs_uri"] = f"gs://{bucket_name}/{blob_path}"
        print(f"Preprocessing artifact uploaded -> {artifact['gcs_uri']}")

    print(f"Preprocessing artifact saved -> {local_path}")
    return artifact


def download_preprocessing_artifact(gcs_uri: str) -> str:
    """Download a preprocessing artifact from GCS to the local preprocessing directory."""
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError("gcs_uri must look like gs://bucket/path/to/file.npz")

    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    local_path = os.path.join(PREPROCESS_DIR, os.path.basename(blob_path))

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(local_path)
    print(f"Preprocessing artifact downloaded -> {local_path}")
    return local_path


def load_preprocessing_artifact(path: str):
    """Load preprocessed tensors saved by save_preprocessing_artifact."""
    with np.load(path) as artifact:
        return (
            artifact["X_train"],
            artifact["X_test"],
            artifact["y_train"],
            artifact["y_test"],
        )


def save_preprocessing_to_bq(X_train, X_test, y_train, y_test):
    """
    Saves preprocessing results to a timestamped BigQuery table.
    """
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")
    timestamp = _timestamp()
    table_name = f"preprocess_{timestamp}"

    # Combine train/test tensors into a single flat table for debugging and inspection.
    train_df = _build_preprocessing_split_df(X_train, y_train, "train")
    test_df = _build_preprocessing_split_df(X_test, y_test, "test")
    result_df = pd.concat([train_df, test_df], ignore_index=True)

    client = bigquery.Client(project=project, location=region)
    table_ref = f"{project}.{dataset}.{table_name}"
    client.load_table_from_dataframe(result_df, table_ref).result()

    # Also save locally
    os.makedirs("results/preprocessing", exist_ok=True)
    result_df.to_csv(f"results/preprocessing/{table_name}.csv", index=False)

    print(f"Preprocessing saved -> BQ: {table_ref}")
    return table_name
