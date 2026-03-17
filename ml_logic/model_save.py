"""Save and load models from Google Cloud Storage"""
import json
import os
import pickle
from datetime import datetime
from google.cloud import storage
from ml_logic.secrets import get_secret


def save_model_to_gcs(model, timestamp=None):
    """Save Keras model to Cloud Storage bucket."""
    bucket_name = get_secret("BUCKET_NAME")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save locally first
    local_path = f"results/models/model_{timestamp}.keras"
    os.makedirs("results/models", exist_ok=True)
    model.save(local_path)

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"models/model_{timestamp}.keras")
    blob.upload_from_filename(local_path)

    print(f"Model saved -> gs://{bucket_name}/models/model_{timestamp}.keras")
    return timestamp


def save_model_artifacts_to_gcs(timestamp, scalers=None, metadata=None):
    """Persist non-model artifacts required to reproduce MM256 inference."""
    bucket_name = get_secret("BUCKET_NAME")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    os.makedirs("results/models", exist_ok=True)
    saved_paths = {}

    if scalers is not None:
        scalers_path = f"results/models/model_{timestamp}_scalers.pkl"
        with open(scalers_path, "wb") as stream:
            pickle.dump(scalers, stream)
        bucket.blob(f"models/model_{timestamp}_scalers.pkl").upload_from_filename(scalers_path)
        saved_paths["scalers"] = scalers_path
        print(f"Scalers saved -> gs://{bucket_name}/models/model_{timestamp}_scalers.pkl")

    if metadata is not None:
        metadata_path = f"results/models/model_{timestamp}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
        bucket.blob(f"models/model_{timestamp}_metadata.json").upload_from_filename(metadata_path)
        saved_paths["metadata"] = metadata_path
        print(f"Metadata saved -> gs://{bucket_name}/models/model_{timestamp}_metadata.json")

    return saved_paths


def load_model_from_gcs(timestamp):
    """Load a specific model version from Cloud Storage."""
    from tensorflow.keras.models import load_model

    bucket_name = get_secret("BUCKET_NAME")
    local_path = f"results/models/model_{timestamp}.keras"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"models/model_{timestamp}.keras")
    os.makedirs("results/models", exist_ok=True)
    blob.download_to_filename(local_path)

    return load_model(local_path)
