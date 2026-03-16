"""Save and load models from Google Cloud Storage"""
import os
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
