"""Save and load models from Google Cloud Storage."""
import json
import os
import pickle
from datetime import datetime

from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage

from ml_logic.secrets import get_secret


def _upload_file_best_effort(local_path: str, blob_path: str) -> dict:
    """Upload a local file to GCS without failing the pipeline on GCS errors."""
    bucket_name = get_secret("BUCKET_NAME")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_path).upload_from_filename(local_path)
        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        print(f"Uploaded -> {gcs_uri}")
        return {"uploaded": True, "gcs_uri": gcs_uri, "local_path": local_path}
    except (GoogleAPIError, OSError) as exc:
        print(
            "WARNING: GCS upload failed; keeping local artifact only.\n"
            f"  file={local_path}\n"
            f"  blob={blob_path}\n"
            f"  error={exc}"
        )
        return {"uploaded": False, "gcs_uri": None, "local_path": local_path, "error": str(exc)}


def save_model_to_gcs(model, timestamp=None):
    """Save Keras model locally and upload it to GCS when possible."""
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save locally first
    local_path = f"results/models/model_{timestamp}.keras"
    os.makedirs("results/models", exist_ok=True)
    model.save(local_path)

    upload_result = _upload_file_best_effort(local_path, f"models/model_{timestamp}.keras")
    if not upload_result["uploaded"]:
        print(f"Model saved locally -> {local_path}")
    return timestamp


def save_model_artifacts_to_gcs(timestamp, scalers=None, metadata=None):
    """Persist non-model artifacts required to reproduce MM256 inference.

    Artifacts are always saved locally; GCS upload is attempted on a best-effort
    basis so a permissions issue does not abort the whole training run.
    """
    os.makedirs("results/models", exist_ok=True)
    saved_paths = {}

    if scalers is not None:
        scalers_path = f"results/models/model_{timestamp}_scalers.pkl"
        with open(scalers_path, "wb") as stream:
            pickle.dump(scalers, stream)
        saved_paths["scalers"] = scalers_path
        upload_result = _upload_file_best_effort(
            scalers_path, f"models/model_{timestamp}_scalers.pkl"
        )
        saved_paths["scalers_gcs_uri"] = upload_result["gcs_uri"]
        if not upload_result["uploaded"]:
            print(f"Scalers saved locally -> {scalers_path}")

    if metadata is not None:
        metadata_path = f"results/models/model_{timestamp}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
        saved_paths["metadata"] = metadata_path
        upload_result = _upload_file_best_effort(
            metadata_path, f"models/model_{timestamp}_metadata.json"
        )
        saved_paths["metadata_gcs_uri"] = upload_result["gcs_uri"]
        if not upload_result["uploaded"]:
            print(f"Metadata saved locally -> {metadata_path}")

    return saved_paths


def load_model_from_gcs(timestamp):
    """Load a specific model version from Cloud Storage."""
    from tensorflow.keras.models import load_model
    from ml_logic.model_mm256 import PinballLoss

    bucket_name = get_secret("BUCKET_NAME")
    local_path = f"results/models/model_{timestamp}.keras"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"models/model_{timestamp}.keras")
    os.makedirs("results/models", exist_ok=True)
    blob.download_to_filename(local_path)

    return load_model(local_path, custom_objects={"PinballLoss": PinballLoss})
