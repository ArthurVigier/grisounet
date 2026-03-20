"""Centralized secret/config access: Secret Manager with .env fallback"""
import os

_cache = {}

_SM_PROJECT = os.environ.get("GCP_COMPUTE_PROJECT") or os.environ.get("GCP_PROJECT") or "grisounet"


def get_secret(secret_id):
    """Read from Secret Manager, fallback to .env for local development."""
    if secret_id in _cache:
        return _cache[secret_id]

    # Try Secret Manager first
    try:
        from google.cloud import secretmanager
        from google.api_core.exceptions import GoogleAPIError

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{_SM_PROJECT}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        value = response.payload.data.decode("UTF-8")
        _cache[secret_id] = value
        return value
    except ImportError:
        print("WARNING: google-cloud-secret-manager not installed, falling back to .env")
    except GoogleAPIError as exc:
        print(f"WARNING: Secret Manager lookup failed for '{secret_id}': {exc}")

    # Fallback to .env
    from dotenv import load_dotenv
    load_dotenv()
    value = os.environ.get(secret_id)
    if value:
        _cache[secret_id] = value
    return value
