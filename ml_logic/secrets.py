"""Centralized secret/config access: Secret Manager with .env fallback"""
import os

_cache = {}

def get_secret(secret_id):
    """Read from Secret Manager, fallback to .env for local development."""
    if secret_id in _cache:
        return _cache[secret_id]

    # Try Secret Manager first
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/grisounet/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        value = response.payload.data.decode("UTF-8")
        _cache[secret_id] = value
        return value
    except Exception:
        pass

    # Fallback to .env
    from dotenv import load_dotenv
    load_dotenv()
    value = os.environ.get(secret_id)
    if value:
        _cache[secret_id] = value
    return value
