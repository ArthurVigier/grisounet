# Grisounet

Methane time-series forecasting project with a small production codepath and a separate research workspace.

## Repository layout

### Production code

- `ml_logic/`: data access, preprocessing, model training, persistence, analysis helpers
- `api/`: FastAPI service layer
- `interface/`: local pipeline orchestration entry points
- `scripts/`: operational helpers for BigQuery and GCS inventory/export
- `results/`: generated outputs written at runtime and ignored by Git

### Research workspace

- `research/notebooks/`: exploratory notebooks and EDA material
- `research/analysis/`: draft analysis scripts, processed artifacts, and supporting files
- `research/references/`: reference PDFs and background material

## Getting started

1. Create a virtual environment.
2. Install the dependency set you need:

   - production/runtime only: `pip install -r requirements.txt`
   - contributor tooling: `pip install -r requirements/dev.txt`
   - notebooks and exploratory analysis: `pip install -r requirements/research.txt`

3. For local development, populate `.env` with the required GCP and storage settings.
4. On the VM, Secret Manager is the source of truth for runtime configuration.
5. Use `make data` to pull the raw modeling dataframe from BigQuery when you need a local snapshot.

## Main workflows

- Pull a local snapshot from BigQuery: `make data`
- Run the end-to-end pipeline: `python -m interface.workflow`
- Start the API locally: `fastapi dev api/fast.py`
- Discover GPU VM creation commands without creating anything: `make gpu_snipe_dry_run`
- Repeatedly probe GCP GPU capacity until a VM is created: `make gpu_snipe`
- Export cloud resources inventory: `python scripts/list_resources.py`
- Fetch BigQuery tables locally: `python scripts/fetch_tables.py`

## GPU sniper

The repository now includes a Compute Engine GPU sniper at `scripts/gpu_sniper.py`.
It is a grisounet-native rewrite inspired by the public `gpu-sniper` project and
uses the repo's current compute defaults instead of hard-coded upstream values.

Default behavior:

- Targets the Compute Engine project from `GCP_COMPUTE_PROJECT` and falls back to `GCP_PROJECT`
- Uses `GCP_REGION` as the zone discovery filter
- Uses `IMAGE_PROJECT` and `IMAGE_FAMILY` for the boot disk image
- Tries both `nvidia-l4` on `g2-standard-4` and `nvidia-tesla-t4` on `n1-standard-4`

Useful commands:

- Dry run the discovery and print the exact `gcloud` commands: `python scripts/gpu_sniper.py --dry-run`
- Only try L4 in western Europe: `python scripts/gpu_sniper.py --gpu nvidia-l4 --region-filter europe-west4`
- Provide a custom target list: `python scripts/gpu_sniper.py --targets-json '[{"gpu_type":"nvidia-l4","machine_type":"g2-standard-4","attach_accelerator":false}]'`

Optional environment variables:

- `GPU_SNIPER_INSTANCE_NAME_BASE`
- `GPU_SNIPER_ZONE_FILTERS`
- `GPU_SNIPER_MAX_RETRIES`
- `GPU_SNIPER_RETRY_DELAY`
- `GPU_SNIPER_MAX_WORKERS`
- `GPU_SNIPER_BOOT_DISK_SIZE_GB`
- `GPU_SNIPER_LABELS`
- `GPU_SNIPER_PROVISIONING_MODEL`
- `GPU_SNIPER_STARTUP_SCRIPT`

## Notes

- The repository currently contains both runnable application code and historical research material, but they are now separated by directory.
- The VM workflow is BigQuery-first and should not rely on files committed under `raw_data/`.
- `venv/`, notebook checkpoints, local outputs, and other machine-specific artifacts should stay out of version control.
- `requirements/full.txt` preserves the previous all-in-one environment as a legacy snapshot; new work should use the split files under `requirements/`.
