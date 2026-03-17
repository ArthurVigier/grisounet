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
- Export cloud resources inventory: `python scripts/list_resources.py`
- Fetch BigQuery tables locally: `python scripts/fetch_tables.py`

## Notes

- The repository currently contains both runnable application code and historical research material, but they are now separated by directory.
- The VM workflow is BigQuery-first and should not rely on files committed under `raw_data/`.
- `venv/`, notebook checkpoints, local outputs, and other machine-specific artifacts should stay out of version control.
- `requirements/full.txt` preserves the previous all-in-one environment as a legacy snapshot; new work should use the split files under `requirements/`.
