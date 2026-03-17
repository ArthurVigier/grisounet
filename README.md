# Grisounet

Methane time-series forecasting project with a small production codepath and a separate research workspace.

## Repository layout

### Production code

- `ml_logic/`: data access, preprocessing, model training, persistence, analysis helpers
- `api/`: FastAPI service layer
- `interface/`: local pipeline orchestration entry points
- `scripts/`: operational helpers for BigQuery and GCS inventory/export
- `raw_data/`: local raw dataset bootstrap area
- `results/`: generated outputs written at runtime and ignored by Git

### Research workspace

- `research/notebooks/`: exploratory notebooks and EDA material
- `research/analysis/`: draft analysis scripts, processed artifacts, and supporting files
- `research/references/`: reference PDFs and background material

## Getting started

1. Install Git LFS.
2. Create a virtual environment.
3. Install the dependency set you need:

   - production/runtime only: `pip install -r requirements.txt`
   - contributor tooling: `pip install -r requirements/dev.txt`
   - notebooks and exploratory analysis: `pip install -r requirements/research.txt`

4. Populate `.env` with the required GCP and storage settings if you are running locally.
5. Run `make data` to fetch and unpack the local methane dataset.

## Main workflows

- Local data preparation: `make data`
- Run the end-to-end pipeline: `python -m interface.workflow`
- Start the API locally: `fastapi dev api/fast.py`
- Export cloud resources inventory: `python scripts/list_resources.py`
- Fetch BigQuery tables locally: `python scripts/fetch_tables.py`

## Notes

- The repository currently contains both runnable application code and historical research material, but they are now separated by directory.
- `venv/`, notebook checkpoints, local outputs, and other machine-specific artifacts should stay out of version control.
- `requirements/full.txt` preserves the previous all-in-one environment as a legacy snapshot; new work should use the split files under `requirements/`.
