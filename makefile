.PHONY: data pull_bq preprocess_split slice_arrays gpu_snipe gpu_snipe_dry_run

data: pull_bq

pull_bq:
	python -c "from ml_logic.data import pull_data_from_bq; pull_data_from_bq()"

load_data_local:
	python -c "from ml_logic.preprocessor import load_data_local; load_data_local()"

preprocess_split:
	python -c "from ml_logic.data import load_modeling_dataframe; from ml_logic.preprocessor import preprocess_split; preprocess_split(load_modeling_dataframe(source='bq'))"

slice_arrays:
	python -c "from ml_logic.data import load_modeling_dataframe; from ml_logic.preprocessor import preprocess_split, slice_arrays; train_data, _, _ = preprocess_split(load_modeling_dataframe(source='bq')); slice_arrays(train_data, 0, 1000)"

gpu_snipe:
	python scripts/gpu_sniper.py

gpu_snipe_dry_run:
	python scripts/gpu_sniper.py --dry-run
