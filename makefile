RAW_DIR := raw_data

.PHONY: data lfs unzip flatten preprocess_split slice_arrays

data: lfs unzip flatten

lfs:
	git lfs install
	git lfs pull

unzip:
	find "$(RAW_DIR)" -name "*.zip" -print0 | while IFS= read -r -d '' file; do \
		dir="$${file%.zip}"; \
		echo "Extraction de $$file dans $$dir"; \
		mkdir -p "$$dir"; \
		unzip -o "$$file" -d "$$dir"; \
	done

flatten:
	@csv=$$(find "$(RAW_DIR)" -type f -name "methane_data.csv" ! -path "$(RAW_DIR)/methane_data.csv" | head -n 1); \
	if [ -n "$$csv" ]; then \
		echo "CSV trouvé: $$csv"; \
		mv "$$csv" "$(RAW_DIR)/methane_data.csv"; \
		echo "Suppression du dossier intermédiaire"; \
		rm -rf "$$(dirname "$$csv")"; \
	fi

load_data_local:
  python -c "from ml_logic.preprocessor import load_data_local; load_data_local()"

preprocess_split:
	python -c "from ml_logic.preprocessor import preprocess_split; preprocess_split()"

slice_arrays:
	python -c "from ml_logic.preprocessor import slice_arrays; slice_arrays()"
