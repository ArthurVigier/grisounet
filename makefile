RAW_DIR := raw_data

.PHONY: data lfs unzip flatten

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
