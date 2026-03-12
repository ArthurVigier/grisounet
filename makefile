RAW_DIR := raw_data

.PHONY: data lfs unzip

data: lfs unzip

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
