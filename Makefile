ROOT ?= data
AS_OF ?= $(shell date +%Y-%m-%d)
PROFILE ?= demo

.PHONY: demo test lint pinot-up pinot-down

# generate -> run -> (Pinot up, load) -> query through the client. Works without Docker:
# the Pinot steps are skipped and the client answers from the curated Parquet files.
demo:
	uv run riskagg generate --profile $(PROFILE) --as-of $(AS_OF) --root $(ROOT)
	uv run riskagg run --profile $(PROFILE) --as-of $(AS_OF) --root $(ROOT)
	@if docker info >/dev/null 2>&1; then \
		$(MAKE) --no-print-directory pinot-up && \
		uv run riskagg load-pinot --as-of $(AS_OF) --root $(ROOT); \
	else \
		echo "Docker is not running: skipping Pinot; the client will answer from Parquet."; \
	fi
	uv run riskagg query var_99_1d --group-by firm_id --as-of $(AS_OF) --root $(ROOT)
	uv run riskagg query var_99_1d --group-by desk_id --top 5 --as-of $(AS_OF) --root $(ROOT)

test:
	uv run pytest -q

lint:
	uv run ruff check .

pinot-up:
	docker compose up -d --wait --wait-timeout 300

pinot-down:
	docker compose down
