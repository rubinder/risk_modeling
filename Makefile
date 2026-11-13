ROOT ?= data
AS_OF ?= $(shell date +%Y-%m-%d)
PROFILE ?= demo
DOCKER ?= docker
RISKAGG ?= uv run riskagg

.PHONY: demo demo-pinot test lint pinot-up pinot-down

# generate -> run -> (Pinot up, load) -> query through the client. Works without Docker, and
# when Pinot cannot be started or loaded: the client then answers from the curated Parquet files.
demo:
	$(RISKAGG) generate --profile $(PROFILE) --as-of $(AS_OF) --root $(ROOT)
	$(RISKAGG) run --profile $(PROFILE) --as-of $(AS_OF) --root $(ROOT)
	@$(MAKE) --no-print-directory demo-pinot AS_OF=$(AS_OF) ROOT=$(ROOT)
	$(RISKAGG) query var_99_1d --group-by firm_id --as-of $(AS_OF) --root $(ROOT)
	$(RISKAGG) query var_99_1d --group-by desk_id --top 5 --as-of $(AS_OF) --root $(ROOT)

# Never fails: a Pinot that cannot be started or loaded only means answers come from Parquet.
demo-pinot:
	@if ! $(DOCKER) info >/dev/null 2>&1; then \
		echo "Docker is not running: skipping Pinot; the client will answer from Parquet."; \
	elif ! $(MAKE) --no-print-directory pinot-up; then \
		echo "Pinot could not be started: skipping it; the client will answer from Parquet."; \
	elif ! $(RISKAGG) load-pinot --as-of $(AS_OF) --root $(ROOT); then \
		echo "Loading Pinot failed: the client will answer from Parquet."; \
	fi

test:
	uv run pytest -q

lint:
	uv run ruff check .

pinot-up:
	$(DOCKER) compose up -d --wait --wait-timeout 300

pinot-down:
	$(DOCKER) compose down
