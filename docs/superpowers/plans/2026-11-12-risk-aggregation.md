# Risk Aggregation Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A laptop-runnable PySpark platform that aggregates synthetic trade data into VaR, stress, CCAR and FRTB metrics from YAML definitions, serves them through a semantic layer, Pinot and an API, and teaches the domain through a study guide.

**Architecture:** Seeded generators write raw Parquet. An engine reduces P&L vectors once to a small reporting grain and computes every declared grouping from it in one pass; measure modules implement the non-generic formulas (FRTB, CCAR). A semantic model YAML is the single source for validation, Pinot schemas, the query builder and the data dictionary. Tasks 1–8 deliver VaR end to end; tasks 9–15 widen.

**Tech Stack:** Python 3.13, PySpark 4.x (local mode), Java 17, uv, pytest, ruff, NumPy, pandas, pyarrow, PyYAML, FastAPI, httpx, typer, Apache Pinot (Docker Compose).

**Spec:** `docs/superpowers/specs/2026-11-12-risk-aggregation-design.md`.

## Global Constraints

- Package `riskagg`, `src/` layout, console script `riskagg`. Work on branch `feat/risk-aggregation`.
- No Python UDFs in the tuned aggregation path. UDFs are allowed in data generation and in the benchmark's naive variants.
- A vector measure (VaR, ES) is never produced by summing metric values, in Spark, in Pinot SQL or in pandas. A structured measure is only looked up.
- Every simplification relative to the regulation is marked with a `SIMPLIFICATION:` comment in code and listed in the topic's study page.
- Sign convention: P&L is positive for gains. VaR, ES, stress loss and capital charges are reported as positive numbers for losses. A VaR may be negative (a node with only gains); never clip it.
- Quantile convention lives only in `riskagg/conventions.py`. No other module ranks or averages tail losses.
- Rolled-up dimensions are the literal string `*`, never null.
- Everything before Pinot runs and is tested with Docker stopped. Pinot tests carry `@pytest.mark.pinot` and skip when the controller is unreachable.
- All randomness is seeded from the profile seed (default 42); the same command twice gives identical Parquet contents.
- Regulatory vocabulary in names: `risk_class`, `bucket`, `liquidity_horizon`, `weighted_sensitivity`, `cet1`, `rwa`.
- Each task ends with `uv run ruff check . && uv run pytest -q` green and a commit.

## Review Focus

1. **A filter or slice that matches no rows** returns an empty result with the right columns; it does not raise or return a zero VaR. (Tests in Tasks 5 and 7.)
2. **P&L arrays of unequal length or containing nulls** fail at the reporting-grain step with an error naming the dataset and the lengths found, not with a wrong number. (Task 5.)
3. **Re-running the same as-of date** leaves both runs on disk, serves only the official one, and never doubles an additive total. (Task 6.)
4. **Pinot stopped or half-loaded** during a query: the client says so and answers from Parquet. (Task 8.)
5. **A node with a single child, or a child that is the whole parent**: diversification is zero and incremental VaR equals parent VaR; no division by zero or empty join. (Task 5.)

## File Structure

```
pyproject.toml  Makefile  docker-compose.yml  .gitignore  README.md
config/semantic_model.yaml          dimensions, hierarchies, measures, tables
config/aggregations.yaml            the self-service definitions
config/reference/frtb_sa.yaml       risk weights, correlations
config/reference/stress_scenarios.yaml
config/reference/ccar.yaml
src/riskagg/
  conventions.py      quantile rules (NumPy and Spark column forms)
  spark.py            get_spark()
  context.py          RunContext, directory layout, official-run registry
  cli.py              typer app
  data/profiles.py    Profile, PROFILES
  data/org.py         hierarchy, trades, risk_factors
  data/market.py      factor model, pnl_vectors, sensitivities
  data/banking.py     macro_scenarios, banking_book, capital_start, default_risk_positions
  semantic/model.py   load + validate semantic model
  semantic/pinot_schema.py   schema + table config generation
  semantic/query.py   Query, build_query, SliceNotAvailable
  semantic/dictionary.py     data dictionary markdown
  engine/spec.py      AggregationSpec loading + validation
  engine/grain.py     reporting grain
  engine/vectors.py   element-wise sum, grouping-sets aggregation
  engine/runner.py    run_all: shared-pass execution
  engine/audit.py     manifest, reconciliations
  measures/var.py  stress.py  frtb_sa.py  frtb_drc.py  frtb_ima.py  ccar.py
  serving/curate.py   write curated tables
  serving/executors.py  PinotExecutor, PandasExecutor
  serving/pinot.py    create tables, ingest
  serving/api.py  serving/client.py
  bench/variants.py  bench/runner.py
tests/  conftest.py, unit tests per module, tests/golden/, tests/reference/ (NumPy)
docs/study/01-var-es.md 02-stress.md 03-ccar.md 04-frtb.md 05-aggregation-engineering.md
docs/benchmarks/report.md   notebooks/replicate.ipynb
```

Data layout under `data/` (git-ignored): `raw/<profile>/as_of_date=<d>/<dataset>/`, `curated/<table>/as_of_date=<d>/run_id=<r>/`, `runs/<run_id>/manifest.json`, `runs/official.json`.

---

### Task 1: Scaffold and quantile conventions

**Files:** Create `pyproject.toml`, `.gitignore`, `src/riskagg/{__init__,spark,conventions}.py`, `tests/conftest.py`, `tests/test_conventions.py`.

**Interfaces — Produces:**
- `get_spark(app_name: str = "riskagg", **conf) -> SparkSession`: local[*], AQE on, `spark.sql.shuffle.partitions` default 8, session timezone UTC.
- `conftest.spark`: session-scoped fixture, `local[2]`, 4 shuffle partitions, UI off.
- `var_rank(n: int, level: float) -> int` = `max(1, ceil(round(n * (1 - level), 9)))`
- `var_from_pnl(pnl: np.ndarray, level: float) -> float`: the `var_rank`-th largest loss.
- `es_from_pnl(pnl, level) -> float`: with `k = n*(1-level)`, `(sum of worst floor(k) losses + (k - floor(k)) * next loss) / k`.
- `var_col(vec: Column, level: float) -> Column`, `es_col(vec: Column, level: float) -> Column`: same rules as native Spark expressions over an `array<double>`.

- [ ] **Step 1:** `git switch -c feat/risk-aggregation`; `uv init --package`-style scaffold with the dependencies in Tech Stack; `.gitignore` covers `data/`, `.venv/`, `spark-warehouse/`, `__pycache__/`.
- [ ] **Step 2: Failing tests**

```python
def test_var_and_es_20_scenarios():
    pnl = np.arange(-10, 10, dtype=float)          # losses 10, 9, 8, 7, ...
    assert var_from_pnl(pnl, 0.90) == 9.0           # 2nd worst
    assert es_from_pnl(pnl, 0.80) == 8.5            # mean of worst 4

def test_headline_levels_250_scenarios():
    pnl = np.arange(250, dtype=float) - 125         # losses 125, 124, 123, ...
    assert var_rank(250, 0.99) == 3
    assert var_from_pnl(pnl, 0.99) == 123.0
    assert es_from_pnl(pnl, 0.975) == pytest.approx(122.36)   # k = 6.25

def test_all_gains_gives_negative_var():
    assert var_from_pnl(np.array([1.0, 2.0, 3.0, 4.0]), 0.75) == -1.0

def test_spark_matches_numpy(spark):               # random 250-vectors, 20 rows, shuffled order
    # var_col / es_col equal var_from_pnl / es_from_pnl to 1e-9 at 0.99 and 0.975
```
- [ ] **Step 3:** Run, expect import failures. **Step 4:** Implement. **Step 5:** Run green. **Step 6:** Commit `chore: scaffold and quantile conventions`.

### Task 2: Organisation, trades and risk factors

**Files:** Create `src/riskagg/data/{profiles,org}.py`, `tests/test_data_org.py`.

**Interfaces — Produces:**
- `Profile(name, seed, n_business_lines, n_desks, n_books, n_trades, n_scenarios, skew_desk_share, windows)`; `PROFILES`: `tiny` (2, 3, 6, 20 trades, 20 scenarios), `demo` (4, 20, 200, 100_000, 250), `bench` (4, 40, 400, 1_000_000, 250, skew 0.4, windows = CURRENT only). `tiny` and `demo` have both windows and skew 0.
- `make_hierarchy(spark, p) -> DataFrame`: `book_id, book_name, desk_id, desk_name, business_line_id, business_line_name, firm_id, legal_entity, region`. One legal entity and region per book; `tiny` has 2 legal entities.
- `make_risk_factors(spark, p) -> DataFrame`: `risk_factor_id, risk_class (GIRR|EQ|FX), factor_type (DELTA|VEGA), bucket, tenor_years, currency, liquidity_horizon, in_reduced_set, daily_vol`. GIRR buckets are currencies with tenors 1, 2, 5, 10, 30; EQ buckets `1, 5, 8, 12`; FX buckets are non-USD currencies. Liquidity horizons: GIRR USD/EUR/GBP/JPY 10, other 20; EQ buckets 5, 12 → 10, buckets 1, 8 → 20; EQ vega 20 or 60; FX major 10, other 20; FX vega 40. About 80% of factors are in the reduced set.
- `make_trades(spark, p, as_of_date) -> DataFrame`: `trade_id (long), book_id, asset_class (RATES|EQUITY|FX), currency, counterparty_id, notional, is_exotic, as_of_date`. Built with `spark.range`, not on the driver. With skew, `skew_desk_share` of trades fall in one desk's books.

- [ ] **Step 1: Failing tests:** tiny row counts exact (6 books, 20 trades); every book has one legal entity and region; determinism (two calls give equal collected rows); bench-profile skew checked on a 10k-trade override: largest desk share within 0.38–0.42.
- [ ] **Steps 2–5:** fail, implement, pass, commit `feat: organisation and trade generators`.

### Task 3: P&L vectors and sensitivities

**Files:** Create `src/riskagg/data/market.py`, `tests/test_data_market.py`.

**Interfaces — Consumes:** Task 2 frames. **Produces:**
- `factor_returns(p, risk_factors_pdf, window) -> np.ndarray` shape `(n_factors, n_scenarios)`: correlated one-day returns (one common factor per risk class plus idiosyncratic); the STRESSED window multiplies volatility by 3 and raises correlation.
- `make_exposures(spark, p, trades, risk_factors) -> DataFrame`: `trade_id, risk_factor_id, exposure`. Each trade has a primary factor in its asset class; non-USD EQUITY and RATES trades also get an FX factor; about 30% of trades are options with a VEGA factor and non-zero gamma.
- `make_pnl_vectors(spark, p, trades, exposures, risk_factors) -> DataFrame`: `trade_id, book_id, asset_class, currency, risk_class, liquidity_horizon, in_reduced_set, window, pnl array<double>`; `pnl[s] = Σ exposure_f × return_f[s]` over the trade's factors in that group.
- `make_sensitivities(spark, p, exposures, risk_factors) -> DataFrame`: `trade_id, book_id, risk_factor_id, risk_class, factor_type, bucket, tenor_years, delta, gamma, vega, curvature_up, curvature_down`. `delta = exposure` on DELTA factors, `vega = exposure` on VEGA factors, zeros elsewhere; curvature figures are positive for a loss and zero for non-options.

- [ ] **Step 1: Failing tests:** every `pnl` has length `p.n_scenarios`; summing a tiny trade's CURRENT vectors equals `exposures @ factor_returns` recomputed in NumPy; two desks' summed vectors have correlation strictly between 0 and 1 (diversification is real); STRESSED vectors have larger standard deviation; determinism.
- [ ] **Steps 2–5:** fail, implement (mapInPandas with the returns matrix broadcast), pass, commit.

### Task 4: Semantic model and aggregation specifications

**Files:** Create `config/semantic_model.yaml`, `config/aggregations.yaml`, `src/riskagg/semantic/model.py`, `src/riskagg/engine/spec.py`, `tests/test_spec.py`.

**Interfaces — Produces:**
- YAML shape:

```yaml
dimensions:   {desk_id: {label: Desk, description: ...}, ...}
hierarchies:  {org: [firm_id, business_line_id, desk_id, book_id]}
grain:        [firm_id, business_line_id, desk_id, book_id, legal_entity, region, asset_class, currency, risk_class]
measures:
  var_99_1d:  {label: ..., description: ..., unit: USD, class: vector, function: var,
               params: {level: 0.99, window: CURRENT, horizon_days: 1}, table: risk_metrics}
tables:       {risk_metrics: {...columns}, pnl_scenarios: {...}}
```
  Vector measures in this task: `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d` (window STRESSED). `class` is `additive | vector | structured`. Structured measures carry `computed_at: [[dims], ...]`.
- `aggregations.yaml`: list of `{name, measure, grouping_sets: [[dim, ...], ...], filters: {dim: [values]}, diversification: bool}`. Ship `var_by_org` (four org levels, diversification on) and `var_by_desk_asset_class`.
- `load_model(path) -> SemanticModel` with `.measure(name) -> Measure`, `.dimensions`, `.grain`, `.hierarchies`.
- `load_specs(path, model) -> list[AggregationSpec]`; `spec_hash(specs, reference_paths) -> str` (sha256 of canonical content, first 12 hex).
- `SpecError` raised with messages naming the spec and the problem.

- [ ] **Step 1: Failing tests**

```python
def test_shipped_config_loads(): ...                      # both YAML files load, 2 specs
def test_unknown_dimension_rejected():                    # match="unknown dimension 'trader'"
def test_vector_measure_outside_grain_rejected():         # grouping by counterparty_id; match="not available for vector"
def test_structured_measure_only_at_computed_levels():    # match="computed only at"
def test_diversification_requires_org_hierarchy_sets(): ...
def test_spec_hash_changes_with_reference_data(tmp_path): ...
```
- [ ] **Steps 2–5:** fail, implement, pass, commit.

### Task 5: Engine and VaR

**Files:** Create `src/riskagg/engine/{grain,vectors,runner}.py`, `src/riskagg/measures/var.py`, `tests/test_engine.py`, `tests/test_var.py`, `tests/reference/np_var.py`.

**Interfaces — Consumes:** Tasks 1–4. **Produces:**
- `build_reporting_grain(pnl_vectors, hierarchy, n_scenarios) -> DataFrame`: one row per `grain` dims + `liquidity_horizon, in_reduced_set, window`, with summed `pnl`; hierarchy joined by broadcast. Raises `ValueError` naming the lengths found if any array length ≠ `n_scenarios` or any element is null.
- `sum_vectors(col: str, n: int) -> Column`: element-wise sum as `array(sum(col[0]), …, sum(col[n-1]))`.
- `aggregate_vectors(grain, grouping_sets, n_scenarios, filters=None) -> DataFrame`: one grouping-sets pass; columns = all grain dims (`*` where rolled up), `node_level` (the set's dims joined with `|`), `window`, `pnl`.
- `METRIC_COLUMNS = [aggregation, measure, node_level, *grain dims, scenario, value]`
- `var.metrics_from_vectors(nodes, spec, measure) -> DataFrame` in `METRIC_COLUMNS`; `horizon_days` scales by `sqrt` (`SIMPLIFICATION`).
- `var.diversification(nodes, spec, measure, hierarchy_levels) -> DataFrame`: per parent node, rows `<measure>__sum_children`, `<measure>__diversification`; per child, `<measure>__incremental` = parent VaR − VaR(parent vector − child vector).
- `run_all(specs, model, grain, n_scenarios) -> DataFrame`: groups vector specs sharing filters, unions their grouping sets into one `aggregate_vectors` call, then derives each spec's metrics. Dispatches other classes through `MEASURE_REGISTRY: dict[str, Callable]`, which later tasks register into.

- [ ] **Step 1: Failing tests**

```python
def test_two_book_diversification(spark):
    # book A pnl [-10, 5, 0, 5], book B [5, -10, 0, 5], same desk, level 0.75 (rank 1)
    # desk VaR == 5; A and B VaR == 10; sum_children == 20; diversification == 15
    # incremental for A == 5 - 10 == -5

def test_single_child_parent(spark):      # diversification == 0, incremental == parent VaR
def test_matches_numpy_reference(spark):  # tiny profile, every node of var_by_org at 0.99/0.975 vs np_var, 1e-9
def test_partition_and_order_invariance(spark):   # repartition(1) vs repartition(7) + orderBy(rand): equal metrics
def test_es_subadditive(spark):           # every parent ES <= sum of children ES + 1e-9
def test_one_pass_for_shared_specs(spark):# run_all plan for both shipped specs contains exactly one Expand node
def test_empty_filter_gives_empty_frame(spark):   # filters desk_id: [NOPE] -> 0 rows, METRIC_COLUMNS
def test_ragged_vectors_rejected(spark):  # match="pnl_vectors.*length"
```
- [ ] **Steps 2–5:** fail, implement, pass, commit `feat: aggregation engine and VaR`.

### Task 6: Run context, audit, curated outputs, CLI

**Files:** Create `src/riskagg/{context,cli}.py`, `src/riskagg/engine/audit.py`, `src/riskagg/serving/curate.py`, `tests/test_run.py`.

**Interfaces — Produces:**
- `RunContext(profile, as_of_date, run_id, spec_hash, root: Path)` with `raw(dataset)`, `curated(table)`, `manifest_path`; `new_run(profile, as_of_date, root, spec_hash)` (run_id = UTC timestamp + 6 hex).
- `official_run(root, as_of_date) -> str | None`; `mark_official(ctx)`.
- `curate.write_risk_metrics(df, ctx)` adds `as_of_date, run_id, spec_hash`; `curate.write_pnl_scenarios(grain, ctx)` explodes the grain summed over `liquidity_horizon` and `in_reduced_set` to `grain dims, window, scenario_idx, pnl`.
- `audit.write_manifest(ctx, inputs, spark)`: per input dataset row count and a content checksum, Spark conf, git commit, spec hash.
- `audit.reconcile(ctx, spark) -> list[Check(name, passed, detail)]`: firm vector = Σ book vectors (1e-6); `pnl_scenarios` firm sum per scenario = firm vector; no `(node, measure)` duplicated. Later tasks append checks.
- CLI: `riskagg generate --profile --as-of`, `riskagg run --profile --as-of` (aggregates, curates, reconciles; marks official only if all checks pass; non-zero exit otherwise).

- [ ] **Step 1: Failing tests:** `generate` then `run` on tiny writes `risk_metrics` and a manifest; running twice leaves two `run_id=` directories and `official.json` names the second; reading `risk_metrics` filtered to the official run gives each `(node, measure)` once; a corrupted grain (one vector altered after curation) makes `reconcile` fail and `run` exit non-zero without changing `official.json`.
- [ ] **Steps 2–5:** fail, implement, pass, commit.

### Task 7: Query builder, executors, Pinot schema generation

**Files:** Create `src/riskagg/semantic/{query,pinot_schema}.py`, `src/riskagg/serving/executors.py`, `tests/test_query.py`.

**Interfaces — Produces:**
- `Query(table, select_dims, aggregate: tuple[str, str] | None, filters: dict, kind: "additive"|"vector_scenarios"|"lookup", post: Measure | None)` with `.to_pinot_sql() -> str`.
- `build_query(model, measure, group_by, filters, as_of_date, run_id) -> Query`. Rules: additive → `SUM(value)`-style over its detail table; vector → `SELECT dims, scenario_idx, SUM(pnl) FROM pnl_scenarios … GROUP BY dims, scenario_idx` with `kind="vector_scenarios"`; structured → lookup in `risk_metrics` when `group_by` matches a `computed_at` set, else `SliceNotAvailable` with an explanation. Every query filters `as_of_date` and `run_id`.
- `finish(query, rows: pd.DataFrame) -> pd.DataFrame`: for `vector_scenarios`, applies `conventions` per group. The quantile is always taken here, never in SQL, so one implementation of the convention serves every path.
- `PandasExecutor(root).execute(query) -> pd.DataFrame` (pyarrow dataset with partition filters); `PinotExecutor(broker_url).execute(query)` (POST `/query/sql`).
- `pinot_schema(model, table) -> dict`, `pinot_table_config(model, table) -> dict`: dims as dimension fields, `value`/`pnl` as metrics, `as_of_date` as the time column, inverted indexes on `node_level`, `desk_id`, `measure`; a star-tree index on the additive detail tables.

- [ ] **Step 1: Failing tests**

```python
def test_vector_sql_sums_by_scenario_only():
    sql = build_query(model, "var_99_1d", ["desk_id"], {}, D, R).to_pinot_sql()
    assert "SUM(pnl)" in sql and "scenario_idx" in sql and "PERCENTILE" not in sql.upper()

def test_var_by_counterparty_refused():           # SliceNotAvailable, match="counterparty"
def test_every_query_pins_date_and_run(): ...
def test_adhoc_var_equals_precomputed(tmp_run):    # PandasExecutor: var_99_1d by desk == risk_metrics desk rows, 1e-9
def test_adhoc_slice_not_precomputed(tmp_run):     # by (region, asset_class) equals NumPy reference
def test_no_matching_rows_is_empty(tmp_run): ...
def test_schema_generated_from_model(): ...        # every model column present, time column set
```
- [ ] **Steps 2–5:** fail, implement, pass, commit.

### Task 8: Pinot, API, client (thin slice complete)

**Files:** Create `docker-compose.yml`, `src/riskagg/serving/{pinot,api,client}.py`, `Makefile`, `tests/test_api.py`, `tests/test_pinot.py`.

**Interfaces — Produces:**
- Compose service `pinot` (pin a current stable `apachepinot/pinot` tag; `QuickStart -type EMPTY`; controller 9000, broker 8000).
- `pinot.load(ctx, model, controller_url)`: creates or updates schema and offline table for each curated table, ingests the official run's files, replacing that as-of date's segments. CLI `riskagg load-pinot`.
- `pinot.is_up(controller_url) -> bool`.
- API: `GET /catalog` (measures, dimensions, valid slices from the model), `GET /metrics/{measure}?group_by=a,b&as_of=…&<dim>=v`, `GET /drilldown/{table}`; 422 with the `SliceNotAvailable` text. CLI `riskagg serve`.
- `RiskClient(base_url=None, root="data")`: `.catalog()`, `.metric(measure, group_by, **filters) -> pd.DataFrame`, `.source` (`"pinot"` or `"parquet"`). Uses Pinot when up, otherwise Parquet with a logged notice.
- `make demo`: generate → run → (compose up, load-pinot if Docker present) → print firm VaR via the client.

- [ ] **Step 1: Failing tests:** FastAPI TestClient returns firm `var_99_1d` equal to `risk_metrics`; unknown measure → 404; refused slice → 422 with the reason; with an unreachable broker URL the client answers from Parquet and `.source == "parquet"`; `@pytest.mark.pinot`: after `load`, Pinot and Parquet executors agree on VaR by desk, and loading the same date twice does not change `SELECT COUNT(*)`.
- [ ] **Step 2–4:** fail, implement, pass (run the Pinot tests with Docker up at least once and record the result in the commit message).
- [ ] **Step 5:** Write `docs/study/01-var-es.md` per spec section 8, with a worked example on `tiny` at 90% VaR and 80% ES, and `tests/golden/test_var_worked_example.py` asserting every number printed in it. Commit.

### Task 9: Stress testing

**Files:** Create `config/reference/stress_scenarios.yaml`, `src/riskagg/measures/stress.py`, `tests/test_stress.py`, `docs/study/02-stress.md`, golden test. Modify model, aggregations, `curate.py`, `audit.py`.

**Interfaces — Produces:**
- Scenario file: `{name, description, global_market_shock: bool, shocks: [{risk_class, factor_type, bucket|currency (optional), shock}]}`; the most specific matching rule wins. Scenarios: `GFC_2008` (global market shock), `RATES_UP_200BP`, `EQUITY_DOWN_30`, `FX_CRISIS`, `VOL_SPIKE`. Units: EQ/FX relative return, GIRR absolute rate change, VEGA absolute volatility change.
- `load_scenarios(path) -> list[Scenario]`; `shock_table(spark, scenarios, risk_factors) -> DataFrame (scenario, risk_factor_id, shock)`.
- `stress_pnl(sensitivities, shocks) -> DataFrame (trade_id, book_id, risk_class, scenario, pnl)`: `delta*shock + 0.5*gamma*shock² ` on DELTA factors, `vega*shock` on VEGA factors; shocks broadcast.
- `stress_by_grain(stress, trades, hierarchy) -> DataFrame` at grain dims × scenario → curated `stress_results`.
- Measures `stress_pnl` (additive, served from `stress_results`) and `worst_stress_loss` (structured, `computed_at` the four org levels; rows carry the worst scenario name in `scenario`).
- Reconciliation: firm stress P&L per scenario = Σ trades.

- [ ] **Step 1: Failing tests**

```python
def test_formula(spark):   # delta 100, gamma 50, shock -0.30 -> -27.75; plus vega 10, vol shock 0.20 -> -25.75
def test_most_specific_shock_wins(spark): ...
def test_additive_across_hierarchy(spark):      # firm == sum of desks, each scenario
def test_worst_of_is_not_additive(spark):       # constructed case: firm worst loss < sum of desk worst losses
def test_missing_shock_means_zero_not_null(spark): ...
def test_query_stress_by_desk_and_scenario(tmp_run): ...
```
- [ ] **Steps 2–5:** fail, implement, pass, study page + golden test, commit.

### Task 10: FRTB SA sensitivities-based method

**Files:** Create `config/reference/frtb_sa.yaml`, `src/riskagg/measures/frtb_sa.py`, `tests/test_frtb_sa.py`, `tests/reference/np_frtb.py`.

**Interfaces — Produces:**
- Reference values (verify each against the Basel MAR21 text online before committing; mark any departure):
  - GIRR delta risk weights by tenor 0.25…30y: 1.7, 1.7, 1.6, 1.3, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1 %, divided by √2 for USD, EUR, GBP, JPY, AUD, CAD, SEK. Intra-bucket ρ = `max(exp(-0.03*|Tk-Tl|/min(Tk,Tl)), 0.40)`. Inter-bucket γ = 0.50.
  - Equity spot risk weights: bucket 1 55%, 5 30%, 8 50%, 12 15%. Intra-bucket ρ: 15% (1), 25% (5, 8), 80% (12). γ: 15% among 1, 5, 8; 45% between 12 and the others.
  - FX: 15%, divided by √2 for USD against EUR, GBP, JPY, AUD, CAD, CHF. γ = 0.60.
  - Vega risk weights: GIRR 100%, EQ buckets 5 and 12 77.78%, buckets 1 and 8 100%, FX 100%. `SIMPLIFICATION:` one vega factor per bucket, no option-maturity correlation.
  - Curvature: `SIMPLIFICATION:` one curvature factor per bucket; `K_b = max(CVR_up, CVR_down, 0)`, `S_b` is the chosen side's sum, cross-bucket uses γ² and ψ = 0 when both `S` are negative.
  - Scenarios: medium as given; high `min(1.25*x, 1)`; low `max(2x - 1, 0.75x)`.
- `load_frtb_ref(path) -> FrtbSaRef`
- `weighted_sensitivities(sens, node_cols, ref) -> DataFrame`: netted per node × risk factor, `ws = sensitivity × rw`.
- `sbm_charges(sens, node_cols, ref) -> DataFrame (node_cols…, risk_class, component DELTA|VEGA|CURVATURE, corr_scenario LOW|MEDIUM|HIGH, bucket_detail-free charge)` using `K_b = sqrt(max(0, ΣΣ ρ ws_k ws_l))` and `sqrt(Σ K_b² + ΣΣ γ S_b S_c)`, with the `S_b = max(min(S_b, K_b), -K_b)` alternative when the sum is negative.
- `sbm_total(charges) -> DataFrame (node_cols…, corr_scenario, charge, is_binding)`; the binding scenario is the largest total.
- `bucket_detail(...)` → curated `frtb_sa_detail` (node, risk_class, component, bucket, corr_scenario, k_b, s_b).

- [ ] **Step 1: Failing tests**

```python
def test_equity_intra_bucket(spark):
    # bucket 5, deltas 100 and -50 -> ws 30, -15
    # MEDIUM 30.0; HIGH approx 29.047; LOW approx 30.923  (LOW binds for a hedged book)
def test_equity_inter_bucket(spark):
    # add bucket 8 delta 50 -> ws 25; MEDIUM total approx 40.466
def test_fx_specified_pair(spark):       # EUR delta 1000 -> approx 106.066
def test_girr_tenor_correlation():       # rho(1y, 2y) approx 0.97045; rho(1y, 30y) == 0.40
def test_netting_before_weighting(spark):# +100 and -100 on one factor in two books: desk charge 0, book charges > 0
def test_not_a_rollup(spark):            # firm charge < sum of desk charges on tiny
def test_matches_numpy_reference(spark): # tiny, desk and firm, all scenarios, 1e-6
def test_null_sensitivities_treated_as_zero(spark): ...
```
- [ ] **Steps 2–5:** fail, implement, pass, commit.

### Task 11: FRTB default risk charge, residual add-on, SA total

**Files:** Create `src/riskagg/data/banking.py` (default-risk positions only in this task), `src/riskagg/measures/frtb_drc.py`, `tests/test_frtb_drc.py`. Modify model, aggregations, `curate.py`, `cli.py`.

**Interfaces — Produces:**
- `make_default_risk_positions(spark, p, trades) -> DataFrame (trade_id, book_id, issuer_id, drc_bucket CORPORATE|SOVEREIGN, rating, seniority SENIOR|NON_SENIOR|EQUITY, direction LONG|SHORT, notional)`.
- `drc(positions, node_cols, ref) -> DataFrame (node_cols…, drc_bucket, hbr, charge)`: JTD = LGD × notional (LGD 75% senior, 100% otherwise); net long against short per issuer; risk weights AAA 0.5%, AA 2%, A 3%, BBB 6%, BB 15%, B 30%, CCC 50%, unrated 15%; `HBR = Σ net long / (Σ net long + Σ |net short|)`; `charge = max(Σ rw·long − HBR·Σ rw·|short|, 0)`. `SIMPLIFICATION:` no maturity scaling, no seniority-specific netting.
- `rrao(trades, node_cols) -> DataFrame`: 1.0% of notional where `is_exotic`.
- `frtb_sa_total(sbm_total, drc, rrao) -> DataFrame`; structured measures `frtb_sa_charge`, `frtb_sbm_charge`, `frtb_drc_charge`, `frtb_rrao_charge`, `computed_at` desk and firm, registered in `MEASURE_REGISTRY`.

- [ ] **Step 1: Failing tests:** long 100 senior BBB and short 40 senior BB → HBR ≈ 0.714286, charge ≈ 1.285714; only shorts → charge 0 and HBR 0 without dividing by zero; RRAO of 1,000,000 exotic notional = 10,000; total = sum of the three at desk and firm on tiny; API refuses `frtb_sa_charge` by book with a 422.
- [ ] **Steps 2–5:** fail, implement, pass, commit.

### Task 12: FRTB internal models approach

**Files:** Create `src/riskagg/measures/frtb_ima.py`, `tests/test_frtb_ima.py`, `docs/study/04-frtb.md`, golden test.

**Interfaces — Consumes:** reporting grain (with `liquidity_horizon, in_reduced_set, window`). **Produces:**
- `lh_adjusted_es(es_by_lh: dict[int, float]) -> float`: `sqrt(ES_10² + Σ_j (ES_j · sqrt((LH_j − LH_{j−1})/10))²)` over 20, 40, 60, 120, where `ES_j` is the ten-day ES of factors with horizon ≥ `LH_j`. Pure function, used by the NumPy reference.
- `ima_es(grain, node_cols, n_scenarios) -> DataFrame (node_cols…, risk_class ('*' = unconstrained), es_rs, es_rc, es_fc, es)`: cumulative "horizon ≥ j" vectors per node, 97.5% ES via `es_col`, ×√10 (`SIMPLIFICATION`), `es = es_rs × max(es_fc/es_rc, 1)`; when `es_rc` is 0 the ratio is 1.
- `imcc(ima_es) -> DataFrame (node_cols…, imcc_unconstrained, imcc_constrained_sum, imcc)` with `imcc = 0.5·unconstrained + 0.5·Σ risk classes`. Structured measure `frtb_imcc` at desk and firm.

- [ ] **Step 1: Failing tests:** `lh_adjusted_es({10: 100, 20: 60, 40: 30, 60: 0, 120: 0}) ≈ 124.097`; `es_rs 150, es_fc 110, es_rc 100 → 165`; unconstrained 165 with constrained sum 200 → 182.5; ratio floored at 1 when full-set ES is below reduced-set ES; tiny profile matches a NumPy reference built from raw vectors at desk and firm.
- [ ] **Steps 2–4:** fail, implement, pass.
- [ ] **Step 5:** Study page covering SA and IMA, including the parts explained but not built (non-modellable risk factors, P&L attribution, backtesting, desk eligibility), with one worked SA bucket and one worked cascade; golden test; commit.

### Task 13: CCAR

**Files:** Extend `src/riskagg/data/banking.py`; create `config/reference/ccar.yaml`, `src/riskagg/measures/ccar.py`, `tests/test_ccar.py`, `docs/study/03-ccar.md`, golden test. Modify model, `curate.py`, `audit.py`.

**Interfaces — Produces:**
- `make_macro_scenarios(spark, p) -> (scenario BASELINE|SEVERELY_ADVERSE, quarter 0..9, unemployment_rate, gdp_growth)`; quarter 0 is the starting point.
- `make_banking_book(spark, p, hierarchy) -> (segment_id, legal_entity, business_line_id, ead, pd_base, lgd, revenue_rate_q, expense_rate_q, risk_weight)`.
- `make_capital_start(spark, p, hierarchy) -> (legal_entity, cet1_start, rwa_trading, dividends_q)`.
- `ccar.yaml`: `pd_unemployment_beta: 0.35`, `revenue_gdp_beta: 2.0`, `tax_rate: 0.21`, `scb_floor: 0.025`.
- `project(banking_book, macro, capital_start, gms_loss_by_entity_bl, params) -> DataFrame` = `ccar_projection (scenario, quarter 1..9, node_level BUSINESS_LINE|LEGAL_ENTITY|FIRM, node_id, ppnr, credit_loss, trading_loss, net_income, dividends, cet1, rwa, cet1_ratio)`; capital columns are null for business lines. Formulas, per quarter:
  - `pd_q = min(1, pd_base/4 × (1 + beta × max(0, unemployment_q − unemployment_0)))` with unemployment in percentage points
  - `credit_loss = ead × pd_q × lgd`; `ppnr = ead × (revenue_rate_q × (1 + gdp_beta × gdp_growth_q) − expense_rate_q)`
  - `trading_loss` = the global-market-shock loss from Task 9 in quarter 1 of SEVERELY_ADVERSE, else 0
  - `net_income = (ppnr − credit_loss − trading_loss) × (1 − tax_rate)`; `cet1_q = cet1_{q−1} + net_income − dividends_q`
  - `rwa = Σ ead × risk_weight + rwa_trading`, held constant; firm = Σ legal entities
  - `SIMPLIFICATION:` constant balance sheet, tax applied symmetrically, no counterparty default component.
- `summary(projection, params) -> DataFrame`: `ccar_min_cet1_ratio` per scenario and `ccar_scb = max(floor, ratio_0 − min ratio(SEVERELY_ADVERSE) + Σ dividends q4..q7 / rwa)` for legal entities and firm; written to `risk_metrics`.
- Reconciliation: firm losses = Σ business lines = Σ legal entities, each scenario and quarter.

- [ ] **Step 1: Failing tests**

```python
def test_one_entity_first_quarter(spark):
    # ead 1000, pd_base 0.04, lgd 0.5, unemployment +4pp -> credit_loss 12.0
    # revenue_rate_q 0.01, gdp -0.02, expense 0.004 -> ppnr 5.6
    # trading_loss 20, dividends 1, cet1_start 120, risk_weight 0.8, rwa_trading 200
    # net_income approx -20.856; cet1 approx 98.144; cet1_ratio approx 0.098144
def test_baseline_has_no_trading_loss(spark): ...
def test_cumulative_path_is_running_sum(spark):   # cet1_q9 == cet1_start + sum(net_income) - 9*dividends
def test_scb_floor(spark):                        # mild scenario -> exactly 0.025
def test_business_line_capital_ratio_refused():   # SliceNotAvailable, match="not allocated"
def test_losses_reconcile_across_cuts(spark): ...
```
- [ ] **Steps 2–5:** fail, implement, pass, study page + golden test, commit.

### Task 14: Benchmark

**Files:** Create `src/riskagg/bench/{variants,runner}.py`, `tests/test_bench.py`, `docs/benchmarks/report.md` (generated), `docs/study/05-aggregation-engineering.md`.

**Interfaces — Produces:**
- `Variant(name, group, build: Callable[[SparkSession, Inputs], DataFrame], conf: dict)`. Groups and variants:
  - `vector_sum`: `explode_regroup` (posexplode, sum by index, collect back) vs `array_index_sum` (the engine's).
  - `rollup`: `per_level_from_trades` (one aggregation from trade vectors per org level) vs `grain_then_grouping_sets`.
  - `skew`: on the hash aggregate and on a sort-merge join of `pnl_vectors` to a desk-keyed table with broadcast disabled: `plain`, `aqe_skew_join`, `salted`.
  - `join`: hierarchy join as sort-merge vs broadcast.
  - `shuffle_partitions`: 200 vs sized to the data.
- `run_benchmark(profile, out_path, repeats=3)`: for each variant, median wall time, shuffle read/write bytes and peak execution memory from the Spark REST status API, and the executed plan's key operators; asserts variants in a group return equal results on a sample before timing. CLI `riskagg bench --profile bench`.
- The report states what was measured, including any technique that gave no gain and why (for example, map-side partial aggregation absorbing key skew in a hash aggregate). No number is written by hand.

- [ ] **Step 1: Failing tests** (on a 5k-trade override): every group's variants agree to 1e-6; the report contains a table row per variant with non-empty timing and shuffle columns.
- [ ] **Steps 2–4:** fail, implement, pass; run `riskagg bench --profile bench` once and commit the generated report.
- [ ] **Step 5:** Study page 05: the reporting-grain idea, declared grouping sets against a full cube, why vectors stay as arrays, non-additivity in the OLAP layer, auditability, and the benchmark findings quoted from the report. Commit.

### Task 15: Data dictionary, notebook, README

**Files:** Create `src/riskagg/semantic/dictionary.py`, `docs/data-dictionary.md` (generated), `notebooks/replicate.ipynb`, `README.md`, `tests/test_docs.py`.

**Interfaces — Produces:**
- `write_dictionary(model, path)`: per measure its business name, description, unit, aggregation class, the slices it may be queried at and the rule for combining it; per dimension and table likewise. CLI `riskagg dictionary`.
- Notebook: firm `var_99_1d` through `RiskClient`, rebuilt from raw `pnl_vectors` in NumPy with `conventions`, asserted equal; the same for one FRTB SA equity bucket from raw sensitivities.
- README: what this is, quick start (`make demo`), architecture sketch, a table mapping each capability the project demonstrates to code and study pages, the list of simplifications, and how to add an aggregation by editing YAML (worked example: VaR by region).

- [ ] **Step 1: Failing tests:** the dictionary lists every measure in the model; adding the README's example spec to a copy of `aggregations.yaml` and running on tiny yields `var_99_1d` rows at `node_level == "region"` with no Python change; the notebook executes top to bottom (`jupyter nbconvert --execute`) on tiny; every `SIMPLIFICATION:` comment in `src/` appears in a study page's simplifications list.
- [ ] **Steps 2–4:** fail, implement, pass.
- [ ] **Step 5:** Full check from a clean clone: `uv sync && make demo` with Docker up and again with Docker stopped; record both outcomes in the commit message. Commit.
