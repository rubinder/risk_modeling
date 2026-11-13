# riskagg: risk aggregation on PySpark

`riskagg` aggregates market and capital risk measures over an organisation hierarchy with
PySpark, and serves them through a semantic query layer, Apache Pinot, an HTTP API and a
Python client. It covers historical-simulation VaR and expected shortfall, stress testing,
a stylised CCAR capital projection and FRTB capital (standardised approach and the IMCC of the
internal models approach).

Which aggregations exist is configuration, not code: measures, dimensions and tables are
declared in `config/semantic_model.yaml`, and the slices to precompute in
`config/aggregations.yaml`. The data is synthetic and seeded. Everything runs on one machine
in Spark local mode. Five study pages explain the risk measures and the engineering, each
with a worked example whose numbers are pinned by a test.

## Prerequisites

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Java 17 (for Spark 4)
- Docker, optional: only for Apache Pinot. Without it the client and API answer from the
  curated Parquet files.

## Quick start

```sh
uv sync
make demo      # generate -> run -> (Pinot up, load) -> two queries through the client
make test      # uv run pytest -q
make lint      # uv run ruff check .
```

`make demo` runs, with `PROFILE=demo` (100,000 trades, 250 scenarios) and `ROOT=data`:

1. `riskagg generate`: the raw datasets as Parquet under `data/raw/`.
2. `riskagg run`: aggregates, writes the curated tables under `data/curated/`, runs the
   reconciliation checks, and marks the run official only if all of them pass.
3. If `docker info` succeeds: `make pinot-up` (one Pinot container, `docker-compose.yml`) and
   `riskagg load-pinot`. Otherwise it prints that Pinot is skipped.
4. `riskagg query var_99_1d --group-by firm_id` and the five desks with the largest VaR. Each
   answer names its source (`pinot` or `parquet`) and the run that produced it.

The demo profile writes about 750 MB under `data/`. `make pinot-down` stops Pinot. Pinot's
ports are bound to 127.0.0.1 because the quick-start container has no authentication.

## Architecture

```
config/semantic_model.yaml   config/aggregations.yaml   config/reference/*.yaml
            |                          |                         |
            v                          v                         v
riskagg generate --> raw/<profile>/as_of_date=<d>/<dataset>      (synthetic, seeded)
            |
riskagg run ------> Spark, local mode
            |         trades -> reporting grain (one pass, vectors summed as arrays)
            |         grain  -> one grouping-sets pass for every declared VaR/ES node
            |         other measures -> MEASURE_REGISTRY (stress, FRTB SA/DRC/IMA, CCAR)
            v
         curated/<table>/as_of_date=<d>/run_id=<r>/     runs/<r>/manifest.json
            |         reconciliation checks -> runs/official.json (only if all pass)
            v
riskagg load-pinot --> Apache Pinot (official run only; schemas generated from the model)
            |
RiskService: semantic query layer; Pinot when it serves the official run, else Parquet
            |
            +--> riskagg serve (FastAPI)  --> RiskClient(base_url=...)
            +--> RiskClient(root=...) in-process  --> riskagg query, notebooks
```

| Path | What it holds |
|---|---|
| `src/riskagg/data/` | Seeded generators: hierarchy, trades, P&L vectors, sensitivities, banking book |
| `src/riskagg/engine/` | Specs, reporting grain, grouping-sets vector aggregation, audit |
| `src/riskagg/measures/` | VaR/ES plans, stress, FRTB SA, DRC/RRAO, IMA, CCAR |
| `src/riskagg/semantic/` | Semantic model, query builder and refusals, Pinot schemas, data dictionary |
| `src/riskagg/serving/` | Curated tables, Pinot loader, executors, service, API, client |
| `src/riskagg/bench/` | Naive and tuned variants, benchmark runner and report |
| `src/riskagg/conventions.py` | The only home of the VaR/ES quantile convention |
| `tests/reference/` | Plain NumPy/loop re-implementations the engine is tested against |
| `docs/study/` | Study pages; `docs/benchmarks/report.md` is generated |

## Three aggregation classes, and why VaRs are never summed

Every measure in `config/semantic_model.yaml` has a class that decides how it may combine:

- **additive** (`stress_pnl`): sums over any of its dimensions. Stress P&L adds across
  trades within one scenario.
- **vector** (`var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`): a quantile of a P&L
  vector. A parent's VaR is not the sum of its children's VaRs: each child's VaR comes from
  its own worst scenarios, and children offset each other within a scenario. So the engine
  sums P&L vectors scenario by scenario and takes the quantile of each node's own vector.
  The query layer does the same for any slice of the grain: Pinot or pandas sums P&L per
  scenario, then `src/riskagg/conventions.py` takes the quantile once, in Python.
- **structured** (worst stress loss, FRTB charges, CCAR ratios and buffer): recomputed at
  each node from that node's own inputs (netted sensitivities, a capital path). These are
  only looked up at the levels in their `computed_at`, never summed, averaged or rebuilt.
  CCAR measures also declare `not_allocated_to: [business_line_id, desk_id, book_id]`.

A query that would need a wrong combination is refused with the reason (HTTP 422, or
`SliceNotAvailable` in Python), for example:

```
var_99_1d cannot be sliced by 'counterparty_id': the measure is non-additive and P&L vectors
are not stored at that grain (available: ['firm_id', 'business_line_id', ...])
ccar_scb cannot be sliced by 'desk_id': capital is not allocated to desks, so the measure
exists only at [['firm_id'], ['firm_id', 'legal_entity']]
```

No star-tree index is built on a Pinot table that serves a non-additive measure, because a
star-tree pre-aggregates by summing. The generated [data dictionary](docs/data-dictionary.md)
lists every measure with its class, the slices it may be queried at and its combining rule.

## What this project demonstrates

| Capability | Where in the code | Where explained | Coverage |
|---|---|---|---|
| Metadata-driven aggregation on Spark | `config/semantic_model.yaml`, `config/aggregations.yaml`, `src/riskagg/engine/spec.py`, `engine/runner.py`, `engine/vectors.py` | [05 §2-3](docs/study/05-aggregation-engineering.md) | Full for the measures built |
| Reducing the cost of high-dimensional aggregations | Reporting grain (`engine/grain.py`), declared grouping sets in one pass instead of a 512-set cube, vectors kept as arrays (`sum_vectors`) | [05 §2-4](docs/study/05-aggregation-engineering.md) | Measured: see Benchmark below |
| Semantic layer for BI and AI queries | `src/riskagg/semantic/model.py`, `semantic/query.py`, `GET /catalog`, `riskagg dictionary` | [05 §5](docs/study/05-aggregation-engineering.md), [data dictionary](docs/data-dictionary.md) | The model, catalogue and refusals are built; no BI tool or language-model front end is connected |
| Spark tuning: partitioning, shuffle, memory | `src/riskagg/bench/variants.py`, `bench/runner.py`, `src/riskagg/spark.py` | [05 §6-7](docs/study/05-aggregation-engineering.md), [report](docs/benchmarks/report.md) | Partial: local mode on one machine, 100,000 trades (limited by local disk space); shuffle bytes, task times and peak execution memory from Spark's REST API; no network I/O, since local-mode shuffles go through the local disk |
| Self-service aggregations without new code | `config/aggregations.yaml`; new VaR/ES levels or windows as `vector` measures in `config/semantic_model.yaml` | [Adding an aggregation](#adding-an-aggregation-by-editing-yaml) | New slices and new VaR/ES parameters need no code; a new structured measure (a new formula) needs a function in `MEASURE_REGISTRY` |
| APIs and curated datasets for UIs and notebooks | `src/riskagg/serving/curate.py`, `serving/api.py`, `serving/client.py`, `notebooks/replicate.ipynb` | [Querying](#querying) | API, client and notebook built; no UI |
| OLAP serving (Apache Pinot) | `src/riskagg/serving/pinot.py`, `semantic/pinot_schema.py`, `serving/executors.py`, `docker-compose.yml` | [05 §5](docs/study/05-aggregation-engineering.md), [01 Q11](docs/study/01-var-es.md) | One quick-start container on one machine; no Druid |
| Risk aggregation for VaR, stress testing, CCAR, FRTB | `src/riskagg/measures/`, `src/riskagg/conventions.py` | [01](docs/study/01-var-es.md), [02](docs/study/02-stress.md), [03](docs/study/03-ccar.md), [04](docs/study/04-frtb.md) | Simplified relative to the rules; every simplification is listed below |
| Auditable, reportable metrics | `src/riskagg/engine/audit.py`, `src/riskagg/context.py` | [05 §8](docs/study/05-aggregation-engineering.md), [Auditability](#auditability) | Full for a single machine |
| Python and SQL | Python throughout; Pinot SQL rendered in `Query.to_pinot_sql` (`semantic/query.py`); Spark through the DataFrame API | - | Java is not used; Spark SQL is written with the DataFrame API rather than SQL strings |
| Performance diagnostics | Benchmark metrics and executed plans from Spark's REST status API (`bench/runner.py`) | [05 §6-7](docs/study/05-aggregation-engineering.md) | Task run times, peak execution memory per task, shuffle bytes and records; not network I/O |
| Engineering practice | Tests against independent references (`tests/reference/`), golden tests that pin the numbers of the study pages' worked examples, reconciliation before a run is official | - | There is no team-leadership artefact here |

## Adding an aggregation by editing YAML

Example: precompute 99% one-day VaR by region. Append this to `config/aggregations.yaml`:

<!-- example-spec:start -->
```yaml
- name: var_by_region
  measure: var_99_1d
  grouping_sets:
    - [region]
  filters: {}
  diversification: false
```
<!-- example-spec:end -->

Then run again:

```sh
uv run riskagg run --profile demo --root data
```

The run validates the spec against the semantic model (an unknown dimension, a dimension the
measure cannot use, or a structured measure at a level outside its `computed_at` fails before
Spark starts), adds `[region]` to the same grouping-sets pass as the other VaR nodes, and
writes `var_99_1d` rows with `node_level` `region` (every other grain dimension `*`) to
`risk_metrics`. The spec hash in every row changes with the spec. Read the stored rows with
the drill-down:

```python
from riskagg.serving.client import RiskClient

RiskClient().drilldown("risk_metrics", measure="var_99_1d", node_level="region")
```

`tests/test_docs.py::test_readme_example_adds_var_by_region_without_code` takes the block
above from this file, appends it to a copy of the config, runs `riskagg run` on the tiny
profile and checks those rows, and that they equal the ad-hoc answer below.

A vector measure can be queried by any subset of the grain without a spec, because the query
layer rebuilds the vectors from `pnl_scenarios`:
`uv run riskagg query var_99_1d --group-by region`. A spec makes the node part of the
official, reconciled output and stores it in Pinot. A new VaR or ES level, window or horizon is
a new `vector` measure in `config/semantic_model.yaml` (`function: var` or `es`, with
`params`), also without code. A new formula, such as a new capital charge, is code: a
function registered in `MEASURE_REGISTRY` (`src/riskagg/engine/runner.py`).

## Querying

Every answer comes from the official run of the as-of date (the latest one by default) and
names its source and run. The rule is the same for the API and the client
(`src/riskagg/serving/service.py`): Pinot answers when it is up and holds exactly that run's
rows for the date; otherwise, or if a Pinot query fails, the curated Parquet files answer and
a warning is logged.

Command line (in-process client):

```sh
uv run riskagg query var_99_1d --group-by desk_id --top 5
uv run riskagg query stress_pnl --group-by scenario
uv run riskagg query ccar_min_cet1_ratio --group-by firm_id
uv run riskagg query var_99_1d --group-by region --url http://127.0.0.1:8080   # over HTTP
```

HTTP API (`uv run riskagg serve`, default `127.0.0.1:8080`):

```sh
curl 'http://127.0.0.1:8080/catalog'
curl 'http://127.0.0.1:8080/metrics/var_99_1d?group_by=desk_id&region=EMEA'
curl 'http://127.0.0.1:8080/metrics/frtb_sa_charge?group_by=firm_id,business_line_id,desk_id'
curl 'http://127.0.0.1:8080/drilldown/frtb_sa_detail?risk_class=EQ&limit=50'
```

Query parameters other than `group_by`, `as_of` and `limit` are filters (`desk_id=DESK001,DESK002`).
Errors: unknown measure, table or date 404; a slice the measure cannot answer 422 with the
reason; anything else malformed 400. A structured measure is looked up at one of its
`computed_at` levels exactly, so FRTB by desk is `group_by=firm_id,business_line_id,desk_id`,
not `group_by=desk_id`.

Python client: in-process with a data root, or over HTTP with a base URL:

```python
from riskagg.serving.client import RiskClient

client = RiskClient(root="data")                      # or RiskClient(base_url="http://127.0.0.1:8080")
desks = client.metric("var_99_1d", ["desk_id"], region="EMEA")
print(client.source, client.as_of_date, client.run_id)
client.metric("ccar_scb", ["firm_id", "legal_entity"])
client.drilldown("ccar_projection", scenario="SEVERELY_ADVERSE")
```

`notebooks/replicate.ipynb` generates and runs the tiny profile into a temporary directory,
asks the client for the firm VaR and one FRTB SA equity bucket charge, and rebuilds both from
the raw data with NumPy (using `riskagg.conventions` for the quantile). It needs no Docker
and no running server:

```sh
uv run jupyter nbconvert --to notebook --execute --output-dir /tmp notebooks/replicate.ipynb
```

## Auditability

- **Run id and as-of date** on every curated row: each table is partitioned by
  `as_of_date=<d>/run_id=<r>`, and a re-run never overwrites another run's partition.
- **Spec hash** on every `risk_metrics` row: a hash of the aggregation specs and of the
  semantic model and reference files (`spec_hash` in `src/riskagg/engine/spec.py`).
- **Manifest** `runs/<run_id>/manifest.json`: row count and order-independent checksum of
  the input datasets (P&L vectors, hierarchy, default-risk positions, banking book, macro
  scenarios, starting capital), the Spark version and the session settings the project sets,
  the git commit, and the reconciliation results (`write_manifest` in
  `src/riskagg/engine/audit.py`).
- **Reconciliation checks** read what was written to disk and use separate code from the
  engine: the firm vector from raw trades equals the sum of the curated book vectors; firm
  VaR equals the quantile of the curated firm P&L; no node appears twice; firm stress P&L
  equals the sum over raw trades; the FRTB SA total equals its components; the firm IMCC is
  rebuilt from raw vectors; CCAR is rebuilt in plain Python.
- **Official run**: only a run that passes every check is written to `runs/official.json`,
  and only official runs are loaded into Pinot or served.

## Benchmark

`riskagg bench --profile bench` times naive against tuned variants of the engine's steps and
writes [docs/benchmarks/report.md](docs/benchmarks/report.md). The figures below are copied
from that report; `tests/test_docs.py` checks each number is in it.

<!-- benchmark:start -->
- Setup: Spark local mode on one Apple M3 (8 logical cores, 16 GiB), `bench` profile of
  100,000 trades, 250 scenarios per vector, 356.5 MiB of P&L vectors.
- Element-wise vector sum: arrays summed in one aggregate took 0.96 s against 12.23 s for
  explode-and-regroup, 12.76× faster.
- Org roll-up: grain then one grouping-sets pass took 1.85 s against 5.00 s for one
  aggregation per level from trades, 2.70× faster.
- Shuffle partitions: 8 (one per core) took 0.92 s against 3.51 s with 200 static partitions,
  3.80× faster.
- Salting and AQE skew handling showed no measurable gain on the skewed aggregate: AQE's skew
  handling rewrites joins, and the partial aggregate before the shuffle already absorbs the skew.
<!-- benchmark:end -->

The benchmark used 100,000 trades rather than millions because of the free disk space on the
machine that generated it, not because of the engine. Local-mode timings do not transfer
directly to a cluster; [05 §7](docs/study/05-aggregation-engineering.md) says what does and
does not.

## Study pages

Suggested reading order:

1. [VaR and expected shortfall](docs/study/01-var-es.md): P&L vectors, the quantile
   convention, and why VaR does not add. Everything else builds on it.
2. [Aggregation engineering](docs/study/05-aggregation-engineering.md): the reporting grain,
   grouping sets, arrays, non-additivity in OLAP, auditability and the benchmark.
3. [Stress testing](docs/study/02-stress.md): additive within a scenario, structured across
   scenarios (worst loss).
4. [FRTB](docs/study/04-frtb.md): SA charges recomputed from netted sensitivities at each node,
   and the IMA liquidity-horizon cascade and IMCC.
5. [CCAR](docs/study/03-ccar.md): a nine-quarter capital path by legal entity, a different
   shape from the trading-book measures.

Then the [data dictionary](docs/data-dictionary.md) and `notebooks/replicate.ipynb`.

## Simplifications

Every simplification relative to the regulation is marked `SIMPLIFICATION:` in the code and
listed on a study page; `tests/test_docs.py` checks that every marker under `src/` is listed.
In summary (details and paragraph references on the pages):

- [VaR and ES](docs/study/01-var-es.md#4-what-this-project-simplifies-compared-with-a-real-bank):
  square-root-of-time scaling for multi-day horizons; a one-factor-per-class market model;
  a synthetic stressed window; scenario P&L linear in factor returns; the firm VaR check
  assumes unfiltered specs.
- [Stress testing](docs/study/02-stress.md#4-what-this-project-simplifies-compared-with-a-real-bank):
  sensitivity-based (delta, gamma, vega) P&L instead of full revaluation; hand-set,
  uncalibrated scenarios; instantaneous shocks.
- [CCAR](docs/study/03-ccar.md#4-what-this-project-simplifies-compared-with-a-real-bank):
  hand-set macro paths; three stylised loan portfolios; constant balance sheet; symmetric tax;
  no counterparty default, operational risk or AT1/T2 instruments; stress capital buffer for
  legal entities too, from one cycle.
- [FRTB](docs/study/04-frtb.md#6-what-this-project-simplifies-compared-with-a-real-bank):
  one curve per currency; equity spot only, buckets 1, 5, 8, 12; one vega and one curvature
  factor per bucket; sensitivities taken as given; curvature up equals down; DRC without P&L
  adjustment, maturity scaling or seniority, two buckets; RRAO at the exotic rate only; IMA
  ES scaled by √10, reduced set drawn at random and untested, liquidity-horizon floors only,
  IMCC only (no SES, DRC, multiplier or 60-day average); the IMCC check assumes unfiltered
  specs.
- [Engineering](docs/study/05-aggregation-engineering.md#9-what-this-project-simplifies-compared-with-a-production-platform):
  local mode; a 100,000-trade benchmark; no incremental processing; no table format with time
  travel; one scenario count.

## Limits

- One machine, Spark local mode. Nothing here was run on a cluster, and no claim is made
  about cluster-scale performance.
- Synthetic, seeded data: 20 trades (`tiny`), 100,000 (`demo`, `bench`).
- Pinot is a single quick-start container without authentication, bound to loopback.
- The API has no authentication or authorisation, and there is no UI.
- Every run recomputes from raw data; there is no incremental processing.
- Java is not used; the project is Python only (PySpark).
- The regulatory measures are simplified as listed above and are for study, not for
  regulatory reporting.
