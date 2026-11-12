# Risk Aggregation Platform: Design

Date: 2026-11-12
Status: awaiting approval

## 1. Intent

A portfolio project demonstrating risk aggregation engineering in
capital markets. It has two jobs:

1. **Show** an interviewer that the author can design a metadata-driven,
   auditable Spark aggregation platform that produces reportable risk metrics
   and serves them through a semantic layer and Apache Pinot.
2. **Teach** the author the risk domain. The author is strong in Spark, has a
   partial grasp of VaR, and is new to stress testing, CCAR and FRTB. Anything
   in the repository is something an interviewer may ask about, so the author
   must be able to explain every number the project produces.

Decided by the author: the project covers VaR, stress testing, CCAR and FRTB;
it teaches as well as demonstrates; Spark is the area of real depth; a real
Pinot instance runs in Docker and the pipeline loads into it.

Everything else in this document is the designer's call unless marked
otherwise, and the author can overrule any of it.

## 2. Goals, non-goals, anti-goals

**Goals**

- One command runs the whole pipeline on a laptop: generate data, aggregate,
  write curated datasets, load Pinot, answer queries.
- A new aggregation is added by editing YAML, with no new Python.
- Each of the four risk areas is implemented with its real aggregation
  structure and is explained in a study guide.
- The Spark work shows measurable gains: a benchmark compares a naive
  implementation with the tuned one and explains the difference from the
  physical plans.
- Every reported number is traceable to a run, a configuration and its inputs.
- Small-scale results are checkable by hand and are checked in tests against
  an independent NumPy reference.

**Non-goals**

- Real market data, pricing models or revaluation. Inputs are synthetic
  sensitivities and P&L vectors.
- Regulatory-grade completeness. FRTB covers a subset of risk classes; CCAR is
  a stylised projection.
- A user interface. The serving end is an API, a Python client and a notebook.
- Running on a cluster. Local mode only, written so that nothing would need to
  change to run on one.

**Anti-goals** (failures even if the code works)

- Maths presented as production-accurate when it is simplified. Every
  simplification is labelled where it occurs, in code and in the study guide.
- Code the author cannot explain. Prefer the plain, readable formulation over
  a clever one, and name things with the regulatory vocabulary.
- A demo that fails without Docker. Pinot is the last stage; everything before
  it runs and is tested without it.
- Summing a non-additive measure anywhere, including in Pinot queries.

## 3. Stack

- Python 3.13, PySpark 4.x in local mode, Java 17 (already installed).
- `uv` for the environment and dependencies, `pytest` for tests, `ruff` for
  linting.
- NumPy and pandas for reference calculations and the notebook.
- Apache Pinot in Docker Compose (single-container quick start is enough).
- FastAPI for the HTTP API.
- Package name `riskagg`, with a `src/` layout and a `riskagg` command.

## 4. Domain model and synthetic data

**Organisation hierarchy:** firm → business line → desk → book. Trades belong
to a book. Other dimensions on a trade: legal entity, region, asset class,
counterparty, currency.

**Generated datasets** (seeded, deterministic, Parquet):

| Dataset | Grain | Used by |
|---|---|---|
| `trades` | one row per trade | all |
| `hierarchy` | one row per book with its ancestors | all |
| `pnl_vectors` | trade × risk class × liquidity horizon × window (current, stressed), array of one-day scenario P&L, with a flag for whether the underlying factors are in the reduced factor set | VaR, ES, FRTB IMA |
| `sensitivities` | trade × risk factor: delta, gamma, vega, and FRTB curvature up/down, with FRTB risk class and bucket | stress, FRTB SA |
| `stress_scenarios` | scenario × risk factor shock | stress, CCAR market shock |
| `macro_scenarios` | scenario (baseline, severely adverse) × quarter × macro variable | CCAR |
| `banking_book` | loan segment: exposure, PD, LGD, revenue and expense drivers | CCAR |
| `default_risk_positions` | trade-level jump-to-default inputs: issuer, rating, seniority, long/short | FRTB default risk charge |

**Scale profiles**

- `tiny`: about 20 trades, 3 desks, 20 scenarios. Checkable by hand; used in
  tests and study-guide worked examples. Because 99% of 20 scenarios is just
  the worst loss, the worked examples use 90% VaR and 80% Expected Shortfall
  to show the mechanics, and say so. Confidence levels are parameters; the
  headline 99% and 97.5% are tested on 250 scenarios against NumPy.
- `demo`: about 100k trades, 250 current-window scenarios plus 250 stressed.
  Runs the full pipeline in a few minutes.
- `bench`: millions of trades, with one deliberately oversized desk to create
  key skew. Used only by the benchmark.

P&L vectors are generated from a small factor model (correlated risk-factor
returns times trade exposures) so that diversification between desks is real
and visible, not noise.

**Vector shape.** Each risk factor has one risk class, one liquidity horizon
and a reduced-set flag, so a trade's P&L splits into one vector per risk class
× liquidity horizon × flag. Summing all of a trade's current-window vectors
gives its total P&L vector, which is what VaR uses. Both windows are generated
for all factors: stressed VaR uses the whole stressed window, and FRTB IMA
uses the reduced-set part of it. Vectors are one-day; the ten-day base horizon
for IMA comes from square-root-of-time scaling, labelled as a simplification
(a bank would use overlapping ten-day returns).

**Reporting grain.** The finest grain the engine persists is book × asset
class × currency × risk class × liquidity horizon. The generator guarantees
each book has exactly one legal entity and one region, so those come from the
hierarchy. Counterparty is not part of the grain: it is available for additive
measures and the default risk charge, which read trade-level data, and a
request for VaR by counterparty is refused. This keeps the persisted result to
a few thousand rows.

**As-of dates and re-runs.** The demo generates one as-of date; the generator
takes a date so more can be added. Each run writes under its own as-of date
and run identifier and never overwrites an earlier run. The run manifest marks
one run per as-of date as official. Only official runs are loaded to Pinot,
replacing that date's segments, and the query builder always filters on as-of
date, so sums can never span runs or dates.

## 5. Risk measures

Each measure is tagged with its **aggregation class**, which drives how the
engine and the semantic layer treat it:

- **additive**: sum across any dimension (stress P&L, notional, sensitivities).
- **vector**: sum the P&L vectors element by element, then apply a function to
  the summed vector (VaR, Expected Shortfall). Never summed after the fact.
- **structured**: a regulatory formula with its own nesting (FRTB SA, CCAR
  capital path). Computed only at the levels the rules define.

### 5.1 VaR and Expected Shortfall (deepest treatment)

- Historical-simulation VaR at 99%, one-day, with ten-day by square-root-of-
  time scaling (labelled as the simplification it is).
- Expected Shortfall at 97.5%.
- Stressed VaR from the stressed window.
- At every hierarchy node: standalone value, sum of children's standalone
  values, and the diversification benefit between them.
- Incremental contribution of each child to the parent's VaR (parent VaR with
  and without the child), to show why contributions are not standalone VaRs.
- Quantile convention fixed in one place and documented, because that is a
  common interview question and a common source of reconciliation breaks. No
  interpolation: with N scenarios, VaR is the loss ranked ceil(N × (1 − level))
  from the worst, so the third-worst of 250 at 99%. Expected Shortfall is the
  mean of the worst N × (1 − level) losses, with a fractional weight on the
  last one when that is not a whole number, which keeps it subadditive.

### 5.2 Stress testing

- A scenario library of named shocks: for example a 2008-style crisis, rates
  up 200 basis points, equities down 30%, a credit-spread widening, a currency
  crisis. Each is a set of shocks per risk factor.
- Stress P&L per trade from sensitivities: delta × shock + ½ × gamma × shock²
  + vega × volatility shock. Labelled as sensitivity-based, not full
  revaluation. The FRTB curvature up/down figures are a separate input used
  only by FRTB SA.
- One scenario in the library is flagged as the global market shock that CCAR
  reuses.
- Aggregated by scenario and any dimension (additive within a scenario).
- Per node: the worst scenario and its loss. "Worst-of" is not additive across
  nodes, and the study guide says so.

### 5.3 CCAR (stylised)

- Two macro scenarios, baseline and severely adverse, over nine quarters.
- Per quarter: pre-provision net revenue from simple macro-linked drivers;
  credit losses as exposure × PD × LGD with PD stressed by the macro path;
  trading losses from the global market shock, taken from the stress engine in
  5.2 and recognised in the first quarter; risk-weighted assets; planned
  capital distributions. The largest-counterparty-default component of the
  real exercise is explained in the study guide but not built.
- Rolled into a CET1 capital and CET1 ratio path per scenario, the minimum
  ratio over the horizon, and a stylised stress capital buffer (start-to-
  trough decline plus four quarters of dividends, floored at 2.5%).
- Losses and revenue are reported by business line, legal entity and firm.
  Capital, risk-weighted assets and the CET1 ratio exist only for legal
  entities and the firm; a request for a business line's capital ratio is
  refused, since capital is not allocated to business lines here.
- The point being shown: CCAR aggregation is across time and across risk
  types (credit, market, revenue) into capital, which is a different shape
  from a cube roll-up.

### 5.4 FRTB

**Standardised Approach**

- Sensitivities-based method for three risk classes: general interest rate
  risk, equity and foreign exchange. Delta, vega and curvature.
- The real nesting: weighted sensitivities → aggregation within a bucket with
  prescribed correlations → aggregation across buckets → three correlation
  scenarios (low, medium, high) → the largest total is the charge.
- Risk weights and correlations live in a reference-data file, taken from the
  Basel text (MAR21) where practical and marked where simplified.
- Default risk charge for non-securitisations, simplified: jump-to-default by
  issuer, netting long and short, risk weights by rating, hedge-benefit ratio.
- Residual risk add-on as a percentage of notional for flagged trades.
- Total: sensitivities-based charge + default risk charge + residual add-on,
  per desk and for the firm.

**Internal Models Approach**

- Expected Shortfall at 97.5% with the liquidity-horizon cascade (10, 20, 40,
  60, 120 days).
- Scaling to the stressed period using the reduced-factor-set ratio.
- Capital charge as the average of the unconstrained figure and the sum of
  per-risk-class figures (rho = 0.5).
- Explained in the study guide but not built: non-modellable risk factors,
  P&L attribution tests, backtesting traffic lights, the desk-level
  eligibility decision.

The point being shown: FRTB SA cannot be produced by a generic cube. The
correlation-weighted nesting has to be expressed as its own plan, and it is
recomputed, never rolled up, at each level where it is reported.

## 6. Aggregation engine

**Specification.** An aggregation is a YAML entry naming: the source dataset,
the measure (which carries its aggregation class), the dimensions or hierarchy
levels to report at, filters, and parameters (confidence level, window,
horizon). The engine validates it against the semantic model and compiles it
to a Spark plan. Adding a new cut of an existing measure needs only YAML.
Adding a new measure *type* needs Python; that boundary is stated in the docs.

**Reducing the work.** The design ideas the engine must embody:

- Aggregate once to the finest reporting grain (book × the declared
  dimensions), persist it, and build every higher level from that small
  result instead of from trades.
- Compute only the groupings the specifications declare, as one grouping-sets
  pass, never the full cube.
- Keep P&L vectors as arrays and sum them element by element inside the
  aggregation, so the row count never multiplies by the scenario count.
- Share one scan and one shuffle among all specifications that read the same
  source at the same grain.

**Spark techniques to demonstrate** (each with a before-and-after in the
benchmark and a paragraph on what the plan shows):

- Element-wise array summation without a Python UDF, against the naive
  explode-and-regroup form.
- Skew handling for the oversized desk: adaptive execution settings and a
  salted two-stage aggregation, compared.
- Broadcast joins for the hierarchy and reference data.
- Partitioning and file layout of the curated outputs by as-of date, and
  shuffle-partition sizing.
- Caching the book-grain result and its storage level.

The exact mechanisms are the builder's to choose, so long as each has a
measured comparison.

**Benchmark.** `riskagg bench` runs naive and tuned variants on the `bench`
profile, records wall time, shuffle read and write sizes, and peak task
memory where available, and writes a Markdown report with the key parts of
each physical plan.

**Auditability.** Every output row carries a run identifier, as-of date and a
hash of the specification and reference data that produced it. A run manifest
records input datasets with row counts and checksums, Spark configuration and
code version. Reconciliation checks run after aggregation: additive measures
at the firm equal the sum of trades; the firm P&L vector equals the sum of
book vectors; no vector measure was produced by summing.

## 7. Semantic layer and serving

**Semantic model.** One YAML file declares dimensions, hierarchies, measures
(with business name, description, unit, aggregation class and the dimensions
each is valid across) and which physical table serves each. It is the single
source for: specification validation, Pinot schema and table configuration
generation, the API's catalogue endpoint, and a generated data dictionary
written for people and for AI query tools.

**Curated datasets** (Parquet, partitioned by as-of date):

- `risk_metrics`: long format, one row per node × measure × scenario where
  relevant. Precomputed at declared nodes. Non-additive measures are served
  only from here, at the nodes where they were computed.
- `pnl_scenarios`: reporting grain (section 4) × window × scenario index ×
  P&L. Additive within a scenario, so VaR for any slice of the reporting-grain
  dimensions can be computed at query time by summing per scenario and then
  taking the quantile. Slices outside those dimensions are refused.
- `stress_results`, `frtb_sa_detail`, `ccar_projection`: detail tables behind
  the headline numbers, for drill-down.

**Pinot.** Docker Compose brings up Pinot. `riskagg load-pinot` creates
schemas and tables generated from the semantic model and batch-ingests the
curated Parquet. Indexing choices (for example a star-tree index on the
additive stress table, sorted or inverted indexes on node and as-of date) are
made deliberately and explained.

**Query path.** A small query builder takes (measure, group-by, filters),
looks the measure up in the semantic model and produces Pinot SQL:

- additive → a plain `SUM`;
- vector → sum by scenario over `pnl_scenarios`, then quantile, or a lookup in
  `risk_metrics` when the slice is a precomputed node;
- structured → lookup only; a request for a slice that was not computed is
  refused with an explanation, not approximated.

**API and client.** FastAPI endpoints for the catalogue, a metric query and a
drill-down, plus a thin Python client. The same queries fall back to the
Parquet files through Spark or pandas when Pinot is not running.

**Notebook.** One notebook that takes a headline number (firm VaR), pulls it
through the client, then rebuilds it from the raw P&L vectors in NumPy and
shows they match. It repeats the exercise briefly for one FRTB SA bucket.

## 8. Study guide

`docs/study/`, one file per topic: VaR and Expected Shortfall, stress
testing, CCAR, FRTB, and aggregation engineering. Each has:

- what the measure is and why regulators ask for it, in plain language;
- how it aggregates and why, with a worked example on the `tiny` profile whose
  numbers match a test;
- what this project simplifies compared with a real bank;
- eight to twelve likely interview questions with short answers and a pointer
  to the code that backs each one.

The README's capability table maps each capability this project demonstrates
to the part of the project that addresses it and gives the commands to run the demo.

## 9. Testing

- Unit tests of every measure on the `tiny` profile against an independent
  NumPy implementation.
- Hand-computed golden values for the worked examples in the study guide.
- Property checks: Expected Shortfall of a parent never exceeds the sum of its
  children's; additive measures reconcile; results are independent of
  partition count and row order.
- Specification tests: valid YAML compiles; an invalid one (unknown dimension,
  summing a vector measure) is rejected with a clear message.
- Query-builder tests on the generated SQL, with no Pinot needed.
- Pinot integration tests, skipped automatically when Pinot is not reachable.

## 10. Commands

`riskagg generate --profile <p>`, `riskagg run`, `riskagg bench`,
`riskagg load-pinot`, `riskagg serve`, and a `make demo` that chains them.

## 11. Left to the builder

- Module layout inside `riskagg`, and the exact YAML schema.
- The factor model behind the synthetic data and the exact row counts.
- Which Spark mechanisms implement each technique in section 6.
- The Pinot ingestion method and index details.
- FRTB reference values where the Basel tables are simplified, provided each
  simplification is marked.
- Build order. Suggested: a thin slice first, taking VaR alone from generated
  data through the engine, curated output, Pinot and the API, so there is
  something demonstrable early. Then widen: stress → FRTB → CCAR → benchmark →
  notebook. Each study-guide page is written alongside its measure.

## 12. Capability table

The README's capability table lists the capabilities this project demonstrates, in its own
words, each mapped to code and study pages.
