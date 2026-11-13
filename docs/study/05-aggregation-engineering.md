# Aggregation engineering: how the Spark engine is built, and what it measurably buys

This page covers the engineering of the aggregation engine rather than a risk measure: the
reporting grain, declared grouping sets against a full cube, why P&L vectors stay as arrays,
how non-additive measures constrain the OLAP layer, how every number is made auditable, and
what a benchmark of naive against tuned variants measured on one machine. Every number in
section 6 is copied from the generated report `docs/benchmarks/report.md`, and
`tests/test_bench.py` checks that each of them appears there. Statements labelled
**reasoning** are not measurements.

## 1. The problem in Spark terms

The input is one P&L vector per trade and (risk class, liquidity horizon, reduced-set flag,
window): an `array<double>` of 250 scenario P&Ls (`pnl_vectors`, see `01-var-es.md`). The
output is VaR and ES at every node the specifications in `config/aggregations.yaml` declare
(firm, business line, desk, book, desk × asset class, ...).

Three facts shape everything below:

1. **VaR and ES are not additive.** A parent's VaR must be computed from the parent's own P&L
   vector, the element-wise sum of its children's vectors. A node cannot be rolled up from
   its children's VaRs, in Spark or anywhere else.
2. **P&L in one scenario is additive.** So vectors can be summed in any order and any
   grouping, which is what makes a shared intermediate result possible.
3. **The cost of a Spark aggregation is mostly what crosses a shuffle**: how many rows and
   how many bytes go through an `Exchange`, and whether a map-side partial aggregate shrinks
   them first. The work inside a task (adding doubles) is cheap by comparison.

## 2. The reporting grain

**Idea.** Reduce the trades once to the finest level any report needs, and build every node
from that small result instead of from trades. The grain here is book × legal entity ×
region × asset class × currency × risk class (the semantic model's `grain`), plus the
`liquidity_horizon`, `in_reduced_set` and `window` columns that FRTB IMA needs.

**Code.** `build_reporting_grain` in `src/riskagg/engine/grain.py`:

- validates the input in one aggregate action first (every array the right length, no null
  arrays or elements, no book missing from the hierarchy) and fails naming what it found,
  instead of producing a wrong number;
- joins the hierarchy with `F.broadcast` (a few hundred books: every task gets a copy, the
  trade side is never shuffled for the join);
- groups by the grain keys with the element-wise vector sum (section 4);
- returns a lazy frame; `riskagg run` caches it (`.cache()`) because VaR, ES, IMA,
  diversification and the curated `pnl_scenarios` all read it.

**Why it pays.** Every later aggregation reads grain rows, not trade rows. The number of grain
rows depends on the number of books and the dimension combinations that exist, not on the
number of trades, so it stops growing once each book holds a few trades of each kind.

**When it does not pay (reasoning).** If a report needs a dimension finer than the grain
(counterparty, trade), the grain cannot answer it; the query layer refuses such a slice
(`SliceNotAvailable` in `src/riskagg/semantic/query.py`) rather than summing VaRs. Adding a
dimension to the grain multiplies its size by roughly that dimension's distinct values per
book, so the grain is a deliberate choice, declared once in `config/semantic_model.yaml`.

## 3. Declared grouping sets against a full cube

**Grouping sets.** `GROUP BY GROUPING SETS ((firm), (firm, bl), (firm, bl, desk), ...)`
computes several groupings in one aggregation. Spark plans it as an `Expand` (each input row
is emitted once per grouping set, with the columns not in that set nulled and a
`spark_grouping_id` added) followed by one `HashAggregate` on the expanded keys, so the whole
thing is one scan and one shuffle.

**Code.** `aggregate_vectors` in `src/riskagg/engine/vectors.py` takes the union of the
grouping sets of all vector specifications that share the same filters (`node_vectors` in
`engine/runner.py`), removes duplicates (`unique_sets`: the same dimensions in another order
are the same node), runs one `groupingSets(...)` aggregation over the grain, and decodes
`grouping_id()` bit by bit into the literal `*` for a rolled-up dimension and a `node_level`
label. `tests/test_engine.py::test_one_pass_for_shared_specs` asserts that the plan of the
whole run for the shipped vector specs has exactly one `Expand`.

**Why not `cube()`.** A cube over the nine grain dimensions is 2⁹ = 512 grouping sets: the
`Expand` would emit every grain row 512 times, almost all of them for nodes nobody asked for.
Rolling up a cube is attractive for additive measures, because any node can then be read
off; for VaR it buys nothing, because every node needs its own vector anyway and a vector
per node per scenario is the expensive part. The engine computes the nodes declared in YAML
and serves other combinations of grain dimensions from per-scenario P&L (section 5).

**Cost model (reasoning).** The `Expand` multiplies grain rows by the number of declared
sets, before the partial aggregate. With five or six sets over a grain that is far smaller
than the trade table, that is cheap; with hundreds of sets, it would be the dominant cost,
and splitting the sets into a few passes, or computing coarse levels from a cached finer
level, would become worth it.

## 4. Why vectors stay as arrays

**The engine's form.** `sum_vectors` in `engine/vectors.py` is
`array(sum(pnl[0]), sum(pnl[1]), ..., sum(pnl[249]))`: 250 ordinary `sum` aggregates over
array elements. In the physical plan this is a `HashAggregate` with
`Functions [250]: [partial_sum(pnl[0]), ...]` below the `Exchange` and the matching final
sums above it. Each aggregation buffer is 250 fixed-width doubles, so it fits Spark's
`UnsafeRow` hash map (the same one an ordinary `sum` uses), and the map-side partial aggregate
sends one row per group per task across the shuffle. The row count never multiplies by the
scenario count. Those two things are the win.

**Whole-stage code generation is off for this aggregate.** In the report's executed plans the
250-sum `HashAggregate`s carry no `*` (the whole-stage codegen marker), while the
`BroadcastHashJoin` below them does. The cause is `spark.sql.codegen.maxFields` (default 100 in
Spark 4.3.0): whole-stage codegen is not used for an operator whose schema has more fields than
that. A check on this Spark version with AQE off: a `groupBy` with 99 element sums is planned as
`*(1) HashAggregate`, with 100 or 250 sums as plain `HashAggregate`, and with 250 sums and
`spark.sql.codegen.maxFields=1000` as `*(1) HashAggregate` again. Raising the limit was not
benchmarked, so whether it would make the aggregate faster is not known here.

**The alternatives a competent engineer might reach for:**

- **Explode and regroup**: `posexplode(pnl)`, `sum` by (key, index), then
  `collect_list(struct(index, value))`, `sort_array`, and take the values. It is correct and
  uses only built-in functions. The plan has `Generate posexplode`, which multiplies rows by
  250 before the first aggregate, and the regrouping is an `ObjectHashAggregate`, because a
  `collect_list` buffer is not fixed-width. Section 6 measures it.
- **A Python UDF or pandas UDAF**: every row crosses to a Python worker and back through
  Arrow. The project forbids UDFs in the tuned path (they are allowed in data generation),
  and this one was not benchmarked.
- **A Scala `Aggregator`** with an `array<double>` buffer is the JVM option worth measuring for
  large vectors (reasoning, not measured): one buffer object instead of 250 columns. It
  needs a JAR on the cluster, which this pure-PySpark project does not ship, and its buffer
  is an object, so it would run as an `ObjectHashAggregate` rather than in the `UnsafeRow`
  map.

**Its costs (reasoning).** The plan, its description and the per-row work all grow linearly
with the scenario count, and above `spark.sql.codegen.maxFields` the aggregate runs without
whole-stage code generation. At several thousand scenarios, an `Aggregator` or a chunked
layout (several array columns of a few hundred elements) becomes worth evaluating; both
would need measuring.

## 5. Non-additivity in the OLAP layer

The serving layer (Pinot, or Parquet when Pinot is not up) must not undo the care above:

- `risk_metrics` holds VaR and ES only at the nodes the specifications computed. Summing
  stored VaRs over a dimension is the classic wrong answer, and nothing in the query layer
  does it.
- `pnl_scenarios` holds the reporting-grain vectors exploded to one row per scenario
  (`write_pnl_scenarios` in `src/riskagg/serving/curate.py`). This is the one place the
  project explodes: Pinot has no element-wise array aggregation, and within one scenario P&L
  is additive. The explosion is of grain rows, not trade rows, so it is affordable. A query
  for VaR at any combination of grain dimensions sums P&L per scenario in SQL (`GROUP BY
  <dims>, scenario_idx`) and applies the quantile afterwards in Python with the shared
  convention functions (`build_query` and `finish` in `src/riskagg/semantic/query.py`).
- No star-tree index is built on a table that serves a non-additive measure (`_star_tree` in
  `src/riskagg/semantic/pinot_schema.py`): a star-tree pre-aggregates metric values by
  summing them.
- A truncated result would silently move a quantile, so partial Pinot responses are refused
  and `finish` checks that every group has every scenario exactly once.
- Structured measures (worst stress loss, FRTB charges, CCAR capital) are only looked up at
  the nodes where they were computed.

## 6. What the benchmark measured

`riskagg bench --profile bench` (`src/riskagg/bench/`) generates the `bench` profile once
(100,000 trades, of which 40% sit in one oversized desk; 186,285 P&L vectors of 250
scenarios, 356.5 MiB of Parquet), then for each group of variants: checks on a sample that
every variant returns the same rows, runs one untimed warm-up round, and times 3 rounds,
each running every variant once in a seeded random order. A run is a `noop` write (every row
computed, nothing written). Shuffle bytes, records, task counts and per-task peak execution
memory come from Spark's REST status API; the executed plan is the adaptive final plan. The
machine was an Apple M3 with 8 cores in local mode, Spark 4.3.0 on Java 17.0.19. All of this
is in the report's header, with the exact endpoints and fields.

Two of the variants run the identical query to the identical executed plan, so their
difference is pure noise: `skew_agg_plain` and `skew_agg_aqe_skew_join` came out at 0.68 and
0.62 s, and the engine's grain query measured three times under three names
(`array_index_sum`, `broadcast`, `sized_to_data`) at 0.96, 0.93 and 0.92 s. Differences of
that order are not findings, and the report says so wherever a difference is smaller than
the run-to-run spread.

### `vector_sum`: arrays against explode-and-regroup

**Compared.** Trade vectors summed to the reporting grain (the same broadcast join and keys),
once with the engine's `array(sum(pnl[0]), ...)` and once with `posexplode`, a `sum` by key
and index, and `collect_list` back into an array.

**Plan.** The exploded variant's final plan, bottom up:

```
+- * Generate posexplode
+- * HashAggregate
+- Exchange hashpartitioning(firm_id, ..., window, i, 8)
+- * HashAggregate
+- ObjectHashAggregate
+- Exchange hashpartitioning(firm_id, ..., window, 8)
+- ObjectHashAggregate
```

Two shuffles instead of one, the first keyed by scenario index as well, and the regrouping
as an `ObjectHashAggregate` (a `collect_list` buffer is not fixed-width). The engine's plan
is `HashAggregate`, `Exchange`, `HashAggregate`.

**Measured.** 12.23 s against 0.96 s, 12.76× faster for the engine. The exploded form wrote
12,640,682 shuffle records (288.2 MiB) against 50,037 (98.3 MiB), and its peak task
execution memory was 544.0 MiB against 48.5 MiB.

**At production scale (reasoning).** The gap is driven by rows, which scale with trades ×
scenarios, so it widens with data and with more scenarios. This is the largest and most
robust result of the benchmark.

### `rollup`: per-level aggregation from trades against the grain and one grouping-sets pass

**Compared.** The four org levels (firm, business line, desk, book), once as four
aggregations of trade vectors unioned, once as the engine does it: trades to the grain, then
one grouping-sets aggregation.

**Plan.** The per-level plan has `Union` over four branches, each with its own
`Scan parquet pnl_vectors`, `BroadcastHashJoin` and `HashAggregate` / `Exchange` /
`HashAggregate`. Three branches have their own `BroadcastExchange` of the hierarchy and the
fourth reuses one (`ReusedExchange`). The engine's
plan scans once, aggregates to the grain, then `Expand` feeds one aggregate keyed by
`spark_grouping_id`.

**Measured.** 5.00 s against 1.85 s, 2.70× faster for the engine. Per-level reads the trade
vectors 4 times and shuffles only 3.5 MiB (each branch partially aggregates to very few
rows); the engine reads them once and shuffles 105.2 MiB, because the grain has many more
rows than the org levels. The engine wins on scans, not on shuffle bytes. The grain was not
cached in this comparison, so the engine's variant paid for building it.

**At production scale (reasoning).** Every extra level or specification costs the per-level
approach another full scan of trade vectors; the engine's cost grows only with grain rows ×
grouping sets. In the real run the grain is also cached and shared with ES, IMA and the
curated outputs, which this comparison does not credit.

### `skew_aggregate`: key skew in a hash aggregate: a technique that gave no gain

**Compared.** Vectors summed by desk, with the oversized desk: plain, with the AQE skew-join
settings, and salted (by desk and one of 16 salt values, then by desk).

**Plan.** The plain plan is `HashAggregate` / `Exchange hashpartitioning(desk_id, 8)` /
`HashAggregate`. The AQE settings left it unchanged: AQE's skew handling rewrites joins, and
this query has none. Salting adds a second `Exchange` and two more `HashAggregate`s.

**Measured.** Plain 0.68 s, AQE settings 0.62 s, salted 1.01 s; both differences are within
the run-to-run spread (salted ranged up to 1.63 s). The plain plan shuffled 160 records
(0.3 MiB): the map-side partial aggregate had already reduced each task's rows to one row
per desk, so the oversized desk never reached a reducer as raw rows. Salting raised that to
2,718 records (5.3 MiB) and added a pass.

**At production scale (reasoning).** The same holds on a cluster for any aggregate that can
be partially combined (sums, counts, element-wise vector sums): skew in the grouping key
does not create a straggler. Salting an aggregate pays when the partial aggregate does not
shrink the hot key: `collect_list` and exact distinct counts do run a partial aggregate
below the `Exchange` (the `vector_sum` plan shows `ObjectHashAggregate` on both sides of it),
but their buffer grows with the input, so the hot key's data crosses the shuffle almost
undiminished. The same applies when partial aggregation is switched off.

### `skew_join`: key skew in a sort-merge join

**Compared.** Trade vectors joined to a desk-keyed table by sort-merge join (broadcast
disabled), then summed by desk: plain (session defaults), AQE skew join forced on, and
salted (16 salt values on the vector side, the desk table replicated once per salt, then a
two-stage sum).

**Plan.** Plain: `SortMergeJoin` with the `HashAggregate` directly on top and no `Exchange`
between them, because the join's `hashpartitioning(desk_id, 8)` already satisfies the
aggregate. With forcing, the final plan has `SortMergeJoin(skew=true)` over an
`AQEShuffleRead, coalesced and skewed`, and AQE inserted an
`Exchange hashpartitioning(desk_id, desk_name, 8)` above the join with a single
`HashAggregate` over it. That aggregate had been planned without a partial stage, so the
joined vectors themselves went through the new shuffle. Salted: a `CartesianProduct` of the
desks and the salts, a join on `(desk_id, salt)`, and two more aggregates.

**Measured.** Plain 1.31 s; AQE forced 1.72 s, with shuffle write 716.9 MiB
against 358.6 MiB; salted 1.60 s. Both time differences are within the run-to-run spread
(the forced variant ranged up to 2.89 s), so no time difference is claimed; the doubled
shuffle is the measured cost. The skew itself did not show as a straggler here: in the plain
run the most uneven stage had a longest task of 0.51 s against a median of 0.42 s.

**At production scale (reasoning).** On a cluster, the one reduce task that holds the
oversized desk would hold 40% of the joined vectors while hundreds of others hold a sliver,
and a split or a salt would pay. The forced AQE split shows the trap to check for in the
plan: splitting a join whose output partitioning a later operator relies on adds a shuffle,
and if the operator above has no partial stage, that shuffle carries the full rows. Better
options at scale are to aggregate before the join where the join key allows it, or to salt
deliberately.

### `join`: the hierarchy as sort-merge against broadcast

**Compared.** The grain aggregation with the hierarchy join as a sort-merge join (broadcast
disabled) and as the engine's broadcast hash join.

**Plan.** Sort-merge: `Exchange hashpartitioning(book_id, 8)` on both sides, `Sort` ×2,
`SortMergeJoin`, and one `HashAggregate` on top; the vectors cross the shuffle raw, before
any aggregation. Broadcast: `BroadcastExchange` of the hierarchy, `BroadcastHashJoin`, then
the partial `HashAggregate` before the only `Exchange`.

**Measured.** The sort-merge join shuffled 362.5 MiB (186,685 records, every trade vector);
the broadcast join shuffled 98.3 MiB (50,037 records). The times, 1.33 s against 0.93 s,
differ by 0.41 s, which is within the run-to-run spread of 0.94 s, so the report does not
claim a time difference.

**At production scale (reasoning).** Locally a shuffle is a disk write and a read, often
from the page cache. On a cluster those 362.5 MiB against 98.3 MiB are network transfer,
and the ratio grows with trades per book, so broadcasting a small dimension is the safe
default; the threshold to watch is the dimension's size in executor memory.

### `shuffle_partitions`: 200 against sized to the data

**Compared.** The grain aggregation with 200 shuffle partitions and AQE coalescing off, with
8 (one per core: the shuffle after partial aggregation is about 98.3 MiB), and with 200
partitions and AQE coalescing on.

**Plan.** The same operators; the difference is the `Exchange`'s partition count, and for
the coalesced variant an `AQEShuffleRead, coalesced` after it. The runs had 216, 24 and 25
tasks.

**Measured.** 3.51 s at 200 static partitions, 0.92 s sized to the data (3.80× faster) and
0.98 s with AQE coalescing (3.57× faster). AQE recovered almost all of the difference.

**At production scale (reasoning).** What was measured is the task count (216 against 24)
and the time; why each of the extra reduce tasks costs so much was not isolated. The
hypothesis is a fixed cost per task (scheduling, and setting up a 250-sum aggregate) paid
for very little data each. On a cluster the right number follows the shuffle size and the
cores: with AQE on, set the initial number generously and let coalescing merge small
partitions; without AQE, size it from the expected shuffle bytes.

## 7. What does not transfer from a local-mode run

The benchmark ran Spark in local mode on one laptop: one JVM is the driver and the only
executor, with eight task slots sharing one heap and one SSD. Read the numbers as evidence
about **plans and bytes**, which do transfer, more than about **seconds**, which do not:

- **Shuffle bytes and records transfer; shuffle cost does not.** Locally, a shuffle is a
  write to and a read from the same disk, often served from the page cache. On a cluster the
  same bytes cross the network, are fetched by remote executors, and can be lost with an
  executor. The sort-merge hierarchy join shuffled 362.5 MiB where the broadcast join
  shuffled 98.3 MiB; on a cluster that costs far more than the small local time difference
  suggests (reasoning).
- **Executor memory does not transfer.** The peak task execution memory column is per task,
  measured; whether a task spills on a cluster depends on the executor's memory and cores,
  which local mode does not have.
- **Skew transfers only in shape.** The oversized desk holds 40% of the trades, but with 40
  desks hashed into 8 partitions on 8 local cores, the hot partition is one of only a handful
  and every task reads from local disk. On a cluster with hundreds of partitions, one task
  holding 40% of the data would run for most of the stage while the others idle (reasoning).
- **Per-task costs may not transfer.** With 200 partitions the run had 216 tasks against 24
  and took several times longer; the cost per extra task was not isolated (hypothesis: a
  fixed setup cost per task, large relative to the little data each reduce task gets here).
- **The data is smaller than designed** (see the simplifications below). Ratios between variants that are bound
  by rows or bytes (the vector sum, the sort-merge join) should hold or widen with size;
  anything bound by fixed per-task or per-query costs shrinks relative to the work
  (reasoning).
- **One machine, one run of the benchmark.** Run-to-run spread is reported, and two variants
  that execute the identical plan are reported as a noise control; a difference smaller than
  either is not a finding.

## 8. Auditability

Every number a user sees can be traced to a run, a configuration and its inputs:

- **Every output row** carries `run_id` and `as_of_date` (the partition directories of every
  curated table, written with dynamic partition overwrite so a re-run never touches another
  run's partition), and `risk_metrics` rows carry `spec_hash`, a hash of the specifications
  and the reference data (`src/riskagg/serving/curate.py`, `spec_hash` in
  `src/riskagg/engine/spec.py`).
- **A run manifest** (`runs/<run_id>/manifest.json`, `write_manifest` in
  `src/riskagg/engine/audit.py`) records each input dataset's row count and an
  order-independent checksum, the Spark version and the session settings that affect results,
  and the git commit of the code.
- **Reconciliation reads what was written to disk**, never in-memory frames, and uses
  different code from the engine: the firm vector from raw trades equals the sum of the
  curated book vectors; firm VaR and ES equal the measure of the curated firm vector; no node
  appears twice; firm stress P&L equals the sum over raw trades; the FRTB SA total equals its
  components; the firm IMCC is rebuilt from raw vectors; CCAR is rebuilt in plain Python
  (`CHECKS` in `audit.py`).
- **A run becomes official only if every check passes** (`mark_official` in
  `src/riskagg/context.py`); the serving layer reads only official runs.

## 9. What this project simplifies compared with a production platform

- **Local mode on one machine** (section 7).
- **A smaller benchmark than designed.** The design called for millions of trades; the
  committed report uses the `bench` profile at 100,000 trades (it was 1,000,000), because the
  machine it was generated on had about 2 GB of free disk and the benchmark's peak disk use is
  several times the input (the input, the shuffle files of a run that moves whole vectors,
  and the previous run's files until Spark deletes them).
- **No incremental processing**: every run recomputes from raw. A production platform would
  aggregate only the books whose trades changed and reuse yesterday's grain for the rest
  (reasoning).
- **No table format with time travel** (Delta, Iceberg): runs are separated by partition
  directories and a manifest instead.
- **One scenario count** for every vector; a production platform has several (for example
  1-day and 10-day overlapping scenarios), each with its own grain table.

## 10. Likely interview questions

1. **How do you sum arrays element-wise in Spark without a UDF?** `array(sum(col[0]), ...,
   sum(col[n-1]))`: native aggregates over array elements, fixed-width buffers in the
   `UnsafeRow` hash map, and map-side partial aggregation. With 250 elements the aggregate
   exceeds `spark.sql.codegen.maxFields` (100), so it runs without whole-stage code
   generation. Code: `sum_vectors` in `src/riskagg/engine/vectors.py`; measured against
   explode-and-regroup in `vector_sum`.
2. **Why not explode to one row per scenario?** The row count is multiplied by the scenario
   count before the first shuffle, and putting the array back together needs a
   `collect_list`, which runs as an `ObjectHashAggregate` and a sort. Code:
   `explode_regroup` in `src/riskagg/bench/variants.py`.
3. **What is a reporting grain and why build one?** The finest level any report needs, built
   once from trades and cached; every node is built from it. Code: `build_reporting_grain` in
   `src/riskagg/engine/grain.py`. The `rollup` group measured the engine's whole approach
   (one scan, the grain, one grouping-sets pass) against four aggregations from trades; it
   does not separate the grain's contribution from the single scan's.
4. **GROUPING SETS, ROLLUP or CUBE?** Grouping sets compute exactly the declared groupings in
   one `Expand` plus one aggregate; ROLLUP is the special case of a hierarchy's prefixes;
   CUBE is every subset (512 for nine dimensions), almost all unused, and useless for VaR
   because each node needs its own vector anyway. Code: `aggregate_vectors`; guarded by
   `test_one_pass_for_shared_specs`.
5. **How do you tell rolled-up rows from real nulls?** `grouping_id()`: one bit per grouping
   column; the engine decodes it into the literal `*` and a `node_level` label, so a null in
   the data can never be mistaken for "all". Code: `aggregate_vectors`.
6. **When does key skew hurt, and when does it not?** It hurts where whole rows of the hot key
   reach one task: a sort-merge join, a window, an aggregate whose partial buffer grows with
   its input (`collect_list`, exact distinct). It does not hurt a hash aggregate with a
   fixed-size partial buffer, because each map task sends one row per key. Measured in
   `skew_aggregate` and `skew_join`.
7. **What does AQE's skew-join handling do, and what can go wrong?** It splits an oversized
   shuffle partition of a sort-merge join into several tasks and replicates the matching side.
   By default it will not do so if the split would need an extra shuffle; forcing it
   (`spark.sql.adaptive.forceOptimizeSkewedJoin`) here added a shuffle of whole vectors after
   the join, because the aggregate above it had no partial stage. Measured in `skew_join`.
8. **How does salting work and what does it cost?** Add a random salt to the hot side's key,
   replicate the other side once per salt, aggregate by (key, salt), then by key. It spreads
   the hot key over many tasks at the price of a replicated build side, more shuffle records
   and a second aggregation. Code: `skew_join_salted` and `skew_agg_salted`.
9. **Broadcast or sort-merge for a dimension join?** Broadcast whenever the dimension fits in
   memory: the fact side is never shuffled for the join, so it can be partially aggregated
   before its only shuffle. Code: the `F.broadcast` in `build_reporting_grain`; measured in
   `join`.
10. **How many shuffle partitions?** Enough that each holds a sensible amount of data and the
    cores stay busy; the default 200 is far too many for a small post-aggregation shuffle and
    too few for a terabyte one. With AQE, set a generous number and let coalescing merge
    them, knowing what coalescing aims at: with the default
    `spark.sql.adaptive.coalescePartitions.parallelismFirst=true`, Spark ignores the advisory
    size (`spark.sql.adaptive.advisoryPartitionSizeInBytes`, 64 MB) and computes a target from
    the default parallelism, usually smaller, to keep every core busy (bounded below by
    `coalescePartitions.minPartitionSize`, 1 MB). The configuration's own description
    recommends setting it to `false` on a busy cluster so the advisory size is respected and
    there are not many small tasks. Measured in `shuffle_partitions`.
11. **How do you serve VaR from an OLAP store when VaR is not additive?** Store it only at
    computed nodes; for other slices store per-scenario P&L, which is additive, sum it per
    scenario in SQL and apply the quantile afterwards; never build a star-tree on it. Code:
    `src/riskagg/semantic/query.py`, `_star_tree` in `src/riskagg/semantic/pinot_schema.py`.
12. **How do you make a risk number auditable?** Run id, as-of date and specification hash on
    every row; a manifest with input checksums, Spark settings and the code version;
    reconciliation from disk with independent code; a run is official only if every check
    passes. Code: `src/riskagg/engine/audit.py`.
13. **How would you benchmark a Spark change honestly?** Compare plans and bytes, not only
    seconds; check the variants return the same rows first; warm up; repeat in a varied
    order; report the spread and a same-plan control; say when a difference is noise and
    which effects a single machine cannot show. Code: `src/riskagg/bench/runner.py`.
