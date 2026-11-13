"""Benchmark variants: a naive and a tuned way to compute the same aggregation, per group.

Every variant of a group returns the same rows (checked before timing). Groups:

- ``vector_sum``: element-wise sum of P&L vectors to the reporting grain, by exploding to one
  row per scenario against the engine's ``array(sum(pnl[i]) ...)``.
- ``rollup``: the four org levels, each aggregated from trade vectors, against the engine's
  grain followed by one grouping-sets pass.
- ``skew_aggregate``: a hash aggregate by desk over the oversized desk: plain, with AQE skew
  settings (which only rewrite joins), salted two-stage.
- ``skew_join``: a sort-merge join of trade vectors to a desk-keyed table (broadcast off),
  then the desk sum: plain, AQE skew join, salted.
- ``join``: the hierarchy join as a sort-merge join against a broadcast hash join.
- ``shuffle_partitions``: the grain aggregation with 200 static partitions, with partitions
  sized to the data, and with 200 coalesced by AQE.

The naive variants are what a competent engineer could reasonably write, not strawmen.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from riskagg.engine.grain import GRAIN_EXTRA
from riskagg.engine.vectors import GRAIN_DIMS, ROLLED_UP, aggregate_vectors, node_level, sum_vectors
from riskagg.semantic.model import load_model

__all__ = ["AQE_SKEW_CONF", "SALT_BUCKETS", "Inputs", "Variant", "variants"]

KEYS = [*GRAIN_DIMS, *GRAIN_EXTRA]
ORG = list(load_model().hierarchies["org"])
ORG_SETS = [ORG[: i + 1] for i in range(len(ORG))]
SALT_BUCKETS = 16
NO_BROADCAST = {
    "spark.sql.autoBroadcastJoinThreshold": "-1",
    "spark.sql.adaptive.autoBroadcastJoinThreshold": "-1",
}
# AQE splits a skewed shuffle partition of a sort-merge join when it is larger than both the
# threshold and factor x the median partition. ``force`` lets it do so even though the
# aggregate after the join then needs a shuffle of its own.
AQE_SKEW_CONF = {
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.forceOptimizeSkewedJoin": "true",
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "2",
    "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "64MB",
}


@dataclass(frozen=True)
class Inputs:
    """Trade-level P&L vectors, the org hierarchy and the vector length."""

    pnl_vectors: DataFrame
    hierarchy: DataFrame
    n_scenarios: int


Build = Callable[[SparkSession, Inputs], DataFrame]


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    build: Build
    conf: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @contextmanager
    def with_conf(self, spark: SparkSession) -> Iterator[None]:
        """Set ``conf`` for the block, then restore exactly: reset or unset each key."""
        explicit = dict(spark.conf.getAll)
        try:
            for k, v in self.conf.items():
                spark.conf.set(k, v)
            yield
        finally:
            for k in self.conf:
                if k in explicit:
                    spark.conf.set(k, explicit[k])
                else:
                    spark.conf.unset(k)


# --- shared pieces --------------------------------------------------------------------------


def _books(inputs: Inputs) -> DataFrame:
    hier_cols = [c for c in GRAIN_DIMS if c not in inputs.pnl_vectors.columns]
    return inputs.hierarchy.select("book_id", *hier_cols)


def _with_org(inputs: Inputs, broadcast: bool = True) -> DataFrame:
    books = _books(inputs)
    return inputs.pnl_vectors.join(F.broadcast(books) if broadcast else books, "book_id")


def _vec(inputs: Inputs, col: str = "pnl") -> Column:
    return sum_vectors(col, inputs.n_scenarios).alias("pnl")


def _grain(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """The engine's grain aggregation (``build_reporting_grain`` without its input checks)."""
    del spark
    return _with_org(inputs).groupBy(*KEYS).agg(_vec(inputs))


def _salt() -> Column:
    return F.pmod(F.xxhash64("trade_id"), F.lit(SALT_BUCKETS)).alias("salt")


# --- vector_sum -----------------------------------------------------------------------------


def explode_regroup(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """One row per scenario, summed by key and index, then collected back into an array."""
    del spark
    exploded = _with_org(inputs).select(*KEYS, F.posexplode("pnl").alias("i", "v"))
    per_index = exploded.groupBy(*KEYS, "i").agg(F.sum("v").alias("v"))
    collected = per_index.groupBy(*KEYS).agg(
        F.sort_array(F.collect_list(F.struct("i", "v"))).alias("s")
    )
    return collected.select(*KEYS, F.col("s.v").alias("pnl"))


# --- rollup ---------------------------------------------------------------------------------


def per_level_from_trades(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """One aggregation of trade vectors per org level, unioned."""
    del spark
    frames = []
    for gs in ORG_SETS:
        agg = _with_org(inputs).groupBy(*gs, "window").agg(_vec(inputs))
        dims = [F.col(d) if d in gs else F.lit(ROLLED_UP).alias(d) for d in GRAIN_DIMS]
        frames.append(agg.select(*dims, F.lit(node_level(gs)).alias("node_level"), "window", "pnl"))
    out = frames[0]
    for df in frames[1:]:
        out = out.unionByName(df)
    return out


def grain_then_grouping_sets(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """The engine: trades to the reporting grain once, then every level in one grouping-sets pass."""
    return aggregate_vectors(_grain(spark, inputs), ORG_SETS, inputs.n_scenarios)


# --- skew -----------------------------------------------------------------------------------


def skew_agg_plain(spark: SparkSession, inputs: Inputs) -> DataFrame:
    del spark
    return _with_org(inputs).groupBy("desk_id").agg(_vec(inputs))


def skew_agg_salted(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """Two stages: by (desk, salt), then by desk."""
    del spark
    first = _with_org(inputs).groupBy("desk_id", _salt()).agg(_vec(inputs))
    return first.groupBy("desk_id").agg(_vec(inputs))


def _desks(inputs: Inputs) -> DataFrame:
    """A desk-keyed table (one row per desk)."""
    return inputs.hierarchy.select("desk_id", "desk_name").distinct()


def skew_join_plain(spark: SparkSession, inputs: Inputs) -> DataFrame:
    del spark
    joined = _with_org(inputs).join(_desks(inputs), "desk_id")
    return joined.groupBy("desk_id", "desk_name").agg(_vec(inputs))


def skew_join_salted(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """Salt the big side, replicate the desk table once per salt, then sum in two stages."""
    big = _with_org(inputs).withColumn("salt", _salt())
    desks = _desks(inputs).crossJoin(spark.range(SALT_BUCKETS).withColumnRenamed("id", "salt"))
    joined = big.join(desks, ["desk_id", "salt"])
    first = joined.groupBy("desk_id", "desk_name", "salt").agg(_vec(inputs))
    return first.groupBy("desk_id", "desk_name").agg(_vec(inputs))


# --- join -----------------------------------------------------------------------------------


def sort_merge(spark: SparkSession, inputs: Inputs) -> DataFrame:
    """The grain with the hierarchy join left to the planner, broadcasting switched off."""
    del spark
    return _with_org(inputs, broadcast=False).groupBy(*KEYS).agg(_vec(inputs))


def variants(spark: SparkSession) -> list[Variant]:
    """Every variant, grouped; ``sized_to_data`` takes one shuffle partition per core."""
    cores = str(spark.sparkContext.defaultParallelism)
    static = {"spark.sql.adaptive.coalescePartitions.enabled": "false"}
    return [
        Variant("explode_regroup", "vector_sum", explode_regroup,
                description="posexplode, sum by (key, index), collect_list back into an array"),
        Variant("array_index_sum", "vector_sum", _grain,
                description="engine: array(sum(pnl[0]), ..., sum(pnl[n-1])) in one aggregate"),
        Variant("per_level_from_trades", "rollup", per_level_from_trades,
                description="one aggregation of trade vectors per org level, unioned"),
        Variant("grain_then_grouping_sets", "rollup", grain_then_grouping_sets,
                description="engine: trades to the grain once, then one grouping-sets pass"),
        Variant("skew_agg_plain", "skew_aggregate", skew_agg_plain,
                description="group by desk; session defaults"),
        Variant("skew_agg_aqe_skew_join", "skew_aggregate", skew_agg_plain, AQE_SKEW_CONF,
                description="the same query with the AQE skew-join settings"),
        Variant("skew_agg_salted", "skew_aggregate", skew_agg_salted,
                description=f"group by (desk, salt of {SALT_BUCKETS}), then by desk"),
        Variant("skew_join_plain", "skew_join", skew_join_plain, NO_BROADCAST,
                description="sort-merge join to the desk table, then group by desk"),
        Variant("skew_join_aqe_skew_join", "skew_join", skew_join_plain,
                {**NO_BROADCAST, **AQE_SKEW_CONF},
                description="the same query with AQE skew-join splitting forced on"),
        Variant("skew_join_salted", "skew_join", skew_join_salted, NO_BROADCAST,
                description=f"join on (desk, salt of {SALT_BUCKETS}), two-stage sum"),
        Variant("sort_merge", "join", sort_merge, NO_BROADCAST,
                description="hierarchy join as a sort-merge join (broadcast disabled)"),
        Variant("broadcast", "join", _grain,
                description="engine: hierarchy broadcast to every task"),
        Variant("static_200", "shuffle_partitions", _grain,
                {**static, "spark.sql.shuffle.partitions": "200"},
                description="200 shuffle partitions, AQE coalescing off"),
        Variant("sized_to_data", "shuffle_partitions", _grain,
                {**static, "spark.sql.shuffle.partitions": cores},
                description=f"{cores} shuffle partitions (one per core), AQE coalescing off"),
        Variant("aqe_coalesced_200", "shuffle_partitions", _grain,
                {"spark.sql.adaptive.coalescePartitions.enabled": "true",
                 "spark.sql.shuffle.partitions": "200"},
                description="200 shuffle partitions, AQE coalescing on"),
    ]  # fmt: skip
