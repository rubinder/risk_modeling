"""Stress testing: scenario shocks, sensitivity-based stress P&L, and the worst scenario.

Stress P&L is additive within a scenario: a node's P&L under one scenario is the sum of its
trades' P&L under that scenario, so any slice can be summed in SQL. "Worst scenario" is not
additive: the worst scenario of a parent is generally not the sum of its children's worst
losses (each child may be hit hardest by a different scenario), so ``worst_stress_loss`` is a
structured measure that is computed at declared nodes and only looked up.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from riskagg.engine.inputs import RunInputs
from riskagg.engine.spec import AggregationSpec
from riskagg.engine.vectors import GRAIN_DIMS, METRIC_COLUMNS, ROLLED_UP, node_level, unique_sets
from riskagg.paths import stress_scenarios_path
from riskagg.semantic.model import SemanticModel

RISK_CLASSES = ("GIRR", "EQ", "FX")
FACTOR_TYPES = ("DELTA", "VEGA")


@dataclass(frozen=True)
class ShockRule:
    risk_class: str
    factor_type: str
    shock: float
    bucket: str | None = None
    currency: str | None = None

    @property
    def specificity(self) -> int:
        return (self.bucket is not None) + (self.currency is not None)

    def matches(self, risk_class: str, factor_type: str, bucket: str | None, currency: str | None):
        return (
            self.risk_class == risk_class
            and self.factor_type == factor_type
            and (self.bucket is None or self.bucket == bucket)
            and (self.currency is None or self.currency == currency)
        )


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    global_market_shock: bool
    shocks: tuple[ShockRule, ...]

    def shock_for(
        self, risk_class: str, factor_type: str, bucket: str | None, currency: str | None
    ) -> float:
        """The shock of the matching rule with the most optional fields; 0.0 if none matches.

        Two equally specific rules that match one factor are ambiguous and raise.
        """
        hits = [r for r in self.shocks if r.matches(risk_class, factor_type, bucket, currency)]
        if not hits:
            return 0.0
        best = max(r.specificity for r in hits)
        top = [r for r in hits if r.specificity == best]
        if len(top) > 1:
            raise ValueError(
                f"scenario '{self.name}': {len(top)} equally specific rules match "
                f"{risk_class}/{factor_type}/bucket={bucket}/currency={currency}"
            )
        return top[0].shock


def _rule(scenario: str, raw: dict) -> ShockRule:
    unknown = set(raw) - {"risk_class", "factor_type", "bucket", "currency", "shock"}
    if unknown:
        raise ValueError(f"scenario '{scenario}': unknown shock fields {sorted(unknown)}")
    if raw.get("risk_class") not in RISK_CLASSES:
        raise ValueError(f"scenario '{scenario}': risk_class must be one of {RISK_CLASSES}")
    if raw.get("factor_type") not in FACTOR_TYPES:
        raise ValueError(f"scenario '{scenario}': factor_type must be one of {FACTOR_TYPES}")
    if not isinstance(raw.get("shock"), int | float) or isinstance(raw.get("shock"), bool):
        raise TypeError(f"scenario '{scenario}': shock must be a number, got {raw.get('shock')!r}")
    bucket = raw.get("bucket")
    return ShockRule(
        raw["risk_class"],
        raw["factor_type"],
        float(raw["shock"]),
        None if bucket is None else str(bucket),
        raw.get("currency"),
    )


# SIMPLIFICATION: scenarios are hypothetical, hand-set shocks, not calibrated to a historical
# episode or to a regulator's published scenario.
def load_scenarios(path: str | Path | None = None) -> list[Scenario]:
    """Scenarios from YAML; names are unique and at most one is the global market shock."""
    raw = yaml.safe_load(stress_scenarios_path(path).read_text()) or []
    scenarios, seen = [], set()
    for r in raw:
        name = r["name"]
        if name in seen:
            raise ValueError(f"duplicate scenario name '{name}'")
        seen.add(name)
        rules = tuple(_rule(name, s) for s in r.get("shocks") or ())
        scenarios.append(
            Scenario(name, r.get("description", ""), bool(r.get("global_market_shock")), rules)
        )
    if sum(s.global_market_shock for s in scenarios) > 1:
        raise ValueError("at most one scenario may be the global market shock")
    return scenarios


_SHOCK_SCHEMA = T.StructType(
    [
        T.StructField("scenario", T.StringType(), False),
        T.StructField("risk_factor_id", T.StringType(), False),
        T.StructField("shock", T.DoubleType(), False),
    ]
)


def shock_table(
    spark: SparkSession, scenarios: Iterable[Scenario], risk_factors: DataFrame
) -> DataFrame:
    """One row per scenario x risk factor, ``shock`` 0.0 where no rule matches (never null).

    Scenarios x risk factors is tiny and independent of the trade count, so it is built on the
    driver.
    """
    factors = risk_factors.select(
        "risk_factor_id", "risk_class", "factor_type", "bucket", "currency"
    ).collect()
    rows = [
        (s.name, f["risk_factor_id"], s.shock_for(f["risk_class"], f["factor_type"], f["bucket"], f["currency"]))
        for s in scenarios
        for f in factors
    ]  # fmt: skip
    return spark.createDataFrame(rows, _SHOCK_SCHEMA)


# SIMPLIFICATION: stress P&L is a second-order Taylor expansion on sensitivities (delta, gamma)
# plus linear vega, not a full revaluation of each trade under the shocked market.
# SIMPLIFICATION: the shock is instantaneous and applied once: no time horizon, no management
# actions, no liquidity-horizon effects and no knock-on moves in unshocked risk factors.
def stress_pnl(sensitivities: DataFrame, shocks: DataFrame) -> DataFrame:
    """Per trade and scenario: ``delta*s + 0.5*gamma*s^2`` on DELTA factors, ``vega*s`` on VEGA.

    Columns: trade_id, book_id, risk_class, scenario, pnl. ``shocks`` is broadcast.
    """
    s = F.col("shock")
    pnl = (
        F.when(
            F.col("factor_type") == "DELTA",
            F.col("delta") * s + F.lit(0.5) * F.col("gamma") * s * s,
        )
        .when(F.col("factor_type") == "VEGA", F.col("vega") * s)
        .otherwise(F.lit(0.0))
    )
    return (
        sensitivities.join(F.broadcast(shocks), "risk_factor_id")
        .groupBy("trade_id", "book_id", "risk_class", "scenario")
        .agg(F.sum(pnl).alias("pnl"))
    )


def stress_by_grain(stress: DataFrame, trades: DataFrame, hierarchy: DataFrame) -> DataFrame:
    """Stress P&L summed to ``GRAIN_DIMS`` x scenario: columns GRAIN_DIMS, scenario, pnl.

    ``asset_class`` and ``currency`` come from the trade, the organisation dimensions from the
    hierarchy (broadcast, one row per book).
    """
    trade_dims = ["asset_class", "currency"]
    org = [d for d in GRAIN_DIMS if d not in (*trade_dims, "book_id", "risk_class")]
    books = F.broadcast(hierarchy.select("book_id", *org))
    keys = [*GRAIN_DIMS, "scenario"]
    return (
        stress.join(trades.select("trade_id", *trade_dims), "trade_id")
        .join(books, "book_id")
        .groupBy(*keys)
        .agg(F.sum("pnl").alias("pnl"))
        .select(*keys, "pnl")
    )


def worst_stress_loss(spec: AggregationSpec, model: SemanticModel, inputs: RunInputs) -> DataFrame:
    """The worst scenario and its loss at every node of ``spec``'s grouping sets.

    Rows carry the scenario name in ``scenario`` and the loss (minus the summed stress P&L)
    in ``value``: positive for a loss, negative when every scenario gains, never clipped.
    Ties go to the scenario name that sorts first. Each node's scenario P&L is summed first
    and the minimum taken after, so the figure is never a sum of children's worst losses.
    """
    stress = inputs.stress_grain()
    for dim, values in spec.filters.items():
        stress = stress.filter(F.col(dim).isin(list(values)))
    frames = []
    for gs in unique_sets(spec.grouping_sets):
        node = stress.groupBy(*gs, "scenario").agg(F.sum("pnl").alias("pnl"))
        rolled = {d: F.lit(ROLLED_UP) for d in GRAIN_DIMS if d not in gs}
        frames.append(node.withColumns(rolled).withColumn("node_level", F.lit(node_level(gs))))
    nodes = frames[0]
    for f in frames[1:]:
        nodes = nodes.unionByName(f)
    first = Window.partitionBy("node_level", *GRAIN_DIMS).orderBy("pnl", "scenario")
    worst = nodes.withColumn("_rank", F.row_number().over(first)).where("_rank = 1")
    return worst.select(
        F.lit(spec.name).alias("aggregation"),
        F.lit(spec.measure).alias("measure"),
        "node_level",
        *GRAIN_DIMS,
        "scenario",
        (-F.col("pnl")).alias("value"),
    ).select(*METRIC_COLUMNS)
