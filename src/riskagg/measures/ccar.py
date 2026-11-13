"""CCAR (stylised): a nine-quarter capital projection under the supervisory macro scenarios, and
the stress capital buffer.

The aggregation is a path through time, not a quantile and not a cube roll-up. Per quarter and
scenario, revenue (PPNR), credit losses and the trading loss of the global market shock are
summed over loan segments into business lines and legal entities; capital (CET1) only exists
for legal entities and the firm, and accumulates quarter by quarter. The capital measures in
``risk_metrics`` (starting ratio, minimum ratio, stress capital buffer) are structured: looked
up at the legal entity and the firm, never at a business line, because capital is not allocated
to business lines.

Formulas, per quarter q = 1..9 (``u`` unemployment in percentage points, ``g`` GDP growth):

- ``pd_q = min(1, pd_base / 4 x (1 + beta x max(0, u_q - u_0)))``
- ``credit_loss = ead x pd_q x lgd``
- ``ppnr = ead x (revenue_rate_q x (1 + gdp_beta x g_q) - expense_rate_q)``
- ``trading_loss`` = the global-market-shock loss in quarter 1 of SEVERELY_ADVERSE, else 0
- ``net_income = (ppnr - credit_loss - trading_loss) x (1 - tax_rate)``
- ``cet1_q = cet1_{q-1} + net_income_q`` (no common dividends: 12 CFR 252.44(c)(1); the
  rule does deduct stated AT1/T2 payments, 252.44(c)(2), but there are none here)
- ``rwa = sum(ead x risk_weight) + rwa_trading``, constant; ``cet1_ratio = cet1 / rwa``
- ``scb = max(floor, ratio_0 - min_q ratio_q(SEVERELY_ADVERSE)
  + sum(planned dividends, dividend quarters) / rwa(trough quarter))``
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from riskagg.data.banking import BASELINE, SEVERELY_ADVERSE
from riskagg.data.org import FIRM_ID
from riskagg.engine.inputs import RunInputs
from riskagg.engine.spec import AggregationSpec
from riskagg.engine.vectors import GRAIN_DIMS, METRIC_COLUMNS, ROLLED_UP, node_level
from riskagg.measures.stress import Scenario
from riskagg.paths import ccar_path
from riskagg.semantic.model import SemanticModel, SpecError

SCENARIOS = (BASELINE, SEVERELY_ADVERSE)
N_QUARTERS = 9
BUSINESS_LINE, LEGAL_ENTITY, FIRM = "BUSINESS_LINE", "LEGAL_ENTITY", "FIRM"
LOSS_COLUMNS = ("ppnr", "credit_loss", "trading_loss", "net_income", "dividends")
CAPITAL_COLUMNS = ("cet1", "rwa", "cet1_ratio")
PROJECTION_COLUMNS = [
    "scenario", "quarter", "node_level", "node_id", *LOSS_COLUMNS, *CAPITAL_COLUMNS,
]  # fmt: skip
START_RATIO, MIN_RATIO, SCB = "ccar_start_cet1_ratio", "ccar_min_cet1_ratio", "ccar_scb"
CCAR_MEASURE_NAMES = (START_RATIO, MIN_RATIO, SCB)
# risk_metrics node levels: the firm, and legal entities within it.
FIRM_LEVEL = ("firm_id",)
ENTITY_LEVEL = ("firm_id", "legal_entity")


@dataclass(frozen=True)
class CcarParams:
    pd_unemployment_beta: float
    revenue_gdp_beta: float
    tax_rate: float
    scb_floor: float
    scb_dividend_quarters: tuple[int, int]


def load_params(path: str | Path | None = None) -> CcarParams:
    raw = yaml.safe_load(ccar_path(path).read_text()) or {}
    first, last = (int(q) for q in raw["scb_dividend_quarters"])
    if not 1 <= first <= last <= N_QUARTERS:
        raise ValueError(f"scb_dividend_quarters must lie in 1..{N_QUARTERS}, got {first}..{last}")
    return CcarParams(
        float(raw["pd_unemployment_beta"]),
        float(raw["revenue_gdp_beta"]),
        float(raw["tax_rate"]),
        float(raw["scb_floor"]),
        (first, last),
    )


def global_market_shock(scenarios: Iterable[Scenario]) -> str:
    """Name of the scenario flagged ``global_market_shock`` in the stress library."""
    flagged = [s.name for s in scenarios if s.global_market_shock]
    if len(flagged) != 1:
        raise ValueError(f"expected one global market shock scenario, found {flagged}")
    return flagged[0]


def gms_loss_by_entity_bl(stress_grain: DataFrame, scenario: str) -> DataFrame:
    """``(legal_entity, business_line_id, trading_loss)`` from the stress P&L of ``scenario``.

    ``trading_loss`` is minus the summed P&L: positive for a loss, negative for a gain (never
    clipped).
    """
    return (
        stress_grain.where(F.col("scenario") == scenario)
        .groupBy("legal_entity", "business_line_id")
        .agg((-F.sum("pnl")).alias("trading_loss"))
    )


def _quarters(macro: DataFrame) -> DataFrame:
    """Projected quarters with the rise in unemployment over quarter 0 of the same scenario."""
    start = macro.where(F.col("quarter") == 0).select(
        "scenario", F.col("unemployment_rate").alias("_u0")
    )
    return (
        macro.where(F.col("quarter").between(1, N_QUARTERS))
        .join(start, "scenario")
        .select(
            "scenario",
            "quarter",
            F.greatest(F.lit(0.0), F.col("unemployment_rate") - F.col("_u0")).alias("_du"),
            "gdp_growth",
        )
    )


def _pair_quarters(
    banking_book: DataFrame, macro: DataFrame, gms: DataFrame, params: CcarParams
) -> DataFrame:
    """PPNR, credit and trading loss and net income per scenario, quarter, entity and line."""
    q = F.broadcast(_quarters(macro))
    pd_q = F.least(
        F.lit(1.0),
        F.col("pd_base") / 4.0 * (1.0 + F.lit(params.pd_unemployment_beta) * F.col("_du")),
    )
    revenue = F.col("revenue_rate_q") * (1.0 + F.lit(params.revenue_gdp_beta) * F.col("gdp_growth"))
    keys = ["scenario", "quarter", "legal_entity", "business_line_id"]
    segments = (
        banking_book.crossJoin(q)
        .groupBy(*keys)
        .agg(
            F.sum(F.col("ead") * (revenue - F.col("expense_rate_q"))).alias("ppnr"),
            F.sum(F.col("ead") * pd_q * F.col("lgd")).alias("credit_loss"),
        )
    )
    # Every (entity, line) pair with a banking book or a trading position, in every quarter.
    pairs = (
        banking_book.select("legal_entity", "business_line_id")
        .unionByName(gms.select("legal_entity", "business_line_id"))
        .distinct()
        .crossJoin(q.select("scenario", "quarter"))
    )
    shocked = (F.col("scenario") == SEVERELY_ADVERSE) & (F.col("quarter") == 1)
    pre_tax = F.col("ppnr") - F.col("credit_loss") - F.col("trading_loss")
    return (
        pairs.join(segments, keys, "left")
        .join(F.broadcast(gms), ["legal_entity", "business_line_id"], "left")
        .select(
            *keys,
            F.coalesce("ppnr", F.lit(0.0)).alias("ppnr"),
            F.coalesce("credit_loss", F.lit(0.0)).alias("credit_loss"),
            F.when(shocked, F.coalesce("trading_loss", F.lit(0.0)))
            .otherwise(F.lit(0.0))
            .alias("trading_loss"),
        )
        # SIMPLIFICATION: tax is applied symmetrically - a pre-tax loss earns a full tax credit
        # (no deferred-tax-asset limits).
        .withColumn("net_income", pre_tax * (1.0 - params.tax_rate))
    )


def _summed(df: DataFrame, keys: list[str]) -> DataFrame:
    return df.groupBy(*keys).agg(*(F.sum(c).alias(c) for c in LOSS_COLUMNS if c in df.columns))


def _null_capital(df: DataFrame) -> DataFrame:
    return df.withColumns({c: F.lit(None).cast("double") for c in CAPITAL_COLUMNS})


# SIMPLIFICATION: constant balance sheet - exposures, risk weights and RWA stay at their
# starting values for all nine quarters (no growth, run-off, or migration of risk weights).
# SIMPLIFICATION: no preferred stock or other additional tier 1 / tier 2 instruments, so no
# stated dividends or interest are deducted from CET1 (the supervisory path deducts them,
# 12 CFR 252.44(c)(2)).
# SIMPLIFICATION: no counterparty default component (the largest-counterparty-default loss of
# the supervisory test) and no operational-risk losses or other trading-book effects beyond
# the one global market shock.
def project(
    banking_book: DataFrame,
    macro: DataFrame,
    capital_start: DataFrame,
    gms_loss_by_entity_bl: DataFrame,
    params: CcarParams,
) -> DataFrame:
    """The ``ccar_projection`` table: ``PROJECTION_COLUMNS``, quarters 1..9, both scenarios.

    Business-line rows carry losses and revenue only (capital columns null). Legal-entity rows
    add CET1, RWA and the CET1 ratio; the firm row sums the legal entities (its ratio is
    summed CET1 over summed RWA). ``dividends`` is the planned common dividend of the quarter:
    it is reported, and used only by the stress capital buffer's dividend add-on, never
    deducted from the projected CET1 (12 CFR 252.44(c)(1)).
    """
    pairs = _pair_quarters(banking_book, macro, gms_loss_by_entity_bl, params)
    bl = _summed(pairs, ["scenario", "quarter", "business_line_id"])
    bl = _null_capital(
        bl.withColumn("dividends", F.lit(None).cast("double"))
        .withColumn("node_level", F.lit(BUSINESS_LINE))
        .withColumnRenamed("business_line_id", "node_id")
    )

    credit_rwa = banking_book.groupBy("legal_entity").agg(
        F.sum(F.col("ead") * F.col("risk_weight")).alias("_rwa_credit")
    )
    capital = capital_start.join(credit_rwa, "legal_entity", "left").select(
        "legal_entity",
        "cet1_start",
        (F.coalesce("_rwa_credit", F.lit(0.0)) + F.col("rwa_trading")).alias("rwa"),
        F.col("dividends_q").alias("dividends"),
    )
    path = Window.partitionBy("scenario", "legal_entity").orderBy("quarter")
    le = (
        _summed(pairs, ["scenario", "quarter", "legal_entity"])
        .join(F.broadcast(capital), "legal_entity")
        .withColumn("cet1", F.col("cet1_start") + F.sum("net_income").over(path))
        .withColumn("cet1_ratio", F.col("cet1") / F.col("rwa"))
        .withColumn("node_level", F.lit(LEGAL_ENTITY))
        .withColumnRenamed("legal_entity", "node_id")
        .drop("cet1_start")
    )
    firm = (
        le.groupBy("scenario", "quarter")
        .agg(*(F.sum(c).alias(c) for c in (*LOSS_COLUMNS, "cet1", "rwa")))
        .withColumn("cet1_ratio", F.col("cet1") / F.col("rwa"))
        .withColumn("node_level", F.lit(FIRM))
        .withColumn("node_id", F.lit(FIRM_ID))
    )
    return (
        bl.select(*PROJECTION_COLUMNS)
        .unionByName(le.select(*PROJECTION_COLUMNS))
        .unionByName(firm.select(*PROJECTION_COLUMNS))
    )


def summary(projection: DataFrame, capital_start: DataFrame, params: CcarParams) -> DataFrame:
    """Capital measures per legal entity and firm: ``(measure, node_level, node_id, scenario,
    value)`` with ``node_level`` LEGAL_ENTITY or FIRM.

    - ``ccar_start_cet1_ratio``: starting CET1 / RWA, scenario ``*``.
    - ``ccar_min_cet1_ratio``: the lowest projected ratio over quarters 1..9, per scenario.
    - ``ccar_scb``: ``max(floor, start - min(SEVERELY_ADVERSE) + dividend add-on)``, where the
      add-on is the planned dividends of ``params.scb_dividend_quarters`` over the RWA of the
      quarter with the lowest ratio (the first such quarter on a tie), scenario ``*``.

    The starting ratio uses the quarter-1 RWA, which equals the starting RWA because the balance
    sheet is held constant.
    """
    # SIMPLIFICATION: the stress capital buffer is computed for each legal entity as well as the
    # firm; the rule sets it for the holding company only (12 CFR 225.8(b)), and one supervisory
    # stress test cycle is used (no averaging of the decline over two cycles).
    capital = projection.where(F.col("node_level") != BUSINESS_LINE)
    entity_start = capital_start.select(
        F.lit(LEGAL_ENTITY).alias("node_level"),
        F.col("legal_entity").alias("node_id"),
        "cet1_start",
    )
    firm_start = capital_start.agg(F.sum("cet1_start").alias("cet1_start")).select(
        F.lit(FIRM).alias("node_level"), F.lit(FIRM_ID).alias("node_id"), "cet1_start"
    )
    node = ["node_level", "node_id"]
    rwa0 = capital.where(F.col("quarter") == 1).select(
        *node, "scenario", F.col("rwa").alias("_rwa")
    )
    start = (
        entity_start.unionByName(firm_start)
        .join(rwa0.where(F.col("scenario") == SEVERELY_ADVERSE).drop("scenario"), node)
        .select(*node, (F.col("cet1_start") / F.col("_rwa")).alias("start_ratio"))
    )
    lowest = Window.partitionBy(*node, "scenario").orderBy("cet1_ratio", "quarter")
    trough = (
        capital.withColumn("_rank", F.row_number().over(lowest))
        .where("_rank = 1")
        .select(
            *node,
            "scenario",
            F.col("cet1_ratio").alias("min_ratio"),
            F.col("rwa").alias("trough_rwa"),
        )
    )
    first, last = params.scb_dividend_quarters
    dividends = (
        capital.where(
            (F.col("scenario") == SEVERELY_ADVERSE) & F.col("quarter").between(first, last)
        )
        .groupBy(*node)
        .agg(F.sum("dividends").alias("planned_dividends"))
    )
    stressed = trough.where(F.col("scenario") == SEVERELY_ADVERSE).drop("scenario")
    scb = (
        start.join(stressed, node)
        .join(dividends, node)
        .select(
            *node,
            F.greatest(
                F.lit(params.scb_floor),
                F.col("start_ratio") - F.col("min_ratio")
                + F.col("planned_dividends") / F.col("trough_rwa"),
            ).alias("value"),
        )
    )  # fmt: skip

    def rows(df: DataFrame, measure: str, scenario: Column, value: str) -> DataFrame:
        return df.select(F.lit(measure).alias("measure"), *node, scenario.alias("scenario"),
                         F.col(value).alias("value"))  # fmt: skip

    return (
        rows(start, START_RATIO, F.lit(ROLLED_UP), "start_ratio")
        .unionByName(rows(trough, MIN_RATIO, F.col("scenario"), "min_ratio"))
        .unionByName(rows(scb, SCB, F.lit(ROLLED_UP), "value"))
    )


def _metric_rows(spec: AggregationSpec, model: SemanticModel, inputs: RunInputs) -> DataFrame:
    """``spec``'s measure from the run's CCAR summary, in ``METRIC_COLUMNS``."""
    if spec.filters:
        raise SpecError(f"spec '{spec.name}': CCAR measures take no filters")
    levels = {frozenset(gs) for gs in spec.grouping_sets}
    wanted = [lvl for lvl, dims in ((FIRM, FIRM_LEVEL), (LEGAL_ENTITY, ENTITY_LEVEL))
              if frozenset(dims) in levels]  # fmt: skip
    df = inputs.ccar_summary().where(
        (F.col("measure") == spec.measure) & F.col("node_level").isin(wanted)
    )
    is_firm = F.col("node_level") == FIRM
    dims = {d: F.lit(ROLLED_UP) for d in GRAIN_DIMS}
    dims["firm_id"] = F.lit(FIRM_ID)
    dims["legal_entity"] = F.when(is_firm, F.lit(ROLLED_UP)).otherwise(F.col("node_id"))
    return df.select(
        F.lit(spec.name).alias("aggregation"),
        "measure",
        F.when(is_firm, node_level(FIRM_LEVEL))
        .otherwise(node_level(ENTITY_LEVEL))
        .alias("node_level"),
        *(dims[d].alias(d) for d in GRAIN_DIMS),
        "scenario",
        "value",
    ).select(*METRIC_COLUMNS)


CCAR_MEASURES = {name: _metric_rows for name in CCAR_MEASURE_NAMES}
