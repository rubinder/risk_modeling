"""FRTB internal models approach: liquidity-adjusted expected shortfall and the IMCC (MAR33).

Per node and per risk-class scope (each risk class alone, and ``*`` for all together):

1. Three sets of one-day P&L from the reporting grain: full set, current window (``fc``);
   reduced set, current (``rc``); reduced set, stressed (``rs``) (MAR33.5-33.6).
2. For each set the cumulative vectors "factors with liquidity horizon >= LH_j", LH_j in 10, 20,
   40, 60, 120 days; the 97.5% ES of each (MAR33.3), scaled to the 10-day base horizon.
3. The liquidity-horizon cascade (MAR33.4):
   ``ES = sqrt(ES_10^2 + sum_{j>=2} (ES_j sqrt((LH_j - LH_{j-1}) / 10))^2)``.
4. Stressed calibration (MAR33.6): ``ES = ES_R,S x max(ES_F,C / ES_R,C, 1)``; the ratio is 1
   when ``ES_R,C`` is 0.
5. ``IMCC = rho x IMCC(C) + (1 - rho) x sum_i IMCC(C_i)``, rho = 0.5 (MAR33.15), where
   IMCC(C) is step 4 for all risk classes together and IMCC(C_i) for risk class i alone
   (MAR33.14: a partial ES with every other risk factor held constant). MAR33.15 writes
   IMCC(C_i) with the bare ratio ``ES_F,C,i / ES_R,C,i``; applying MAR33.6's floor at 1 to
   each risk class's ratio too is this project's reading, and it can only raise the charge.

Every node is recomputed from its own summed vectors, never rolled up from the nodes below:
ES does not add, and the firm's unconstrained ES diversifies across desks.

Not built (explained on the study page): non-modellable risk factors and their stress scenario
capital (MAR33.16-33.17), the IMA default risk charge (MAR33.18-33.39), the 60-day average and
multiplier (MAR33.41-33.42), PLA tests and backtesting (MAR32) and desk eligibility.
"""

import itertools
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from riskagg.conventions import es_col
from riskagg.engine.inputs import RunInputs
from riskagg.engine.spec import AggregationSpec
from riskagg.engine.vectors import (
    GRAIN_DIMS,
    METRIC_COLUMNS,
    ROLLED_UP,
    node_level,
    sum_vectors,
    unique_sets,
)
from riskagg.semantic.model import SemanticModel, SpecError

log = logging.getLogger(__name__)

LEVEL = 0.975  # MAR33.3
LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)  # MAR33.4 Table 1
BASE_HORIZON = 10  # MAR33.4: T = 10 days
RHO = 0.5  # MAR33.15
ALL_CLASSES = ROLLED_UP
# ES set -> (window, reduced set only). MAR33.5-33.6.
ES_SETS: dict[str, tuple[str, bool]] = {
    "fc": ("CURRENT", False),
    "rc": ("CURRENT", True),
    "rs": ("STRESSED", True),
}
STRESSED = ES_SETS["rs"][0]

# SIMPLIFICATION: the 10-day ES is the one-day ES of the scenario vector scaled by sqrt(10).
# MAR33.4 requires ES over 10-day risk factor changes (overlapping observations allowed),
# "without scaling from a shorter horizon".
_TEN_DAY = math.sqrt(BASE_HORIZON)
# SIMPLIFICATION: the reduced set is not tested against MAR33.5's requirement that ES_R,C
# explain at least 75% of ES_F,C on average over the preceding 12 weeks (a run holds one day).
# SIMPLIFICATION: liquidity horizons are the MAR33.12 floors per risk factor; no desk-level
# extension and no cap at the instrument's maturity (MAR33.12(3) and FAQ4).
# SIMPLIFICATION: the capital reported is the IMCC alone: no stress scenario capital for
# non-modellable risk factors (MAR33.16-33.17), no IMA default risk charge (MAR33.18-33.39),
# no 60-day average or multiplier (MAR33.41-33.42), and every desk is treated as eligible
# (no PLA test or backtesting, MAR32).


def _cascade_weights() -> dict[int, float]:
    """sqrt((LH_j - LH_{j-1}) / T), with weight 1 for the base horizon (MAR33.4)."""
    weights = {LIQUIDITY_HORIZONS[0]: 1.0}
    for prev, lh in itertools.pairwise(LIQUIDITY_HORIZONS):
        weights[lh] = math.sqrt((lh - prev) / BASE_HORIZON)
    return weights


CASCADE_WEIGHTS = _cascade_weights()


def lh_adjusted_es(es_by_lh: Mapping[int, float]) -> float:
    """The liquidity-horizon cascade of MAR33.4 over 10-day ES values keyed by LH_j.

    ``es_by_lh[LH_j]`` is the ES of the factors whose liquidity horizon is at least LH_j
    (``es_by_lh[10]`` is the ES of all factors). A missing horizon counts as 0.
    """
    # Each ES_j is squared, so a scope whose ES is negative (gains only in its tail) still
    # gets a positive liquidity-adjusted ES, and so a positive charge.
    return math.sqrt(sum((es_by_lh.get(lh, 0.0) * w) ** 2 for lh, w in CASCADE_WEIGHTS.items()))


def scaled_es(es_rs: float, es_fc: float, es_rc: float) -> float:
    """MAR33.6: ``ES_R,S x max(ES_F,C / ES_R,C, 1)``; the ratio is 1 when ``ES_R,C`` is 0."""
    ratio = 1.0 if es_rc == 0 else es_fc / es_rc
    return es_rs * max(ratio, 1.0)


def stressed_scaling(df: DataFrame) -> DataFrame:
    """``df`` plus column ``es`` from its ``es_rs``, ``es_fc`` and ``es_rc`` (see ``scaled_es``)."""
    ratio = F.when(F.col("es_rc") == 0, F.lit(1.0)).otherwise(F.col("es_fc") / F.col("es_rc"))
    return df.withColumn("es", F.col("es_rs") * F.greatest(ratio, F.lit(1.0)))


def _targets() -> Column:
    """The (ES set, LH_j) pairs a grain row contributes to, as an array of structs.

    A row with horizon ``h`` belongs to every cumulative vector with ``LH_j <= h``. A horizon
    outside MAR33.4's list is an error, never silently bucketed.
    """
    lh = F.col("liquidity_horizon")
    unknown = F.raise_error(F.format_string("unknown liquidity horizon %s", lh.cast("string")))
    lh = F.when(lh.isin(*LIQUIDITY_HORIZONS), lh).otherwise(unknown)
    window, reduced = F.col("window"), F.col("in_reduced_set")
    out = []
    for name, (win, reduced_only) in ES_SETS.items():
        member = (window == win) & reduced if reduced_only else window == win
        for j in LIQUIDITY_HORIZONS:
            pair = F.struct(F.lit(name).alias("es_set"), F.lit(j).alias("liquidity_horizon"))
            out.append(F.when(member & (lh >= j), pair))
    return F.filter(F.array(*out), lambda t: t.isNotNull())


def es_by_horizon(grain: DataFrame, node_cols: list[str], n_scenarios: int) -> DataFrame:
    """Columns node_cols, risk_class (``*`` = all), es_set, liquidity_horizon, es.

    ``es`` is the 10-day 97.5% ES of the cumulative vector of factors with horizon at least
    ``liquidity_horizon`` in the ES set. One aggregation over the grain: each grain row is
    repeated once per (set, LH_j) it belongs to (at most 15) and every vector, per risk class
    and for all classes, comes out of one grouping-sets aggregation of element-wise sums.
    Combinations with no grain rows are absent (their ES is 0 in the cascade).
    """
    keys = [*node_cols, "es_set", "liquidity_horizon"]
    exploded = grain.select(
        *node_cols, "risk_class", "pnl", F.explode(_targets()).alias("_t")
    ).select(*node_cols, "risk_class", "_t.es_set", "_t.liquidity_horizon", "pnl")
    agg = exploded.groupingSets([[*keys, "risk_class"], keys], *keys, "risk_class").agg(
        sum_vectors("pnl", n_scenarios).alias("pnl"), F.grouping("risk_class").alias("_all")
    )
    risk_class = F.when(F.col("_all") == 1, F.lit(ALL_CLASSES)).otherwise(F.col("risk_class"))
    return agg.select(
        *node_cols,
        risk_class.alias("risk_class"),
        "es_set",
        "liquidity_horizon",
        (es_col(F.col("pnl"), LEVEL) * F.lit(_TEN_DAY)).alias("es"),
    )


def ima_from_horizons(by_horizon: DataFrame) -> DataFrame:
    """Columns node_cols, risk_class, es_fc, es_rc, es_rs, es from ``es_by_horizon`` rows."""
    node_cols = [
        c
        for c in by_horizon.columns
        if c not in ("risk_class", "es_set", "liquidity_horizon", "es")
    ]
    weight = F.create_map(*[x for lh, w in CASCADE_WEIGHTS.items() for x in (F.lit(lh), F.lit(w))])
    # Squared, as in ``lh_adjusted_es``: a negative ES_j still adds a positive term.
    term = F.pow(F.col("es") * weight[F.col("liquidity_horizon")], 2)
    sets = [
        F.sqrt(F.coalesce(F.sum(F.when(F.col("es_set") == name, term)), F.lit(0.0))).alias(
            f"es_{name}"
        )
        for name in ES_SETS
    ]
    cascaded = by_horizon.groupBy(*node_cols, "risk_class").agg(*sets)
    return stressed_scaling(cascaded).select(
        *node_cols, "risk_class", "es_fc", "es_rc", "es_rs", "es"
    )


def ima_es(grain: DataFrame, node_cols: list[str], n_scenarios: int) -> DataFrame:
    """Columns node_cols, risk_class (``*`` = unconstrained), es_rs, es_rc, es_fc, es.

    Each of ``es_fc``, ``es_rc`` and ``es_rs`` is liquidity-horizon adjusted (MAR33.4); ``es``
    is the stress-calibrated ES (MAR33.6).
    """
    return ima_from_horizons(es_by_horizon(grain, node_cols, n_scenarios)).select(
        *node_cols, "risk_class", "es_rs", "es_rc", "es_fc", "es"
    )


def imcc(ima: DataFrame) -> DataFrame:
    """Columns node_cols, imcc_unconstrained, imcc_constrained_sum, imcc (MAR33.15).

    ``ima`` carries node_cols, risk_class and es (at least). The constrained figure is the sum
    of the per-risk-class ES (no diversification across classes); ``imcc`` weights the two
    with rho = 0.5.
    """
    node_cols = [c for c in ima.columns if c not in ("risk_class", "es_rs", "es_rc", "es_fc", "es")]
    is_all = F.col("risk_class") == ALL_CLASSES
    agg = ima.groupBy(*node_cols).agg(
        F.coalesce(F.max(F.when(is_all, F.col("es"))), F.lit(0.0)).alias("imcc_unconstrained"),
        F.coalesce(F.sum(F.when(~is_all, F.col("es"))), F.lit(0.0)).alias("imcc_constrained_sum"),
    )
    value = F.lit(RHO) * F.col("imcc_unconstrained") + F.lit(1 - RHO) * F.col(
        "imcc_constrained_sum"
    )
    return agg.select(*node_cols, "imcc_unconstrained", "imcc_constrained_sum", value.alias("imcc"))


# --- the run --------------------------------------------------------------------------------

IMA_MEASURES = {
    "frtb_imcc": "imcc",
    "frtb_imcc_unconstrained": "imcc_unconstrained",
    "frtb_imcc_constrained_sum": "imcc_constrained_sum",
}


@dataclass
class ImaLevel:
    """IMA frames at one node level, cached by the run; both empty without a stressed window."""

    node_cols: list[str]
    detail: DataFrame  # ima_es
    imcc: DataFrame
    cached: list[DataFrame]


def build_ima_level(inputs: RunInputs, node_cols: list[str]) -> ImaLevel:
    """``ima_es`` and ``imcc`` at ``node_cols`` from the run's reporting grain.

    Without any stressed-window rows (the bench profile generates only CURRENT) there is no
    ES_R,S, so no node gets an IMCC: the frames are empty rather than reporting 0.
    """
    if inputs.grain is None or inputs.n_scenarios is None:
        raise ValueError("FRTB IMA measures need the run's grain and n_scenarios")
    grain = inputs.grain
    if grain.where(F.col("window") == STRESSED).limit(1).count() == 0:
        log.warning(
            "FRTB IMA: the reporting grain has no %s window, so no IMCC is reported at %s",
            STRESSED, node_cols,
        )  # fmt: skip
        grain = grain.where(F.lit(False))
    detail = ima_es(grain, node_cols, inputs.n_scenarios).cache()
    charge = imcc(detail).cache()
    return ImaLevel(list(node_cols), detail, charge, [detail, charge])


def _ima_rows(
    column: str, spec: AggregationSpec, model: SemanticModel, inputs: RunInputs
) -> DataFrame:
    del model
    if spec.filters:
        raise SpecError(f"spec '{spec.name}': FRTB measures do not support filters")
    frames = []
    for gs in unique_sets(spec.grouping_sets):
        level = inputs.ima_level(list(gs))
        rolled = {d: F.lit(ROLLED_UP) for d in GRAIN_DIMS if d not in gs}
        frames.append(
            level.imcc.select(
                F.lit(spec.name).alias("aggregation"),
                F.lit(spec.measure).alias("measure"),
                F.lit(node_level(gs)).alias("node_level"),
                *gs,
                F.lit(ROLLED_UP).alias("scenario"),
                F.col(column).alias("value"),
            )
            .withColumns(rolled)
            .select(*METRIC_COLUMNS)
        )
    rows = frames[0]
    for f in frames[1:]:
        rows = rows.unionByName(f)
    return rows


def _measure_fn(column: str):
    def fn(spec: AggregationSpec, model: SemanticModel, inputs: RunInputs) -> DataFrame:
        return _ima_rows(column, spec, model, inputs)

    return fn


FRTB_IMA_MEASURES = {name: _measure_fn(col) for name, col in IMA_MEASURES.items()}


def ima_detail(inputs: RunInputs, levels: list[list[str]]) -> DataFrame:
    """``ima_es`` at every level with rolled-up org dims ``*``.

    Columns: node_level, firm_id, business_line_id, desk_id, risk_class, es_fc, es_rc, es_rs, es.
    """
    org = ("firm_id", "business_line_id", "desk_id")
    frames = []
    for gs in levels:
        rolled = {d: F.lit(ROLLED_UP) for d in org if d not in gs}
        frames.append(
            inputs.ima_level(gs)
            .detail.withColumns(rolled)
            .withColumn("node_level", F.lit(node_level(gs)))
            .select("node_level", *org, "risk_class", "es_fc", "es_rc", "es_rs", "es")
        )
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out
