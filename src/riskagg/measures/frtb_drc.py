"""FRTB standardised approach: default risk charge, residual risk add-on and the SA total.

The SA capital charge of a node is the sum of three components (MAR20.4):

* the sensitivities-based method (``frtb_sa``), taken at the binding correlation scenario;
* the default risk charge for non-securitisations (MAR22);
* the residual risk add-on (MAR23).

All are structured measures: each reported node is recomputed from the netted positions at that
node, never rolled up from the nodes below it (netting across desks lowers the charge, and the
SBM's square roots do not add).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from riskagg.engine.inputs import RunInputs
from riskagg.engine.spec import AggregationSpec
from riskagg.engine.vectors import GRAIN_DIMS, METRIC_COLUMNS, ROLLED_UP, node_level, unique_sets
from riskagg.measures import frtb_sa
from riskagg.paths import frtb_drc_path
from riskagg.semantic.model import SemanticModel, SpecError

# SIMPLIFICATION: gross JTD = LGD x notional: no P&L adjustment (MAR22.10-22.11), so an
# instrument whose market value moved is still measured at its notional.
# SIMPLIFICATION: no maturity scaling: every position is treated as one year or longer, so
# long and short positions net in full (MAR22.19-22.20 scale shorter positions by maturity).
# SIMPLIFICATION: long and short positions in one issuer net regardless of seniority; MAR22.19
# allows offsetting only where the short ranks the same as or below the long.
# SIMPLIFICATION: two buckets, CORPORATE and SOVEREIGN; no local governments and municipalities
# bucket (MAR22.22(3)), and no hedge benefit across buckets (MAR22.26, so this is exact).
# SIMPLIFICATION: the residual risk add-on uses the 1.0% exotic-underlying rate only
# (MAR23.8(2)(a)), applied to the gross notional of trades flagged ``is_exotic``; the 0.1% rate
# for other residual risks (MAR23.8(2)(b)) is not applied.

FRTB_COMPONENT_MEASURES = {
    "frtb_sa_charge": "charge",
    "frtb_sbm_charge": "sbm_charge",
    "frtb_drc_charge": "drc_charge",
    "frtb_rrao_charge": "rrao_charge",
}
_SIGN = {"LONG": 1.0, "SHORT": -1.0}


@dataclass(frozen=True)
class DrcRef:
    """Default risk and residual add-on parameters from ``config/reference/frtb_drc.yaml``."""

    lgd: dict[str, float]  # seniority -> LGD (MAR22.12)
    risk_weights: dict[str, float]  # rating -> default risk weight (MAR22.24)
    buckets: tuple[str, ...]
    rrao_exotic: float  # MAR23.8(2)(a)
    rrao_other: float  # MAR23.8(2)(b), carried for reference, not applied


def load_drc_ref(path: str | Path | None = None) -> DrcRef:
    raw = yaml.safe_load(frtb_drc_path(path).read_text())
    drc, rr = raw["default_risk"], raw["residual_risk_add_on"]
    return DrcRef(
        lgd={str(k).upper(): float(v) for k, v in drc["lgd"].items()},
        risk_weights={str(k).upper(): float(v) for k, v in drc["risk_weights"].items()},
        buckets=tuple(drc["buckets"]),
        rrao_exotic=float(rr["exotic"]),
        rrao_other=float(rr["other"]),
    )


def _lookup(table: dict[str, float], key: Column, what: str) -> Column:
    """Value of ``table`` at ``key``; an unknown key is an error, never a silent default."""
    m = F.create_map(*[x for k, v in table.items() for x in (F.lit(k), F.lit(v))])
    value = m[key]
    message = F.format_string("no default risk %s for '%s'", F.lit(what), key)
    return F.when(value.isNull(), F.raise_error(message)).otherwise(value)


def drc(positions: DataFrame, node_cols: list[str], ref: DrcRef) -> DataFrame:
    """Columns node_cols, drc_bucket, hbr, charge (MAR22.19-22.26).

    ``positions`` carries ``node_cols`` plus issuer_id, drc_bucket, rating, seniority,
    direction and notional. Per node x issuer the signed JTDs (``LGD x notional``, negative for
    shorts) are netted; the net longs and net shorts of a bucket are summed separately,
    ``HBR = long / (long + |short|)`` (0 where there is no long), and
    ``charge = max(sum rw x long - HBR x sum rw x |short|, 0)``.
    """
    sign = F.when(F.col("direction") == "LONG", 1.0).when(F.col("direction") == "SHORT", -1.0)
    sign = F.when(sign.isNull(), F.raise_error("direction must be LONG or SHORT")).otherwise(sign)
    jtd = sign * _lookup(ref.lgd, F.upper(F.col("seniority")), "LGD") * F.col("notional")
    net = positions.groupBy(*node_cols, "issuer_id", "drc_bucket", "rating").agg(
        F.sum(jtd).alias("net_jtd")
    )
    rw = _lookup(ref.risk_weights, F.upper(F.col("rating")), "risk weight")
    long_ = F.greatest(F.col("net_jtd"), F.lit(0.0))
    short = F.greatest(-F.col("net_jtd"), F.lit(0.0))
    agg = net.groupBy(*node_cols, "drc_bucket").agg(
        F.sum(long_).alias("_long"),
        F.sum(short).alias("_short"),
        F.sum(rw * long_).alias("_rw_long"),
        F.sum(rw * short).alias("_rw_short"),
    )
    total = F.col("_long") + F.col("_short")
    hbr = F.when(total > 0, F.col("_long") / total).otherwise(F.lit(0.0))
    charge = F.greatest(F.col("_rw_long") - hbr * F.col("_rw_short"), F.lit(0.0))
    return agg.select(*node_cols, "drc_bucket", hbr.alias("hbr"), charge.alias("charge"))


def rrao(trades: DataFrame, node_cols: list[str], ref: DrcRef | None = None) -> DataFrame:
    """Columns node_cols, charge: ``rate x gross notional`` of the exotic trades (MAR23.8)."""
    ref = ref or load_drc_ref()
    return (
        trades.where(F.col("is_exotic"))
        .groupBy(*node_cols)
        .agg((F.sum("notional") * F.lit(ref.rrao_exotic)).alias("charge"))
    )


def frtb_sa_total(sbm_total: DataFrame, drc: DataFrame, rrao: DataFrame) -> DataFrame:
    """Columns node_cols, sbm_charge, drc_charge, rrao_charge, charge (the SA total).

    The SBM component is the binding correlation scenario's charge (MAR21.7); the DRC is summed
    over buckets (MAR22.26). A node missing from a component has 0 for it.
    """
    node_cols = [c for c in sbm_total.columns if c not in ("corr_scenario", "charge", "is_binding")]
    if not node_cols:
        raise ValueError("frtb_sa_total needs at least one node column")
    sbm = sbm_total.where("is_binding").select(*node_cols, F.col("charge").alias("sbm_charge"))
    d = drc.groupBy(*node_cols).agg(F.sum("charge").alias("drc_charge"))
    r = rrao.select(*node_cols, F.col("charge").alias("rrao_charge"))
    out = sbm.join(d, node_cols, "full_outer").join(r, node_cols, "full_outer")
    parts = ("sbm_charge", "drc_charge", "rrao_charge")
    out = out.fillna(0.0, subset=list(parts))
    return out.select(
        *node_cols, *parts, sum((F.col(c) for c in parts), F.lit(0.0)).alias("charge")
    )


@dataclass
class FrtbLevel:
    """Everything FRTB at one node level; ``detail`` and ``total`` are cached by the run."""

    node_cols: list[str]
    detail: DataFrame  # frtb_sa.bucket_detail
    total: DataFrame  # frtb_sa_total
    cached: list[DataFrame]


def build_level(inputs: RunInputs, node_cols: list[str]) -> FrtbLevel:
    """SBM, DRC and RRAO at ``node_cols``, each netted at that level (never rolled up).

    The bucket detail and the total are cached, so the SBM is computed once for the charge
    rows of all four measures and for the curated detail table. ``RunInputs.release``
    unpersists them.
    """
    missing = [
        n
        for n in ("sensitivities", "trades", "hierarchy", "default_risk_positions")
        if getattr(inputs, n) is None
    ]
    if missing:
        raise ValueError(f"FRTB SA measures need run inputs {missing}, which are not set")
    sa_ref, drc_ref = frtb_sa.load_frtb_ref(), load_drc_ref()
    sens = frtb_sa.attach_org(inputs.sensitivities, inputs.hierarchy, node_cols)
    detail = frtb_sa.bucket_detail(sens, node_cols, sa_ref).cache()
    sbm = frtb_sa.sbm_total(frtb_sa.charges_from_detail(detail, sa_ref))
    trades = frtb_sa.attach_org(inputs.trades, inputs.hierarchy, node_cols)
    positions = frtb_sa.attach_org(inputs.default_risk_positions, inputs.hierarchy, node_cols)
    total = frtb_sa_total(sbm, drc(positions, node_cols, drc_ref), rrao(trades, node_cols, drc_ref))
    total = total.cache()
    return FrtbLevel(node_cols, detail, total, [detail, total])


def _frtb_rows(
    column: str, spec: AggregationSpec, model: SemanticModel, inputs: RunInputs
) -> DataFrame:
    del model
    if spec.filters:
        raise SpecError(f"spec '{spec.name}': FRTB measures do not support filters")
    frames = []
    for gs in unique_sets(spec.grouping_sets):
        level = inputs.frtb_level(list(gs))
        rolled = {d: F.lit(ROLLED_UP) for d in GRAIN_DIMS if d not in gs}
        frames.append(
            level.total.select(
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
        return _frtb_rows(column, spec, model, inputs)

    return fn


FRTB_MEASURES = {name: _measure_fn(col) for name, col in FRTB_COMPONENT_MEASURES.items()}


def sa_detail(inputs: RunInputs, levels: list[list[str]]) -> DataFrame:
    """``bucket_detail`` at every level in ``GRAIN``-style columns, rolled-up dims ``*``.

    Columns: node_level, firm_id, business_line_id, desk_id, risk_class, component, bucket,
    corr_scenario, k_b, s_b.
    """
    org = ("firm_id", "business_line_id", "desk_id")
    frames = []
    for gs in levels:
        rolled = {d: F.lit(ROLLED_UP) for d in org if d not in gs}
        frames.append(
            inputs.frtb_level(gs)
            .detail.withColumns(rolled)
            .withColumn("node_level", F.lit(node_level(gs)))
            .select(
                "node_level",
                *org,
                "risk_class",
                "component",
                "bucket",
                "corr_scenario",
                "k_b",
                "s_b",
            )
        )
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out
