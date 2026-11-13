"""FRTB standardised approach: the sensitivities-based method (Basel MAR21).

The nesting, recomputed at every node where it is reported (never rolled up):

1. Net sensitivities per node x risk factor across trades (MAR21.4(2)).
2. Weight them: ``WS_k = s_k x RW_k`` (MAR21.4(3)).
3. Within each bucket: ``K_b = sqrt(max(0, sum_k sum_l rho_kl WS_k WS_l))``, ``rho_kk = 1``
   (MAR21.4(4)), as a self-join of the node's weighted sensitivities inside the bucket.
4. Across buckets: ``sqrt(sum_b K_b^2 + sum_b sum_{c!=b} gamma_bc S_b S_c)``, ``S_b = sum WS_k``;
   if the sum is negative, ``S_b`` is clamped to ``[-K_b, K_b]`` (MAR21.4(5)). A self-join of the
   node's buckets inside a risk class.
5. Curvature per MAR21.5 with squared correlations (MAR21.100-21.101) and psi.
6. Steps 3-5 under three correlation scenarios (MAR21.6); per scenario the delta, vega and
   curvature charges of all risk classes are summed and the largest total is the charge (MAR21.7).

Sign convention: delta and vega are P&L per unit move of the factor (relative move for EQ and FX,
absolute rate change for GIRR); curvature_up/curvature_down are CVR amounts, positive for loss.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from riskagg.paths import frtb_sa_path

RISK_CLASSES = ("GIRR", "EQ", "FX")
COMPONENTS = ("DELTA", "VEGA", "CURVATURE")
SCENARIOS = ("LOW", "MEDIUM", "HIGH")
SENSITIVITY_COLUMNS = ("delta", "vega", "curvature_up", "curvature_down")
SQRT2 = math.sqrt(2.0)

# SIMPLIFICATION: one curve per currency for GIRR: no inflation or cross-currency basis risk
# factors (MAR21.8(2)-(3)) and no different-curve correlation (99.90%, MAR21.45/21.47).
# SIMPLIFICATION: equity spot only, no repo rate risk factor (MAR21.12); buckets 1, 5, 8, 12 only.
# SIMPLIFICATION: one vega factor per bucket, no option-maturity or underlying-maturity
# dimension (MAR21.8(4), MAR21.93-21.94): the option-maturity correlation is taken as 1.
# SIMPLIFICATION: one curvature factor per bucket (the bucket's CVRs summed); exact for GIRR
# with one curve per currency and for FX, and for EQ while each bucket holds one issuer.
# SIMPLIFICATION: sensitivities are taken as given, not computed by the prescribed bumps
# (MAR21.18-21.26).


@dataclass(frozen=True)
class FrtbSaRef:
    """Risk weights and correlations from ``config/reference/frtb_sa.yaml`` (MAR21)."""

    reporting_currency: str
    high_multiplier: float
    high_cap: float
    low_slope: float
    low_intercept: float
    low_floor_multiplier: float
    girr_delta_rw: dict[float, float]
    girr_specified_currencies: tuple[str, ...]
    girr_theta: float
    girr_rho_floor: float
    girr_gamma: float
    girr_vega_rw: float
    eq_delta_rw: dict[str, float]
    eq_rho: dict[str, float]
    eq_gamma_1_to_10: float
    eq_gamma_with_11: float
    eq_gamma_12_13: float
    eq_gamma_otherwise: float
    eq_vega_rw: dict[str, float]
    fx_delta_rw: float
    fx_specified_currencies: tuple[str, ...]
    fx_gamma: float
    fx_vega_rw: float


def load_frtb_ref(path: str | Path | None = None) -> FrtbSaRef:
    raw = yaml.safe_load(frtb_sa_path(path).read_text())
    rep = raw["reporting_currency"]
    sc = raw["correlation_scenarios"]
    girr, eq, fx = raw["GIRR"], raw["EQ"], raw["FX"]
    eq_buckets = {str(b): v for b, v in eq["delta"]["buckets"].items()}
    eq_gamma = eq["delta"]["gamma"]
    pairs = [p.split("/") for p in fx["delta"]["specified_pairs"]]
    fx_specified = sorted({b if a == rep else a for a, b in pairs if rep in (a, b)})
    # MAR21.44 footnote 12: the domestic reporting currency is a specified currency.
    girr_specified = sorted({*girr["delta"]["specified_currencies"], rep})
    return FrtbSaRef(
        reporting_currency=rep,
        high_multiplier=float(sc["high"]["multiplier"]),
        high_cap=float(sc["high"]["cap"]),
        low_slope=float(sc["low"]["slope"]),
        low_intercept=float(sc["low"]["intercept"]),
        low_floor_multiplier=float(sc["low"]["floor_multiplier"]),
        girr_delta_rw={float(t): float(w) for t, w in girr["delta"]["risk_weights"].items()},
        girr_specified_currencies=tuple(girr_specified),
        girr_theta=float(girr["delta"]["tenor_correlation"]["theta"]),
        girr_rho_floor=float(girr["delta"]["tenor_correlation"]["floor"]),
        girr_gamma=float(girr["delta"]["gamma"]),
        girr_vega_rw=float(girr["vega"]["risk_weight"]),
        eq_delta_rw={b: float(v["risk_weight"]) for b, v in eq_buckets.items()},
        eq_rho={b: float(v["rho"]) for b, v in eq_buckets.items()},
        eq_gamma_1_to_10=float(eq_gamma["buckets_1_to_10"]),
        eq_gamma_with_11=float(eq_gamma["with_bucket_11"]),
        eq_gamma_12_13=float(eq_gamma["between_12_and_13"]),
        eq_gamma_otherwise=float(eq_gamma["otherwise"]),
        eq_vega_rw={str(b): float(v) for b, v in eq["vega"]["risk_weights"].items()},
        fx_delta_rw=float(fx["delta"]["risk_weight"]),
        fx_specified_currencies=tuple(fx_specified),
        fx_gamma=float(fx["delta"]["gamma"]),
        fx_vega_rw=float(fx["vega"]["risk_weight"]),
    )


def girr_delta_rho(t_k: float, t_l: float, ref: FrtbSaRef) -> float:
    """MAR21.46 footnote 13: ``max(exp(-theta |Tk - Tl| / min(Tk, Tl)), 40%)``."""
    if t_k == t_l:
        return 1.0
    return max(math.exp(-ref.girr_theta * abs(t_k - t_l) / min(t_k, t_l)), ref.girr_rho_floor)


def attach_org(sens: DataFrame, hierarchy: DataFrame, cols: list[str]) -> DataFrame:
    """Add organisation columns from the hierarchy by ``book_id`` (broadcast, one row per book).

    A book missing from the hierarchy is an error, never a silently dropped row (as in
    ``engine.grain``): the check is one extra pass over the distinct books of ``sens``.
    """
    missing = [c for c in cols if c not in sens.columns]
    if not missing:
        return sens
    books = hierarchy.select("book_id", *missing)
    unknown = (
        sens.select("book_id")
        .distinct()
        .join(F.broadcast(books.select("book_id")), "book_id", "left_anti")
        .limit(10)
        .collect()
    )
    if unknown:
        raise ValueError(
            f"book_id missing from hierarchy: {sorted(str(r['book_id']) for r in unknown)}"
        )
    return sens.join(F.broadcast(books), "book_id")


# --- column expressions -------------------------------------------------------------------


def _map(d: dict) -> Column:
    return F.create_map(*[x for k, v in d.items() for x in (F.lit(k), F.lit(v))])


def _scenario(x: Column, scenario: Column, ref: FrtbSaRef) -> Column:
    """MAR21.6: medium as given; high ``min(1.25x, 1)``; low ``max(2x - 1, 0.75x)``."""
    return (
        F.when(scenario == "HIGH", F.least(x * ref.high_multiplier, F.lit(ref.high_cap)))
        .when(
            scenario == "LOW",
            F.greatest(x * ref.low_slope + ref.low_intercept, x * ref.low_floor_multiplier),
        )
        .otherwise(x)
    )


def _risk_weight(ref: FrtbSaRef) -> Column:
    """Risk weight per factor (MAR21.42/44, 21.77, 21.87/88, 21.92); unknown -> error."""
    rc, comp, bucket = F.col("risk_class"), F.col("component"), F.col("bucket")
    girr_div = F.when(bucket.isin(*ref.girr_specified_currencies), SQRT2).otherwise(1.0)
    fx_div = F.when(bucket.isin(*ref.fx_specified_currencies), SQRT2).otherwise(1.0)
    delta = (
        F.when(rc == "GIRR", _map(ref.girr_delta_rw)[F.col("tenor_years")] / girr_div)
        .when(rc == "EQ", _map(ref.eq_delta_rw)[bucket])
        .when(rc == "FX", F.lit(ref.fx_delta_rw) / fx_div)
    )
    vega = (
        F.when(rc == "GIRR", F.lit(ref.girr_vega_rw))
        .when(rc == "EQ", _map(ref.eq_vega_rw)[bucket])
        .when(rc == "FX", F.lit(ref.fx_vega_rw))
    )
    rw = F.when(comp == "DELTA", delta).when(comp == "VEGA", vega)
    message = F.format_string(
        "no FRTB SA risk weight for %s %s bucket %s tenor %s",
        rc, comp, bucket, F.coalesce(F.col("tenor_years").cast("string"), F.lit("-")),
    )  # fmt: skip
    return F.when(rw.isNull(), F.raise_error(message)).otherwise(rw)


def _intra_rho(ref: FrtbSaRef) -> Column:
    """Medium rho between two distinct factors of one bucket (columns ``t_a``, ``t_b``)."""
    rc, comp = F.col("risk_class"), F.col("component")
    t_a, t_b = F.col("t_a"), F.col("t_b")
    girr = F.greatest(
        F.exp(-ref.girr_theta * F.abs(t_a - t_b) / F.least(t_a, t_b)), F.lit(ref.girr_rho_floor)
    )
    # Vega (MAR21.94): rho = delta rho x option-maturity rho, the latter 1 here (one vega
    # factor per bucket). GIRR vega (MAR21.93) with no maturity dimensions: exp(0) = 1.
    # FX has one factor per bucket; two factor ids in one FX bucket are the same rate.
    return (
        F.when((rc == "GIRR") & (comp == "DELTA"), girr)
        .when(rc == "EQ", _map(ref.eq_rho)[F.col("bucket")])
        .otherwise(F.lit(1.0))
    )


def _gamma(ref: FrtbSaRef) -> Column:
    """Medium gamma between buckets ``b`` and ``c`` (MAR21.50, 21.80, 21.89, 21.95)."""
    rc = F.col("risk_class")
    b, c = F.col("b").cast("int"), F.col("c").cast("int")
    eq = (
        F.when((b == 11) | (c == 11), ref.eq_gamma_with_11)
        .when(b.between(1, 10) & c.between(1, 10), ref.eq_gamma_1_to_10)
        .when(((b == 12) & (c == 13)) | ((b == 13) & (c == 12)), ref.eq_gamma_12_13)
        .otherwise(ref.eq_gamma_otherwise)
    )
    return F.when(rc == "GIRR", ref.girr_gamma).when(rc == "EQ", eq).when(rc == "FX", ref.fx_gamma)


def _scenarios(sens: DataFrame) -> DataFrame:
    spark = sens.sparkSession
    return F.broadcast(spark.createDataFrame([(s,) for s in SCENARIOS], "corr_scenario string"))


def _clean(sens: DataFrame) -> DataFrame:
    return sens.fillna(0.0, subset=list(SENSITIVITY_COLUMNS))


# --- the method ---------------------------------------------------------------------------


def weighted_sensitivities(sens: DataFrame, node_cols: list[str], ref: FrtbSaRef) -> DataFrame:
    """Net per node x risk factor (MAR21.4(2)), then ``weighted_sensitivity = s x RW``.

    Columns: node_cols, risk_class, component (DELTA|VEGA), bucket, risk_factor_id,
    tenor_years, sensitivity, risk_weight, weighted_sensitivity. Null sensitivities are 0.
    """
    keys = [*node_cols, "risk_class", "component", "bucket", "risk_factor_id", "tenor_years"]
    s = F.when(F.col("factor_type") == "DELTA", F.col("delta")).otherwise(F.col("vega"))
    netted = (
        _clean(sens)
        .filter(F.col("factor_type").isin("DELTA", "VEGA"))
        .withColumn("component", F.col("factor_type"))
        .groupBy(*keys)
        .agg(F.sum(s).alias("sensitivity"))
    )
    return netted.withColumn("risk_weight", _risk_weight(ref)).withColumn(
        "weighted_sensitivity", F.col("sensitivity") * F.col("risk_weight")
    )


def _delta_vega_buckets(sens: DataFrame, node_cols: list[str], ref: FrtbSaRef) -> DataFrame:
    """K_b and S_b per node, risk class, DELTA|VEGA, bucket and scenario (MAR21.4(4))."""
    ws = weighted_sensitivities(sens, node_cols, ref).select(
        *node_cols, "risk_class", "component", "bucket", "risk_factor_id", "tenor_years",
        "weighted_sensitivity",
    )  # fmt: skip
    keys = [*node_cols, "risk_class", "component", "bucket"]
    a = ws.withColumnsRenamed(
        {"risk_factor_id": "k_a", "tenor_years": "t_a", "weighted_sensitivity": "ws_a"}
    )
    b = ws.withColumnsRenamed(
        {"risk_factor_id": "k_b_id", "tenor_years": "t_b", "weighted_sensitivity": "ws_b"}
    )
    same = F.col("k_a") == F.col("k_b_id")
    rho = F.when(same, F.lit(1.0)).otherwise(
        _scenario(_intra_rho(ref), F.col("corr_scenario"), ref)
    )
    return (
        a.join(b, keys)
        .crossJoin(_scenarios(sens))
        .groupBy(*keys, "corr_scenario")
        .agg(
            F.sum(rho * F.col("ws_a") * F.col("ws_b")).alias("_k2"),
            F.sum(F.when(same, F.col("ws_a")).otherwise(0.0)).alias("s_b"),
        )
        .select(
            *keys,
            "corr_scenario",
            F.sqrt(F.greatest(F.lit(0.0), F.col("_k2"))).alias("k_b"),
            "s_b",
        )
    )


def _curvature_buckets(sens: DataFrame, node_cols: list[str]) -> DataFrame:
    """K_b and S_b per node, risk class, bucket and scenario for curvature (MAR21.5(3)).

    With one curvature factor per bucket ``K_b^+ = max(CVR^+, 0)`` (the rho term vanishes),
    ``K_b = max(K_b^+, K_b^-)``; up is selected when ``K_b^+ > K_b^-``, or on a tie when
    ``CVR^+ > CVR^-``; ``S_b`` is the selected CVR. Independent of the scenario.
    """
    keys = [*node_cols, "risk_class", "bucket"]
    up, down = F.col("cvr_up"), F.col("cvr_down")
    k_up, k_down = F.greatest(up, F.lit(0.0)), F.greatest(down, F.lit(0.0))
    up_selected = (k_up > k_down) | ((k_up == k_down) & (up > down))
    return (
        _clean(sens)
        .filter(
            (F.col("factor_type") == "DELTA")
            & ((F.col("curvature_up") != 0) | (F.col("curvature_down") != 0))
        )
        .groupBy(*keys)
        .agg(F.sum("curvature_up").alias("cvr_up"), F.sum("curvature_down").alias("cvr_down"))
        .crossJoin(_scenarios(sens))
        .select(
            *node_cols,
            "risk_class",
            F.lit("CURVATURE").alias("component"),
            "bucket",
            "corr_scenario",
            F.greatest(k_up, k_down).alias("k_b"),
            F.when(up_selected, up).otherwise(down).alias("s_b"),
        )
    )


def bucket_detail(sens: DataFrame, node_cols: list[str], ref: FrtbSaRef) -> DataFrame:
    """Columns node_cols, risk_class, component, bucket, corr_scenario, k_b, s_b.

    ``s_b`` is the plain sum of weighted sensitivities (delta, vega) or the selected CVR
    (curvature), before any clamping by the alternative specification.
    """
    cols = [*node_cols, "risk_class", "component", "bucket", "corr_scenario", "k_b", "s_b"]
    return (
        _delta_vega_buckets(sens, node_cols, ref)
        .select(*cols)
        .unionByName(_curvature_buckets(sens, node_cols).select(*cols))
    )


def sbm_charges(sens: DataFrame, node_cols: list[str], ref: FrtbSaRef) -> DataFrame:
    """Columns node_cols, risk_class, component, corr_scenario, charge.

    One row per node x risk class x component that has sensitivities, per scenario. ``sens``
    must already carry ``node_cols`` (see ``attach_org``); ``[]`` is the whole portfolio.
    """
    return charges_from_detail(bucket_detail(sens, node_cols, ref), ref)


def charges_from_detail(detail: DataFrame, ref: FrtbSaRef) -> DataFrame:
    """The across-bucket step of ``sbm_charges`` from an (optionally cached) ``bucket_detail``.

    Lets a run compute the bucket detail once and use it for both the charges and the curated
    detail table.
    """
    node_cols = [
        c
        for c in detail.columns
        if c not in ("risk_class", "component", "bucket", "corr_scenario", "k_b", "s_b")
    ]
    keys = [*node_cols, "risk_class", "component", "corr_scenario"]
    b = detail.withColumnsRenamed({"bucket": "b", "k_b": "k1", "s_b": "s1"})
    c = detail.withColumnsRenamed({"bucket": "c", "k_b": "k2", "s_b": "s2"})
    curv = F.col("component") == "CURVATURE"
    # MAR21.101: curvature gamma is the square of the delta gamma; scenarios apply after.
    gamma_medium = F.when(curv, F.pow(_gamma(ref), 2)).otherwise(_gamma(ref))
    gamma = _scenario(gamma_medium, F.col("corr_scenario"), ref)
    # MAR21.5(4)(b): psi = 0 when both S are negative (curvature only).
    psi = F.when(curv & (F.col("s1") < 0) & (F.col("s2") < 0), 0.0).otherwise(1.0)
    same = F.col("b") == F.col("c")
    s1_alt = F.greatest(F.least(F.col("s1"), F.col("k1")), -F.col("k1"))
    s2_alt = F.greatest(F.least(F.col("s2"), F.col("k2")), -F.col("k2"))
    k2_term = F.col("k1") * F.col("k1")
    term = F.when(same, k2_term).otherwise(gamma * F.col("s1") * F.col("s2") * psi)
    term_alt = F.when(same, k2_term).otherwise(gamma * s1_alt * s2_alt)
    total, total_alt = F.col("_total"), F.col("_total_alt")
    # MAR21.4(5)(b): delta/vega use the clamped S_b when the sum is negative. Curvature has no
    # alternative; its sum is floored at 0 (MAR21.5(4)). MAR21.4(5) states no floor for the
    # alternative sum; the floor here only keeps the square root defined.
    chosen = F.when(~curv & (total < 0), total_alt).otherwise(total)
    return (
        b.join(c, keys)
        .groupBy(*keys)
        .agg(F.sum(term).alias("_total"), F.sum(term_alt).alias("_total_alt"))
        .select(*keys, F.sqrt(F.greatest(F.lit(0.0), chosen)).alias("charge"))
    )


def sbm_total(charges: DataFrame) -> DataFrame:
    """Columns node_cols, corr_scenario, charge, is_binding (MAR21.7).

    Per node and scenario the charges of every risk class and component are summed; the
    binding scenario is the largest total, ties going to LOW, then MEDIUM, then HIGH.
    """
    node_cols = [
        c
        for c in charges.columns
        if c not in ("risk_class", "component", "corr_scenario", "charge")
    ]
    order = F.array_position(F.array(*[F.lit(s) for s in SCENARIOS]), F.col("corr_scenario"))
    w = Window.partitionBy(*node_cols) if node_cols else Window.partitionBy(F.lit(1))
    totals = charges.groupBy(*node_cols, "corr_scenario").agg(F.sum("charge").alias("charge"))
    rank = F.row_number().over(w.orderBy(F.col("charge").desc(), order))
    return totals.select(*node_cols, "corr_scenario", "charge", (rank == 1).alias("is_binding"))
