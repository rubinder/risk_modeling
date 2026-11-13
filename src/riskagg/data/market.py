"""Market-data generators: factor returns, exposures, P&L vectors and sensitivities.

Units: an exposure is the P&L per unit move of its risk factor, so a trade's P&L in
scenario ``s`` is ``sum(exposure_f * return_f[s])``.
"""

import zlib
from collections.abc import Iterator

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from riskagg.data.org import EQ_CCY, GIRR_TENORS, _pick, _uniform
from riskagg.data.profiles import Profile

# SIMPLIFICATION: one common factor per risk class plus one market-wide factor; a single
# correlation level per window instead of an estimated correlation matrix.
CLASS_CORRELATION = {"CURRENT": 0.5, "STRESSED": 0.8}
# SIMPLIFICATION: the stressed window is the current factor model with volatility tripled and
# correlation raised, not a historical 12-month period of significant stress chosen by the bank.
VOL_MULTIPLIER = {"CURRENT": 1.0, "STRESSED": 3.0}
MARKET_WEIGHT = 0.3  # share of a class factor's variance that is market-wide
OPTION_SHARE = 0.3
LONG_SHARE = 0.7
VEGA_SCALE = 0.05
# SIMPLIFICATION: curvature uses a symmetric quadratic repricing at one shock per risk class
# (GIRR is an absolute rate move, EQ and FX relative moves) instead of full revaluation.
CURVATURE_SHOCK = {"GIRR": 0.017, "EQ": 0.30, "FX": 0.15}
_EQ_BUCKET_BY_CCY = {ccy: bucket for bucket, ccy in EQ_CCY.items()}


def factor_returns(p: Profile, risk_factors: pd.DataFrame, window: str) -> np.ndarray:
    """Correlated one-day returns, shape ``(n_factors, n_scenarios)``.

    Rows follow ``risk_factor_id`` sorted ascending. A market factor, a factor per risk class
    and an idiosyncratic term drive each factor; STRESSED triples volatility and raises
    correlation. Seeded from the profile seed and the window name.
    """
    rf = risk_factors.sort_values("risk_factor_id").reset_index(drop=True)
    rng = np.random.default_rng([p.seed, zlib.crc32(window.encode())])
    rho = CLASS_CORRELATION.get(window, CLASS_CORRELATION["CURRENT"])
    mult = VOL_MULTIPLIER.get(window, 1.0)
    classes = sorted(rf["risk_class"].unique())
    market = rng.standard_normal(p.n_scenarios)
    class_z = {c: rng.standard_normal(p.n_scenarios) for c in classes}
    idio = rng.standard_normal((len(rf), p.n_scenarios))
    common = np.stack(
        [
            np.sqrt(MARKET_WEIGHT) * market + np.sqrt(1 - MARKET_WEIGHT) * class_z[c]
            for c in rf["risk_class"]
        ]
    )
    z = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
    return rf["daily_vol"].to_numpy()[:, None] * mult * z


def make_exposures(
    spark: SparkSession, p: Profile, trades: DataFrame, risk_factors: DataFrame
) -> DataFrame:
    """Columns ``trade_id, book_id, risk_factor_id, exposure`` (book_id feeds sensitivities)."""
    del spark, risk_factors  # factor ids follow the naming rules of make_risk_factors

    def u(salt: str) -> Column:
        return _uniform(F.col("trade_id"), p.seed, salt)

    ccy = F.col("currency")
    asset = F.col("asset_class")
    notional = F.col("notional")
    sign = F.when(u("sign") < LONG_SHARE, 1.0).otherwise(-1.0)
    tenor = _pick(GIRR_TENORS, F.floor(u("tenor") * len(GIRR_TENORS)))
    eq_bucket = F.create_map(
        *[x for c, b in _EQ_BUCKET_BY_CCY.items() for x in (F.lit(c), F.lit(b))]
    )[ccy]
    is_option = u("option") < OPTION_SHARE

    t = trades.select(
        "trade_id",
        "book_id",
        asset.alias("asset_class"),
        ccy.alias("currency"),
        notional.alias("notional"),
        sign.alias("sign"),
        tenor.alias("tenor"),
        eq_bucket.alias("eq_bucket"),
        is_option.alias("is_option"),
    )
    primary_id = (
        F.when(asset == "RATES", F.format_string("GIRR.%s.%dY", ccy, F.col("tenor").cast("int")))
        .when(asset == "EQUITY", F.concat(F.lit("EQ."), F.col("eq_bucket")))
        .otherwise(F.concat(F.lit("FX."), ccy))
    )
    primary_exposure = F.when(asset == "RATES", F.col("notional") * F.col("tenor")).otherwise(
        F.col("notional")
    )
    vega_id = (
        F.when(asset == "RATES", F.format_string("GIRR.%s.VEGA", ccy))
        .when(asset == "EQUITY", F.concat(F.lit("EQ."), F.col("eq_bucket"), F.lit(".VEGA")))
        .otherwise(F.concat(F.lit("FX."), ccy, F.lit(".VEGA")))
    )
    cols = ["trade_id", "book_id"]
    parts = [
        t.select(
            *cols,
            primary_id.alias("risk_factor_id"),
            (F.col("sign") * primary_exposure).alias("exposure"),
        ),
        t.filter(asset.isin("RATES", "EQUITY") & (ccy != "USD")).select(
            *cols,
            F.concat(F.lit("FX."), ccy).alias("risk_factor_id"),
            (F.col("sign") * F.col("notional")).alias("exposure"),
        ),
        t.filter(F.col("is_option")).select(
            *cols,
            vega_id.alias("risk_factor_id"),
            (F.col("sign") * F.col("notional") * VEGA_SCALE).alias("exposure"),
        ),
    ]
    out = parts[0].unionByName(parts[1]).unionByName(parts[2])
    return out.select("trade_id", "book_id", "risk_factor_id", "exposure")


def make_pnl_vectors(
    spark: SparkSession,
    p: Profile,
    trades: DataFrame,
    exposures: DataFrame,
    risk_factors: DataFrame,
) -> DataFrame:
    """One row per trade x (risk_class, liquidity_horizon, in_reduced_set) x window."""
    # SIMPLIFICATION: scenario P&L is linear in the factor returns (exposure x return), not a
    # full revaluation of each trade under each historical scenario.
    rf_pdf = risk_factors.toPandas().sort_values("risk_factor_id").reset_index(drop=True)
    returns = {w: factor_returns(p, rf_pdf, w) for w in p.windows}
    rf_idx = spark.createDataFrame(
        rf_pdf.assign(factor_idx=np.arange(len(rf_pdf), dtype="int32"))[
            ["risk_factor_id", "risk_class", "liquidity_horizon", "in_reduced_set", "factor_idx"]
        ]
    )
    meta = trades.select("trade_id", "asset_class", "currency")
    groups = (
        exposures.join(F.broadcast(rf_idx), "risk_factor_id")
        .join(meta, "trade_id")
        .groupBy(
            "trade_id",
            "book_id",
            "asset_class",
            "currency",
            "risk_class",
            "liquidity_horizon",
            "in_reduced_set",
        )
        .agg(
            F.sort_array(F.collect_list(F.struct("factor_idx", "exposure"))).alias("legs"),
        )
        .select(
            "trade_id",
            "book_id",
            "asset_class",
            "currency",
            "risk_class",
            "liquidity_horizon",
            "in_reduced_set",
            F.transform("legs", lambda x: x["factor_idx"]).alias("idx"),
            F.transform("legs", lambda x: x["exposure"]).alias("exp"),
        )
        .withColumn("window", F.explode(F.array(*[F.lit(w) for w in p.windows])))
    )
    schema = (
        "trade_id long, book_id string, asset_class string, currency string, "
        "risk_class string, liquidity_horizon int, in_reduced_set boolean, "
        "window string, pnl array<double>"
    )
    keep = [f.name for f in groups.schema.fields if f.name not in ("idx", "exp")]

    def run(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in batches:
            pnl = [
                np.asarray(e) @ returns[w][np.asarray(i, dtype=int)]
                for i, e, w in zip(pdf["idx"], pdf["exp"], pdf["window"], strict=True)
            ]
            out = pdf[keep].copy()
            out["pnl"] = pnl
            yield out[[*keep, "pnl"]]

    return groups.mapInPandas(run, schema)


def make_sensitivities(
    spark: SparkSession, p: Profile, exposures: DataFrame, risk_factors: DataFrame
) -> DataFrame:
    """Per trade x risk factor sensitivities; options are trades with a VEGA exposure."""
    del spark
    rf = F.broadcast(
        risk_factors.select("risk_factor_id", "risk_class", "factor_type", "bucket", "tenor_years")
    )
    df = exposures.join(rf, "risk_factor_id")
    is_delta = F.col("factor_type") == "DELTA"
    is_option = (
        F.max((F.col("factor_type") == "VEGA").cast("int")).over(Window.partitionBy("trade_id"))
        == 1
    )
    df = df.withColumn("is_option", is_option)
    key = F.concat_ws("|", "trade_id", "risk_factor_id")
    shock = F.create_map(*[x for c, s in CURVATURE_SHOCK.items() for x in (F.lit(c), F.lit(s))])[
        F.col("risk_class")
    ]
    has_gamma = is_delta & F.col("is_option")
    gamma_sign = F.when(_uniform(key, p.seed, "gamma_sign") < 0.5, 1.0).otherwise(-1.0)
    gamma = F.when(
        has_gamma,
        gamma_sign * F.abs("exposure") * (F.lit(1.0) + _uniform(key, p.seed, "gamma_size")) * 2.0,
    ).otherwise(0.0)
    df = df.withColumn("gamma", gamma)
    curvature = F.when(
        has_gamma, F.greatest(F.lit(0.0), -0.5 * F.col("gamma") * shock * shock)
    ).otherwise(0.0)
    return df.select(
        "trade_id",
        "book_id",
        "risk_factor_id",
        "risk_class",
        "factor_type",
        "bucket",
        "tenor_years",
        F.when(is_delta, F.col("exposure")).otherwise(0.0).alias("delta"),
        "gamma",
        F.when(~is_delta, F.col("exposure")).otherwise(0.0).alias("vega"),
        curvature.alias("curvature_up"),
        curvature.alias("curvature_down"),
    )
