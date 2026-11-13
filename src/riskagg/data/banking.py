"""Banking-book data: default risk positions derived from the trades (FRTB), and the loan
segments, macro scenarios and starting capital of the CCAR projection."""

import random

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from riskagg.data.org import _pick, _uniform
from riskagg.data.profiles import Profile

N_CORPORATE_ISSUERS = 24
N_SOVEREIGN_ISSUERS = 6
N_ISSUERS = N_CORPORATE_ISSUERS + N_SOVEREIGN_ISSUERS
_CORPORATE_RATINGS = {"AAA": 1, "AA": 2, "A": 4, "BBB": 6, "BB": 4, "B": 2, "CCC": 1, "UNRATED": 2}
_SOVEREIGN_RATINGS = {"AAA": 2, "AA": 2, "A": 2, "BBB": 2, "BB": 1, "B": 1}
DEBT_SHARE = 0.5  # of RATES trades that carry a default risk position
SOVEREIGN_SHARE = 0.3  # of debt positions
SENIOR_SHARE = 0.7  # of debt positions
LONG_SHARE = 0.6


def issuer_table(seed: int) -> list[tuple[str, str, str]]:
    """``(issuer_id, drc_bucket, rating)``: corporates first, then sovereigns; one rating each."""
    rng = random.Random(f"issuers-{seed}")
    rows = []
    for i in range(N_ISSUERS):
        corporate = i < N_CORPORATE_ISSUERS
        weights = _CORPORATE_RATINGS if corporate else _SOVEREIGN_RATINGS
        rating = rng.choices(list(weights), weights=list(weights.values()))[0]
        rows.append((f"ISS{i + 1:03d}", "CORPORATE" if corporate else "SOVEREIGN", rating))
    return rows


def make_default_risk_positions(spark: SparkSession, p: Profile, trades: DataFrame) -> DataFrame:
    """Default risk positions derived from ``trades``, deterministically (hash of the trade id).

    EQUITY trades become EQUITY-seniority positions on corporate issuers; a share of RATES
    trades become SENIOR or NON_SENIOR debt positions on corporate or sovereign issuers; FX
    trades have none. Columns: trade_id, book_id, issuer_id, drc_bucket, rating, seniority,
    direction, notional.
    """
    del spark
    issuers = issuer_table(p.seed)

    def u(salt: str):
        return _uniform(F.col("trade_id"), p.seed, salt)

    asset = F.col("asset_class")
    is_equity = asset == "EQUITY"
    is_debt = (asset == "RATES") & (u("drc") < DEBT_SHARE)
    sovereign = is_debt & (u("sovereign") < SOVEREIGN_SHARE)
    idx = (
        F.when(sovereign, N_CORPORATE_ISSUERS + F.floor(u("issuer") * N_SOVEREIGN_ISSUERS))
        .otherwise(F.floor(u("issuer") * N_CORPORATE_ISSUERS))
        .cast("int")
    )
    seniority = (
        F.when(is_equity, "EQUITY")
        .when(u("seniority") < SENIOR_SHARE, "SENIOR")
        .otherwise("NON_SENIOR")
    )
    return trades.where(is_equity | is_debt).select(
        "trade_id",
        "book_id",
        _pick([i[0] for i in issuers], idx).alias("issuer_id"),
        _pick([i[1] for i in issuers], idx).alias("drc_bucket"),
        _pick([i[2] for i in issuers], idx).alias("rating"),
        seniority.alias("seniority"),
        F.when(u("direction") < LONG_SHARE, "LONG").otherwise("SHORT").alias("direction"),
        "notional",
    )


# --- CCAR -------------------------------------------------------------------------------------

BASELINE, SEVERELY_ADVERSE = "BASELINE", "SEVERELY_ADVERSE"
# Quarter 0 is the starting point (the last actual quarter); 1..9 are the projected quarters.
# Unemployment rate in percent; real GDP growth annualised, as a decimal (-0.08 = -8%).
# SIMPLIFICATION: two hand-set paths of two macro variables, not the Federal Reserve's
# published supervisory scenarios (which have many more domestic and international variables).
MACRO_PATHS: dict[str, tuple[tuple[float, float], ...]] = {
    BASELINE: (
        (4.0, 0.020), (4.1, 0.019), (4.2, 0.018), (4.2, 0.018), (4.3, 0.019),
        (4.3, 0.020), (4.3, 0.020), (4.2, 0.021), (4.2, 0.021), (4.1, 0.021),
    ),
    SEVERELY_ADVERSE: (
        (4.0, 0.020), (5.5, -0.060), (7.0, -0.080), (8.3, -0.050), (9.3, -0.020),
        (10.0, 0.000), (10.0, 0.010), (9.7, 0.020), (9.3, 0.025), (8.9, 0.025),
    ),
}  # fmt: skip

# Loan segment types: ranges (low, high) of annual PD, LGD, risk weight and quarterly revenue
# and expense rates (fractions of EAD), and each type's share of a segment pair's exposure.
# SIMPLIFICATION: three stylised portfolios with uniform random parameters; a bank models
# hundreds of segments with loss models per product (PD/LGD/EAD by vintage, collateral, ...).
SEGMENT_TYPES: dict[str, dict[str, tuple[float, float] | float]] = {
    "CORPORATE": {"share": 0.50, "pd": (0.010, 0.030), "lgd": (0.35, 0.45), "rw": (0.90, 1.00),
                  "revenue": (0.0060, 0.0080), "expense": (0.0030, 0.0040)},
    "MORTGAGE": {"share": 0.38, "pd": (0.005, 0.015), "lgd": (0.15, 0.25), "rw": (0.35, 0.50),
                 "revenue": (0.0040, 0.0060), "expense": (0.0020, 0.0030)},
    "CARDS": {"share": 0.12, "pd": (0.030, 0.060), "lgd": (0.75, 0.90), "rw": (0.90, 1.00),
              "revenue": (0.0250, 0.0350), "expense": (0.0100, 0.0140)},
}  # fmt: skip
EAD_PER_TRADE = 20_000_000.0  # banking-book exposure per trade of the profile: about 3x trading
RWA_TRADING_PER_TRADE = 5_000_000.0  # market-risk RWA per trade of the profile
CET1_TARGET = (0.11, 0.13)  # starting CET1 ratio range
DIVIDEND_RATE_Q = (0.005, 0.008)  # planned quarterly common dividend, fraction of CET1

_MACRO_SCHEMA = T.StructType(
    [
        T.StructField("scenario", T.StringType(), False),
        T.StructField("quarter", T.IntegerType(), False),
        T.StructField("unemployment_rate", T.DoubleType(), False),
        T.StructField("gdp_growth", T.DoubleType(), False),
    ]
)


def make_macro_scenarios(spark: SparkSession, p: Profile) -> DataFrame:
    """``(scenario, quarter 0..9, unemployment_rate, gdp_growth)``; the same for every profile."""
    del p
    rows = [
        (scenario, q, u, g)
        for scenario, path in MACRO_PATHS.items()
        for q, (u, g) in enumerate(path)
    ]
    return spark.createDataFrame(rows, _MACRO_SCHEMA)


def _between(u: Column, bounds: tuple[float, float], digits: int) -> Column:
    lo, hi = bounds
    return F.round(F.lit(lo) + F.lit(hi - lo) * u, digits)


def _books_per(hierarchy: DataFrame, *keys: str) -> DataFrame:
    return hierarchy.groupBy(*keys).agg(F.count(F.lit(1)).alias("_books"))


def make_banking_book(spark: SparkSession, p: Profile, hierarchy: DataFrame) -> DataFrame:
    """One loan segment per segment type and (legal entity, business line) pair that has books.

    A pair's exposure is ``EAD_PER_TRADE x n_trades`` times its share of the books, split by the
    type's share and scaled by a draw in [0.5, 1.5). Every draw is ``_uniform`` on the segment
    id with the profile seed. Columns: segment_id, legal_entity, business_line_id, ead,
    pd_base (annual), lgd, revenue_rate_q, expense_rate_q, risk_weight.
    """
    n_books = hierarchy.count()
    types = spark.createDataFrame(
        [(t, float(v["share"])) for t, v in SEGMENT_TYPES.items()],
        "segment_type string, share double",
    )
    pairs = _books_per(hierarchy, "legal_entity", "business_line_id").crossJoin(types)
    seg = F.concat_ws("|", "legal_entity", "business_line_id", "segment_type")

    def u(salt: str) -> Column:
        return _uniform(F.col("segment_id"), p.seed, salt)

    def param(name: str, digits: int) -> Column:
        col = None
        for t, v in SEGMENT_TYPES.items():
            value = _between(u(name), v[name], digits)
            cond = F.col("segment_type") == t
            col = F.when(cond, value) if col is None else col.when(cond, value)
        return col

    scale = F.lit(EAD_PER_TRADE * p.n_trades / n_books) * F.col("_books") * F.col("share")
    return (
        pairs.withColumn("segment_id", seg)
        .select(
            "segment_id",
            "legal_entity",
            "business_line_id",
            F.round(scale * (F.lit(0.5) + u("ead")), 2).alias("ead"),
            param("pd", 4).alias("pd_base"),
            param("lgd", 4).alias("lgd"),
            param("revenue", 5).alias("revenue_rate_q"),
            param("expense", 5).alias("expense_rate_q"),
            param("rw", 4).alias("risk_weight"),
        )
        .orderBy("segment_id")
    )


def make_capital_start(spark: SparkSession, p: Profile, hierarchy: DataFrame) -> DataFrame:
    """Starting capital per legal entity: ``(legal_entity, cet1_start, rwa_trading, dividends_q)``.

    ``rwa_trading`` is ``RWA_TRADING_PER_TRADE x n_trades`` times the entity's share of the
    books, scaled by a draw in [0.8, 1.2). ``cet1_start`` is a starting ratio drawn from
    ``CET1_TARGET`` times the entity's total RWA (credit RWA of its banking book plus
    ``rwa_trading``), and ``dividends_q`` a planned quarterly common dividend drawn from
    ``DIVIDEND_RATE_Q`` times ``cet1_start``. Draws are ``_uniform`` on the entity id.
    """
    n_books = hierarchy.count()
    credit = (
        make_banking_book(spark, p, hierarchy)
        .groupBy("legal_entity")
        .agg(F.sum(F.col("ead") * F.col("risk_weight")).alias("_rwa_credit"))
    )

    def u(salt: str) -> Column:
        return _uniform(F.col("legal_entity"), p.seed, salt)

    share = F.col("_books") / F.lit(float(n_books))
    rwa_trading = F.round(
        F.lit(RWA_TRADING_PER_TRADE * p.n_trades) * share * (F.lit(0.8) + F.lit(0.4) * u("rwa")), 2
    )
    cet1 = F.round(
        _between(u("cet1"), CET1_TARGET, 4) * (F.col("_rwa_credit") + F.col("rwa_trading")), 2
    )
    return (
        _books_per(hierarchy, "legal_entity")
        .join(credit, "legal_entity")
        .withColumn("rwa_trading", rwa_trading)
        .withColumn("cet1_start", cet1)
        .select(
            "legal_entity",
            "cet1_start",
            "rwa_trading",
            F.round(_between(u("dividend"), DIVIDEND_RATE_Q, 4) * F.col("cet1_start"), 2).alias(
                "dividends_q"
            ),
        )
        .orderBy("legal_entity")
    )
