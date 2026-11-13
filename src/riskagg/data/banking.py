"""Banking-book style positions derived from the trades (default risk positions only so far)."""

import random

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

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
