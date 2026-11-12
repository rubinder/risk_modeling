"""Organisation hierarchy, risk-factor universe and trade generators."""

import datetime
import random

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from riskagg.data.profiles import Profile

MAJOR_CCY = ("USD", "EUR", "GBP", "JPY")
OTHER_CCY = ("BRL", "ZAR")
ALL_CCY = MAJOR_CCY + OTHER_CCY
FX_BUCKETS = tuple(c for c in ALL_CCY if c != "USD")
GIRR_TENORS = (1, 2, 5, 10, 30)
EQ_BUCKETS = ("1", "5", "8", "12")
EQ_CCY = {"1": "USD", "5": "EUR", "8": "JPY", "12": "GBP"}
EQ_DELTA_LH = {"1": 20, "5": 10, "8": 20, "12": 10}
EQ_VEGA_LH = {"1": 60, "5": 20, "8": 60, "12": 20}
FX_VEGA_LH = 40
REGIONS = ("EMEA", "AMER", "APAC")
N_COUNTERPARTIES = 50
REDUCED_SET_SHARE = 0.8


def _array(values) -> Column:
    return F.array(*[F.lit(v) for v in values])


def _pick(values, index: Column) -> Column:
    """Element of a literal array at a 0-based index column."""
    return F.element_at(_array(values), index.cast("int") + 1)


def _uniform(key: Column, seed: int, salt: str) -> Column:
    """Deterministic U[0,1) from a key, independent of partitioning."""
    h = F.xxhash64(key, F.lit(seed), F.lit(salt))
    return F.pmod(h, F.lit(1_000_000)) / 1_000_000.0


def _desk_index(book_idx: int, p: Profile) -> int:
    return book_idx * p.n_desks // p.n_books


def _desk_index_col(p: Profile) -> Column:
    """Desk index of the book numbered by the ``id`` column (integer division)."""
    return F.expr(f"(id * {p.n_desks}) div {p.n_books}")


def _bl_index_col(p: Profile) -> Column:
    return F.expr(f"(((id * {p.n_desks}) div {p.n_books}) * {p.n_business_lines}) div {p.n_desks}")


def make_hierarchy(spark: SparkSession, p: Profile) -> DataFrame:
    n_legal_entities = 2 if p.name == "tiny" else 4
    i = F.col("id")
    desk = _desk_index_col(p)
    bl = _bl_index_col(p)
    return spark.range(p.n_books).select(
        F.format_string("BOOK%04d", i + 1).alias("book_id"),
        F.format_string("Book %d", i + 1).alias("book_name"),
        F.format_string("DESK%03d", desk + 1).alias("desk_id"),
        F.format_string("Desk %d", desk + 1).alias("desk_name"),
        F.format_string("BL%02d", bl + 1).alias("business_line_id"),
        F.format_string("Business Line %d", bl + 1).alias("business_line_name"),
        F.lit("FIRM").alias("firm_id"),
        F.format_string("LE%02d", (i % n_legal_entities) + 1).alias("legal_entity"),
        _pick(REGIONS, i % len(REGIONS)).alias("region"),
    )


_RF_SCHEMA = T.StructType(
    [
        T.StructField("risk_factor_id", T.StringType(), False),
        T.StructField("risk_class", T.StringType(), False),
        T.StructField("factor_type", T.StringType(), False),
        T.StructField("bucket", T.StringType(), False),
        T.StructField("tenor_years", T.DoubleType(), True),
        T.StructField("currency", T.StringType(), False),
        T.StructField("liquidity_horizon", T.IntegerType(), False),
        T.StructField("in_reduced_set", T.BooleanType(), False),
        T.StructField("daily_vol", T.DoubleType(), False),
    ]
)

_VOL_RANGE = {"GIRR": (0.0005, 0.002), "EQ": (0.01, 0.03), "FX": (0.004, 0.012)}


def _risk_factor_specs() -> list[tuple]:
    """(id, class, type, bucket, tenor, currency, liquidity_horizon) in a fixed order."""
    specs: list[tuple] = []
    for ccy in ALL_CCY:
        lh = 10 if ccy in MAJOR_CCY else 20
        for tenor in GIRR_TENORS:
            specs.append((f"GIRR.{ccy}.{tenor}Y", "GIRR", "DELTA", ccy, float(tenor), ccy, lh))
    for bucket in EQ_BUCKETS:
        ccy = EQ_CCY[bucket]
        specs.append((f"EQ.{bucket}", "EQ", "DELTA", bucket, None, ccy, EQ_DELTA_LH[bucket]))
    for ccy in FX_BUCKETS:
        lh = 10 if ccy in MAJOR_CCY else 20
        specs.append((f"FX.{ccy}", "FX", "DELTA", ccy, None, ccy, lh))
    for ccy in ALL_CCY:  # GIRR vega uses the bucket's delta horizon
        lh = 10 if ccy in MAJOR_CCY else 20
        specs.append((f"GIRR.{ccy}.VEGA", "GIRR", "VEGA", ccy, None, ccy, lh))
    for bucket in EQ_BUCKETS:
        specs.append(
            (f"EQ.{bucket}.VEGA", "EQ", "VEGA", bucket, None, EQ_CCY[bucket], EQ_VEGA_LH[bucket])
        )
    for ccy in FX_BUCKETS:
        specs.append((f"FX.{ccy}.VEGA", "FX", "VEGA", ccy, None, ccy, FX_VEGA_LH))
    return specs


def make_risk_factors(spark: SparkSession, p: Profile) -> DataFrame:
    rng = random.Random(p.seed)
    rows = []
    seen_delta_reduced: set[str] = set()
    for spec in _risk_factor_specs():
        _, risk_class, factor_type = spec[:3]
        lo, hi = _VOL_RANGE[risk_class]
        reduced = rng.random() < REDUCED_SET_SHARE
        vol = round(rng.uniform(lo, hi), 6)
        # Guarantee a reduced-set DELTA factor per class: the first DELTA factor of each class.
        if factor_type == "DELTA" and risk_class not in seen_delta_reduced:
            reduced = True
        if factor_type == "DELTA" and reduced:
            seen_delta_reduced.add(risk_class)
        rows.append((*spec, reduced, vol))
    return spark.createDataFrame(rows, _RF_SCHEMA)


def make_trades(spark: SparkSession, p: Profile, as_of_date: datetime.date) -> DataFrame:
    tid = F.col("id")

    def u(salt: str) -> Column:
        return _uniform(tid, p.seed, salt)

    books_in_hot_desk = sum(1 for b in range(p.n_books) if _desk_index(b, p) == 0)
    if p.skew_desk_share > 0 and books_in_hot_desk < p.n_books:
        in_hot = u("skew") < p.skew_desk_share
        hot = F.floor(u("book") * books_in_hot_desk)
        rest = books_in_hot_desk + F.floor(u("book") * (p.n_books - books_in_hot_desk))
        book_idx = F.when(in_hot, hot).otherwise(rest)
    else:
        book_idx = F.floor(u("book") * p.n_books)

    ccy_u = u("ccy")
    asset_u = u("asset")
    asset_class = F.when(asset_u < 0.4, "RATES").when(asset_u < 0.7, "EQUITY").otherwise("FX")
    currency = (
        F.when(asset_class == "FX", _pick(FX_BUCKETS, F.floor(ccy_u * len(FX_BUCKETS))))
        .when(asset_class == "EQUITY", _pick(MAJOR_CCY, F.floor(ccy_u * len(MAJOR_CCY))))
        .otherwise(_pick(ALL_CCY, F.floor(ccy_u * len(ALL_CCY))))
    )
    return spark.range(p.n_trades).select(
        tid.alias("trade_id"),
        F.format_string("BOOK%04d", book_idx.cast("int") + 1).alias("book_id"),
        asset_class.alias("asset_class"),
        currency.alias("currency"),
        F.format_string("CP%03d", F.floor(u("cpty") * N_COUNTERPARTIES).cast("int") + 1).alias(
            "counterparty_id"
        ),
        F.round(F.lit(1_000_000.0) * F.exp(F.lit(3.0) * u("notional")), 2).alias("notional"),
        (u("exotic") < 0.05).alias("is_exotic"),
        F.lit(as_of_date).alias("as_of_date"),
    )
