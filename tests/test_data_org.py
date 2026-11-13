import dataclasses
import datetime

from pyspark.sql import functions as F

from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES

AS_OF = datetime.date(2026, 11, 12)


def test_profiles_match_brief():
    tiny, demo, bench = PROFILES["tiny"], PROFILES["demo"], PROFILES["bench"]
    assert (tiny.n_business_lines, tiny.n_desks, tiny.n_books) == (2, 3, 6)
    assert (tiny.n_trades, tiny.n_scenarios) == (20, 20)
    assert (demo.n_business_lines, demo.n_desks, demo.n_books) == (4, 20, 200)
    assert (demo.n_trades, demo.n_scenarios) == (100_000, 250)
    assert (bench.n_business_lines, bench.n_desks, bench.n_books) == (4, 40, 400)
    assert (bench.n_trades, bench.n_scenarios, bench.skew_desk_share) == (1_000_000, 250, 0.4)
    assert bench.windows == ("CURRENT",)
    for p in (tiny, demo):
        assert p.windows == ("CURRENT", "STRESSED") and p.skew_desk_share == 0
        assert p.seed == 42


def test_tiny_counts_and_book_attributes(spark):
    p = PROFILES["tiny"]
    h = make_hierarchy(spark, p)
    assert h.count() == 6
    assert h.select("book_id").distinct().count() == 6
    assert h.select("legal_entity").distinct().count() == 2
    per_book = h.groupBy("book_id").agg(
        F.countDistinct("legal_entity").alias("le"), F.countDistinct("region").alias("rg")
    )
    assert per_book.filter((F.col("le") != 1) | (F.col("rg") != 1)).count() == 0
    assert h.select("desk_id").distinct().count() == 3
    assert h.select("business_line_id").distinct().count() == 2
    assert make_trades(spark, p, AS_OF).count() == 20


def test_trades_reference_known_books_and_are_deterministic(spark):
    p = PROFILES["tiny"]
    t1 = make_trades(spark, p, AS_OF).collect()
    t2 = make_trades(spark.newSession(), p, AS_OF).repartition(5).orderBy("trade_id").collect()
    assert sorted(t1, key=lambda r: r.trade_id) == t2
    books = {r.book_id for r in make_hierarchy(spark, p).collect()}
    assert {r.book_id for r in t1} <= books
    assert all(r.as_of_date == AS_OF for r in t1)
    assert make_risk_factors(spark, p).collect() == make_risk_factors(spark, p).collect()


def test_skew_on_bench_override(spark):
    p = dataclasses.replace(PROFILES["bench"], n_trades=10_000)
    t = make_trades(spark, p, AS_OF)
    h = make_hierarchy(spark, p).select("book_id", "desk_id")
    shares = t.join(h, "book_id").groupBy("desk_id").count().collect()
    assert 0.38 <= max(r["count"] for r in shares) / 10_000 <= 0.42


def test_risk_factor_rules(spark):
    rf = make_risk_factors(spark, PROFILES["tiny"])
    rows = rf.collect()
    assert len({r.risk_factor_id for r in rows}) == len(rows)
    girr = [r for r in rows if r.risk_class == "GIRR" and r.factor_type == "DELTA"]
    assert len(girr) == 6 * 5 and {r.tenor_years for r in girr} == {1, 2, 5, 10, 30}
    # MAR33.12 Table 2. GIRR: specified currencies 10, unspecified (BRL, ZAR) 20.
    for r in girr:
        assert r.liquidity_horizon == (10 if r.bucket in {"USD", "EUR", "GBP", "JPY"} else 20)
    # Equity buckets 1, 5, 8 and 12 are all large cap or large-cap indices: price 10.
    eq = {r.bucket: r for r in rows if r.risk_class == "EQ" and r.factor_type == "DELTA"}
    assert {b: x.liquidity_horizon for b, x in eq.items()} == {"1": 10, "5": 10, "8": 10, "12": 10}
    # FX: USD/EUR, USD/GBP, USD/JPY, USD/BRL and USD/ZAR are all specified pairs: 10.
    fx = [r for r in rows if r.risk_class == "FX" and r.factor_type == "DELTA"]
    assert {r.bucket for r in fx} == {"EUR", "GBP", "JPY", "BRL", "ZAR"}
    assert {r.liquidity_horizon for r in fx} == {10}
    # Volatilities: interest rate 60, equity (large cap) 20, FX 40.
    vega = [r for r in rows if r.factor_type == "VEGA"]
    assert all(r.tenor_years is None for r in vega)
    assert {r.liquidity_horizon for r in vega if r.risk_class == "GIRR"} == {60}
    assert {r.liquidity_horizon for r in vega if r.risk_class == "EQ"} == {20}
    assert {r.liquidity_horizon for r in vega if r.risk_class == "FX"} == {40}
    assert {r.liquidity_horizon for r in rows} == {10, 20, 40, 60}
    for cls in ("GIRR", "EQ", "FX"):
        assert any(
            r.in_reduced_set for r in rows if r.risk_class == cls and r.factor_type == "DELTA"
        )
    assert all(r.daily_vol > 0 for r in rows)
