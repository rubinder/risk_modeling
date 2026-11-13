"""Golden values for the worked example in ``docs/study/02-stress.md``.

Every number the worked example prints is a constant here. Each is recomputed from the tiny
profile through the real code (shock table, ``stress_pnl``, ``stress_by_grain``, ``run_all``
with the shipped ``worst_stress_by_org`` specification) and compared, and each is looked up
in the study page, so the page and the code cannot drift apart. Money is printed rounded to
whole dollars.
"""

import datetime

import pytest
from pyspark.sql import functions as F

from riskagg.data.market import make_exposures, make_sensitivities
from riskagg.data.org import make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine.inputs import RunInputs
from riskagg.engine.runner import run_all
from riskagg.engine.spec import AggregationSpec, load_specs
from riskagg.engine.vectors import GRAIN_DIMS
from riskagg.measures.stress import (
    Scenario,
    ShockRule,
    load_scenarios,
    shock_table,
    stress_by_grain,
    stress_pnl,
    worst_stress_loss,
)
from riskagg.paths import REPO_ROOT
from riskagg.semantic.model import load_model

STUDY_PAGE = REPO_ROOT / "docs/study/02-stress.md"
SCENARIOS = ["EQUITY_DOWN_30", "FX_CRISIS", "GFC_2008", "RATES_UP_200BP", "VOL_SPIKE"]
N_RISK_FACTORS = 54

# Stress P&L by scenario (whole dollars), in SCENARIOS order.
DESK002_BOOKS = {
    "BOOK0003": [0, -215491, -505741, 315169, 0],
    "BOOK0004": [5535602, 8788, 6856323, 560735, 262730],
}
DESK002 = [5535602, -206703, 6350582, 875904, 262730]
DESKS = {
    "DESK001": [-1338961, -1089169, -4485980, 3085543, 97051],
    "DESK002": DESK002,
    "DESK003": [1212374, -4191554, -11111876, 10572677, 0],
}
FIRM = [5409016, -5487426, -9247274, 14534124, 359781]
# Worst loss and its scenario.
WORST = {
    "DESK001": ("GFC_2008", 4485980),
    "DESK002": ("FX_CRISIS", 206703),
    "DESK003": ("GFC_2008", 11111876),
    "FIRM": ("GFC_2008", 9247274),
}
BOOK_WORST = {"BOOK0003": ("GFC_2008", 505741), "BOOK0004": ("FX_CRISIS", -8788)}
BOOKS_WORST_SUM = 496953
DESKS_WORST_SUM = 15804559


def _money(x: float) -> str:
    return f"{x:,.0f}"


@pytest.fixture(scope="module")
def tiny(spark, tiny_market):
    _, hierarchy, _ = tiny_market
    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12)).cache()
    rf = make_risk_factors(spark, p)
    sens = make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf)
    shocks = shock_table(spark, load_scenarios(), rf)
    grain = stress_by_grain(stress_pnl(sens, shocks), trades, hierarchy).cache()
    pnl = {}
    for key, dim in (("book", "book_id"), ("desk", "desk_id"), ("firm", "firm_id")):
        rows = grain.groupBy(dim, "scenario").agg(F.sum("pnl").alias("pnl")).collect()
        for r in rows:
            pnl.setdefault(r[dim] if key != "firm" else "FIRM", {})[r.scenario] = r.pnl
    inputs = RunInputs(sensitivities=sens, trades=trades, hierarchy=hierarchy, shocks=shocks)
    model = load_model()
    specs = [s for s in load_specs(None, model) if s.name == "worst_stress_by_org"]
    worst = {}
    for r in run_all(specs, model, grain, 20, inputs).collect():
        node = {"firm_id": "FIRM"}.get(r.node_level)
        node = node or (r.book_id if r.node_level.endswith("book_id") else r.desk_id)
        worst[node] = (r.scenario, r.value)
    yield pnl, worst, shocks
    grain.unpersist()
    trades.unpersist()


def test_shock_table_size(tiny):
    *_, shocks = tiny
    assert shocks.count() == len(SCENARIOS) * N_RISK_FACTORS


def test_one_trade_by_hand(spark):
    sc = Scenario("S", "", False, (ShockRule("EQ", "DELTA", -0.30), ShockRule("EQ", "VEGA", 0.20)))
    rf = spark.createDataFrame(
        [("EQ.1", "EQ", "DELTA", "1", "USD"), ("EQ.1.VEGA", "EQ", "VEGA", "1", "USD")],
        "risk_factor_id string, risk_class string, factor_type string, bucket string, "
        "currency string",
    )
    sens = spark.createDataFrame(
        [
            ("T", "B", "EQ.1", "EQ", "DELTA", 100.0, 50.0, 0.0),
            ("T", "B", "EQ.1.VEGA", "EQ", "VEGA", 0.0, 0.0, 10.0),
        ],
        "trade_id string, book_id string, risk_factor_id string, risk_class string, "
        "factor_type string, delta double, gamma double, vega double",
    )
    shocks = shock_table(spark, [sc], rf)
    assert stress_pnl(sens.where("factor_type = 'DELTA'"), shocks).first().pnl == -27.75
    assert stress_pnl(sens, shocks).first().pnl == pytest.approx(-25.75)


def test_which_shock_applies(spark):
    scenarios = {s.name: s for s in load_scenarios()}
    fx = ("FX", "DELTA")
    assert scenarios["FX_CRISIS"].shock_for(*fx, "BRL", "BRL") == -0.20
    assert scenarios["FX_CRISIS"].shock_for(*fx, "EUR", "EUR") == -0.05
    assert scenarios["GFC_2008"].shock_for(*fx, "BRL", "BRL") == -0.25
    assert scenarios["GFC_2008"].shock_for(*fx, "EUR", "EUR") == -0.10
    assert scenarios["GFC_2008"].global_market_shock


def test_desk002_table(tiny):
    pnl, *_ = tiny
    for node, expected in {**DESK002_BOOKS, "DESK002": DESK002}.items():
        assert [_money(pnl[node].get(s, 0.0)) for s in SCENARIOS] == [
            _money(x) for x in expected
        ], node
    for i, s in enumerate(SCENARIOS):  # the desk is the sum of its books, per scenario
        assert pnl["DESK002"][s] == pytest.approx(
            sum(pnl[b].get(s, 0.0) for b in DESK002_BOOKS), abs=1e-6
        )
        assert _money(DESK002[i]) == _money(sum(v[i] for v in DESK002_BOOKS.values()))


def test_firm_table(tiny):
    pnl, *_ = tiny
    for desk, expected in DESKS.items():
        assert [_money(pnl[desk].get(s, 0.0)) for s in SCENARIOS] == [_money(x) for x in expected]
    assert [_money(pnl["FIRM"][s]) for s in SCENARIOS] == [_money(x) for x in FIRM]
    for s in SCENARIOS:
        assert pnl["FIRM"][s] == pytest.approx(sum(pnl[d].get(s, 0.0) for d in DESKS), abs=1e-6)


def test_worst_scenarios_and_non_additivity(tiny):
    pnl, worst, _ = tiny
    for node, (scenario, loss) in {**WORST, **BOOK_WORST}.items():
        got = worst[node]
        assert (got[0], _money(got[1])) == (scenario, _money(loss)), node
        # The stored figure is the loss of the lowest summed scenario P&L.
        low = min(pnl[node].items(), key=lambda kv: (kv[1], kv[0]))
        assert got[0] == low[0] and got[1] == pytest.approx(-low[1])
    books = sum(worst[b][1] for b in BOOK_WORST)
    assert _money(books) == _money(BOOKS_WORST_SUM) and books > worst["DESK002"][1]
    desks = sum(worst[d][1] for d in DESKS)
    assert _money(desks) == _money(DESKS_WORST_SUM) and desks > worst["FIRM"][1]


def test_constructed_counter_example(spark):
    rows = [("D1", "A", -100.0), ("D1", "B", 50.0), ("D2", "A", 50.0), ("D2", "B", -100.0)]
    spec = AggregationSpec(
        "w", "worst_stress_loss", (("firm_id",), ("firm_id", "business_line_id", "desk_id")), {}, False
    )  # fmt: skip
    data = [
        ("FIRM", "BL1", d, f"B-{d}", "LE1", "EMEA", "FX", "EUR", "FX", sc, pnl)
        for d, sc, pnl in rows
    ]
    frame = spark.createDataFrame(data, [*GRAIN_DIMS, "scenario", "pnl"])
    inputs = RunInputs(_stress_grain=frame)
    got = {(r.node_level, r.desk_id): (r.scenario, r.value) for r in
           worst_stress_loss(spec, load_model(), inputs).collect()}  # fmt: skip
    assert got[("firm_id|business_line_id|desk_id", "D1")] == ("A", 100.0)
    assert got[("firm_id|business_line_id|desk_id", "D2")] == ("B", 100.0)
    assert got[("firm_id", "*")] == ("A", 50.0)  # A and B both give -50; A sorts first


def test_study_page_prints_these_numbers():
    text = STUDY_PAGE.read_text()
    for node in (*DESK002_BOOKS, *DESKS):
        assert node in text
    for i, s in enumerate(SCENARIOS):
        row = (
            f"| {s} | {_money(DESK002_BOOKS['BOOK0003'][i])} | "
            f"{_money(DESK002_BOOKS['BOOK0004'][i])} | {_money(DESK002[i])} |"
        )
        assert row in text, row
        cols = " | ".join(_money(DESKS[d][i]) for d in DESKS)
        row = f"| {s} | {cols} | {_money(FIRM[i])} |"
        assert row in text, row
    for value in (
        *(v for _, v in WORST.values()),
        BOOK_WORST["BOOK0003"][1],
        BOOK_WORST["BOOK0004"][1],
        BOOKS_WORST_SUM,
        DESKS_WORST_SUM,
    ):
        assert _money(value) in text, value
    for literal in ("-27.75", "-25.75", "**-0.20**", "**-0.05**", "5 × 54"):
        assert literal in text, literal
    counter = (
        "| A | -100 | 50 | -50 |",
        "| B | 50 | -100 | -50 |",
    )
    for row in counter:
        assert row in text, row
