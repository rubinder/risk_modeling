"""Golden values for the worked example in ``docs/study/03-ccar.md``.

Every number the worked example prints is a constant here. Each is recomputed from the tiny
profile through the real code (the banking-book generators, the stress P&L of the global market
shock through ``RunInputs.stress_grain``, ``ccar.project`` and ``ccar.summary``) and compared,
and each is looked up in the study page, so the page and the code cannot drift apart. Money is
printed rounded to whole dollars, ratios as percentages to four decimals.
"""

import datetime

import pytest
from pyspark.sql import functions as F

from riskagg.data.banking import make_banking_book, make_capital_start, make_macro_scenarios
from riskagg.data.market import make_exposures, make_sensitivities
from riskagg.data.org import make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine.inputs import RunInputs
from riskagg.measures import ccar
from riskagg.measures.stress import load_scenarios, shock_table
from riskagg.paths import REPO_ROOT

STUDY_PAGE = REPO_ROOT / "docs/study/03-ccar.md"
SA, BASE = ccar.SEVERELY_ADVERSE, ccar.BASELINE

# --- one loan segment, severely adverse quarter 1 ---------------------------------------------
SEGMENT = "LE02|BL02|CORPORATE"
SEG_INPUTS = {"ead": 27692533, "pd_base": "0.0203", "lgd": "0.3763",
              "revenue_rate_q": "0.00656", "expense_rate_q": "0.00307"}  # fmt: skip
UNEMPLOYMENT_RISE = "1.5"  # 5.5% - 4.0%
SEG_PD_Q = "0.007739375"  # 0.0203 / 4 x (1 + 0.35 x 1.5)
SEG_REVENUE_Q = "0.0057728"  # 0.00656 x (1 + 2.0 x -0.06)
SEG_CREDIT_LOSS, SEG_PPNR = 80650, 74847

# --- LE02, severely adverse quarter 1 -----------------------------------------------------------
LE02_Q1 = {"ppnr": 561291, "credit_loss": 525500, "trading_loss": 10682964}
LE02_PRE_TAX, LE02_NET_INCOME = -10647174, -8411267
LE02_CET1_START, LE02_CET1_Q1, LE02_RWA = 18681691, 10270424, 157917928
LE02_RATIO_Q1 = "6.5036%"
# quarter -> (net income, CET1, CET1 ratio), severely adverse
LE02_PATH = {
    1: (-8411267, 10270424, "6.5036%"),
    2: (-156025, 10114399, "6.4048%"),
    3: (-217817, 9896582, "6.2669%"),
    4: (-251025, 9645556, "6.1080%"),
    5: (-276340, 9369216, "5.9330%"),
    6: (-255650, 9113566, "5.7711%"),
    7: (-206376, 8907190, "5.6404%"),
    8: (-157919, 8749271, "5.5404%"),
    9: (-119807, 8629464, "5.4645%"),
}

# --- losses by cut, summed over the nine severely adverse quarters ----------------------------
# node -> (ppnr, credit_loss, trading_loss, net_income)
SA_TOTALS = {
    "BL01": (11088004, 14010136, -1864602, -835448),
    "BL02": (5130949, 6031534, 11111876, -9489844),
    "LE01": (10093534, 11874877, -1435691, -273065),
    "LE02": (6125419, 8166793, 10682964, -10052227),
    "FIRM": (16218953, 20041670, 9247274, -10325292),
}

# --- capital measures -----------------------------------------------------------------------
# node -> (start ratio, min ratio SA, decline, dividend add-on, before floor, scb)
SCB = {
    "LE01": ("11.4000%", "11.2822%", "0.1178%", "0.2417%", "0.3594%", "2.5000%"),
    "LE02": ("11.8300%", "5.4645%", "6.3655%", "0.3170%", "6.6825%", "6.6825%"),
    "FIRM": ("11.5742%", "8.9253%", "2.6489%", "0.2722%", "2.9211%", "2.9211%"),
}
# Planned dividends of the add-on quarters (5..8) and the trough quarter's RWA.
ADD_ON_DIVIDENDS = {"LE01": 560402, "LE02": 500669, "FIRM": 1061071}
RWA = {"LE01": 231877637, "LE02": 157917928, "FIRM": 389795565}
BASELINE_MIN = {"LE01": "11.6554%", "LE02": "12.0359%", "FIRM": "11.8096%"}
RWA_WEIGHTED_SCB = "4.1945%"  # average of the entity buffers weighted by RWA: not the firm's


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _pct(x: float) -> str:
    return f"{x:.4%}"


@pytest.fixture(scope="module")
def tiny(spark, tiny_market):
    _, hierarchy, _ = tiny_market
    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12)).cache()
    rf = make_risk_factors(spark, p)
    inputs = RunInputs(
        sensitivities=make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf),
        trades=trades,
        hierarchy=hierarchy,
        shocks=shock_table(spark, load_scenarios(), rf),
        banking_book=make_banking_book(spark, p, hierarchy),
        macro_scenarios=make_macro_scenarios(spark, p),
        capital_start=make_capital_start(spark, p, hierarchy),
    )
    proj = inputs.ccar_projection().cache()
    rows = {(r.scenario, r.quarter, r.node_id): r for r in proj.collect()}
    summary = {(r.measure, r.node_id, r.scenario): r.value for r in inputs.ccar_summary().collect()}
    yield inputs, rows, summary
    proj.unpersist()
    trades.unpersist()


def test_one_segment_first_quarter(spark, tiny):
    inputs, _, _ = tiny
    seg = inputs.banking_book.where(F.col("segment_id") == SEGMENT)
    r = seg.first()
    assert _money(r.ead) == _money(SEG_INPUTS["ead"])
    for k in ("pd_base", "lgd", "revenue_rate_q", "expense_rate_q"):
        assert str(r[k]) == SEG_INPUTS[k]
    macro = {(m.scenario, m.quarter): m for m in inputs.macro_scenarios.collect()}
    rise = macro[(SA, 1)].unemployment_rate - macro[(SA, 0)].unemployment_rate
    assert f"{rise:.1f}" == UNEMPLOYMENT_RISE
    pd_q = r.pd_base / 4 * (1 + 0.35 * rise)
    assert f"{pd_q:.9f}" == SEG_PD_Q
    assert f"{r.revenue_rate_q * (1 + 2.0 * macro[(SA, 1)].gdp_growth):.7f}" == SEG_REVENUE_Q
    # The real projection of that one segment alone (no trading loss).
    gms = spark.createDataFrame(
        [], "legal_entity string, business_line_id string, trading_loss double"
    )
    alone = ccar.project(seg, inputs.macro_scenarios, inputs.capital_start, gms, ccar.load_params())
    q1 = alone.where(f"scenario = '{SA}' and quarter = 1 and node_level = 'BUSINESS_LINE'").first()
    assert _money(q1.credit_loss) == _money(SEG_CREDIT_LOSS)
    assert _money(q1.ppnr) == _money(SEG_PPNR)


def test_le02_first_quarter_and_path(tiny):
    _, rows, _ = tiny
    q1 = rows[(SA, 1, "LE02")]
    for k, v in LE02_Q1.items():
        assert _money(q1[k]) == _money(v)
    pre_tax = q1.ppnr - q1.credit_loss - q1.trading_loss
    assert _money(pre_tax) == _money(LE02_PRE_TAX)
    assert _money(q1.net_income) == _money(LE02_NET_INCOME)
    assert q1.net_income == pytest.approx(pre_tax * 0.79)
    assert _money(q1.cet1 - q1.net_income) == _money(LE02_CET1_START)
    assert _money(q1.cet1) == _money(LE02_CET1_Q1) and _money(q1.rwa) == _money(LE02_RWA)
    assert _pct(q1.cet1_ratio) == LE02_RATIO_Q1
    for q, (ni, cet1, ratio) in LE02_PATH.items():
        r = rows[(SA, q, "LE02")]
        assert (_money(r.net_income), _money(r.cet1), _pct(r.cet1_ratio)) == (
            _money(ni), _money(cet1), ratio
        )  # fmt: skip
        assert r.dividends == pytest.approx(125167.33)  # planned, not deducted


def test_losses_by_cut_reconcile(tiny):
    _, rows, _ = tiny
    cols = ("ppnr", "credit_loss", "trading_loss", "net_income")
    got = {}
    for node, expected in SA_TOTALS.items():
        got[node] = tuple(sum(rows[(SA, q, node)][c] for q in range(1, 10)) for c in cols)
        assert tuple(_money(x) for x in got[node]) == tuple(_money(x) for x in expected)
    for i in range(4):
        firm = got["FIRM"][i]
        assert got["BL01"][i] + got["BL02"][i] == pytest.approx(firm)
        assert got["LE01"][i] + got["LE02"][i] == pytest.approx(firm)


def test_capital_measures(tiny):
    _, rows, summary = tiny
    params = ccar.load_params()
    assert params.scb_dividend_quarters == (5, 8)
    for node, (start, low, decline, add_on, raw, scb) in SCB.items():
        s = summary[(ccar.START_RATIO, node, "*")]
        m = summary[(ccar.MIN_RATIO, node, SA)]
        trough = min(range(1, 10), key=lambda q: (rows[(SA, q, node)].cet1_ratio, q))
        assert trough == 9
        divs = sum(rows[(SA, q, node)].dividends for q in range(5, 9))
        trough_rwa = rows[(SA, trough, node)].rwa
        assert _money(divs) == _money(ADD_ON_DIVIDENDS[node])
        assert _money(trough_rwa) == _money(RWA[node])
        assert (_pct(s), _pct(m), _pct(s - m), _pct(divs / trough_rwa)) == (
            start, low, decline, add_on
        )  # fmt: skip
        assert _pct(s - m + divs / trough_rwa) == raw
        assert _pct(summary[(ccar.SCB, node, "*")]) == scb
        assert _pct(summary[(ccar.MIN_RATIO, node, BASE)]) == BASELINE_MIN[node]
    assert summary[(ccar.SCB, "LE01", "*")] == 0.025
    weighted = sum(summary[(ccar.SCB, le, "*")] * rows[(SA, 1, le)].rwa for le in ("LE01", "LE02"))
    assert _pct(weighted / rows[(SA, 1, "FIRM")].rwa) == RWA_WEIGHTED_SCB


def test_study_page_prints_these_numbers():
    text = STUDY_PAGE.read_text()
    money = [
        SEG_INPUTS["ead"], SEG_CREDIT_LOSS, SEG_PPNR, *LE02_Q1.values(), LE02_PRE_TAX,
        LE02_NET_INCOME, LE02_CET1_START, LE02_CET1_Q1, LE02_RWA,
        *(x for ni, cet1, _ in LE02_PATH.values() for x in (ni, cet1)),
        *(x for t in SA_TOTALS.values() for x in t),
        *ADD_ON_DIVIDENDS.values(), *RWA.values(),
    ]  # fmt: skip
    for value in money:
        assert _money(abs(value)) in text, value
    literals = [
        *(v for k, v in SEG_INPUTS.items() if k != "ead"), UNEMPLOYMENT_RISE, SEG_PD_Q,
        SEG_REVENUE_Q, LE02_RATIO_Q1, *(r for *_, r in LE02_PATH.values()),
        *(x for t in SCB.values() for x in t), *BASELINE_MIN.values(), RWA_WEIGHTED_SCB,
    ]  # fmt: skip
    for literal in literals:
        assert literal in text, literal
