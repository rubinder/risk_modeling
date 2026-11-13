"""Why diversification between business lines is about zero on the demo profile.

Every number the "Diversification on the demo profile" passage of ``docs/study/01-var-es.md``
quotes is recomputed here from the seeded generators and looked up on the page. P&L is linear
in the factor returns, so a business line's P&L vector is its net exposure per risk factor
times the factor returns: the same vectors the engine builds from ``pnl_vectors``, without
writing them (a few seconds for the 100,000 demo trades).
"""

import datetime

import numpy as np
import pytest
from pyspark.sql import functions as F

from riskagg.conventions import var_from_pnl
from riskagg.data.market import factor_returns, make_exposures
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.paths import REPO_ROOT

STUDY_PAGE = REPO_ROOT / "docs/study/01-var-es.md"
P = PROFILES["demo"]

# (firm VaR in billions, sum of the business lines' VaRs in billions, benefit as % of the sum,
#  third-worst scenario of the firm, of each business line)
WINDOWS = {
    "STRESSED": ("21.07", "21.07", "0.00%", 64, {"BL01": 64, "BL02": 64, "BL03": 64, "BL04": 64}),
    "CURRENT": ("6.04", "6.06", "0.34%", 84, {"BL01": 78, "BL02": 84, "BL03": 84, "BL04": 84}),
}
MIN_CORRELATION = "0.999"
NET_LONG_SHARE, NET_OVER_GROSS, N_FACTORS, TRADES_PER_BOOK = "84%", "44%", 54, "500"


@pytest.fixture(scope="module")
def demo(spark):
    trades = make_trades(spark, P, datetime.date(2026, 11, 12))
    rf = make_risk_factors(spark, P)
    exposures = make_exposures(spark, P, trades, rf).cache()
    bl = make_hierarchy(spark, P).select("book_id", "business_line_id")
    net = (
        exposures.join(bl, "book_id")
        .groupBy("business_line_id", "risk_factor_id")
        .agg(F.sum("exposure").alias("e"))
        .toPandas()
    )
    books = (
        exposures.groupBy("book_id", "risk_factor_id")
        .agg(F.sum("exposure").alias("net"), F.sum(F.abs("exposure")).alias("gross"))
        .toPandas()
    )
    n_trades = trades.count()
    exposures.unpersist()
    rf_pdf = rf.toPandas().sort_values("risk_factor_id").reset_index(drop=True)
    idx = {r: i for i, r in enumerate(rf_pdf["risk_factor_id"])}
    vectors = {}
    for window in WINDOWS:
        returns = factor_returns(P, rf_pdf, window)
        per_bl = {}
        for name, g in net.groupby("business_line_id"):
            e = np.zeros(len(rf_pdf))
            e[[idx[f] for f in g["risk_factor_id"]]] = g["e"].to_numpy()
            per_bl[name] = e @ returns
        vectors[window] = per_bl
    return vectors, books, len(rf_pdf), n_trades / books["book_id"].nunique()


def _third_worst(v: np.ndarray) -> int:
    return int(np.argsort(v, kind="stable")[2])  # VaR 99% of 250 is the 3rd-worst loss


def test_business_lines_are_comonotonic(demo):
    vectors, _, _, _ = demo
    for window, (firm_bn, sum_bn, pct, firm_scen, bl_scen) in WINDOWS.items():
        bls = vectors[window]
        firm = sum(bls.values())
        firm_var = var_from_pnl(firm, 0.99)
        total = sum(var_from_pnl(v, 0.99) for v in bls.values())
        assert (f"{firm_var / 1e9:.2f}", f"{total / 1e9:.2f}") == (firm_bn, sum_bn), window
        assert f"{100 * (total - firm_var) / total:.2f}%" == pct, window
        assert _third_worst(firm) == firm_scen
        assert {b: _third_worst(v) for b, v in bls.items()} == bl_scen, window
        assert min(np.corrcoef(v, firm)[0, 1] for v in bls.values()) > float(MIN_CORRELATION)


def test_books_are_alike_and_net_long(demo):
    _, books, n_factors, trades_per_book = demo
    assert n_factors == N_FACTORS and f"{trades_per_book:.0f}" == TRADES_PER_BOOK
    assert books.groupby("book_id").size().median() == N_FACTORS  # a typical book holds them all
    assert f"{100 * (books['net'] > 0).mean():.0f}%" == NET_LONG_SHARE
    assert f"{100 * (books['net'] / books['gross']).median():.0f}%" == NET_OVER_GROSS


def test_page_quotes_these_numbers():
    text = " ".join(STUDY_PAGE.read_text().split())
    for firm_bn, sum_bn, pct, firm_scen, bl_scen in WINDOWS.values():
        for literal in (firm_bn, sum_bn, pct, f"scenario {firm_scen}"):
            assert literal in text, literal
    for literal in (MIN_CORRELATION, NET_LONG_SHARE, NET_OVER_GROSS, f"{N_FACTORS} risk factors",
                    f"{TRADES_PER_BOOK} trades", "scenario 78"):  # fmt: skip
        assert literal in text, literal
