import dataclasses
import datetime

import numpy as np
from pyspark.sql import functions as F

from riskagg.data.market import (
    factor_returns,
    make_exposures,
    make_pnl_vectors,
    make_sensitivities,
)
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES

AS_OF = datetime.date(2026, 11, 12)
P = dataclasses.replace(PROFILES["tiny"], n_trades=600)


def _frames(spark, p=P):
    trades = make_trades(spark, p, AS_OF)
    rf = make_risk_factors(spark, p)
    return trades, rf, make_exposures(spark, p, trades, rf)


def _desk_sums(spark, window):
    trades, rf, expo = _frames(spark)
    pnl = make_pnl_vectors(spark, P, trades, expo, rf).filter(F.col("window") == window)
    desks = make_hierarchy(spark, P).select("book_id", "desk_id")
    sums: dict[str, np.ndarray] = {}
    for r in pnl.join(desks, "book_id").select("desk_id", "pnl").collect():
        sums[r.desk_id] = sums.get(r.desk_id, 0) + np.array(r.pnl)
    return sums


def test_factor_returns_shape_order_and_stress(spark):
    rf = make_risk_factors(spark, P).toPandas()
    cur = factor_returns(P, rf, "CURRENT")
    stress = factor_returns(P, rf, "STRESSED")
    assert cur.shape == (len(rf), P.n_scenarios)
    assert np.array_equal(cur, factor_returns(P, rf.sample(frac=1, random_state=1), "CURRENT"))
    assert not np.array_equal(cur, stress)
    assert stress.std() > 2 * cur.std()
    same_class = rf.sort_values("risk_factor_id").reset_index(drop=True)
    idx = same_class.index[same_class.risk_class == "GIRR"].to_numpy()

    def mean_corr(r):
        return np.corrcoef(r[idx]).mean()

    assert mean_corr(stress) > mean_corr(cur)


def test_exposure_rules(spark):
    trades, rf, expo = _frames(spark)
    t = trades.collect()
    e = expo.groupBy("trade_id").agg(F.collect_list("risk_factor_id").alias("f")).collect()
    factors = {r.trade_id: r.f for r in e}
    known = {r.risk_factor_id for r in rf.collect()}
    assert set(factors) == {r.trade_id for r in t}
    assert {f for fs in factors.values() for f in fs} <= known
    for r in t:
        fs = factors[r.trade_id]
        has_fx = f"FX.{r.currency}" in fs
        if r.asset_class in ("RATES", "EQUITY") and r.currency != "USD":
            assert has_fx
        if r.asset_class == "FX":
            assert has_fx
    options = sum(any(f.endswith(".VEGA") for f in fs) for fs in factors.values())
    assert 0.2 <= options / len(t) <= 0.4


def test_tiny_trade_pnl_matches_numpy(spark):
    p = PROFILES["tiny"]
    trades, rf, expo = _frames(spark, p)
    pnl = make_pnl_vectors(spark, p, trades, expo, rf).filter(F.col("window") == "CURRENT")
    assert all(len(r.pnl) == p.n_scenarios for r in pnl.collect())
    rf_pdf = rf.toPandas().sort_values("risk_factor_id").reset_index(drop=True)
    ret = factor_returns(p, rf_pdf, "CURRENT")
    pos = {f: i for i, f in enumerate(rf_pdf.risk_factor_id)}
    tid = 3
    expected = sum(
        r.exposure * ret[pos[r.risk_factor_id]] for r in expo.collect() if r.trade_id == tid
    )
    got = sum(np.array(r.pnl) for r in pnl.filter(F.col("trade_id") == tid).collect())
    assert np.allclose(got, expected)


def test_windows_group_rows_and_stress_volatility(spark):
    trades, rf, expo = _frames(spark)
    pnl = make_pnl_vectors(spark, P, trades, expo, rf)
    assert {r.window for r in pnl.select("window").distinct().collect()} == {"CURRENT", "STRESSED"}
    fine = pnl.groupBy("trade_id", "risk_class", "liquidity_horizon", "in_reduced_set", "window")
    assert fine.count().count() == pnl.count()
    cur, stress = (
        np.concatenate(list(map(np.asarray, s.values())))
        for s in (_desk_sums(spark, "CURRENT"), _desk_sums(spark, "STRESSED"))
    )
    assert stress.std() > cur.std()


def test_desks_diversify(spark):
    sums = _desk_sums(spark, "CURRENT")
    ids = sorted(sums)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            assert 0 < np.corrcoef(sums[ids[a]], sums[ids[b]])[0, 1] < 1


def test_sensitivities(spark):
    _, rf, expo = _frames(spark)
    s = make_sensitivities(spark, P, expo, rf)
    rows = s.collect()
    assert len(rows) == expo.count()
    option_trades = {r.trade_id for r in rows if r.factor_type == "VEGA"}
    for r in rows:
        assert None not in (r.delta, r.gamma, r.vega, r.curvature_up, r.curvature_down)
        assert r.curvature_up >= 0 and r.curvature_down >= 0
        if r.factor_type == "VEGA":
            assert r.delta == 0 and r.gamma == 0 and r.vega != 0
        else:
            assert r.vega == 0 and r.delta != 0
            if r.trade_id in option_trades:
                assert r.gamma != 0
            else:
                assert r.gamma == 0 and r.curvature_up == 0 and r.curvature_down == 0
    e = {(r.trade_id, r.risk_factor_id): r.exposure for r in expo.collect()}
    assert all(e[(r.trade_id, r.risk_factor_id)] == (r.delta or r.vega) for r in rows)
    assert any(r.curvature_up > 0 for r in rows)


def test_determinism_across_partitioning(spark):
    trades, rf, expo = _frames(spark)
    a = make_pnl_vectors(spark, P, trades, expo, rf)
    b = make_pnl_vectors(spark, P, trades.repartition(7), expo.repartition(5), rf)
    key = ["trade_id", "risk_class", "liquidity_horizon", "in_reduced_set", "window"]
    ra = sorted((tuple(r[k] for k in key), tuple(r.pnl)) for r in a.collect())
    rb = sorted((tuple(r[k] for k in key), tuple(r.pnl)) for r in b.collect())
    assert ra == rb
