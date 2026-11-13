import datetime
import math

import pytest
from pyspark.sql import functions as F

from reference import np_frtb
from riskagg.data.market import make_exposures, make_sensitivities
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.measures.frtb_sa import (
    attach_org,
    bucket_detail,
    girr_delta_rho,
    load_frtb_ref,
    sbm_charges,
    sbm_total,
    weighted_sensitivities,
)

SENS_SCHEMA = (
    "trade_id long, book_id string, desk_id string, risk_factor_id string, risk_class string, "
    "factor_type string, bucket string, tenor_years double, delta double, gamma double, "
    "vega double, curvature_up double, curvature_down double"
)


@pytest.fixture(scope="module")
def ref():
    return load_frtb_ref()


def _row(rf, rc, bucket, *, ft="DELTA", tenor=None, delta=0.0, vega=0.0, up=0.0, down=0.0,
         book="B1", desk="D1", trade=1):  # fmt: skip
    return (trade, book, desk, rf, rc, ft, bucket, tenor, delta, 0.0, vega, up, down)


def _sens(spark, rows):
    return spark.createDataFrame(rows, SENS_SCHEMA)


def _charges(df, component="DELTA", **node):
    out = {}
    for r in df.collect():
        if r["component"] != component or any(r[k] != v for k, v in node.items()):
            continue
        out[(r["risk_class"], r["corr_scenario"])] = r["charge"]
    return out


def test_equity_intra_bucket(spark, ref):
    """Bucket 5 (RW 30%, rho 25%), two issuers: ws 30 and -15.

    MEDIUM: 900 + 225 - 2*0.25*450 = 900 -> 30. HIGH rho 0.3125: 1125 - 281.25 -> 29.0474.
    LOW rho max(-0.5, 0.1875) = 0.1875: 1125 - 168.75 -> 30.9233. LOW binds for a hedge.
    """
    sens = _sens(spark, [
        _row("EQ.5.A", "EQ", "5", delta=100.0),
        _row("EQ.5.B", "EQ", "5", delta=-50.0),
    ])  # fmt: skip
    charges = sbm_charges(sens, [], ref)
    got = _charges(charges)
    assert got[("EQ", "MEDIUM")] == pytest.approx(30.0)
    assert got[("EQ", "HIGH")] == pytest.approx(math.sqrt(843.75))
    assert got[("EQ", "LOW")] == pytest.approx(math.sqrt(956.25))
    assert got[("EQ", "HIGH")] == pytest.approx(29.047375, abs=1e-6)
    assert got[("EQ", "LOW")] == pytest.approx(30.923292, abs=1e-6)
    total = {r["corr_scenario"]: r for r in sbm_total(charges).collect()}
    assert [s for s, r in total.items() if r["is_binding"]] == ["LOW"]
    assert total["LOW"]["charge"] == pytest.approx(math.sqrt(956.25))


def test_equity_inter_bucket(spark, ref):
    """Bucket 8 delta 50 -> ws 25 (K 25, S 25); bucket 5 K 30, S 15; gamma 15%.

    900 + 625 + 2*0.15*15*25 = 1637.5 -> 40.466.
    """
    sens = _sens(spark, [
        _row("EQ.5.A", "EQ", "5", delta=100.0),
        _row("EQ.5.B", "EQ", "5", delta=-50.0),
        _row("EQ.8", "EQ", "8", delta=50.0),
    ])  # fmt: skip
    got = _charges(sbm_charges(sens, [], ref))
    assert got[("EQ", "MEDIUM")] == pytest.approx(math.sqrt(1637.5))
    assert got[("EQ", "MEDIUM")] == pytest.approx(40.466035, abs=1e-6)
    detail = {
        (r["bucket"], r["corr_scenario"]): (r["k_b"], r["s_b"])
        for r in bucket_detail(sens, [], ref).collect()
    }
    assert detail[("5", "MEDIUM")] == pytest.approx((30.0, 15.0))
    assert detail[("8", "LOW")] == pytest.approx((25.0, 25.0))


def test_fx_specified_pair(spark, ref):
    """EUR, BRL (MAR21.88 fn 22: both specified against USD): 1000 * 15% / sqrt(2) = 106.066.

    PLN is not a specified pair: 1000 * 15% = 150.
    """
    for ccy, expected in (("EUR", 106.066017), ("BRL", 106.066017), ("PLN", 150.0)):
        sens = _sens(spark, [_row(f"FX.{ccy}", "FX", ccy, delta=1000.0)])
        got = _charges(sbm_charges(sens, [], ref))
        for scenario in ("LOW", "MEDIUM", "HIGH"):
            assert got[("FX", scenario)] == pytest.approx(expected, abs=1e-6), ccy


def test_girr_tenor_correlation(ref):
    assert girr_delta_rho(1.0, 2.0, ref) == pytest.approx(math.exp(-0.03))
    assert girr_delta_rho(1.0, 2.0, ref) == pytest.approx(0.97045, abs=1e-5)
    # MAR21.46 Table 2: 1y/30y is 41.9% (exp(-0.87)); the 40% floor binds e.g. at 0.25y/10y.
    assert girr_delta_rho(1.0, 30.0, ref) == pytest.approx(math.exp(-0.87))
    assert girr_delta_rho(1.0, 30.0, ref) == pytest.approx(0.418952, abs=1e-6)
    assert girr_delta_rho(0.25, 10.0, ref) == 0.40
    assert girr_delta_rho(5.0, 5.0, ref) == 1.0


# MAR21.46 Table 2, row by row, in percent.
_TABLE_2 = [
    [100.0, 97.0, 91.4, 81.1, 71.9, 56.6, 40.0, 40.0, 40.0, 40.0],
    [97.0, 100.0, 97.0, 91.4, 86.1, 76.3, 56.6, 41.9, 40.0, 40.0],
    [91.4, 97.0, 100.0, 97.0, 94.2, 88.7, 76.3, 65.7, 56.6, 41.9],
    [81.1, 91.4, 97.0, 100.0, 98.5, 95.6, 88.7, 82.3, 76.3, 65.7],
    [71.9, 86.1, 94.2, 98.5, 100.0, 98.0, 93.2, 88.7, 84.4, 76.3],
    [56.6, 76.3, 88.7, 95.6, 98.0, 100.0, 97.0, 94.2, 91.4, 86.1],
    [40.0, 56.6, 76.3, 88.7, 93.2, 97.0, 100.0, 98.5, 97.0, 94.2],
    [40.0, 41.9, 65.7, 82.3, 88.7, 94.2, 98.5, 100.0, 99.0, 97.0],
    [40.0, 40.0, 56.6, 76.3, 84.4, 91.4, 97.0, 99.0, 100.0, 98.5],
    [40.0, 40.0, 41.9, 65.7, 76.3, 86.1, 94.2, 97.0, 98.5, 100.0],
]


def test_girr_correlation_reproduces_basel_table_2(ref):
    tenors = sorted(ref.girr_delta_rw)
    assert tenors == [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    for i, tk in enumerate(tenors):
        for j, tl in enumerate(tenors):
            assert round(100 * girr_delta_rho(tk, tl, ref), 1) == _TABLE_2[i][j], (tk, tl)


def test_risk_weights(spark, ref):
    """GIRR USD 5y: 1.1%/sqrt(2) (specified); BRL 5y: 1.1%; EQ bucket 1 vega 77.78%."""
    sens = _sens(spark, [
        _row("GIRR.USD.5Y", "GIRR", "USD", tenor=5.0, delta=1000.0),
        _row("GIRR.BRL.5Y", "GIRR", "BRL", tenor=5.0, delta=1000.0),
        _row("GIRR.EUR.0.25Y", "GIRR", "EUR", tenor=0.25, delta=1000.0),
        _row("EQ.1.VEGA", "EQ", "1", ft="VEGA", vega=100.0),
        _row("FX.EUR.VEGA", "FX", "EUR", ft="VEGA", vega=100.0),
    ])  # fmt: skip
    ws = {r["risk_factor_id"]: r for r in weighted_sensitivities(sens, [], ref).collect()}
    assert ws["GIRR.USD.5Y"]["weighted_sensitivity"] == pytest.approx(11.0 / math.sqrt(2))
    assert ws["GIRR.BRL.5Y"]["weighted_sensitivity"] == pytest.approx(11.0)
    assert ws["GIRR.EUR.0.25Y"]["weighted_sensitivity"] == pytest.approx(17.0 / math.sqrt(2))
    assert ws["EQ.1.VEGA"]["weighted_sensitivity"] == pytest.approx(77.78)
    assert ws["EQ.1.VEGA"]["component"] == "VEGA"
    assert ws["FX.EUR.VEGA"]["weighted_sensitivity"] == pytest.approx(100.0)


def test_unknown_bucket_fails(spark, ref):
    sens = _sens(spark, [_row("EQ.11", "EQ", "11", delta=1.0)])
    with pytest.raises(Exception, match="no FRTB SA risk weight"):
        weighted_sensitivities(sens, [], ref).collect()


def test_alternative_specification_when_sum_negative(spark, ref):
    """Bucket 1: 20 issuers, delta 20 -> ws 11 each; bucket 12: 2 indices, delta -300 -> -45.

    MEDIUM: K1^2 = 20*121 + 380*0.15*121 = 9317, S1 = 220; K12^2 = 2*2025 + 2*0.8*2025 = 7290,
    S12 = -90; gamma 45%: 9317 + 7290 - 0.9*220*90 = -1213 < 0, so S_b is clamped to
    [-K_b, K_b]: 9317 + 7290 - 0.9*sqrt(9317)*sqrt(7290).
    """
    rows = [_row(f"EQ.1.{i}", "EQ", "1", delta=20.0) for i in range(20)]
    rows += [_row(f"EQ.12.{i}", "EQ", "12", delta=-300.0) for i in range(2)]
    got = _charges(sbm_charges(_sens(spark, rows), [], ref))
    assert 9317 + 7290 - 0.9 * 220 * 90 < 0
    expected = math.sqrt(9317 + 7290 - 0.9 * math.sqrt(9317) * math.sqrt(7290))
    assert got[("EQ", "MEDIUM")] == pytest.approx(expected)


def test_curvature(spark, ref):
    """USD: CVR up 10, down 4 -> K 10, up selected, S 10. EUR: up -6, down -2 -> K 0, tie,
    up not > down so down selected, S -2. gamma = 0.5^2 = 0.25, scenarios on the square:
    MEDIUM 100 - 2*0.25*20 = 90; HIGH 0.3125 -> 87.5; LOW max(-0.5, 0.1875) -> 92.5.
    """
    sens = _sens(spark, [
        _row("GIRR.USD.5Y", "GIRR", "USD", tenor=5.0, up=6.0, down=1.0, trade=1),
        _row("GIRR.USD.10Y", "GIRR", "USD", tenor=10.0, up=4.0, down=3.0, trade=2),
        _row("GIRR.EUR.5Y", "GIRR", "EUR", tenor=5.0, up=-6.0, down=-2.0),
    ])  # fmt: skip
    got = _charges(sbm_charges(sens, [], ref), "CURVATURE")
    assert got[("GIRR", "MEDIUM")] == pytest.approx(math.sqrt(90.0))
    assert got[("GIRR", "HIGH")] == pytest.approx(math.sqrt(87.5))
    assert got[("GIRR", "LOW")] == pytest.approx(math.sqrt(92.5))
    detail = {
        (r["bucket"], r["corr_scenario"]): (r["k_b"], r["s_b"])
        for r in bucket_detail(sens, [], ref).filter(F.col("component") == "CURVATURE").collect()
    }
    assert detail[("USD", "MEDIUM")] == pytest.approx((10.0, 10.0))
    assert detail[("EUR", "MEDIUM")] == pytest.approx((0.0, -2.0))


def test_curvature_psi_zero_when_both_negative(spark, ref):
    """Both buckets select a negative S with K 0; psi = 0 so the cross term vanishes."""
    sens = _sens(spark, [
        _row("GIRR.USD.5Y", "GIRR", "USD", tenor=5.0, up=-2.0, down=-3.0),
        _row("GIRR.EUR.5Y", "GIRR", "EUR", tenor=5.0, up=-3.0, down=-4.0),
    ])  # fmt: skip
    got = _charges(sbm_charges(sens, [], ref), "CURVATURE")
    assert got == {("GIRR", s): 0.0 for s in ("LOW", "MEDIUM", "HIGH")}


def test_netting_before_weighting(spark, ref):
    sens = _sens(spark, [
        _row("EQ.5", "EQ", "5", delta=100.0, book="B1", trade=1),
        _row("EQ.5", "EQ", "5", delta=-100.0, book="B2", trade=2),
    ])  # fmt: skip
    desk = sbm_charges(sens, ["desk_id"], ref)
    assert {r["charge"] for r in desk.collect()} == {0.0}
    book = sbm_charges(sens, ["book_id"], ref)
    assert book.count() == 6
    assert all(r["charge"] == pytest.approx(30.0) for r in book.collect())


def test_null_sensitivities_treated_as_zero(spark, ref):
    rows = [
        (1, "B1", "D1", "EQ.5.A", "EQ", "DELTA", "5", None, 100.0, None, None, None, None),
        (2, "B1", "D1", "EQ.5.B", "EQ", "DELTA", "5", None, None, None, None, None, None),
    ]
    zeros = [tuple(0.0 if v is None and i >= 8 else v for i, v in enumerate(r)) for r in rows]
    a = sorted(tuple(r) for r in sbm_charges(_sens(spark, rows), [], ref).collect())
    b = sorted(tuple(r) for r in sbm_charges(_sens(spark, zeros), [], ref).collect())
    assert a == b
    assert _charges(sbm_charges(_sens(spark, rows), [], ref))[("EQ", "MEDIUM")] == 30.0


@pytest.fixture(scope="module")
def tiny_sens(spark):
    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12))
    rf = make_risk_factors(spark, p)
    sens = make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf)
    hierarchy = make_hierarchy(spark, p)
    return attach_org(sens, hierarchy, ["desk_id"]).cache(), hierarchy


def _totals(df, node_cols):
    return {
        (tuple(r[c] for c in node_cols), r["corr_scenario"]): (r["charge"], r["is_binding"])
        for r in df.collect()
    }


def test_not_a_rollup(tiny_sens, ref):
    sens, _ = tiny_sens
    desks = _totals(sbm_total(sbm_charges(sens, ["desk_id"], ref)), ["desk_id"])
    firm = _totals(sbm_total(sbm_charges(sens, [], ref)), [])
    firm_charge = next(c for (_, _), (c, b) in firm.items() if b)
    desk_sum = sum(c for (_, _), (c, b) in desks.items() if b)
    assert len({k[0] for k in desks}) == 3
    assert 0 < firm_charge < desk_sum


@pytest.mark.parametrize("node_cols", [[], ["desk_id"]])
def test_matches_numpy_reference(tiny_sens, ref, node_cols):
    sens, _ = tiny_sens
    pdf = sens.toPandas()
    charges = sbm_charges(sens, node_cols, ref)
    expected = np_frtb.sbm_charges(pdf, node_cols)
    got = {
        (tuple(r[c] for c in node_cols), r["risk_class"], r["component"], r["corr_scenario"]):
            r["charge"]
        for r in charges.collect()
    }  # fmt: skip
    assert set(got) == set(expected)
    assert {k[2] for k in got} == {"DELTA", "VEGA", "CURVATURE"}
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-6), key
    totals = _totals(sbm_total(charges), node_cols)
    exp_totals = np_frtb.sbm_total(expected)
    assert set(totals) == set(exp_totals)
    for key, (value, binding) in exp_totals.items():
        assert totals[key][0] == pytest.approx(value, rel=1e-9, abs=1e-6), key
        assert totals[key][1] == binding, key
