"""Golden values for the worked examples in ``docs/study/04-frtb.md``.

Every number the worked examples print is a constant here. Each is recomputed from the tiny
profile through the real code (``frtb_sa`` for the SBM bucket, ``frtb_drc`` for the default
risk bucket and the SA totals, ``frtb_ima`` for the liquidity-horizon cascade and the IMCC) and
compared, and each is looked up in the study page, so the page and the code cannot drift apart.
Money is printed rounded to whole dollars.
"""

import datetime
import math

import pytest
from pyspark.sql import functions as F

from riskagg.data.banking import make_default_risk_positions
from riskagg.data.market import make_exposures, make_sensitivities
from riskagg.data.org import make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine.grain import build_reporting_grain
from riskagg.engine.inputs import RunInputs
from riskagg.measures import frtb_drc, frtb_ima, frtb_sa
from riskagg.paths import REPO_ROOT

STUDY_PAGE = REPO_ROOT / "docs/study/04-frtb.md"
DESK_COLS = ["firm_id", "business_line_id", "desk_id"]
DESK = "DESK002"
SCENARIOS = ("LOW", "MEDIUM", "HIGH")

# --- SBM: DESK002, GIRR delta -----------------------------------------------------------------
# risk factor -> (net sensitivity, risk weight in %, weighted sensitivity)
GIRR_WS = {
    "GIRR.USD.1Y": (15758470, "1.1314%", 178287),
    "GIRR.USD.2Y": (24818521, "0.9192%", 228141),
    "GIRR.GBP.1Y": (3291693, "1.1314%", 37241),
}
RHO_1Y_2Y = {"LOW": "0.9409", "MEDIUM": "0.9704", "HIGH": "1.0000"}
GAMMA = {"LOW": "0.375", "MEDIUM": "0.500", "HIGH": "0.625"}
K_USD = {"LOW": 400469, "MEDIUM": 403460, "HIGH": 406428}
S_USD, K_GBP = 406428, 37241
GIRR_DELTA = {"LOW": 416070, "MEDIUM": 423441, "HIGH": 430686}
FX_DELTA = {"LOW": 1689667, "MEDIUM": 1455020, "HIGH": 1174390}
SBM_DESK002 = {"LOW": 6496321, "MEDIUM": 6342869, "HIGH": 6141510}

# --- SA totals: (sbm, drc, rrao, total) per desk and for the firm ----------------------------
SA_TOTALS = {
    "DESK001": (7363134, 0, 0, 7363134),
    "DESK002": (6496321, 879046, 0, 7375367),
    "DESK003": (10194253, 1207208, 140010, 11541471),
    "FIRM": (18751325, 599295, 140010, 19490629),
}
SA_DESK_SUMS = (24053709, 2086254, 140010, 26279972)

# --- DRC: DESK002, CORPORATE bucket ---------------------------------------------------------
# issuer -> (rating, seniority, direction, notional, signed JTD, risk weight %)
DRC_POSITIONS = {
    "ISS009": ("AAA", "NON_SENIOR", "LONG", 15758470, 15758470, "0.5%"),
    "ISS019": ("BBB", "EQUITY", "LONG", 16943207, 16943207, "6%"),
    "ISS013": ("BB", "EQUITY", "SHORT", 1508801, -1508801, "15%"),
}
DRC_LONG, DRC_SHORT = 32701677, 1508801
DRC_RW_LONG, DRC_RW_SHORT = 1095385, 226320
DRC_HBR = "0.9559"
DRC_CHARGE = 879046
SOVEREIGN_SHORT_JTD = 2468770  # ISS030 AAA senior short 3,291,693 x 75%
ISS013_DESK001_JTD, ISS013_FIRM_NET = 3759109, 2250308

# --- IMA: DESK002 -----------------------------------------------------------------------------
# ES set -> 10-day ES of "factors with horizon >= LH_j" for LH_j 10, 20, 40, 60 (120 is empty)
CASCADE = {
    "fc": (1238368, 52871, 52871, 2125),
    "rc": (904182, 52871, 52871, 2125),
    "rs": (3489718, 114030, 114030, 3782),
}
LH_ES = {"fc": 1241753, "rc": 908812, "rs": 3495306}
RATIO = "1.3663"
ES_DESK002 = 4775800
# risk class -> (es_fc, es_rc, es_rs, es)
CLASS_ES = {
    "EQ": (910398, 665721, 4107845, 5617634),
    "FX": (722776, 608532, 1324056, 1572629),
    "GIRR": (217024, 174517, 634934, 789581),
}
IMCC_DESK002 = (4775800, 7979845, 6377822)  # unconstrained, constrained sum, IMCC
IMCC = {"DESK001": 5783239, "DESK002": 6377822, "DESK003": 15967860, "FIRM": 20297172}
IMCC_DESK_SUM = 28128922


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _pct(x: float, places: int = 4) -> str:
    return f"{100 * x:.{places}f}%"


@pytest.fixture(scope="module")
def sa(spark, tiny_market):
    _, hierarchy, _ = tiny_market
    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12)).cache()
    rf = make_risk_factors(spark, p)
    sens = make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf).cache()
    positions = make_default_risk_positions(spark, p, trades).cache()
    inputs = RunInputs(
        sensitivities=sens, trades=trades, hierarchy=hierarchy, default_risk_positions=positions
    )
    yield inputs
    inputs.release()
    for df in (trades, sens, positions):
        df.unpersist()


@pytest.fixture(scope="module")
def ima(tiny_market):
    pnl, hierarchy, n = tiny_market
    grain = build_reporting_grain(pnl, hierarchy, n).cache()
    by_lh = frtb_ima.es_by_horizon(grain, DESK_COLS, n).where(F.col("desk_id") == DESK).collect()
    out = {
        "by_lh": {(r.risk_class, r.es_set, r.liquidity_horizon): r.es for r in by_lh},
        "es": {
            (r.desk_id, r.risk_class): r for r in frtb_ima.ima_es(grain, DESK_COLS, n).collect()
        },
        "imcc": {
            r.desk_id: r for r in frtb_ima.imcc(frtb_ima.ima_es(grain, DESK_COLS, n)).collect()
        },
        "firm": frtb_ima.imcc(frtb_ima.ima_es(grain, ["firm_id"], n)).first(),
    }
    yield out
    grain.unpersist()


def test_sbm_girr_bucket(sa):
    ref = frtb_sa.load_frtb_ref()
    sens = frtb_sa.attach_org(sa.sensitivities, sa.hierarchy, DESK_COLS)
    ws = {
        r.risk_factor_id: r
        for r in frtb_sa.weighted_sensitivities(sens, DESK_COLS, ref)
        .where((F.col("desk_id") == DESK) & (F.col("risk_class") == "GIRR"))
        .where(F.col("component") == "DELTA")
        .collect()
    }
    assert set(ws) == set(GIRR_WS)
    for rf, (s, rw, w) in GIRR_WS.items():
        assert (_money(ws[rf].sensitivity), _pct(ws[rf].risk_weight), _money(ws[rf].weighted_sensitivity)) == (
            _money(s), rw, _money(w)
        )  # fmt: skip
    # MAR21.42 Table 1 divided by sqrt(2) for specified currencies (MAR21.44).
    assert ws["GIRR.USD.1Y"].risk_weight == pytest.approx(0.016 / math.sqrt(2))
    assert ws["GIRR.USD.2Y"].risk_weight == pytest.approx(0.013 / math.sqrt(2))

    # The tenor correlation and its three scenarios (MAR21.46 footnote 13, MAR21.6).
    rho = frtb_sa.girr_delta_rho(1.0, 2.0, ref)
    assert rho == pytest.approx(math.exp(-0.03))
    rhos = {"LOW": max(2 * rho - 1, 0.75 * rho), "MEDIUM": rho, "HIGH": min(1.25 * rho, 1.0)}
    gammas = {"LOW": max(2 * 0.5 - 1, 0.75 * 0.5), "MEDIUM": 0.5, "HIGH": min(1.25 * 0.5, 1.0)}
    for sc in SCENARIOS:
        assert f"{rhos[sc]:.4f}" == RHO_1Y_2Y[sc] and f"{gammas[sc]:.3f}" == GAMMA[sc]

    detail = (
        frtb_sa.bucket_detail(sens, DESK_COLS, ref)
        .where((F.col("desk_id") == DESK) & (F.col("risk_class") == "GIRR"))
        .where(F.col("component") == "DELTA")
        .cache()
    )
    kb = {(r.bucket, r.corr_scenario): r for r in detail.collect()}
    w1, w2 = ws["GIRR.USD.1Y"].weighted_sensitivity, ws["GIRR.USD.2Y"].weighted_sensitivity
    for sc in SCENARIOS:
        usd = kb[("USD", sc)]
        assert _money(usd.k_b) == _money(K_USD[sc]) and _money(usd.s_b) == _money(S_USD)
        assert usd.k_b == pytest.approx(math.sqrt(w1**2 + w2**2 + 2 * rhos[sc] * w1 * w2))
        gbp = kb[("GBP", sc)]
        assert _money(gbp.k_b) == _money(gbp.s_b) == _money(K_GBP)
    fx_detail = (
        frtb_sa.bucket_detail(sens, DESK_COLS, ref)
        .where((F.col("desk_id") == DESK) & (F.col("risk_class") == "FX"))
        .where(F.col("component") == "DELTA")
        .collect()
    )
    charges = {
        (r.risk_class, r.component, r.corr_scenario): r.charge
        for r in frtb_sa.charges_from_detail(
            frtb_sa.bucket_detail(sens, DESK_COLS, ref).where(F.col("desk_id") == DESK), ref
        ).collect()
    }
    for sc in SCENARIOS:
        girr = charges[("GIRR", "DELTA", sc)]
        assert _money(girr) == _money(GIRR_DELTA[sc])
        usd, gbp = kb[("USD", sc)], kb[("GBP", sc)]
        across = usd.k_b**2 + gbp.k_b**2 + 2 * gammas[sc] * usd.s_b * gbp.s_b
        assert girr == pytest.approx(math.sqrt(across))
        assert _money(charges[("FX", "DELTA", sc)]) == _money(FX_DELTA[sc])
        # DESK002 is short GBP and long JPY against USD: the hedge weakens as correlation falls.
        fx = {r.bucket: r.s_b for r in fx_detail if r.corr_scenario == sc}
        assert fx["GBP"] < 0 < fx["JPY"]
    detail.unpersist()


def test_sa_totals_and_binding_scenario(sa):
    level = sa.frtb_level(DESK_COLS)
    totals = {r.desk_id: r for r in level.total.collect()}
    firm = sa.frtb_level(["firm_id"]).total.first()
    for node, row in (*totals.items(), ("FIRM", firm)):
        got = (row.sbm_charge, row.drc_charge, row.rrao_charge, row.charge)
        assert tuple(_money(x) for x in got) == tuple(_money(x) for x in SA_TOTALS[node]), node
    sums = [sum(getattr(r, c) for r in totals.values()) for c in
            ("sbm_charge", "drc_charge", "rrao_charge", "charge")]  # fmt: skip
    assert [_money(x) for x in sums] == [_money(x) for x in SA_DESK_SUMS]
    ref = frtb_sa.load_frtb_ref()
    sens = frtb_sa.attach_org(sa.sensitivities, sa.hierarchy, DESK_COLS)
    sbm = frtb_sa.sbm_total(
        frtb_sa.charges_from_detail(frtb_sa.bucket_detail(sens, DESK_COLS, ref), ref)
    ).where(F.col("desk_id") == DESK)
    by_sc = {r.corr_scenario: r for r in sbm.collect()}
    for sc in SCENARIOS:
        assert _money(by_sc[sc].charge) == _money(SBM_DESK002[sc])
    assert [sc for sc in SCENARIOS if by_sc[sc].is_binding] == ["LOW"]


def test_drc_bucket(sa):
    ref = frtb_drc.load_drc_ref()
    pos = frtb_sa.attach_org(sa.default_risk_positions, sa.hierarchy, DESK_COLS)
    rows = {r.issuer_id: r for r in pos.where(F.col("desk_id") == DESK).collect()}
    for issuer, (rating, seniority, direction, notional, jtd, rw) in DRC_POSITIONS.items():
        r = rows[issuer]
        assert (r.drc_bucket, r.rating, r.seniority, r.direction) == (
            "CORPORATE", rating, seniority, direction
        )  # fmt: skip
        signed = (1 if direction == "LONG" else -1) * ref.lgd[seniority] * r.notional
        assert (_money(r.notional), _money(signed)) == (_money(notional), _money(jtd))
        assert f"{100 * ref.risk_weights[rating]:g}%" == rw
    sovereign = [r for r in rows.values() if r.drc_bucket == "SOVEREIGN"]
    assert len(sovereign) == 1 and sovereign[0].direction == "SHORT"
    assert _money(ref.lgd[sovereign[0].seniority] * sovereign[0].notional) == _money(
        SOVEREIGN_SHORT_JTD
    )
    longs = sum(v[4] for v in DRC_POSITIONS.values() if v[4] > 0)
    assert _money(longs) == _money(DRC_LONG)
    charge = {
        r.drc_bucket: r for r in frtb_drc.drc(pos, DESK_COLS, ref).collect() if r.desk_id == DESK
    }
    corp = charge["CORPORATE"]
    assert f"{corp.hbr:.4f}" == DRC_HBR
    assert corp.hbr == pytest.approx(DRC_LONG / (DRC_LONG + DRC_SHORT), abs=1e-6)
    assert _money(corp.charge) == _money(DRC_CHARGE)
    assert corp.charge == pytest.approx(DRC_RW_LONG - corp.hbr * DRC_RW_SHORT, abs=1.0)
    assert (charge["SOVEREIGN"].hbr, charge["SOVEREIGN"].charge) == (0.0, 0.0)
    # Issuer ISS013 is long at DESK001 (senior) and short at DESK002 (equity): it nets at firm.
    all_pos = [r for r in pos.where(F.col("issuer_id") == "ISS013").collect()]
    desk001 = [r for r in all_pos if r.desk_id == "DESK001"]
    assert len(desk001) == 1 and desk001[0].seniority == "SENIOR"
    jtd001 = ref.lgd["SENIOR"] * desk001[0].notional
    assert _money(jtd001) == _money(ISS013_DESK001_JTD)
    assert _money(jtd001 - DRC_POSITIONS["ISS013"][3]) == _money(ISS013_FIRM_NET)


def test_ima_cascade_and_imcc(ima):
    by_lh = ima["by_lh"]
    horizons = (10, 20, 40, 60)
    for es_set, values in CASCADE.items():
        got = tuple(by_lh.get(("*", es_set, lh), 0.0) for lh in horizons)
        assert tuple(_money(x) for x in got) == tuple(_money(x) for x in values), es_set
        assert ("*", es_set, 120) not in by_lh
        assert frtb_ima.lh_adjusted_es(dict(zip(horizons, got, strict=True))) == pytest.approx(
            getattr(ima["es"][(DESK, "*")], f"es_{es_set}")
        )
    row = ima["es"][(DESK, "*")]
    assert {k: _money(getattr(row, f"es_{k}")) for k in LH_ES} == {
        k: _money(v) for k, v in LH_ES.items()
    }
    assert f"{row.es_fc / row.es_rc:.4f}" == RATIO
    assert _money(row.es) == _money(ES_DESK002)
    for rc, values in CLASS_ES.items():
        r = ima["es"][(DESK, rc)]
        assert tuple(_money(x) for x in (r.es_fc, r.es_rc, r.es_rs, r.es)) == tuple(
            _money(x) for x in values
        ), rc
    m = ima["imcc"][DESK]
    got = (m.imcc_unconstrained, m.imcc_constrained_sum, m.imcc)
    assert tuple(_money(x) for x in got) == tuple(_money(x) for x in IMCC_DESK002)
    for desk in ("DESK001", "DESK002", "DESK003"):
        assert _money(ima["imcc"][desk].imcc) == _money(IMCC[desk])
    assert _money(ima["firm"].imcc) == _money(IMCC["FIRM"])
    # DESK003's only equity factor is outside the reduced set: its stress-calibrated EQ ES is 0.
    eq3 = ima["es"][("DESK003", "EQ")]
    assert eq3.es_fc > 0 and eq3.es_rc == 0 and eq3.es_rs == 0 and eq3.es == 0
    assert _money(sum(ima["imcc"][d].imcc for d in ima["imcc"])) == _money(IMCC_DESK_SUM)


def test_study_page_prints_these_numbers():
    text = STUDY_PAGE.read_text()
    money = [
        *(v for s, _, w in GIRR_WS.values() for v in (s, w)),
        *K_USD.values(), S_USD, K_GBP, *GIRR_DELTA.values(), *FX_DELTA.values(),
        *SBM_DESK002.values(),
        *(x for t in SA_TOTALS.values() for x in t), *SA_DESK_SUMS,
        *(x for v in DRC_POSITIONS.values() for x in (v[3], abs(v[4]))),
        DRC_LONG, DRC_SHORT, DRC_RW_LONG, DRC_RW_SHORT, DRC_CHARGE, SOVEREIGN_SHORT_JTD,
        ISS013_DESK001_JTD, ISS013_FIRM_NET,
        *(x for t in CASCADE.values() for x in t), *LH_ES.values(), ES_DESK002,
        *(x for t in CLASS_ES.values() for x in t), *IMCC_DESK002, *IMCC.values(), IMCC_DESK_SUM,
    ]  # fmt: skip
    for value in money:
        if value:
            assert _money(value) in text, value
    for literal in (
        *(rw for _, rw, _ in GIRR_WS.values()),
        *RHO_1Y_2Y.values(), *GAMMA.values(), DRC_HBR, RATIO,
        *(v[5] for v in DRC_POSITIONS.values()),
    ):  # fmt: skip
        assert literal in text, literal
