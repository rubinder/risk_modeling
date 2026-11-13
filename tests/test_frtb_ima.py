import logging
import math
import shutil

import pytest
from fastapi.testclient import TestClient
from pyspark.sql import functions as F

from reference import np_ima
from riskagg.context import RunContext, run_context
from riskagg.engine import audit
from riskagg.engine.grain import build_reporting_grain
from riskagg.engine.inputs import RunInputs
from riskagg.engine.runner import MEASURE_REGISTRY
from riskagg.engine.spec import load_specs
from riskagg.measures.frtb_ima import (
    IMA_MEASURES,
    es_by_horizon,
    ima_es,
    imcc,
    lh_adjusted_es,
    scaled_es,
    stressed_scaling,
)
from riskagg.semantic.model import load_model
from riskagg.semantic.pinot_schema import pinot_table_config
from riskagg.serving.api import create_app

DOWN = "http://127.0.0.1:9"
DESK = ["firm_id", "business_line_id", "desk_id"]
GRAIN_SCHEMA = (
    "desk_id string, risk_class string, liquidity_horizon int, in_reduced_set boolean, "
    "window string, pnl array<double>"
)
ROOT10 = math.sqrt(10)


# --- the formulas -------------------------------------------------------------------------


def test_liquidity_horizon_cascade():
    # sqrt(100^2 + (60 x 1)^2 + (30 x sqrt(2))^2) = sqrt(15400)
    got = lh_adjusted_es({10: 100, 20: 60, 40: 30, 60: 0, 120: 0})
    assert got == pytest.approx(124.097, abs=1e-3)
    assert got == pytest.approx(math.sqrt(15400))
    # Missing horizons count as zero; a lone base-horizon ES passes through.
    assert lh_adjusted_es({10: 100}) == pytest.approx(100.0)
    # The 120-day bucket is weighted sqrt((120 - 60) / 10) = sqrt(6).
    assert lh_adjusted_es({10: 0, 120: 10}) == pytest.approx(10 * math.sqrt(6))


def test_stressed_scaling_pure():
    assert scaled_es(150, 110, 100) == pytest.approx(165.0)
    assert scaled_es(150, 90, 100) == pytest.approx(150.0)  # ratio floored at 1 (MAR33.6)
    assert scaled_es(150, 50, 0) == pytest.approx(150.0)  # es_rc 0: ratio 1


def test_stressed_scaling_spark(spark):
    df = spark.createDataFrame(
        [("a", 150.0, 110.0, 100.0), ("b", 150.0, 90.0, 100.0), ("c", 150.0, 50.0, 0.0)],
        "k string, es_rs double, es_fc double, es_rc double",
    )
    got = {r["k"]: r["es"] for r in stressed_scaling(df).collect()}
    assert got == {"a": pytest.approx(165.0), "b": pytest.approx(150.0), "c": pytest.approx(150.0)}


def test_imcc_weights_unconstrained_and_constrained_equally(spark):
    es = spark.createDataFrame(
        [("D1", "*", 165.0), ("D1", "GIRR", 120.0), ("D1", "EQ", 80.0), ("D2", "*", 10.0),
         ("D2", "FX", 10.0)],
        "desk_id string, risk_class string, es double",
    )  # fmt: skip
    got = {r["desk_id"]: r.asDict() for r in imcc(es).collect()}
    assert got["D1"] == {
        "desk_id": "D1",
        "imcc_unconstrained": pytest.approx(165.0),
        "imcc_constrained_sum": pytest.approx(200.0),
        "imcc": pytest.approx(182.5),
    }
    assert got["D2"]["imcc"] == pytest.approx(10.0)


def _grain(spark, rows):
    return spark.createDataFrame(rows, GRAIN_SCHEMA)


@pytest.fixture(scope="module")
def hand_grain(spark):
    """Four scenarios; at 97.5% the tail is 0.1 of a scenario, so ES = the worst loss.

    D1 EQ:  CURRENT  10d reduced     [-10, 0, 0, 0]
            CURRENT  20d not reduced [0, -5, 0, 0]
            STRESSED 10d reduced     [-30, 0, 0, 0]
            STRESSED 20d not reduced [-99, 0, 0, 0]  (not in any set)
    D1 FX:  CURRENT  10d reduced     [0, 0, -8, 0]
            STRESSED 10d reduced     [0, 0, -8, 0]
    """
    return _grain(spark, [
        ("D1", "EQ", 10, True, "CURRENT", [-10.0, 0.0, 0.0, 0.0]),
        ("D1", "EQ", 20, False, "CURRENT", [0.0, -5.0, 0.0, 0.0]),
        ("D1", "EQ", 10, True, "STRESSED", [-30.0, 0.0, 0.0, 0.0]),
        ("D1", "EQ", 20, False, "STRESSED", [-99.0, 0.0, 0.0, 0.0]),
        ("D1", "FX", 10, True, "CURRENT", [0.0, 0.0, -8.0, 0.0]),
        ("D1", "FX", 10, True, "STRESSED", [0.0, 0.0, -8.0, 0.0]),
    ])  # fmt: skip


def test_es_by_horizon_builds_cumulative_ten_day_vectors(hand_grain):
    rows = es_by_horizon(hand_grain, ["desk_id"], 4).collect()
    got = {(r["risk_class"], r["es_set"], r["liquidity_horizon"]): r["es"] for r in rows}
    assert got == {
        ("EQ", "fc", 10): pytest.approx(10 * ROOT10),
        ("EQ", "fc", 20): pytest.approx(5 * ROOT10),
        ("EQ", "rc", 10): pytest.approx(10 * ROOT10),
        ("EQ", "rs", 10): pytest.approx(30 * ROOT10),
        ("FX", "fc", 10): pytest.approx(8 * ROOT10),
        ("FX", "rc", 10): pytest.approx(8 * ROOT10),
        ("FX", "rs", 10): pytest.approx(8 * ROOT10),
        # all classes: [-10, -5, -8, 0] current, [-30, 0, -8, 0] stressed reduced
        ("*", "fc", 10): pytest.approx(10 * ROOT10),
        ("*", "fc", 20): pytest.approx(5 * ROOT10),
        ("*", "rc", 10): pytest.approx(10 * ROOT10),
        ("*", "rs", 10): pytest.approx(30 * ROOT10),
    }


def test_ima_es_by_hand(hand_grain):
    got = {r["risk_class"]: r.asDict() for r in ima_es(hand_grain, ["desk_id"], 4).collect()}
    eq = got["EQ"]
    assert eq["es_fc"] == pytest.approx(math.sqrt(1250))  # sqrt(100 + 25) x sqrt(10)
    assert eq["es_rc"] == pytest.approx(math.sqrt(1000))
    assert eq["es_rs"] == pytest.approx(math.sqrt(9000))
    assert eq["es"] == pytest.approx(math.sqrt(9000) * math.sqrt(1.25))  # = sqrt(11250)
    assert got["FX"]["es"] == pytest.approx(8 * ROOT10)  # ratio 1
    assert got["*"]["es"] == pytest.approx(math.sqrt(11250))
    (node,) = imcc(ima_es(hand_grain, ["desk_id"], 4)).collect()
    assert node["imcc_unconstrained"] == pytest.approx(math.sqrt(11250))
    assert node["imcc_constrained_sum"] == pytest.approx(math.sqrt(11250) + 8 * ROOT10)
    assert node["imcc"] == pytest.approx(math.sqrt(11250) + 4 * ROOT10)


def test_unknown_liquidity_horizon_is_an_error(spark):
    bad = _grain(spark, [("D1", "EQ", 15, True, "CURRENT", [-1.0, 0.0, 0.0, 0.0])])
    with pytest.raises(Exception, match="liquidity horizon"):
        ima_es(bad, ["desk_id"], 4).collect()


# --- the tiny profile against the NumPy reference --------------------------------------------


@pytest.fixture(scope="module")
def tiny(spark, tiny_market):
    pnl, hierarchy, n = tiny_market
    grain = build_reporting_grain(pnl, hierarchy, n).cache()
    yield grain, pnl.toPandas(), hierarchy.toPandas(), n
    grain.unpersist()


@pytest.mark.parametrize("node_cols", [["firm_id"], DESK])
def test_tiny_matches_numpy_reference(tiny, node_cols):
    grain, pnl_pdf, hier_pdf, n = tiny
    expected = np_ima.ima_es(pnl_pdf, hier_pdf, node_cols)
    es = ima_es(grain, node_cols, n).cache()
    got = {(tuple(r[c] for c in node_cols), r["risk_class"]): r.asDict() for r in es.collect()}
    assert got.keys() == expected.keys()
    assert {rc for _, rc in got} == {"*", "GIRR", "EQ", "FX"}
    for key, exp in expected.items():
        for col in ("es_fc", "es_rc", "es_rs", "es"):
            assert got[key][col] == pytest.approx(exp[col], rel=1e-9, abs=1e-6), (key, col)
        assert got[key]["es_fc"] > 0
        if key[1] == "*":
            assert got[key]["es_rs"] > 0
        if got[key]["es_rc"] == 0:
            # No reduced-set factor in this scope (DESK003 EQ on tiny): es_rs is 0 too, so the
            # stress-calibrated ES is 0, which MAR33.5's 75% explanatory test exists to catch.
            assert got[key]["es_rs"] == 0 and got[key]["es"] == 0
    exp_imcc = np_ima.imcc(expected)
    rows = imcc(es).collect()
    assert len(rows) == len(exp_imcc)
    for r in rows:
        exp = exp_imcc[tuple(r[c] for c in node_cols)]
        for col in ("imcc_unconstrained", "imcc_constrained_sum", "imcc"):
            assert r[col] == pytest.approx(exp[col], rel=1e-9), col
        # Diversification across risk classes: the unconstrained ES is below the sum.
        assert r["imcc_unconstrained"] <= r["imcc_constrained_sum"]
    es.unpersist()


def test_no_stressed_window_gives_no_rows(spark, tiny, caplog):
    """Without a stressed window (the bench profile) IMA is not computed, never reported as 0,
    and the run says so in a warning."""
    grain = tiny[0].where("window = 'CURRENT'")
    model = load_model()
    specs = {s.name: s for s in load_specs(None, model)}
    inputs = RunInputs(grain=grain, n_scenarios=tiny[3])
    try:
        with caplog.at_level(logging.WARNING, logger="riskagg.measures.frtb_ima"):
            rows = MEASURE_REGISTRY["frtb_imcc"](specs["frtb_imcc_by_org"], model, inputs)
            assert rows.count() == 0
        assert any(
            r.levelno == logging.WARNING and "no STRESSED window" in r.getMessage()
            for r in caplog.records
        ), caplog.text
    finally:
        inputs.release()


# --- the run --------------------------------------------------------------------------------


def _stored(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    df = spark.read.parquet(str(ctx.curated("risk_metrics")))
    out: dict[tuple, dict[str, float]] = {}
    for r in df.where(F.col("measure").isin(*IMA_MEASURES)).collect():
        assert r["scenario"] == "*"
        out.setdefault((r["node_level"], r["desk_id"]), {})[r["measure"]] = r["value"]
    return out


def test_run_stores_imcc_at_firm_and_desk(spark, tmp_run, tiny):
    nodes = _stored(spark, tmp_run)
    assert {lvl for lvl, _ in nodes} == {"firm_id", "firm_id|business_line_id|desk_id"}
    _, pnl_pdf, hier_pdf, _ = tiny
    for level, cols in (("firm_id", ["firm_id"]), ("firm_id|business_line_id|desk_id", DESK)):
        expected = np_ima.imcc(np_ima.ima_es(pnl_pdf, hier_pdf, cols))
        got = {k: v for k, v in nodes.items() if k[0] == level}
        assert len(got) == len(expected)
        for node, exp in expected.items():
            m = got[(level, node[-1] if level != "firm_id" else "*")]
            assert m["frtb_imcc"] == pytest.approx(exp["imcc"], rel=1e-9)
            assert m["frtb_imcc_unconstrained"] == pytest.approx(exp["imcc_unconstrained"])
            assert m["frtb_imcc_constrained_sum"] == pytest.approx(exp["imcc_constrained_sum"])
    firm = nodes[("firm_id", "*")]["frtb_imcc"]
    desks = sum(v["frtb_imcc"] for k, v in nodes.items() if k[0] != "firm_id")
    assert firm != pytest.approx(desks)  # recomputed at the firm, not the sum of desks


def test_ima_detail_table_curated(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    df = spark.read.parquet(str(ctx.curated("frtb_ima_detail")))
    cols = set(tmp_run.model.tables["frtb_ima_detail"]["columns"]) - {"as_of_date", "run_id"}
    assert set(df.columns) == cols
    assert {r[0] for r in df.select("node_level").distinct().collect()} == {
        "firm_id", "firm_id|business_line_id|desk_id"
    }  # fmt: skip
    stored = _stored(spark, tmp_run)
    for r in df.where("risk_class = '*'").collect():
        assert stored[(r["node_level"], r["desk_id"])]["frtb_imcc_unconstrained"] == (
            pytest.approx(r["es"])
        )
    assert df.where("es < es_rs - 1e-9").count() == 0  # the ratio is floored at 1
    assert (
        "starTreeIndexConfigs"
        not in (pinot_table_config(tmp_run.model, "frtb_ima_detail")["tableIndexConfig"])
    )


def test_registry_and_audit(spark, tmp_run):
    for name in IMA_MEASURES:
        assert name in MEASURE_REGISTRY
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    check = audit.check_frtb_imcc_from_raw(ctx, spark)
    assert check.passed, check.detail
    assert audit.check_frtb_imcc_from_raw in audit.CHECKS


def test_audit_catches_a_wrong_stored_imcc(spark, tmp_run, tmp_path):
    """A firm IMCC written from the wrong frame (here: doubled) fails the reconciliation."""
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    bad = RunContext(ctx.profile, ctx.as_of_date, ctx.run_id, ctx.spec_hash, tmp_path)
    shutil.copytree(ctx.raw("pnl_vectors"), bad.raw("pnl_vectors"))
    metrics = spark.read.parquet(str(ctx.curated("risk_metrics")))
    doubled = F.when(
        (F.col("measure") == "frtb_imcc") & (F.col("node_level") == "firm_id"), F.col("value") * 2
    ).otherwise(F.col("value"))
    metrics.withColumn("value", doubled).write.parquet(str(bad.curated("risk_metrics")))
    check = audit.check_frtb_imcc_from_raw(bad, spark)
    assert not check.passed and "frtb_imcc" in check.detail


def test_api_answers_firm_and_desk_and_refuses_book(tmp_run):
    api = TestClient(create_app(tmp_run.root, broker_url=DOWN, controller_url=DOWN))
    firm = api.get("/metrics/frtb_imcc", params={"group_by": "firm_id"})
    assert firm.status_code == 200, firm.text
    assert firm.json()["rows"][0]["value"] > 0
    desk = api.get("/metrics/frtb_imcc", params={"group_by": ",".join(DESK)})
    assert desk.status_code == 200 and len(desk.json()["rows"]) == 3
    book = api.get("/metrics/frtb_imcc", params={"group_by": ",".join([*DESK, "book_id"])})
    assert book.status_code == 422
    drill = api.get("/drilldown/frtb_ima_detail", params={"limit": 3})
    assert drill.status_code == 200, drill.text
