import datetime

import pytest
from fastapi.testclient import TestClient
from pyspark.sql import functions as F

from riskagg.context import run_context
from riskagg.data.banking import N_ISSUERS, make_default_risk_positions
from riskagg.data.market import make_exposures, make_sensitivities
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine import audit
from riskagg.engine.runner import MEASURE_REGISTRY
from riskagg.measures.frtb_drc import (
    FRTB_COMPONENT_MEASURES,
    drc,
    frtb_sa_total,
    load_drc_ref,
    rrao,
)
from riskagg.measures.frtb_sa import attach_org, charges_from_detail, load_frtb_ref, sbm_total
from riskagg.semantic.pinot_schema import pinot_schema, pinot_table_config
from riskagg.serving.api import create_app
from riskagg.serving.service import RiskService

POS_SCHEMA = (
    "trade_id long, book_id string, desk_id string, issuer_id string, drc_bucket string, "
    "rating string, seniority string, direction string, notional double"
)
DOWN = "http://127.0.0.1:9"
DESK = ["firm_id", "business_line_id", "desk_id"]


@pytest.fixture(scope="module")
def ref():
    return load_drc_ref()


def _pos(spark, rows):
    return spark.createDataFrame(rows, POS_SCHEMA)


def test_reference_values_follow_basel():
    r = load_drc_ref()
    assert r.lgd == {"SENIOR": 0.75, "NON_SENIOR": 1.0, "EQUITY": 1.0}  # MAR22.12
    assert r.risk_weights == {  # MAR22.24
        "AAA": 0.005, "AA": 0.02, "A": 0.03, "BBB": 0.06, "BB": 0.15, "B": 0.30,
        "CCC": 0.50, "UNRATED": 0.15, "DEFAULTED": 1.0,
    }  # fmt: skip
    assert (r.rrao_exotic, r.rrao_other) == (0.01, 0.001)  # MAR23.8(2)


def test_hedge_benefit_and_charge(spark, ref):
    """Long 100 senior BBB, short 40 senior BB: JTD 75 and 30; HBR 75/105."""
    pos = _pos(spark, [
        (1, "B1", "D1", "I1", "CORPORATE", "BBB", "SENIOR", "LONG", 100.0),
        (2, "B1", "D1", "I2", "CORPORATE", "BB", "SENIOR", "SHORT", 40.0),
    ])  # fmt: skip
    (row,) = drc(pos, ["desk_id"], ref).collect()
    assert row["drc_bucket"] == "CORPORATE"
    assert row["hbr"] == pytest.approx(75 / 105)
    assert row["charge"] == pytest.approx(75 * 0.06 - (75 / 105) * 30 * 0.15)
    assert row["charge"] == pytest.approx(1.285714, abs=1e-6)


def test_only_shorts_give_zero_without_dividing_by_zero(spark, ref):
    pos = _pos(spark, [(1, "B1", "D1", "I1", "CORPORATE", "BB", "SENIOR", "SHORT", 50.0)])
    (row,) = drc(pos, ["desk_id"], ref).collect()
    assert row["hbr"] == 0.0 and row["charge"] == 0.0


def test_only_longs_charge_the_full_weighted_jtd(spark, ref):
    pos = _pos(spark, [(1, "B1", "D1", "I1", "SOVEREIGN", "A", "NON_SENIOR", "LONG", 200.0)])
    (row,) = drc(pos, ["desk_id"], ref).collect()
    assert row["hbr"] == 1.0 and row["charge"] == pytest.approx(200 * 1.0 * 0.03)


def test_same_issuer_nets_and_lgd_follows_seniority(spark, ref):
    """Senior long 100 (JTD 75) against equity short 50 (JTD 50) in one issuer nets to long 25."""
    pos = _pos(spark, [
        (1, "B1", "D1", "I1", "CORPORATE", "BBB", "SENIOR", "LONG", 100.0),
        (2, "B1", "D1", "I1", "CORPORATE", "BBB", "EQUITY", "SHORT", 50.0),
    ])  # fmt: skip
    (row,) = drc(pos, ["desk_id"], ref).collect()
    assert row["charge"] == pytest.approx(25 * 0.06) and row["hbr"] == 1.0


def test_buckets_are_charged_separately_and_nodes_apart(spark, ref):
    pos = _pos(spark, [
        (1, "B1", "D1", "I1", "CORPORATE", "BBB", "SENIOR", "LONG", 100.0),
        (2, "B2", "D2", "I2", "SOVEREIGN", "AA", "SENIOR", "LONG", 100.0),
        (3, "B3", "D2", "I1", "CORPORATE", "BBB", "SENIOR", "SHORT", 100.0),
    ])  # fmt: skip
    got = {
        (r["desk_id"], r["drc_bucket"]): r["charge"] for r in drc(pos, ["desk_id"], ref).collect()
    }
    assert got == {
        ("D1", "CORPORATE"): pytest.approx(75 * 0.06),
        ("D2", "SOVEREIGN"): pytest.approx(75 * 0.02),
        ("D2", "CORPORATE"): 0.0,
    }
    (firm,) = (
        drc(pos.withColumn("f", F.lit("F")), ["f"], ref).where("drc_bucket='CORPORATE'").collect()
    )
    assert firm["charge"] == 0.0  # the short at D2 nets the long at D1 at firm level


def test_unknown_rating_or_seniority_is_an_error(spark, ref):
    bad = _pos(spark, [(1, "B1", "D1", "I1", "CORPORATE", "XX", "SENIOR", "LONG", 1.0)])
    with pytest.raises(Exception, match="risk weight"):
        drc(bad, ["desk_id"], ref).collect()
    bad = _pos(spark, [(1, "B1", "D1", "I1", "CORPORATE", "A", "JUNIOR", "LONG", 1.0)])
    with pytest.raises(Exception, match="LGD"):
        drc(bad, ["desk_id"], ref).collect()


def test_rrao_is_one_percent_of_exotic_notional(spark, ref):
    trades = spark.createDataFrame(
        [("D1", 600_000.0, True), ("D1", 400_000.0, True), ("D1", 5_000_000.0, False),
         ("D2", 1.0, False)],
        "desk_id string, notional double, is_exotic boolean",
    )  # fmt: skip
    got = {r["desk_id"]: r["charge"] for r in rrao(trades, ["desk_id"], ref).collect()}
    assert got == {"D1": pytest.approx(10_000.0)}


def test_total_counts_missing_components_as_zero(spark):
    sbm = spark.createDataFrame(
        [("D1", "LOW", 5.0, False), ("D1", "HIGH", 9.0, True), ("D2", "HIGH", 3.0, True)],
        "desk_id string, corr_scenario string, charge double, is_binding boolean",
    )
    d = spark.createDataFrame(
        [
            ("D1", "CORPORATE", 0.5, 1.0),
            ("D1", "SOVEREIGN", 0.5, 2.0),
            ("D3", "CORPORATE", 0.0, 4.0),
        ],
        "desk_id string, drc_bucket string, hbr double, charge double",
    )
    r = spark.createDataFrame([("D2", 7.0)], "desk_id string, charge double")
    got = {x["desk_id"]: x.asDict() for x in frtb_sa_total(sbm, d, r).collect()}
    assert got["D1"] == {"desk_id": "D1", "sbm_charge": 9.0, "drc_charge": 3.0,
                         "rrao_charge": 0.0, "charge": 12.0}  # fmt: skip
    assert got["D2"]["charge"] == 10.0 and got["D3"]["charge"] == 4.0


def test_attach_org_fails_loudly_on_unknown_book(spark):
    hierarchy = spark.createDataFrame([("B1", "D1")], "book_id string, desk_id string")
    ok = spark.createDataFrame([("B1", 1.0)], "book_id string, x double")
    assert attach_org(ok, hierarchy, ["desk_id"]).count() == 1
    bad = spark.createDataFrame([("B1", 1.0), ("B9", 2.0)], "book_id string, x double")
    with pytest.raises(ValueError, match="B9"):
        attach_org(bad, hierarchy, ["desk_id"])


def test_default_risk_positions_derivation(spark):
    p = PROFILES["demo"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12)).limit(3000).cache()
    pos = make_default_risk_positions(spark, p, trades)
    again = make_default_risk_positions(spark, p, trades)
    assert sorted(map(tuple, pos.collect())) == sorted(map(tuple, again.collect()))
    joined = pos.join(
        trades.select("trade_id", "asset_class", F.col("notional").alias("trade_notional")),
        "trade_id",
    )
    assert (
        joined.count() == pos.count() and pos.select("trade_id").distinct().count() == pos.count()
    )
    assert joined.where("asset_class = 'FX'").count() == 0
    assert joined.where("(asset_class = 'EQUITY') != (seniority = 'EQUITY')").count() == 0
    assert joined.where("asset_class = 'EQUITY' and drc_bucket != 'CORPORATE'").count() == 0
    assert joined.where("notional != trade_notional").count() == 0
    assert {r[0] for r in pos.select("direction").distinct().collect()} == {"LONG", "SHORT"}
    assert {r[0] for r in pos.select("seniority").distinct().collect()} == {
        "EQUITY", "SENIOR", "NON_SENIOR",
    }  # fmt: skip
    assert {r[0] for r in pos.select("drc_bucket").distinct().collect()} == {
        "CORPORATE", "SOVEREIGN",
    }  # fmt: skip
    issuers = pos.groupBy("issuer_id").agg(
        F.countDistinct("rating").alias("r"), F.countDistinct("drc_bucket").alias("b")
    )
    assert issuers.where("r != 1 or b != 1").count() == 0
    assert issuers.count() <= N_ISSUERS
    trades.unpersist()


# --- the tiny run ---------------------------------------------------------------------------


def _metrics(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    df = spark.read.parquet(str(ctx.curated("risk_metrics")))
    rows = df.where(F.col("measure").isin(*FRTB_COMPONENT_MEASURES)).collect()
    out: dict[tuple, dict[str, float]] = {}
    for r in rows:
        assert r["scenario"] == "*"
        out.setdefault((r["node_level"], r["desk_id"]), {})[r["measure"]] = r["value"]
    return out


def test_total_is_sum_of_components_at_desk_and_firm_on_tiny(spark, tmp_run):
    nodes = _metrics(spark, tmp_run)
    levels = {lvl for lvl, _ in nodes}
    assert levels == {"firm_id", "firm_id|business_line_id|desk_id"}
    assert len([k for k in nodes if k[0] == "firm_id"]) == 1
    assert len([k for k in nodes if k[0] != "firm_id"]) >= 3
    for key, m in nodes.items():
        assert set(m) == set(FRTB_COMPONENT_MEASURES), key
        parts = m["frtb_sbm_charge"] + m["frtb_drc_charge"] + m["frtb_rrao_charge"]
        assert m["frtb_sa_charge"] == pytest.approx(parts, rel=1e-9, abs=1e-9), key
        assert m["frtb_sbm_charge"] > 0 and m["frtb_drc_charge"] >= 0
    firm = nodes[("firm_id", "*")]
    assert firm["frtb_drc_charge"] > 0 and firm["frtb_rrao_charge"] > 0


def test_detail_table_curated_at_both_levels(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    df = spark.read.parquet(str(ctx.curated("frtb_sa_detail")))
    assert set(df.columns) == set(tmp_run.model.tables["frtb_sa_detail"]["columns"]) - {
        "as_of_date", "run_id"
    }  # fmt: skip
    assert {r[0] for r in df.select("node_level").distinct().collect()} == {
        "firm_id", "firm_id|business_line_id|desk_id"
    }  # fmt: skip
    firm = df.where("node_level = 'firm_id'")
    assert firm.where("desk_id != '*' or business_line_id != '*'").count() == 0
    assert {r[0] for r in df.select("corr_scenario").distinct().collect()} == {
        "LOW", "MEDIUM", "HIGH"
    }  # fmt: skip
    assert df.where("k_b < 0").count() == 0
    # The stored SBM charge of every node equals the binding scenario's total rebuilt from that
    # node's own curated detail (not from any other level).
    stored = {
        (r["node_level"], r["desk_id"]): r["value"]
        for r in spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where("measure = 'frtb_sbm_charge'")
        .collect()
    }
    for level, cols in (("firm_id", ["firm_id"]), ("firm_id|business_line_id|desk_id", DESK)):
        node = df.where(F.col("node_level") == level).select(
            *cols, "risk_class", "component", "bucket", "corr_scenario", "k_b", "s_b"
        )
        rebuilt = sbm_total(charges_from_detail(node, load_frtb_ref())).where("is_binding")
        got = {(level, r["desk_id"] if "desk_id" in cols else "*"): r["charge"]
               for r in rebuilt.collect()}  # fmt: skip
        assert got.keys() == {k for k in stored if k[0] == level}
        for key, value in got.items():
            assert stored[key] == pytest.approx(value, rel=1e-9, abs=1e-9), key


def test_registry_and_shipped_specs():
    for name in FRTB_COMPONENT_MEASURES:
        assert name in MEASURE_REGISTRY


def test_reconciliation_includes_frtb_check(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    check = audit.check_frtb_sa_is_sum_of_components(ctx, spark)
    assert check.passed, check.detail


def test_detail_table_has_no_star_tree_and_two_metric_columns(tmp_run):
    model = tmp_run.model
    assert (
        "starTreeIndexConfigs"
        not in pinot_table_config(model, "frtb_sa_detail")["tableIndexConfig"]
    )
    schema = pinot_schema(model, "frtb_sa_detail")
    assert [f["name"] for f in schema["metricFieldSpecs"]] == ["k_b", "s_b"]
    drill = RiskService(tmp_run.root, model, DOWN, DOWN).drilldown("frtb_sa_detail", limit=5)
    assert len(drill.rows) == 5 and {"k_b", "s_b"} <= set(drill.rows.columns)


def test_api_answers_firm_and_desk_and_refuses_book(tmp_run):
    api = TestClient(create_app(tmp_run.root, broker_url=DOWN, controller_url=DOWN))
    firm = api.get("/metrics/frtb_sa_charge", params={"group_by": "firm_id"})
    assert firm.status_code == 200, firm.text
    (row,) = firm.json()["rows"]
    assert row["value"] > 0
    desk = api.get("/metrics/frtb_sa_charge", params={"group_by": ",".join(DESK)})
    assert desk.status_code == 200 and len(desk.json()["rows"]) >= 3
    book = api.get("/metrics/frtb_sa_charge", params={"group_by": ",".join([*DESK, "book_id"])})
    assert book.status_code == 422 and "frtb_sa_charge" in book.json()["detail"]
    assert api.get("/metrics/frtb_sa_charge", params={"group_by": "book_id"}).status_code == 422


def test_sbm_total_reused_by_level(spark):
    """build_level is memoised per node level and released."""
    from riskagg.engine.inputs import RunInputs

    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12))
    hierarchy = make_hierarchy(spark, p)
    rf = make_risk_factors(spark, p)
    sens = make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf)
    inputs = RunInputs(
        sensitivities=sens, trades=trades, hierarchy=hierarchy,
        default_risk_positions=make_default_risk_positions(spark, p, trades),
    )  # fmt: skip
    a = inputs.frtb_level(DESK)
    assert inputs.frtb_level(list(DESK)) is a
    assert a.total.count() >= 3
    inputs.release()
    assert inputs._frtb_levels == {}


def test_nodes_are_recomputed_not_rolled_up_through_the_run_path(spark):
    """Two desks hold offsetting positions; the firm rows come from netting at firm level.

    Desk D1: long 100 senior BBB issuer X (JTD 75, DRC 75 x 6% = 4.5), EQ bucket 5 delta +100
    (WS 30, SBM 30), exotic trade of 1,000,000 (RRAO 10,000). Desk D2: short 100 senior BBB
    issuer X (no long: DRC 0), EQ bucket 5 delta -100 (SBM 30). Firm: X nets to 0, DRC 0; the
    deltas cancel, SBM 0; RRAO 10,000 (additive). Desk sums would be DRC 4.5 and SBM 60.
    """
    from riskagg.engine.inputs import RunInputs
    from riskagg.engine.runner import MEASURE_REGISTRY
    from riskagg.engine.spec import load_specs
    from riskagg.semantic.model import load_model

    hierarchy = spark.createDataFrame(
        [("B1", "D1", "BL1", "FIRM"), ("B2", "D2", "BL1", "FIRM")],
        "book_id string, desk_id string, business_line_id string, firm_id string",
    )
    sens = spark.createDataFrame(
        [(1, "B1", "EQ.5.A", "EQ", "DELTA", "5", None, 100.0, 0.0, 0.0, 0.0, 0.0),
         (2, "B2", "EQ.5.A", "EQ", "DELTA", "5", None, -100.0, 0.0, 0.0, 0.0, 0.0)],
        "trade_id long, book_id string, risk_factor_id string, risk_class string, "
        "factor_type string, bucket string, tenor_years double, delta double, gamma double, "
        "vega double, curvature_up double, curvature_down double",
    )  # fmt: skip
    trades = spark.createDataFrame(
        [(1, "B1", 1_000_000.0, True), (2, "B2", 5_000_000.0, False)],
        "trade_id long, book_id string, notional double, is_exotic boolean",
    )
    positions = spark.createDataFrame(
        [(1, "B1", "X", "CORPORATE", "BBB", "SENIOR", "LONG", 100.0),
         (2, "B2", "X", "CORPORATE", "BBB", "SENIOR", "SHORT", 100.0)],
        "trade_id long, book_id string, issuer_id string, drc_bucket string, rating string, "
        "seniority string, direction string, notional double",
    )  # fmt: skip
    inputs = RunInputs(
        sensitivities=sens, trades=trades, hierarchy=hierarchy, default_risk_positions=positions
    )
    model = load_model()
    specs = {s.name: s for s in load_specs(None, model)}

    def rows(measure: str, spec: str) -> dict[tuple[str, str], float]:
        df = MEASURE_REGISTRY[measure](specs[spec], model, inputs)
        return {(r["node_level"], r["desk_id"]): r["value"] for r in df.collect()}

    firm, desk = ("firm_id", "*"), "firm_id|business_line_id|desk_id"
    try:
        drc_rows = rows("frtb_drc_charge", "frtb_drc_by_org")
        assert drc_rows[firm] == pytest.approx(0.0)
        assert drc_rows[(desk, "D1")] == pytest.approx(4.5)
        assert drc_rows[(desk, "D2")] == pytest.approx(0.0)
        assert drc_rows[firm] != pytest.approx(drc_rows[(desk, "D1")] + drc_rows[(desk, "D2")])

        sbm_rows = rows("frtb_sbm_charge", "frtb_sbm_by_org")
        assert sbm_rows[firm] == pytest.approx(0.0)
        assert sbm_rows[(desk, "D1")] == pytest.approx(30.0)
        assert sbm_rows[(desk, "D2")] == pytest.approx(30.0)
        assert sbm_rows[firm] != pytest.approx(sbm_rows[(desk, "D1")] + sbm_rows[(desk, "D2")])

        rrao_rows = rows("frtb_rrao_charge", "frtb_rrao_by_org")
        assert rrao_rows[firm] == pytest.approx(10_000.0)
        assert rrao_rows[(desk, "D1")] == pytest.approx(10_000.0)

        sa_rows = rows("frtb_sa_charge", "frtb_sa_by_org")
        assert sa_rows[firm] == pytest.approx(10_000.0)
        assert sa_rows[(desk, "D1")] == pytest.approx(30.0 + 4.5 + 10_000.0)
        assert sa_rows[(desk, "D2")] == pytest.approx(30.0)
    finally:
        inputs.release()
