import dataclasses
import shutil

import pytest
from fastapi.testclient import TestClient
from pyspark.sql import functions as F

from riskagg.context import RunContext, run_context
from riskagg.data.banking import (
    MACRO_PATHS,
    SEGMENT_TYPES,
    make_banking_book,
    make_capital_start,
    make_macro_scenarios,
)
from riskagg.data.profiles import PROFILES
from riskagg.engine import audit
from riskagg.engine.runner import MEASURE_REGISTRY
from riskagg.measures import ccar
from riskagg.paths import ccar_path
from riskagg.semantic.model import SpecError, load_model
from riskagg.semantic.pinot_schema import pinot_table_config
from riskagg.semantic.query import SliceNotAvailable, build_query
from riskagg.serving.api import create_app

DOWN = "http://127.0.0.1:1"
PARAMS = ccar.load_params()
BOOK_COLS = (
    "segment_id string, legal_entity string, business_line_id string, ead double, "
    "pd_base double, lgd double, revenue_rate_q double, expense_rate_q double, risk_weight double"
)
MACRO_COLS = "scenario string, quarter int, unemployment_rate double, gdp_growth double"
CAPITAL_COLS = "legal_entity string, cet1_start double, rwa_trading double, dividends_q double"
GMS_COLS = "legal_entity string, business_line_id string, trading_loss double"


def _macro(spark, paths):
    rows = [(s, q, u, g) for s, path in paths.items() for q, (u, g) in enumerate(path)]
    return spark.createDataFrame(rows, MACRO_COLS)


def _flat(u0, u, g):
    """Quarter 0 at ``u0``, then nine quarters at ``u`` and ``g``."""
    return [(u0, 0.0), *[(u, g)] * 9]


def _one_entity(spark, sa_path, baseline_path=None, trading_loss=20.0, dividends=1.0):
    """The brief's hand example: one segment, one legal entity, one business line."""
    book = spark.createDataFrame(
        [("S1", "LE1", "BL1", 1000.0, 0.04, 0.5, 0.01, 0.004, 0.8)], BOOK_COLS
    )
    macro = _macro(
        spark,
        {ccar.BASELINE: baseline_path or _flat(4.0, 4.0, 0.0), ccar.SEVERELY_ADVERSE: sa_path},
    )
    capital = spark.createDataFrame([("LE1", 120.0, 200.0, dividends)], CAPITAL_COLS)
    gms = spark.createDataFrame([("LE1", "BL1", trading_loss)], GMS_COLS)
    return book, macro, capital, gms


def _rows(df, level, scenario):
    rows = df.where((F.col("node_level") == level) & (F.col("scenario") == scenario)).collect()
    return sorted(rows, key=lambda r: r["quarter"])


def _summary(spark, inputs):
    book, macro, capital, gms = inputs
    proj = ccar.project(book, macro, capital, gms, PARAMS)
    return proj, {
        (r["measure"], r["node_level"], r["scenario"]): r["value"]
        for r in ccar.summary(proj, capital, PARAMS).collect()
    }


# --- the formulas, by hand ------------------------------------------------------------------


def test_one_entity_first_quarter(spark):
    # ead 1000, pd_base 0.04, lgd 0.5, unemployment +4pp -> credit_loss 12.0
    # revenue_rate_q 0.01, gdp -0.02, expense 0.004 -> ppnr 5.6
    # trading_loss 20, cet1_start 120, risk_weight 0.8, rwa_trading 200
    # net_income (5.6 - 12 - 20) x 0.79 = -20.856; no dividend deducted (12 CFR 252.44(c)(1))
    proj = ccar.project(*_one_entity(spark, _flat(4.0, 8.0, -0.02)), PARAMS)
    for level in (ccar.BUSINESS_LINE, ccar.LEGAL_ENTITY, ccar.FIRM):
        q1 = _rows(proj, level, ccar.SEVERELY_ADVERSE)[0]
        assert q1["quarter"] == 1
        assert q1["credit_loss"] == pytest.approx(12.0)
        assert q1["ppnr"] == pytest.approx(5.6)
        assert q1["trading_loss"] == pytest.approx(20.0)
        assert q1["net_income"] == pytest.approx(-20.856)
    le = _rows(proj, ccar.LEGAL_ENTITY, ccar.SEVERELY_ADVERSE)[0]
    assert le["dividends"] == 1.0  # planned, reported, not deducted
    assert le["rwa"] == pytest.approx(1000.0)
    assert le["cet1"] == pytest.approx(99.144)
    assert le["cet1_ratio"] == pytest.approx(0.099144)
    bl = _rows(proj, ccar.BUSINESS_LINE, ccar.SEVERELY_ADVERSE)[0]
    assert all(bl[c] is None for c in ("cet1", "rwa", "cet1_ratio", "dividends"))
    assert set(proj.columns) == set(ccar.PROJECTION_COLUMNS)


def test_pd_is_capped_at_one_and_never_falls_below_base(spark):
    # +300pp would give pd_q = 0.01 x 106 > 1; a fall in unemployment does not lower the PD.
    up = ccar.project(*_one_entity(spark, _flat(4.0, 304.0, 0.0)), PARAMS)
    assert _rows(up, ccar.FIRM, ccar.SEVERELY_ADVERSE)[1]["credit_loss"] == pytest.approx(500.0)
    down = ccar.project(*_one_entity(spark, _flat(4.0, 3.0, 0.0)), PARAMS)
    assert _rows(down, ccar.FIRM, ccar.SEVERELY_ADVERSE)[1]["credit_loss"] == pytest.approx(5.0)


def test_baseline_has_no_trading_loss(spark):
    proj = ccar.project(*_one_entity(spark, _flat(4.0, 8.0, -0.02)), PARAMS)
    base = proj.where(F.col("scenario") == ccar.BASELINE)
    assert base.count() == 3 * 9
    assert base.where(F.col("trading_loss") != 0).count() == 0
    sa = proj.where((F.col("scenario") == ccar.SEVERELY_ADVERSE) & (F.col("quarter") > 1))
    assert sa.where(F.col("trading_loss") != 0).count() == 0


def test_cumulative_path_is_running_sum(spark):
    proj = ccar.project(*_one_entity(spark, list(MACRO_PATHS[ccar.SEVERELY_ADVERSE])), PARAMS)
    for scenario in ccar.SCENARIOS:
        path = _rows(proj, ccar.LEGAL_ENTITY, scenario)
        assert [r["quarter"] for r in path] == list(range(1, 10))
        running = 120.0
        for r in path:
            running += r["net_income"]
            assert r["cet1"] == pytest.approx(running)
            assert r["cet1_ratio"] == pytest.approx(r["cet1"] / 1000.0)
        assert path[-1]["cet1"] == pytest.approx(120.0 + sum(r["net_income"] for r in path))


def test_scb_floor(spark):
    # A mild scenario: a trading gain and benign macro; the ratio never falls -> exactly 2.5%.
    mild = _one_entity(spark, _flat(4.0, 4.0, 0.0), trading_loss=-5.0)
    _, s = _summary(spark, mild)
    assert s[(ccar.SCB, ccar.LEGAL_ENTITY, "*")] == 0.025
    assert s[(ccar.SCB, ccar.FIRM, "*")] == 0.025
    assert s[(ccar.START_RATIO, ccar.LEGAL_ENTITY, "*")] == pytest.approx(0.12)


def test_scb_above_floor_by_hand(spark):
    # Each SA quarter after the first: credit 12, ppnr 5.6 -> net income -5.056; quarter 1 adds
    # the 20 trading loss: -20.856. Trough at quarter 9: 120 - 20.856 - 8 x 5.056 = 58.696.
    proj, s = _summary(spark, _one_entity(spark, _flat(4.0, 8.0, -0.02), dividends=1.5))
    trough = 120.0 - 20.856 - 8 * 5.056
    assert _rows(proj, ccar.LEGAL_ENTITY, ccar.SEVERELY_ADVERSE)[-1]["cet1"] == pytest.approx(
        trough
    )
    first, last = PARAMS.scb_dividend_quarters
    add_on = (last - first + 1) * 1.5 / 1000.0
    expected = 0.12 - trough / 1000.0 + add_on
    assert s[(ccar.MIN_RATIO, ccar.LEGAL_ENTITY, ccar.SEVERELY_ADVERSE)] == pytest.approx(
        trough / 1000.0
    )
    assert s[(ccar.SCB, ccar.LEGAL_ENTITY, "*")] == pytest.approx(expected)
    assert s[(ccar.SCB, ccar.FIRM, "*")] == pytest.approx(expected)


def test_dividend_quarters_follow_the_amended_rule():
    # 91 FR 62636, effective 2026-12-01: "fifth through eighth quarters" (was fourth-seventh).
    assert PARAMS.scb_dividend_quarters == (5, 8)
    assert PARAMS.scb_floor == 0.025
    assert (PARAMS.pd_unemployment_beta, PARAMS.revenue_gdp_beta, PARAMS.tax_rate) == (
        0.35, 2.0, 0.21
    )  # fmt: skip


def test_bad_dividend_quarters_rejected(tmp_path):
    text = ccar_path().read_text().replace("[5, 8]", "[5, 10]")
    path = tmp_path / "ccar.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="scb_dividend_quarters"):
        ccar.load_params(path)


def test_global_market_shock_is_gfc():
    from riskagg.measures.stress import load_scenarios

    assert ccar.global_market_shock(load_scenarios()) == "GFC_2008"


# --- the semantic layer ---------------------------------------------------------------------


@pytest.mark.parametrize("measure", ccar.CCAR_MEASURE_NAMES)
def test_business_line_capital_ratio_refused(measure):
    model = load_model()
    for group_by in (["firm_id", "business_line_id"], ["business_line_id"]):
        with pytest.raises(SliceNotAvailable, match="not allocated"):
            build_query(model, measure, group_by, {}, "2026-11-12", "r")
    with pytest.raises(SliceNotAvailable, match="not allocated"):
        build_query(model, measure, ["firm_id"], {"desk_id": ["DESK001"]}, "2026-11-12", "r")
    q = build_query(model, measure, ["firm_id", "legal_entity"], {}, "2026-11-12", "r")
    assert q.kind == "lookup" and q.filters["node_level"] == ["firm_id|legal_entity"]


def test_not_allocated_to_is_validated(tmp_path):
    from riskagg.paths import semantic_model_path

    base = semantic_model_path().read_text()
    bad = base.replace(
        "    not_allocated_to: [business_line_id, desk_id, book_id]\n  ccar_min",
        "    not_allocated_to: [business_line_id, trader]\n  ccar_min",
        1,
    )
    path = tmp_path / "model.yaml"
    path.write_text(bad)
    with pytest.raises(SpecError, match="unknown dimension 'trader'"):
        load_model(path)
    clash = base.replace(
        "    not_allocated_to: [business_line_id, desk_id, book_id]\n  ccar_min",
        "    not_allocated_to: [legal_entity]\n  ccar_min",
        1,
    )
    path.write_text(clash)
    with pytest.raises(SpecError, match="not allocated to"):
        load_model(path)


def test_projection_table_has_no_star_tree():
    model = load_model()
    assert not any(m.table == "ccar_projection" for m in model.measures.values())
    assert (
        "starTreeIndexConfigs"
        not in pinot_table_config(model, "ccar_projection")["tableIndexConfig"]
    )


# --- the generators -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ccar(spark, tiny_market):
    _, hierarchy, _ = tiny_market
    p = PROFILES["tiny"]
    return (
        make_banking_book(spark, p, hierarchy),
        make_macro_scenarios(spark, p),
        make_capital_start(spark, p, hierarchy),
        hierarchy,
    )


def test_generators_shape_and_determinism(spark, tiny_ccar):
    book, macro, capital, hierarchy = tiny_ccar
    p = PROFILES["tiny"]
    pairs = hierarchy.select("legal_entity", "business_line_id").distinct().count()
    assert book.count() == pairs * len(SEGMENT_TYPES)
    assert book.select("segment_id").distinct().count() == book.count()
    assert macro.count() == 2 * 10
    assert capital.count() == hierarchy.select("legal_entity").distinct().count()
    again = (
        make_banking_book(spark, p, hierarchy),
        make_macro_scenarios(spark, p),
        make_capital_start(spark, p, hierarchy),
    )
    for a, b in zip((book, macro, capital), again, strict=True):
        assert a.collect() == b.collect()
    for r in book.collect():
        bounds = SEGMENT_TYPES[r["segment_id"].split("|")[2]]
        assert bounds["pd"][0] <= r["pd_base"] <= bounds["pd"][1]
        assert r["ead"] > 0
    ratios = {r["legal_entity"]: r for r in capital.collect()}
    assert all(0 < r["dividends_q"] < r["cet1_start"] for r in ratios.values())


@pytest.fixture(scope="module")
def tiny_projection(spark, tiny_ccar):
    from riskagg.data.market import make_exposures, make_sensitivities
    from riskagg.data.org import make_risk_factors, make_trades
    from riskagg.engine.inputs import RunInputs
    from riskagg.measures.stress import load_scenarios, shock_table

    book, macro, capital, hierarchy = tiny_ccar
    p = PROFILES["tiny"]
    import datetime

    trades = make_trades(spark, p, datetime.date(2026, 11, 12)).cache()
    rf = make_risk_factors(spark, p)
    inputs = RunInputs(
        sensitivities=make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf),
        trades=trades,
        hierarchy=hierarchy,
        shocks=shock_table(spark, load_scenarios(), rf),
        banking_book=book,
        macro_scenarios=macro,
        capital_start=capital,
    )
    proj = inputs.ccar_projection().cache()
    yield inputs, proj
    proj.unpersist()
    trades.unpersist()


def test_losses_reconcile_across_cuts(tiny_projection):
    _, proj = tiny_projection
    totals = (
        proj.groupBy("scenario", "quarter", "node_level")
        .agg(*(F.sum(c).alias(c) for c in ("ppnr", "credit_loss", "trading_loss", "net_income")))
        .collect()
    )
    by = {}
    for r in totals:
        by.setdefault((r["scenario"], r["quarter"]), {})[r["node_level"]] = r
    assert len(by) == 2 * 9
    for levels in by.values():
        assert set(levels) == {ccar.BUSINESS_LINE, ccar.LEGAL_ENTITY, ccar.FIRM}
        for c in ("ppnr", "credit_loss", "trading_loss", "net_income"):
            firm = levels[ccar.FIRM][c]
            assert levels[ccar.BUSINESS_LINE][c] == pytest.approx(firm, rel=1e-12, abs=1e-6)
            assert levels[ccar.LEGAL_ENTITY][c] == pytest.approx(firm, rel=1e-12, abs=1e-6)


def test_trading_loss_is_the_gfc_stress_loss_and_gains_are_kept(tiny_projection):
    inputs, proj = tiny_projection
    gfc = {
        r["legal_entity"]: -r["pnl"]
        for r in inputs.stress_grain()
        .where(F.col("scenario") == "GFC_2008")
        .groupBy("legal_entity")
        .agg(F.sum("pnl").alias("pnl"))
        .collect()
    }
    q1 = {
        r["node_id"]: r["trading_loss"]
        for r in _rows(proj, ccar.LEGAL_ENTITY, ccar.SEVERELY_ADVERSE)
        if r["quarter"] == 1
    }
    assert q1.keys() == gfc.keys()
    for le, loss in gfc.items():
        assert q1[le] == pytest.approx(loss)
    assert min(q1.values()) < 0  # tiny: GFC_2008 is a net gain for one legal entity


def test_firm_ratio_is_not_an_average_of_entity_ratios(tiny_projection):
    _, proj = tiny_projection
    for scenario in ccar.SCENARIOS:
        firm = _rows(proj, ccar.FIRM, scenario)
        entities = proj.where(
            (F.col("node_level") == ccar.LEGAL_ENTITY) & (F.col("scenario") == scenario)
        ).collect()
        for f in firm:
            les = [r for r in entities if r["quarter"] == f["quarter"]]
            assert f["cet1"] == pytest.approx(sum(r["cet1"] for r in les))
            assert f["cet1_ratio"] == pytest.approx(f["cet1"] / sum(r["rwa"] for r in les))


# --- the run --------------------------------------------------------------------------------


def test_run_stores_capital_measures_at_entity_and_firm(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    metrics = spark.read.parquet(str(ctx.curated("risk_metrics"))).where(
        F.col("measure").isin(*ccar.CCAR_MEASURE_NAMES)
    )
    rows = metrics.collect()
    assert {r["node_level"] for r in rows} == {"firm_id", "firm_id|legal_entity"}
    assert all(r["business_line_id"] == "*" and r["desk_id"] == "*" for r in rows)
    counts = {m: sum(r["measure"] == m for r in rows) for m in ccar.CCAR_MEASURE_NAMES}
    assert counts == {ccar.START_RATIO: 3, ccar.MIN_RATIO: 6, ccar.SCB: 3}
    assert all(r["value"] >= 0.025 for r in rows if r["measure"] == ccar.SCB)
    detail = spark.read.parquet(str(ctx.curated("ccar_projection")))
    cols = set(tmp_run.model.tables["ccar_projection"]["columns"]) - {"as_of_date", "run_id"}
    assert set(detail.columns) == cols
    assert detail.count() == 2 * 9 * (2 + 2 + 1)  # two lines, two entities, the firm


def test_registry_and_audit(spark, tmp_run):
    for name in ccar.CCAR_MEASURE_NAMES:
        assert name in MEASURE_REGISTRY
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    check = audit.check_ccar_from_raw(ctx, spark)
    assert check.passed, check.detail
    assert audit.check_ccar_from_raw in audit.CHECKS


def _copy_inputs(ctx, bad):
    for name in ("banking_book", "macro_scenarios", "capital_start"):
        shutil.copytree(ctx.raw(name), bad.raw(name))
    for table in ("stress_results",):
        shutil.copytree(ctx.curated(table), bad.curated(table))


@pytest.mark.parametrize(
    ("table", "column", "where"),
    [
        ("ccar_projection", "credit_loss", "node_level = 'BUSINESS_LINE' and quarter = 3"),
        ("ccar_projection", "cet1", "node_level = 'LEGAL_ENTITY' and quarter = 9"),
        ("risk_metrics", "value", "measure = 'ccar_scb' and node_level = 'firm_id'"),
    ],
)
def test_audit_catches_a_wrong_stored_value(spark, tmp_run, tmp_path, table, column, where):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    bad = dataclasses.replace(ctx, root=tmp_path)
    assert isinstance(bad, RunContext)
    _copy_inputs(ctx, bad)
    for t in ("ccar_projection", "risk_metrics"):
        df = spark.read.parquet(str(ctx.curated(t)))
        if t == table:
            df = df.withColumn(
                column, F.when(F.expr(where), F.col(column) * 1.01 + 1.0).otherwise(F.col(column))
            )
        df.write.parquet(str(bad.curated(t)))
    check = audit.check_ccar_from_raw(bad, spark)
    assert not check.passed, check.detail


def test_api_answers_entities_and_refuses_business_lines(tmp_run):
    api = TestClient(create_app(tmp_run.root, broker_url=DOWN, controller_url=DOWN))
    firm = api.get("/metrics/ccar_min_cet1_ratio", params={"group_by": "firm_id"})
    assert firm.status_code == 200, firm.text
    assert {r["scenario"] for r in firm.json()["rows"]} == set(ccar.SCENARIOS)
    les = api.get("/metrics/ccar_scb", params={"group_by": "firm_id,legal_entity"})
    assert les.status_code == 200 and len(les.json()["rows"]) == 2
    bl = api.get("/metrics/ccar_scb", params={"group_by": "firm_id,business_line_id"})
    assert bl.status_code == 422 and "not allocated" in bl.text
    drill = api.get("/drilldown/ccar_projection", params={"limit": 5})
    assert drill.status_code == 200, drill.text


def test_nullable_capital_columns_get_pinot_null_handling():
    from riskagg.semantic.pinot_schema import pinot_schema
    from riskagg.semantic.query import build_drilldown

    model = load_model()
    assert pinot_schema(model, "ccar_projection")["enableColumnBasedNullHandling"] is True
    assert "enableColumnBasedNullHandling" not in pinot_schema(model, "risk_metrics")
    sql = build_drilldown(model, "ccar_projection", {}, "2026-11-12", "r", 5).to_pinot_sql()
    assert sql.startswith("SET enableNullHandling = true; SELECT ")
    plain = build_drilldown(model, "risk_metrics", {}, "2026-11-12", "r", 5).to_pinot_sql()
    assert plain.startswith("SELECT ")
