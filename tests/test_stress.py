import datetime

import pandas as pd
import pytest
from pyspark.sql import functions as F

from riskagg.context import run_context
from riskagg.data.market import make_exposures, make_sensitivities
from riskagg.data.org import make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine import audit
from riskagg.engine.inputs import RunInputs
from riskagg.engine.runner import run_all
from riskagg.engine.spec import AggregationSpec, load_specs
from riskagg.engine.vectors import GRAIN_DIMS, METRIC_COLUMNS
from riskagg.measures.stress import (
    Scenario,
    ShockRule,
    load_scenarios,
    shock_table,
    stress_by_grain,
    stress_pnl,
    worst_stress_loss,
)
from riskagg.semantic.model import load_model
from riskagg.semantic.pinot_schema import pinot_table_config
from riskagg.semantic.query import SliceNotAvailable
from riskagg.serving.service import RiskService

ORG = ("firm_id", "business_line_id", "desk_id", "book_id")


def _factors(spark):
    rows = [
        ("EQ.1", "EQ", "DELTA", "1", "USD"),
        ("EQ.1.VEGA", "EQ", "VEGA", "1", "USD"),
        ("FX.BRL", "FX", "DELTA", "BRL", "BRL"),
        ("FX.EUR", "FX", "DELTA", "EUR", "EUR"),
        ("GIRR.USD.5Y", "GIRR", "DELTA", "USD", "USD"),
    ]
    return spark.createDataFrame(
        rows, "risk_factor_id string, risk_class string, factor_type string, bucket string, "
        "currency string"
    )  # fmt: skip


def _shocks(spark, scenarios):
    return {
        (r.scenario, r.risk_factor_id): r.shock
        for r in shock_table(spark, scenarios, _factors(spark)).collect()
    }


def test_shipped_scenario_library():
    scenarios = {s.name: s for s in load_scenarios()}
    assert list(scenarios) == [
        "GFC_2008", "RATES_UP_200BP", "EQUITY_DOWN_30", "FX_CRISIS", "VOL_SPIKE"
    ]  # fmt: skip
    assert [n for n, s in scenarios.items() if s.global_market_shock] == ["GFC_2008"]
    gfc = scenarios["GFC_2008"]
    assert gfc.shock_for("EQ", "DELTA", "1", "USD") == -0.40
    assert gfc.shock_for("GIRR", "DELTA", "USD", "USD") == -0.015
    assert gfc.shock_for("FX", "DELTA", "EUR", "EUR") == -0.10
    assert gfc.shock_for("FX", "DELTA", "ZAR", "ZAR") == -0.25
    assert gfc.shock_for("GIRR", "VEGA", "USD", "USD") == 0.30
    assert scenarios["VOL_SPIKE"].shock_for("EQ", "DELTA", "1", "USD") == 0.0
    assert scenarios["RATES_UP_200BP"].shock_for("GIRR", "DELTA", "BRL", "BRL") == 0.02
    assert scenarios["FX_CRISIS"].shock_for("FX", "DELTA", "BRL", "BRL") == -0.20
    assert scenarios["FX_CRISIS"].shock_for("FX", "DELTA", "GBP", "GBP") == -0.05


def test_formula(spark):
    """delta 100, gamma 50, shock -0.30: -30 + 0.5*50*0.09 = -27.75; vega 10 * 0.20 = +2."""
    sc = Scenario(
        "S", "", False,
        (ShockRule("EQ", "DELTA", -0.30), ShockRule("EQ", "VEGA", 0.20)),
    )  # fmt: skip
    sens = spark.createDataFrame(
        [
            ("T1", "B1", "EQ.1", "EQ", "DELTA", 100.0, 50.0, 0.0),
            ("T1", "B1", "EQ.1.VEGA", "EQ", "VEGA", 0.0, 0.0, 10.0),
        ],
        "trade_id string, book_id string, risk_factor_id string, risk_class string, "
        "factor_type string, delta double, gamma double, vega double",
    )
    shocks = shock_table(spark, [sc], _factors(spark))
    (row,) = stress_pnl(sens.where("factor_type = 'DELTA'"), shocks).collect()
    assert row.pnl == pytest.approx(-27.75)
    (row,) = stress_pnl(sens, shocks).collect()
    assert (row.trade_id, row.book_id, row.risk_class, row.scenario) == ("T1", "B1", "EQ", "S")
    assert row.pnl == pytest.approx(-25.75)


def test_most_specific_shock_wins(spark):
    rules = (
        ShockRule("FX", "DELTA", -0.05),
        ShockRule("FX", "DELTA", -0.20, currency="BRL"),
        ShockRule("EQ", "DELTA", -0.10),
        ShockRule("EQ", "DELTA", -0.30, bucket="1"),
        ShockRule("EQ", "DELTA", -0.50, bucket="1", currency="USD"),
    )
    shocks = _shocks(spark, [Scenario("S", "", False, rules)])
    assert shocks[("S", "FX.BRL")] == -0.20  # currency rule beats the general rule
    assert shocks[("S", "FX.EUR")] == -0.05
    assert shocks[("S", "EQ.1")] == -0.50  # two optional fields beat one, and none
    rule_order_reversed = Scenario("S", "", False, rules[::-1])
    assert _shocks(spark, [rule_order_reversed]) == shocks  # rule order is irrelevant


def test_equally_specific_rules_are_ambiguous():
    sc = Scenario(
        "S", "", False,
        (ShockRule("EQ", "DELTA", -0.1, bucket="1"), ShockRule("EQ", "DELTA", -0.2, currency="USD")),
    )  # fmt: skip
    with pytest.raises(ValueError, match="equally specific"):
        sc.shock_for("EQ", "DELTA", "1", "USD")


def test_missing_shock_means_zero_not_null(spark):
    sc = Scenario("S", "", False, (ShockRule("EQ", "DELTA", -0.3),))
    table = shock_table(spark, [sc], _factors(spark))
    assert table.count() == 5  # one row per scenario x risk factor, matched or not
    assert table.where(F.col("shock").isNull()).count() == 0
    shocks = {r.risk_factor_id: r.shock for r in table.collect()}
    assert shocks["FX.BRL"] == 0.0 and shocks["EQ.1.VEGA"] == 0.0 and shocks["EQ.1"] == -0.3
    sens = spark.createDataFrame(
        [("T1", "B1", "FX.BRL", "FX", "DELTA", 100.0, 5.0, 0.0)],
        "trade_id string, book_id string, risk_factor_id string, risk_class string, "
        "factor_type string, delta double, gamma double, vega double",
    )
    (row,) = stress_pnl(sens, table).collect()
    assert row.pnl == 0.0


def test_scenario_file_validation(tmp_path):
    def load(text):
        path = tmp_path / "s.yaml"
        path.write_text(text)
        return load_scenarios(path)

    with pytest.raises(ValueError, match="duplicate scenario"):
        load("- {name: A, shocks: []}\n- {name: A, shocks: []}\n")
    with pytest.raises(ValueError, match="at most one"):
        load("- {name: A, global_market_shock: true}\n- {name: B, global_market_shock: true}\n")
    with pytest.raises(ValueError, match="risk_class"):
        load("- {name: A, shocks: [{risk_class: CSR, factor_type: DELTA, shock: 0.1}]}\n")
    with pytest.raises(TypeError, match="shock must be a number"):
        load("- {name: A, shocks: [{risk_class: EQ, factor_type: DELTA, shock: big}]}\n")


@pytest.fixture(scope="module")
def tiny_stress(spark, tiny_market):
    """Tiny-profile sensitivities, trades, hierarchy and the real shock table."""
    _, hierarchy, _ = tiny_market
    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12)).cache()
    rf = make_risk_factors(spark, p)
    sens = make_sensitivities(spark, p, make_exposures(spark, p, trades, rf), rf).cache()
    shocks = shock_table(spark, load_scenarios(), rf)
    stress = stress_pnl(sens, shocks)
    grain = stress_by_grain(stress, trades, hierarchy).cache()
    yield (
        stress,
        grain,
        RunInputs(sensitivities=sens, trades=trades, hierarchy=hierarchy, shocks=shocks),
        trades,
        sens,
    )
    grain.unpersist()
    trades.unpersist()
    sens.unpersist()


def test_additive_across_hierarchy(tiny_stress):
    stress, grain, *_ = tiny_stress
    assert grain.columns == [*GRAIN_DIMS, "scenario", "pnl"]
    firm = {
        r.scenario: r.pnl
        for r in grain.groupBy("scenario").agg(F.sum("pnl").alias("pnl")).collect()
    }
    by_trade = {r.scenario: r.pnl for r in stress.groupBy("scenario").agg(F.sum("pnl").alias("pnl")).collect()}  # fmt: skip
    assert set(firm) == {s.name for s in load_scenarios()}
    desks = grain.groupBy("desk_id", "scenario").agg(F.sum("pnl").alias("pnl")).collect()
    for scenario, total in firm.items():
        assert total == pytest.approx(by_trade[scenario], rel=1e-9, abs=1e-6)
        assert sum(r.pnl for r in desks if r.scenario == scenario) == pytest.approx(
            total, rel=1e-9, abs=1e-6
        )
    assert any(abs(v) > 1 for v in firm.values())


def _stress_frame(spark, rows):
    """rows: (desk, scenario, pnl) -> a stress grain with one book per desk."""
    data = [
        ("FIRM", "BL1", desk, f"B-{desk}", "LE1", "EMEA", "FX", "EUR", "FX", sc, pnl)
        for desk, sc, pnl in rows
    ]
    return spark.createDataFrame(data, [*GRAIN_DIMS, "scenario", "pnl"])


def _worst(spark, rows, sets=None):
    model = load_model()
    sets = sets or (("firm_id",), ("firm_id", "business_line_id", "desk_id"))
    spec = AggregationSpec("w", "worst_stress_loss", sets, {}, False)
    inputs = RunInputs(_stress_grain=_stress_frame(spark, rows))
    out = worst_stress_loss(spec, model, inputs)
    assert out.columns == METRIC_COLUMNS
    return {(r.node_level, r.desk_id): (r.scenario, r.value) for r in out.collect()}


def test_worst_of_is_not_additive(spark):
    """Each desk's worst scenario is different; netting makes the firm's worst smaller."""
    rows = [
        ("D1", "A", -100.0), ("D1", "B", 50.0),
        ("D2", "A", 50.0), ("D2", "B", -100.0),
    ]  # fmt: skip
    worst = _worst(spark, rows)
    d1 = worst[("firm_id|business_line_id|desk_id", "D1")]
    d2 = worst[("firm_id|business_line_id|desk_id", "D2")]
    firm = worst[("firm_id", "*")]
    assert d1 == ("A", 100.0) and d2 == ("B", 100.0)
    assert firm == ("A", 50.0)  # A: -100 + 50 = -50 and B: 50 - 100 = -50; tie -> name order
    assert firm[1] < d1[1] + d2[1]


def test_worst_tie_goes_to_first_name_and_gains_are_not_clipped(spark):
    rows = [("D1", "Z", -10.0), ("D1", "M", -10.0), ("D1", "A", -5.0)]
    assert _worst(spark, rows)[("firm_id", "*")] == ("M", 10.0)
    gains = [("D1", "A", 7.0), ("D1", "B", 3.0), ("D2", "A", 1.0), ("D2", "B", 2.0)]
    worst = _worst(spark, gains)
    assert worst[("firm_id", "*")] == ("B", -5.0)  # A: +8, B: +5 -> least gain, negative loss
    assert worst[("firm_id|business_line_id|desk_id", "D1")] == ("B", -3.0)


def test_worst_stress_loss_rows_on_tiny(spark, tiny_stress):
    _, grain, inputs, *_ = tiny_stress
    model = load_model()
    specs = [s for s in load_specs(None, model) if s.name == "worst_stress_by_org"]
    out = run_all(specs, model, grain, 20, inputs)
    rows = out.collect()
    assert {(r.aggregation, r.measure) for r in rows} == {
        ("worst_stress_by_org", "worst_stress_loss")
    }
    assert {r.node_level for r in rows} == {"|".join(ORG[: i + 1]) for i in range(4)}
    firm = {r.scenario: r.pnl for r in grain.groupBy("scenario").agg(F.sum("pnl").alias("pnl")).collect()}  # fmt: skip
    (firm_row,) = [r for r in rows if r.node_level == "firm_id"]
    scenario, loss = min(((s, -v) for s, v in firm.items()), key=lambda t: (-t[1], t[0]))
    assert firm_row.scenario == scenario and firm_row.value == pytest.approx(loss)
    for r in rows:  # rolled-up dimensions are '*', never null
        live = set(r.node_level.split("|"))
        assert all((r[d] == "*") == (d not in live) for d in GRAIN_DIMS)


def test_unset_inputs_are_reported():
    with pytest.raises(ValueError, match="sensitivities.*trades.*hierarchy.*shocks"):
        RunInputs().stress_grain()


def _expected_desk_scenario(run) -> pd.DataFrame:
    """Independent pandas computation of stress P&L by desk x scenario from the raw files."""
    raw = run.root / "raw" / "tiny" / f"as_of_date={run.as_of_date.isoformat()}"
    sens = pd.read_parquet(raw / "sensitivities")
    factors = pd.read_parquet(raw / "risk_factors").set_index("risk_factor_id")
    hier = pd.read_parquet(raw / "hierarchy").set_index("book_id")["desk_id"]
    records = []
    for sc in load_scenarios():
        shock = sens["risk_factor_id"].map(
            lambda f, sc=sc: sc.shock_for(
                factors.at[f, "risk_class"], factors.at[f, "factor_type"],
                factors.at[f, "bucket"], factors.at[f, "currency"],
            )
        )  # fmt: skip
        delta = sens["delta"] * shock + 0.5 * sens["gamma"] * shock**2
        vega = sens["vega"] * shock
        pnl = delta.where(sens["factor_type"] == "DELTA", vega)
        by_desk = pnl.groupby(sens["book_id"].map(hier)).sum()
        records += [(d, sc.name, v) for d, v in by_desk.items()]
    return pd.DataFrame(records, columns=["desk_id", "scenario", "value"])


def test_query_stress_by_desk_and_scenario(tmp_run):
    svc = RiskService(tmp_run.root, tmp_run.model)
    ans = svc.metric("stress_pnl", ["desk_id", "scenario"])
    assert ans.run_id == tmp_run.run_id
    got = ans.rows.sort_values(["desk_id", "scenario"]).reset_index(drop=True)
    want = _expected_desk_scenario(tmp_run).sort_values(["desk_id", "scenario"])
    assert (
        got[["desk_id", "scenario"]].values.tolist()
        == want[["desk_id", "scenario"]].values.tolist()
    )
    assert got["value"].tolist() == pytest.approx(want["value"].tolist(), rel=1e-9, abs=1e-6)
    # Additive: the firm total per scenario is the sum of the desks, and a filter slices it.
    firm = svc.metric("stress_pnl", ["scenario"]).rows.set_index("scenario")["value"]
    assert firm.to_dict() == pytest.approx(want.groupby("scenario")["value"].sum().to_dict())
    one = svc.metric("stress_pnl", ["desk_id"], {"scenario": ["GFC_2008"]}).rows
    gfc = want[want["scenario"] == "GFC_2008"].set_index("desk_id")["value"]
    assert one.set_index("desk_id")["value"].to_dict() == pytest.approx(gfc.to_dict())
    total = svc.metric("stress_pnl", [], {"scenario": ["GFC_2008"]}).rows
    assert total["value"].iloc[0] == pytest.approx(gfc.sum())


def test_stress_pnl_refuses_slices_outside_its_dimensions(tmp_run):
    svc = RiskService(tmp_run.root, tmp_run.model)
    with pytest.raises(SliceNotAvailable):
        svc.metric("stress_pnl", ["counterparty_id"])


def test_worst_stress_loss_is_looked_up_not_rebuilt(tmp_run):
    svc = RiskService(tmp_run.root, tmp_run.model)
    desk = ["firm_id", "business_line_id", "desk_id"]
    by_desk = svc.metric("worst_stress_loss", desk).rows
    want = _expected_desk_scenario(tmp_run)
    for r in by_desk.itertuples():
        losses = -want[want["desk_id"] == r.desk_id].set_index("scenario")["value"]
        assert r.value == pytest.approx(losses.max(), rel=1e-9)
    firm = svc.metric("worst_stress_loss", ["firm_id"]).rows["value"].iloc[0]
    assert firm < by_desk["value"].sum()  # worst-of is not additive across the desks
    with pytest.raises(SliceNotAvailable, match="only precomputed"):
        svc.metric("worst_stress_loss", ["desk_id", "scenario"])


def test_reconciliation_includes_stress_check(spark, tmp_run):
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    assert audit.check_stress_firm_pnl_is_sum_of_trades in audit.CHECKS
    checks = {c.name: c for c in audit.reconcile(ctx, spark)}
    check = checks["stress_firm_pnl_equals_sum_of_trades"]
    assert check.passed, check.detail
    assert "5 scenarios compared" in check.detail


def test_star_tree_applies_to_stress_results_only():
    model = load_model()
    star = pinot_table_config(model, "stress_results")["tableIndexConfig"]["starTreeIndexConfigs"]
    assert star[0]["functionColumnPairs"] == ["SUM__pnl"]
    assert "scenario" in star[0]["dimensionsSplitOrder"]
    for table in ("risk_metrics", "pnl_scenarios"):
        assert "starTreeIndexConfigs" not in pinot_table_config(model, table)["tableIndexConfig"]


def test_shipped_specs_and_measures():
    model = load_model()
    names = {s.name for s in load_specs(None, model)}
    assert {"worst_stress_by_org", "var_by_org"} <= names
    assert "stress_pnl" not in {s.measure for s in load_specs(None, model)}  # served by query
    m = model.measure("stress_pnl")
    assert (m.measure_class, m.table) == ("additive", "stress_results")
    assert m.dimensions == (*model.grain, "scenario")


DESK_LEVEL = ["firm_id", "business_line_id", "desk_id"]


def _want_worst(run) -> dict[str, tuple[str, float]]:
    want = _expected_desk_scenario(run)
    out = {}
    for desk, g in want.groupby("desk_id"):
        low = g.sort_values(["value", "scenario"]).iloc[0]
        out[desk] = (low["scenario"], -low["value"])
    return out


def test_lookup_sql_selects_the_scenario_only_for_scenario_bearing_measures():
    from riskagg.semantic.query import build_query

    model = load_model()
    d = datetime.date(2026, 11, 12)
    q = build_query(model, "worst_stress_loss", DESK_LEVEL, {}, d, "r1")
    assert q.columns == [*DESK_LEVEL, "scenario", "value"]
    assert 'SELECT "firm_id", "business_line_id", "desk_id", "scenario", "value" FROM' in (
        q.to_pinot_sql()
    )
    assert "scenario" in q.to_pinot_sql()
    other = dataclasses_replace_structured(model)
    plain = build_query(other, "worst_stress_loss", DESK_LEVEL, {}, d, "r1")
    assert plain.columns == [*DESK_LEVEL, "value"] and '"scenario"' not in plain.to_pinot_sql()


def dataclasses_replace_structured(model):
    import dataclasses

    m = dataclasses.replace(model.measure("worst_stress_loss"), carries_scenario=False)
    return dataclasses.replace(model, measures={**model.measures, m.name: m})


def test_worst_stress_loss_scenario_through_executor_service_and_api(tmp_run):
    from fastapi.testclient import TestClient

    from riskagg.semantic.query import build_query
    from riskagg.serving.api import create_app
    from riskagg.serving.client import RiskClient
    from riskagg.serving.executors import PandasExecutor

    want = _want_worst(tmp_run)
    q = build_query(
        tmp_run.model, "worst_stress_loss", DESK_LEVEL, {}, tmp_run.as_of_date, tmp_run.run_id
    )
    raw = PandasExecutor(tmp_run.root).execute(q)
    assert list(raw.columns) == [*DESK_LEVEL, "scenario", "value"]
    down = "http://127.0.0.1:9"
    body = TestClient(create_app(tmp_run.root, broker_url=down, controller_url=down)).get(
        "/metrics/worst_stress_loss", params={"group_by": ",".join(DESK_LEVEL)}
    ).json()  # fmt: skip
    got = {r["desk_id"]: (r["scenario"], r["value"]) for r in body["rows"]}
    assert set(got) == set(want)
    for desk, (scenario, loss) in want.items():
        assert got[desk][0] == scenario and got[desk][1] == pytest.approx(loss, rel=1e-9)
    frame = RiskClient(root=tmp_run.root).metric("worst_stress_loss", DESK_LEVEL)
    assert list(frame.columns) == [*DESK_LEVEL, "scenario", "value"]
    firm = RiskClient(root=tmp_run.root).metric("worst_stress_loss", ["firm_id"])
    assert list(firm.columns) == ["firm_id", "scenario", "value"] and len(firm) == 1
    # Measures that do not carry a scenario are unchanged.
    var = RiskClient(root=tmp_run.root).metric("var_99_1d", ["desk_id"])
    assert list(var.columns) == ["desk_id", "value"]
