import dataclasses
import math

import pytest

from reference import np_var
from riskagg.engine.grain import GRAIN_DIMS, build_reporting_grain
from riskagg.engine.runner import run_all
from riskagg.engine.spec import AggregationSpec, load_specs
from riskagg.engine.vectors import METRIC_COLUMNS, aggregate_vectors
from riskagg.measures import var
from riskagg.semantic.model import Measure, load_model

ORG = ("firm_id", "business_line_id", "desk_id", "book_id")
PNL_SCHEMA = (
    "trade_id long, book_id string, asset_class string, currency string, risk_class string, "
    "liquidity_horizon int, in_reduced_set boolean, window string, pnl array<double>"
)
HIER_SCHEMA = (
    "book_id string, book_name string, desk_id string, desk_name string, "
    "business_line_id string, business_line_name string, firm_id string, "
    "legal_entity string, region string"
)


def _model_with(measure: Measure):
    model = load_model()
    return dataclasses.replace(model, measures={**model.measures, measure.name: measure})


def _measure(name="var_75", function="var", level=0.75, window="CURRENT", horizon_days=1):
    params = {"level": level, "window": window, "horizon_days": horizon_days}
    return Measure(name, name, "", "USD", "vector", function, params, "risk_metrics", GRAIN_DIMS)


def _books(spark, vectors: dict[str, list[float]], desks: dict[str, str] | None = None):
    desks = desks or {b: "D1" for b in vectors}
    hier = [(b, b, d, d, "BL1", "BL1", "FIRM", "LE1", "EMEA") for b, d in desks.items()]
    pnl = [
        (i, b, "RATES", "USD", "GIRR", 10, True, "CURRENT", v)
        for i, (b, v) in enumerate(vectors.items())
    ]
    return spark.createDataFrame(pnl, PNL_SCHEMA), spark.createDataFrame(hier, HIER_SCHEMA)


def _by_measure(df):
    out: dict[tuple[str, str], float] = {}
    for r in df.collect():
        node = r.book_id if r.book_id != "*" else r.desk_id
        out[(r.measure, node)] = r.value
    return out


def _desk_book_run(spark, vectors, desks=None):
    pnl, hier = _books(spark, vectors, desks)
    m = _measure()
    spec = AggregationSpec("t", m.name, (ORG[:3], ORG), {}, True)
    return _by_measure(run_all([spec], _model_with(m), build_reporting_grain(pnl, hier, 4), 4))


def test_two_book_diversification(spark):
    got = _desk_book_run(spark, {"A": [-10.0, 5.0, 0.0, 5.0], "B": [5.0, -10.0, 0.0, 5.0]})
    assert got[("var_75", "D1")] == 5.0
    assert got[("var_75", "A")] == 10.0
    assert got[("var_75", "B")] == 10.0
    assert got[("var_75__sum_children", "D1")] == 20.0
    assert got[("var_75__diversification", "D1")] == 15.0
    assert got[("var_75__incremental", "A")] == -5.0
    assert got[("var_75__incremental", "B")] == -5.0
    assert ("var_75__incremental", "D1") not in got  # top level of the spec has no parent
    assert len(got) == 7


def test_single_child_parent(spark):
    got = _desk_book_run(
        spark,
        {"A": [-10.0, 5.0, 0.0, 5.0], "B": [5.0, -10.0, 0.0, 5.0]},
        desks={"A": "D1", "B": "D2"},
    )
    for desk, book in (("D1", "A"), ("D2", "B")):
        assert got[("var_75__diversification", desk)] == 0.0
        assert got[("var_75__sum_children", desk)] == got[("var_75", desk)] == 10.0
        assert got[("var_75__incremental", book)] == got[("var_75", desk)]


def _keyed(df):
    out = {}
    for r in df.collect():
        out[(r.measure, r.node_level, tuple(r[d] for d in GRAIN_DIMS))] = r.value
    return out


@pytest.mark.parametrize(
    ("name", "function", "level"), [("var_99_1d", "var", 0.99), ("es_975_1d", "es", 0.975)]
)
def test_matches_numpy_reference(tiny_market, name, function, level):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    (spec,) = [s for s in load_specs(None, model) if s.name == "var_by_org"]
    spec = dataclasses.replace(spec, measure=name)
    got = _keyed(run_all([spec], model, build_reporting_grain(pnl, hierarchy, n), n))

    pnl_pdf, hier_pdf = pnl.toPandas(), hierarchy.toPandas()
    sets = spec.grouping_sets
    ref = np_var.node_metrics(pnl_pdf, hier_pdf, GRAIN_DIMS, sets, "CURRENT", function, level)
    expected = {(name, lvl, node): v for (lvl, node), v in ref.items()}
    div = np_var.diversification(pnl_pdf, hier_pdf, GRAIN_DIMS, sets, "CURRENT", function, level)
    expected |= {(f"{name}__{kind}", lvl, node): v for (kind, lvl, node), v in div.items()}

    assert len(ref) == 1 + 2 + 3 + 6  # firm, business lines, desks, books (tiny)
    assert got.keys() == expected.keys()
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, abs=1e-9), k


def test_es_subadditive(tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    (spec,) = [s for s in load_specs(None, model) if s.name == "var_by_org"]
    spec = dataclasses.replace(spec, measure="es_975_1d")
    rows = run_all([spec], model, build_reporting_grain(pnl, hierarchy, n), n).collect()
    value = {(r.measure, r.node_level, tuple(r[d] for d in GRAIN_DIMS)): r.value for r in rows}
    parents = [(lvl, node) for (m, lvl, node) in value if m == "es_975_1d__sum_children"]
    assert len(parents) == 1 + 2 + 3  # firm, business lines, desks
    for lvl, node in parents:
        parent_es = value[("es_975_1d", lvl, node)]
        assert parent_es <= value[("es_975_1d__sum_children", lvl, node)] + 1e-9


def test_horizon_scaling_and_window(tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    grain = build_reporting_grain(pnl, hierarchy, n)
    nodes = aggregate_vectors(grain, [ORG[:3]], n)
    spec = AggregationSpec("d", "var_99_1d", (ORG[:3],), {}, False)

    def values(measure_name):
        df = var.metrics_from_vectors(nodes, spec, model.measure(measure_name))
        assert df.columns == METRIC_COLUMNS
        return {r.desk_id: r.value for r in df.collect()}

    one_day, ten_day, stressed = values("var_99_1d"), values("var_99_10d"), values("svar_99_1d")
    assert len(one_day) == 3
    for desk, v in one_day.items():
        assert ten_day[desk] == pytest.approx(v * math.sqrt(10), abs=1e-9)
    ref = np_var.node_metrics(
        pnl.toPandas(), hierarchy.toPandas(), GRAIN_DIMS, [ORG[:3]], "STRESSED", "var", 0.99
    )
    assert {node[2]: v for (_, node), v in ref.items()} == pytest.approx(stressed, abs=1e-9)
