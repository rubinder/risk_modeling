import dataclasses

import numpy as np
import pytest
from pyspark.sql import functions as F

from riskagg.engine.grain import GRAIN_DIMS, build_reporting_grain
from riskagg.engine.runner import MEASURE_REGISTRY, node_vectors, run_all
from riskagg.engine.spec import AggregationSpec, SpecError, load_specs
from riskagg.engine.vectors import METRIC_COLUMNS, aggregate_vectors, sum_vectors
from riskagg.semantic.model import load_model

PNL_SCHEMA = (
    "trade_id long, book_id string, asset_class string, currency string, risk_class string, "
    "liquidity_horizon int, in_reduced_set boolean, window string, pnl array<double>"
)
HIER_SCHEMA = (
    "book_id string, book_name string, desk_id string, desk_name string, "
    "business_line_id string, business_line_name string, firm_id string, "
    "legal_entity string, region string"
)


def _vector_specs(model):
    """The shipped vector specs: stress specs need run inputs the P&L tests do not build."""
    return [
        s for s in load_specs(None, model) if model.measure(s.measure).measure_class == "vector"
    ]


def _hierarchy(spark):
    return spark.createDataFrame(
        [("B1", "Book 1", "D1", "Desk 1", "BL1", "BL 1", "FIRM", "LE1", "EMEA")], HIER_SCHEMA
    )


def _pnl(spark, vectors):
    rows = [
        (i, "B1", "RATES", "USD", "GIRR", 10, True, "CURRENT", v) for i, v in enumerate(vectors)
    ]
    return spark.createDataFrame(rows, PNL_SCHEMA)


def _metrics(df):
    keys = ["aggregation", "measure", "node_level", *GRAIN_DIMS, "scenario"]
    return {tuple(r[k] for k in keys): r["value"] for r in df.collect()}


def test_metric_columns():
    model = load_model()
    assert GRAIN_DIMS == model.grain
    assert METRIC_COLUMNS == [
        "aggregation", "measure", "node_level", *model.grain, "scenario", "value",
    ]  # fmt: skip


def test_grain_sums_trades_and_joins_hierarchy(spark):
    pnl = _pnl(spark, [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    grain = build_reporting_grain(pnl, _hierarchy(spark), 3)
    assert grain.columns == [*GRAIN_DIMS, "liquidity_horizon", "in_reduced_set", "window", "pnl"]
    (row,) = grain.collect()
    assert row.pnl == [11.0, 22.0, 33.0]
    assert (row.firm_id, row.business_line_id, row.desk_id, row.legal_entity, row.region) == (
        "FIRM", "BL1", "D1", "LE1", "EMEA",
    )  # fmt: skip


def test_sum_vectors_is_elementwise(spark):
    df = spark.createDataFrame([("a", [1.0, 2.0]), ("a", [3.0, 5.0])], "k string, v array<double>")
    (row,) = df.groupBy("k").agg(sum_vectors("v", 2).alias("s")).collect()
    assert row.s == [4.0, 7.0]


def test_ragged_vectors_rejected(spark):
    pnl = _pnl(spark, [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match=r"pnl_vectors.*length.*\[3, 4\]"):
        build_reporting_grain(pnl, _hierarchy(spark), 4)


def test_null_elements_rejected(spark):
    pnl = _pnl(spark, [[1.0, None, 3.0]])
    with pytest.raises(ValueError, match=r"pnl_vectors.*null"):
        build_reporting_grain(pnl, _hierarchy(spark), 3)


def test_book_missing_from_hierarchy_rejected(spark):
    pnl = _pnl(spark, [[1.0, 2.0]]).withColumn("book_id", F.lit("B9"))
    with pytest.raises(ValueError, match=r"pnl_vectors.*book_id missing from hierarchy"):
        build_reporting_grain(pnl, _hierarchy(spark), 2)


def test_empty_input_is_not_an_error(spark):
    grain = build_reporting_grain(spark.createDataFrame([], PNL_SCHEMA), _hierarchy(spark), 4)
    assert grain.count() == 0


def test_aggregate_vectors_rolls_up_with_star(tiny_market):
    pnl, hierarchy, n = tiny_market
    grain = build_reporting_grain(pnl, hierarchy, n)
    nodes = aggregate_vectors(grain, [("firm_id",), ("desk_id", "asset_class")], n)
    assert nodes.columns == [*GRAIN_DIMS, "node_level", "window", "pnl"]
    rows = nodes.collect()
    levels = {r.node_level for r in rows}
    assert levels == {"firm_id", "desk_id|asset_class"}
    for r in rows:
        assert len(r.pnl) == n
        live = set(r.node_level.split("|"))
        for d in GRAIN_DIMS:
            assert (r[d] == "*") == (d not in live)
    firm = {r.window: np.array(r.pnl) for r in rows if r.node_level == "firm_id"}
    assert set(firm) == {"CURRENT", "STRESSED"}
    for w, vec in firm.items():
        total = sum(np.array(r.pnl) for r in pnl.filter(F.col("window") == w).collect())
        np.testing.assert_allclose(vec, total, atol=1e-9)


def test_filters_apply_before_aggregation(tiny_market):
    pnl, hierarchy, n = tiny_market
    grain = build_reporting_grain(pnl, hierarchy, n)
    nodes = aggregate_vectors(grain, [("firm_id",)], n, filters={"desk_id": ["DESK001"]})
    desk = aggregate_vectors(grain, [("desk_id",)], n).filter(F.col("desk_id") == "DESK001")
    a = {r.window: r.pnl for r in nodes.collect()}
    b = {r.window: r.pnl for r in desk.collect()}
    assert set(a) == set(b) == {"CURRENT", "STRESSED"}
    for w, vec in a.items():
        np.testing.assert_allclose(vec, b[w], atol=1e-9)


def test_one_pass_for_shared_specs(spark, tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    specs = _vector_specs(model)
    grain = build_reporting_grain(pnl, hierarchy, n)
    out = run_all(specs, model, grain, n)
    # AQE's initial plan: what runs, before any exchange reuse could hide a repeated branch.
    plan = out._jdf.queryExecution().executedPlan().toString()
    assert plan.count("Expand") == 1
    assert plan.count("InMemoryTableScan") == 2  # pnl_vectors and hierarchy, once each
    assert out.count() > 0


def test_node_vectors_union_both_specs(tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    specs = _vector_specs(model)
    nodes = node_vectors(specs, model, build_reporting_grain(pnl, hierarchy, n), n)
    levels = {r.node_level for r in nodes.select("node_level").distinct().collect()}
    assert levels == {
        "firm_id",
        "firm_id|business_line_id",
        "firm_id|business_line_id|desk_id",
        "firm_id|business_line_id|desk_id|book_id",
        "desk_id|asset_class",
    }


def test_run_all_covers_both_shipped_specs(tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    out = run_all(_vector_specs(model), model, build_reporting_grain(pnl, hierarchy, n), n)
    assert out.columns == METRIC_COLUMNS
    rows = out.collect()
    assert {r.scenario for r in rows} == {"*"}
    by_spec = {(r.aggregation, r.measure) for r in rows}
    assert by_spec == {
        ("var_by_org", "var_99_1d"),
        ("var_by_org", "var_99_1d__sum_children"),
        ("var_by_org", "var_99_1d__diversification"),
        ("var_by_org", "var_99_1d__incremental"),
        ("var_by_desk_asset_class", "var_99_1d"),
    }
    keys = [(r.aggregation, r.measure, r.node_level, *[r[d] for d in GRAIN_DIMS]) for r in rows]
    assert len(keys) == len(set(keys))
    firm_incremental = [
        r for r in rows if r.measure.endswith("__incremental") and r.node_level == "firm_id"
    ]
    assert firm_incremental == []


def test_partition_and_order_invariance(spark, tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    specs = _vector_specs(model)
    one = run_all(specs, model, build_reporting_grain(pnl.repartition(1), hierarchy, n), n)
    shuffled = pnl.repartition(7).orderBy(F.rand(seed=7))
    seven = run_all(specs, model, build_reporting_grain(shuffled, hierarchy, n), n)
    a, b = _metrics(one), _metrics(seven)
    assert a.keys() == b.keys() and a
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-9)


def test_empty_filter_gives_empty_frame(tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    (spec,) = [s for s in load_specs(None, model) if s.name == "var_by_org"]
    spec = dataclasses.replace(spec, filters={"desk_id": ("NOPE",)})
    out = run_all([spec], model, build_reporting_grain(pnl, hierarchy, n), n)
    assert out.columns == METRIC_COLUMNS
    assert out.count() == 0


def test_unregistered_non_vector_measure_raises(tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    stress = dataclasses.replace(model.measure("var_99_1d"), name="x", measure_class="additive")
    model = dataclasses.replace(model, measures={**model.measures, "x": stress})
    spec = AggregationSpec("x_by_desk", "x", (("desk_id",),), {}, False)
    assert "x" not in MEASURE_REGISTRY
    with pytest.raises(SpecError, match="x_by_desk.*additive.*'x'.*MEASURE_REGISTRY"):
        run_all([spec], model, build_reporting_grain(pnl, hierarchy, n), n)


def test_registered_measure_is_dispatched(spark, tiny_market):
    pnl, hierarchy, n = tiny_market
    model = load_model()
    extra = dataclasses.replace(model.measure("var_99_1d"), name="y", measure_class="additive")
    model = dataclasses.replace(model, measures={**model.measures, "y": extra})
    spec = AggregationSpec("y_total", "y", (("firm_id",),), {}, False)
    row = ("y_total", "y", "firm_id", "FIRM", *["*"] * (len(GRAIN_DIMS) - 1), "*", 1.0)

    def fake(spec_, model_, inputs_):
        return spark.createDataFrame([row], METRIC_COLUMNS)

    MEASURE_REGISTRY["y"] = fake
    try:
        out = run_all([spec], model, build_reporting_grain(pnl, hierarchy, n), n)
        assert [tuple(r) for r in out.collect()] == [row]
    finally:
        del MEASURE_REGISTRY["y"]
