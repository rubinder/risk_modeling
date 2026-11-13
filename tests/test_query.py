import datetime
import json
import math

import httpx
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest
import yaml

from riskagg.semantic.model import SpecError, load_model
from riskagg.semantic.pinot_schema import pinot_schema, pinot_table_config
from riskagg.semantic.query import (
    Query,
    QueryError,
    SliceNotAvailable,
    build_drilldown,
    build_query,
    finish,
)
from riskagg.serving.executors import PandasExecutor, PinotError, PinotExecutor

D = datetime.date(2026, 11, 12)
R = "20261112T000000Z-abc123"


@pytest.fixture(scope="module")
def model():
    return load_model()


def _col(name, role, type_="STRING"):
    return {"type": type_, "role": role}


@pytest.fixture
def mini_model(tmp_path):
    """Additive and structured measures, which do not ship yet, over small tables."""
    raw = yaml.safe_load(load_model_text())
    raw["measures"]["notional"] = {
        "label": "Notional", "class": "additive", "function": "sum", "unit": "USD",
        "table": "positions", "dimensions": ["desk_id", "currency"],
    }  # fmt: skip
    raw["measures"]["ccar_loss"] = {
        "label": "Loss", "class": "structured", "table": "risk_metrics",
        "computed_at": [["desk_id", "asset_class"], ["firm_id"]],
    }  # fmt: skip
    raw["tables"]["positions"] = {
        "columns": {
            "as_of_date": _col("as_of_date", "time", "DATE"),
            "run_id": _col("run_id", "dimension"),
            "desk_id": _col("desk_id", "dimension"),
            "currency": _col("currency", "dimension"),
            "counterparty_id": _col("counterparty_id", "dimension"),
            "notional": _col("notional", "metric", "DOUBLE"),
        }
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_model(path)


def load_model_text():
    from riskagg.paths import semantic_model_path

    return semantic_model_path().read_text()


def test_vector_sql_sums_by_scenario_only(model):
    sql = build_query(model, "var_99_1d", ["desk_id"], {}, D, R).to_pinot_sql()
    assert 'SUM("pnl")' in sql and "scenario_idx" in sql and "PERCENTILE" not in sql.upper()
    assert 'GROUP BY "desk_id", "scenario_idx"' in sql
    assert """"window" IN ('CURRENT')""" in sql and 'FROM "pnl_scenarios"' in sql


def test_var_by_counterparty_refused(model):
    with pytest.raises(SliceNotAvailable, match="counterparty_id") as e:
        build_query(model, "var_99_1d", ["counterparty_id"], {}, D, R)
    assert "non-additive" in str(e.value)
    with pytest.raises(SliceNotAvailable, match="counterparty_id"):
        build_query(model, "var_99_1d", ["desk_id"], {"counterparty_id": ["c1"]}, D, R)


def test_unknown_measure_and_dimension(model):
    with pytest.raises(SpecError, match="nope"):
        build_query(model, "nope", [], {}, D, R)
    with pytest.raises(QueryError, match="bogus"):
        build_query(model, "var_99_1d", ["bogus"], {}, D, R)
    with pytest.raises(QueryError, match="bogus"):
        build_query(model, "var_99_1d", [], {"bogus": ["x"]}, D, R)
    with pytest.raises(QueryError, match="list of strings"):
        build_query(model, "var_99_1d", [], {"desk_id": "D1"}, D, R)


def test_every_query_pins_date_and_run(model, mini_model):
    queries = [
        build_query(model, "var_99_1d", ["desk_id"], {"region": ["EMEA"]}, D, R),
        build_query(mini_model, "notional", ["desk_id"], {}, D, R),
        build_query(mini_model, "ccar_loss", ["asset_class", "desk_id"], {}, D, R),
    ]
    for q in queries:
        sql = q.to_pinot_sql()
        assert """"as_of_date" IN ('2026-11-12')""" in sql and f""""run_id" IN ('{R}')""" in sql


def test_filters_render_as_quoted_in_lists(model):
    sql = build_query(model, "var_99_1d", [], {"desk_id": ["a", "O'Brien"]}, D, R).to_pinot_sql()
    assert """"desk_id" IN ('a', 'O''Brien')""" in sql
    assert 'GROUP BY "scenario_idx"' in sql


def test_unvalidated_identifier_is_never_interpolated():
    q = Query("pnl_scenarios", ("desk_id; DROP",), ("SUM", "pnl"), {}, "vector_scenarios", None)
    with pytest.raises(QueryError, match="identifier"):
        q.to_pinot_sql()


def test_additive_query(mini_model):
    q = build_query(mini_model, "notional", ["desk_id"], {"currency": ["USD"]}, D, R)
    assert q.kind == "additive"
    sql = q.to_pinot_sql()
    assert sql.startswith('SELECT "desk_id", SUM("notional") AS "value" FROM "positions" WHERE ')
    assert """"currency" IN ('USD')""" in sql and 'GROUP BY "desk_id"' in sql
    with pytest.raises(SliceNotAvailable, match="counterparty_id"):
        build_query(mini_model, "notional", ["counterparty_id"], {}, D, R)
    total = build_query(mini_model, "notional", [], {}, D, R).to_pinot_sql()
    assert "GROUP BY" not in total


def test_additive_extra_fixed_filters_are_general():
    q = Query(
        "t", ("a",), ("SUM", "x"), {"measure": ["m1"], "scenario": ["s", "t"]}, "additive", None
    )
    sql = q.to_pinot_sql()
    assert """"measure" IN ('m1')""" in sql and """"scenario" IN ('s', 't')""" in sql


def test_structured_lookup_matches_any_order(mini_model):
    q = build_query(mini_model, "ccar_loss", ["desk_id", "asset_class"], {"desk_id": ["D1"]}, D, R)
    assert q.kind == "lookup" and q.select_dims == ("desk_id", "asset_class")
    sql = q.to_pinot_sql()
    assert 'FROM "risk_metrics"' in sql and """"measure" IN ('ccar_loss')""" in sql
    assert """"node_level" IN ('desk_id|asset_class')""" in sql
    assert "SUM" not in sql
    q2 = build_query(mini_model, "ccar_loss", ["asset_class", "desk_id"], {}, D, R)
    assert """"node_level" IN ('desk_id|asset_class')""" in q2.to_pinot_sql()
    with pytest.raises(SliceNotAvailable, match="computed_at|precomputed"):
        build_query(mini_model, "ccar_loss", ["region"], {}, D, R)
    with pytest.raises(SliceNotAvailable):
        build_query(mini_model, "ccar_loss", ["firm_id"], {"desk_id": ["D1"]}, D, R)


def test_adhoc_var_equals_precomputed(tmp_run):
    m = tmp_run.model
    ex = PandasExecutor(tmp_run.root)
    q = build_query(m, "var_99_1d", ["desk_id"], {}, tmp_run.as_of_date, tmp_run.run_id)
    got = finish(q, ex.execute(q)).set_index("desk_id")["value"]
    metrics = (
        ds.dataset(tmp_run.root / "curated/risk_metrics", partitioning="hive")
        .to_table()
        .to_pandas()
    )
    pre = metrics[
        (metrics.run_id == tmp_run.run_id)
        & (metrics.measure == "var_99_1d")
        & (metrics.node_level == "firm_id|business_line_id|desk_id")
    ].set_index("desk_id")["value"]
    assert len(pre) > 1 and set(got.index) == set(pre.index)
    for desk, v in pre.items():
        assert got[desk] == pytest.approx(v, abs=1e-9)


def _reference(tmp_run, dims, level, window="CURRENT"):
    pnl = (
        ds.dataset(tmp_run.root / "curated/pnl_scenarios", partitioning="hive")
        .to_table()
        .to_pandas()
    )
    pnl = pnl[(pnl.run_id == tmp_run.run_id) & (pnl.window == window)]
    out = {}
    for key, g in (
        pnl.groupby([*dims, "scenario_idx"])["pnl"].sum().groupby(level=list(range(len(dims))))
    ):
        losses = np.sort(-g.to_numpy())[::-1]
        rank = max(1, math.ceil(round(len(losses) * (1 - level), 9)))
        out[key] = losses[rank - 1]
    return out


def test_adhoc_slice_not_precomputed(tmp_run):
    m = tmp_run.model
    q = build_query(
        m, "var_99_1d", ["region", "asset_class"], {}, tmp_run.as_of_date, tmp_run.run_id
    )
    got = finish(q, PandasExecutor(tmp_run.root).execute(q))
    assert list(got.columns) == ["region", "asset_class", "value"]
    ref = _reference(tmp_run, ["region", "asset_class"], 0.99)
    assert len(got) == len(ref) > 1
    for row in got.itertuples():
        assert row.value == pytest.approx(ref[(row.region, row.asset_class)], abs=1e-9)


def test_horizon_scaling_filters_and_es(tmp_run):
    m, ex = tmp_run.model, PandasExecutor(tmp_run.root)
    kw = (tmp_run.as_of_date, tmp_run.run_id)
    qs = {n: build_query(m, n, [], {}, *kw) for n in ("var_99_1d", "var_99_10d")}
    one, ten = (finish(q, ex.execute(q)) for q in qs.values())
    assert len(one) == 1 and ten.value[0] == pytest.approx(one.value[0] * math.sqrt(10))
    q = build_query(m, "es_975_1d", ["region"], {"asset_class": ["__none__"]}, *kw)
    assert finish(q, ex.execute(q)).empty


def test_no_matching_rows_is_empty(tmp_run):
    m = tmp_run.model
    q = build_query(
        m, "var_99_1d", ["desk_id"], {"region": ["Atlantis"]}, tmp_run.as_of_date, tmp_run.run_id
    )
    rows = PandasExecutor(tmp_run.root).execute(q)
    out = finish(q, rows)
    assert out.empty and list(out.columns) == ["desk_id", "value"]
    q = build_query(m, "var_99_1d", ["desk_id"], {}, tmp_run.as_of_date, "no-such-run")
    assert finish(q, PandasExecutor(tmp_run.root).execute(q)).empty


def test_pandas_executor_additive_and_lookup(tmp_path):
    ds_dir = tmp_path / "curated/positions/as_of_date=2026-11-12/run_id=r1"
    ds_dir.mkdir(parents=True)
    pd.DataFrame(
        {"desk_id": ["a", "a", "b"], "currency": ["USD", "EUR", "USD"], "notional": [1.0, 2.0, 4.0]}
    ).to_parquet(ds_dir / "part-0.parquet")
    q = Query(
        "positions", ("desk_id",), ("SUM", "notional"),
        {"as_of_date": ["2026-11-12"], "run_id": ["r1"], "currency": ["USD", "EUR"]},
        "additive", None,
    )  # fmt: skip
    out = PandasExecutor(tmp_path).execute(q).sort_values("desk_id")
    assert list(out.columns) == ["desk_id", "value"] and list(out.value) == [3.0, 4.0]
    q = Query(
        "positions", ("desk_id",), ("SUM", "notional"),
        {"as_of_date": ["2026-11-12"], "run_id": ["r2"]}, "additive", None,
    )  # fmt: skip
    assert PandasExecutor(tmp_path).execute(q).empty


def _pinot(handler):
    return PinotExecutor(
        "http://broker:8099", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_pinot_executor_posts_sql_and_builds_frame(model):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"resultTable": {
            "dataSchema": {"columnNames": ["desk_id", "scenario_idx", "pnl"],
                           "columnDataTypes": ["STRING", "INT", "DOUBLE"]},
            "rows": [["d1", 0, -1.5], ["d1", 1, 2.0]]}})  # fmt: skip

    q = build_query(model, "var_99_1d", ["desk_id"], {}, D, R)
    df = _pinot(handler).execute(q)
    assert seen["url"] == "http://broker:8099/query/sql" and seen["body"] == {
        "sql": q.to_pinot_sql()
    }
    assert list(df.columns) == ["desk_id", "scenario_idx", "pnl"] and len(df) == 2
    assert finish(q, df)["value"][0] == pytest.approx(1.5)


def test_pinot_executor_raises_on_exceptions(model):
    def handler(request):
        return httpx.Response(200, json={"exceptions": [{"errorCode": 150, "message": "bad sql"}]})

    q = build_query(model, "var_99_1d", ["desk_id"], {}, D, R)
    with pytest.raises(PinotError, match="bad sql"):
        _pinot(handler).execute(q)
    with pytest.raises(PinotError, match="500"):
        _pinot(lambda r: httpx.Response(500, text="boom")).execute(q)


def test_pinot_executor_empty_result(model):
    def handler(request):
        return httpx.Response(200, json={"resultTable": {
            "dataSchema": {"columnNames": ["desk_id", "scenario_idx", "pnl"],
                           "columnDataTypes": ["STRING", "INT", "DOUBLE"]}, "rows": []}})  # fmt: skip

    q = build_query(model, "var_99_1d", ["desk_id"], {}, D, R)
    df = _pinot(handler).execute(q)
    assert df.empty and list(df.columns) == ["desk_id", "scenario_idx", "pnl"]


def test_schema_generated_from_model(model):
    for table, spec in model.tables.items():
        schema = pinot_schema(model, table)
        assert schema["schemaName"] == table
        fields = [
            f["name"]
            for key in ("dimensionFieldSpecs", "metricFieldSpecs", "dateTimeFieldSpecs")
            for f in schema.get(key, [])
        ]
        assert sorted(fields) == sorted(spec["columns"])
        (time,) = schema["dateTimeFieldSpecs"]
        assert time == {
            "name": "as_of_date", "dataType": "STRING",
            "format": "1:DAYS:SIMPLE_DATE_FORMAT:yyyy-MM-dd", "granularity": "1:DAYS",
        }  # fmt: skip
    pnl = pinot_schema(model, "pnl_scenarios")
    assert [f["name"] for f in pnl["metricFieldSpecs"]] == ["pnl"]
    assert {"name": "scenario_idx", "dataType": "INT"} in pnl["dimensionFieldSpecs"]
    with pytest.raises(SpecError, match="nope"):
        pinot_schema(model, "nope")


def test_table_config(model, mini_model):
    cfg = pinot_table_config(model, "risk_metrics")
    assert cfg["tableName"] == "risk_metrics" and cfg["tableType"] == "OFFLINE"
    assert cfg["segmentsConfig"]["timeColumnName"] == "as_of_date"
    assert cfg["segmentsConfig"]["replication"] == "1"
    assert cfg["tableIndexConfig"]["invertedIndexColumns"] == ["node_level", "desk_id", "measure"]
    assert "starTreeIndexConfigs" not in cfg["tableIndexConfig"]  # vector measures: no star-tree
    pnl = pinot_table_config(model, "pnl_scenarios")
    assert pnl["tableIndexConfig"]["invertedIndexColumns"] == ["desk_id"]
    assert "starTreeIndexConfigs" not in pnl["tableIndexConfig"]
    pos = pinot_table_config(mini_model, "positions")
    (tree,) = pos["tableIndexConfig"]["starTreeIndexConfigs"]
    assert tree["functionColumnPairs"] == ["SUM__notional"]
    assert (
        tree["dimensionsSplitOrder"][0] == "as_of_date"
        and "desk_id" in tree["dimensionsSplitOrder"]
    )
    # risk_metrics also serves a structured measure here, so it never gets a star-tree.
    assert (
        "starTreeIndexConfigs"
        not in pinot_table_config(mini_model, "risk_metrics")["tableIndexConfig"]
    )


def _vector_rows(dims_rows):
    return pd.DataFrame(dims_rows, columns=["desk_id", "scenario_idx", "pnl"])


def test_finish_requires_complete_scenario_sets(model):
    q = build_query(model, "var_99_1d", ["desk_id"], {}, D, R)
    full = [("a", i, float(i)) for i in range(5)]
    assert len(finish(q, _vector_rows(full))) == 1
    with pytest.raises(QueryError, match="'b'.*4 scenarios"):  # gap: 0,1,2,4
        finish(q, _vector_rows(full + [("b", i, 1.0) for i in (0, 1, 2, 4)]))
    with pytest.raises(QueryError, match="'b'"):  # duplicate scenario
        finish(q, _vector_rows(full + [("b", i, 1.0) for i in (0, 1, 2, 3, 3)]))
    with pytest.raises(QueryError, match="'b'.*3 scenarios"):  # shorter than the other group
        finish(q, _vector_rows(full + [("b", i, 1.0) for i in range(3)]))
    total = build_query(model, "var_99_1d", [], {}, D, R)
    with pytest.raises(QueryError, match="scenario"):
        finish(total, pd.DataFrame({"scenario_idx": [0, 2], "pnl": [1.0, 2.0]}))


def test_finish_empty_vector_result_has_right_columns(model):
    for dims in (["desk_id"], []):
        q = build_query(model, "var_99_1d", dims, {}, D, R)
        out = finish(q, pd.DataFrame(columns=q.columns))
        assert out.empty and list(out.columns) == [*dims, "value"]


@pytest.mark.parametrize("flag", ["numGroupsLimitReached", "partialResult"])
def test_pinot_executor_refuses_truncated_results(model, flag):
    def handler(request):
        return httpx.Response(200, json={flag: True, "resultTable": {
            "dataSchema": {"columnNames": ["desk_id", "scenario_idx", "pnl"]},
            "rows": [["d1", 0, 1.0]]}})  # fmt: skip

    q = build_query(model, "var_99_1d", ["desk_id"], {}, D, R)
    with pytest.raises(PinotError, match=flag):
        _pinot(handler).execute(q)


def test_pinot_executor_refuses_row_limit_hit(model, monkeypatch):
    monkeypatch.setattr("riskagg.serving.executors.MAX_ROWS", 2)
    rows = [["d1", 0, 1.0], ["d1", 1, 1.0]]

    def handler(request):
        return httpx.Response(200, json={"resultTable": {
            "dataSchema": {"columnNames": ["desk_id", "scenario_idx", "pnl"]}, "rows": rows}})  # fmt: skip

    q = build_query(model, "var_99_1d", ["desk_id"], {}, D, R)
    with pytest.raises(PinotError, match="row limit"):
        _pinot(handler).execute(q)


def test_empty_total_is_same_on_both_backends(mini_model, tmp_path):
    q = build_query(mini_model, "notional", [], {}, D, R)
    assert 'COUNT(*) AS "row_count"' in q.to_pinot_sql()

    def handler(count):
        def respond(request):
            return httpx.Response(200, json={"resultTable": {
                "dataSchema": {"columnNames": ["value", "row_count"]},
                "rows": [[None if count == 0 else 5.0, count]]}})  # fmt: skip

        return respond

    empty = _pinot(handler(0)).execute(q)
    assert empty.empty and list(empty.columns) == ["value"]
    full = _pinot(handler(2)).execute(q)
    assert list(full.columns) == ["value"] and list(full.value) == [5.0]
    leaf = tmp_path / "curated/positions/as_of_date=2026-11-12/run_id=other"
    leaf.mkdir(parents=True)
    pd.DataFrame({"notional": [1.0]}).to_parquet(leaf / "p.parquet")
    pandas_empty = PandasExecutor(tmp_path).execute(q)
    assert pandas_empty.empty and list(pandas_empty.columns) == list(empty.columns)


def test_drilldown_sql_and_validation(model):
    d = build_drilldown(model, "risk_metrics", {"measure": ["var_99_1d"]}, D, R, 50)
    sql = d.to_pinot_sql()
    assert sql.startswith('SELECT "as_of_date", "run_id", "spec_hash", "aggregation", "measure"')
    assert sql.endswith(
        f"""FROM "risk_metrics" WHERE "as_of_date" IN ('2026-11-12') AND "run_id" IN ('{R}') """
        """AND "measure" IN ('var_99_1d') LIMIT 50"""
    )
    assert d.columns == tuple(model.tables["risk_metrics"]["columns"])
    with pytest.raises(QueryError, match="unknown table"):
        build_drilldown(model, "nope", {}, D, R, 10)
    for bad in ({"value": ["1"]}, {"run_id": ["x"]}, {"as_of_date": ["2026-01-01"]}):
        with pytest.raises(QueryError, match="cannot be filtered"):
            build_drilldown(model, "risk_metrics", bad, D, R, 10)
    with pytest.raises(QueryError, match="list of strings"):
        build_drilldown(model, "risk_metrics", {"measure": "var_99_1d"}, D, R, 10)
    with pytest.raises(QueryError, match="limit"):
        build_drilldown(model, "risk_metrics", {}, D, R, 0)


def test_pandas_executor_drilldown(tmp_run):
    d = build_drilldown(
        tmp_run.model, "pnl_scenarios", {"desk_id": ["DESK002"], "window": ["STRESSED"]},
        tmp_run.as_of_date, tmp_run.run_id, 30,
    )  # fmt: skip
    rows = PandasExecutor(tmp_run.root).fetch(d)
    assert len(rows) == 30 and list(rows.columns) == list(d.columns)
    assert set(rows.desk_id) == {"DESK002"} and set(rows.window) == {"STRESSED"}
    assert set(rows.run_id) == {tmp_run.run_id}


def test_vector_table_comes_from_the_model(model, tmp_path):
    from reference.models import prefixed

    assert model.vector_table == "pnl_scenarios"
    q = build_query(prefixed(model, "t_"), "var_99_1d", ["desk_id"], {}, D, R)
    assert q.table == "t_pnl_scenarios" and 'FROM "t_pnl_scenarios"' in q.to_pinot_sql()
    raw = yaml.safe_load(load_model_text())
    raw["vector_table"] = "nope"
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SpecError, match="vector_table 'nope'"):
        load_model(path)
