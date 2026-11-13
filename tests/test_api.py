"""API and client against the tiny ``tmp_run`` root, answered from Parquet (no Pinot needed)."""

import datetime
import json
import logging

import httpx
import pyarrow.dataset as ds
import pytest
from fastapi.testclient import TestClient

from riskagg.context import mark_official, run_context
from riskagg.semantic.query import QueryError, SliceNotAvailable
from riskagg.serving import service as service_mod
from riskagg.serving.api import create_app
from riskagg.serving.client import RiskClient
from riskagg.serving.executors import PandasExecutor
from riskagg.serving.service import NotFound, resolve_run

DOWN = "http://127.0.0.1:9"  # nothing listens on the discard port


@pytest.fixture(scope="module")
def api(tmp_run):
    return TestClient(create_app(tmp_run.root, broker_url=DOWN, controller_url=DOWN))


def _precomputed(tmp_run, measure, node_level):
    metrics = (
        ds.dataset(tmp_run.root / "curated/risk_metrics", partitioning="hive").to_table().to_pandas()
    )
    rows = metrics[
        (metrics.run_id == tmp_run.run_id)
        & (metrics.measure == measure)
        & (metrics.node_level == node_level)
    ]
    return rows


def test_firm_var_equals_risk_metrics(api, tmp_run):
    (expected,) = _precomputed(tmp_run, "var_99_1d", "firm_id")["value"]
    r = api.get("/metrics/var_99_1d", params={"group_by": "firm_id"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["measure"] == "var_99_1d" and body["source"] == "parquet"
    assert body["as_of_date"] == tmp_run.as_of_date.isoformat()
    assert body["run_id"] == tmp_run.run_id
    assert body["rows"] == [{"firm_id": "FIRM", "value": pytest.approx(expected, abs=1e-9)}]
    total = api.get("/metrics/var_99_1d").json()["rows"]
    assert total == [{"value": pytest.approx(expected, abs=1e-9)}]


def test_var_by_desk_with_filters(api, tmp_run):
    pre = _precomputed(tmp_run, "var_99_1d", "firm_id|business_line_id|desk_id")
    pre = dict(zip(pre.desk_id, pre.value, strict=True))
    desks = sorted(pre)[:2]
    r = api.get("/metrics/var_99_1d", params={"group_by": "desk_id", "desk_id": ",".join(desks)})
    rows = r.json()["rows"]
    assert [row["desk_id"] for row in rows] == desks
    for row in rows:
        assert row["value"] == pytest.approx(pre[row["desk_id"]], abs=1e-9)
    repeated = api.get(
        "/metrics/var_99_1d", params=[("group_by", "desk_id"), *(("desk_id", d) for d in desks)]
    )
    assert repeated.json()["rows"] == rows


def test_errors(api):
    r = api.get("/metrics/nope")
    assert r.status_code == 404 and "nope" in r.json()["detail"]
    r = api.get("/metrics/var_99_1d", params={"group_by": "counterparty_id"})
    assert r.status_code == 422
    assert "counterparty_id" in r.json()["detail"] and "non-additive" in r.json()["detail"]
    r = api.get("/metrics/var_99_1d", params={"group_by": "bogus"})
    assert r.status_code == 400 and "bogus" in r.json()["detail"]
    r = api.get("/metrics/var_99_1d", params={"as_of": "2001-01-01"})
    assert r.status_code == 404 and "2001-01-01" in r.json()["detail"]
    r = api.get("/metrics/var_99_1d", params={"as_of": "yesterday"})
    assert r.status_code == 400
    assert api.get("/drilldown/nope").status_code == 404


def test_catalog(api, tmp_run):
    body = api.get("/catalog").json()
    var = body["measures"]["var_99_1d"]
    assert var["class"] == "vector" and var["unit"] == "USD"
    assert var["group_by"] == {"any_subset_of": list(tmp_run.model.grain)}
    assert "counterparty_id" in body["dimensions"]
    assert body["hierarchies"]["org"] == ["firm_id", "business_line_id", "desk_id", "book_id"]
    assert set(body["tables"]) == set(tmp_run.model.tables)
    # where each measure is precomputed, and the derived rows stored beside it
    assert set(body["derived_rows"]) == {"sum_children", "diversification", "incremental"}
    (stored,) = [s for s in var["stored_in_risk_metrics"] if s["aggregation"] == "var_by_org"]
    assert stored["node_levels"][0] == "firm_id"
    assert stored["derived_measures"] == [
        "var_99_1d__sum_children", "var_99_1d__diversification", "var_99_1d__incremental",
    ]  # fmt: skip
    assert body["measures"]["stress_pnl"]["stored_in_risk_metrics"] == []
    es = body["measures"]["es_975_1d"]["stored_in_risk_metrics"]
    assert [s["aggregation"] for s in es] == ["es_by_org"]


def test_drilldown(api, tmp_run):
    r = api.get("/drilldown/pnl_scenarios", params={"desk_id": "DESK001", "window": "CURRENT"})
    body = r.json()
    assert r.status_code == 200 and body["table"] == "pnl_scenarios"
    rows = body["rows"]
    assert rows and len(rows) <= 1000
    assert {row["desk_id"] for row in rows} == {"DESK001"}
    assert {row["run_id"] for row in rows} == {tmp_run.run_id}
    assert {row["as_of_date"] for row in rows} == {tmp_run.as_of_date.isoformat()}
    assert list(rows[0]) == list(tmp_run.model.tables["pnl_scenarios"]["columns"])
    limited = api.get("/drilldown/pnl_scenarios", params={"limit": 7}).json()["rows"]
    assert len(limited) == 7
    assert api.get("/drilldown/pnl_scenarios", params={"pnl": "1"}).status_code == 400
    assert api.get("/drilldown/pnl_scenarios", params={"limit": 0}).status_code == 400


def test_client_without_pinot_answers_from_parquet(tmp_run, caplog):
    client = RiskClient(root=tmp_run.root, broker_url=DOWN, controller_url=DOWN)
    with caplog.at_level(logging.WARNING, logger="riskagg.serving"):
        df = client.metric("var_99_1d", "firm_id")
    assert client.source == "parquet" and client.run_id == tmp_run.run_id
    (expected,) = _precomputed(tmp_run, "var_99_1d", "firm_id")["value"]
    assert list(df.columns) == ["firm_id", "value"] and df.value[0] == pytest.approx(expected)
    assert "Parquet" in caplog.text
    assert client.catalog()["measures"]["var_99_1d"]["class"] == "vector"
    with pytest.raises(SliceNotAvailable):
        client.metric("var_99_1d", ["counterparty_id"])


def test_client_falls_back_when_pinot_fails_mid_query(tmp_run, monkeypatch, caplog):
    """Pinot looks available, but the broker refuses the connection when queried."""
    client = RiskClient(root=tmp_run.root, broker_url=DOWN, controller_url=DOWN)
    monkeypatch.setattr(service_mod.RiskService, "pinot_serves", lambda self, *a: True)
    with caplog.at_level(logging.WARNING, logger="riskagg.serving"):
        df = client.metric("var_99_1d", ["desk_id"], desk_id="DESK001")
    assert client.source == "parquet" and len(df) == 1
    assert "Pinot query failed" in caplog.text


def test_client_over_http(api, tmp_run):
    client = RiskClient(base_url="http://testserver", http=api)
    df = client.metric("var_99_1d", ["desk_id"], desk_id=["DESK001", "DESK002"])
    assert client.source == "parquet" and client.run_id == tmp_run.run_id
    assert client.as_of_date == tmp_run.as_of_date
    local = RiskClient(root=tmp_run.root, broker_url=DOWN, controller_url=DOWN)
    assert df.equals(local.metric("var_99_1d", ["desk_id"], desk_id=["DESK001", "DESK002"]))
    assert client.catalog() == json.loads(json.dumps(local.catalog()))
    empty = client.metric("var_99_1d", ["desk_id"], region="Atlantis")
    assert empty.empty and list(empty.columns) == ["desk_id", "value"]
    with pytest.raises(SliceNotAvailable, match="counterparty_id"):
        client.metric("var_99_1d", ["counterparty_id"])
    with pytest.raises(NotFound):
        client.metric("nope")
    with pytest.raises(QueryError):
        client.metric("var_99_1d", ["bogus"])
    rows = client.drilldown("risk_metrics", limit=5, measure="var_99_1d")
    assert len(rows) == 5 and set(rows.measure) == {"var_99_1d"}


def test_resolve_run_defaults_to_latest_official(tmp_run, tmp_path):
    assert resolve_run(tmp_run.root, None) == (tmp_run.as_of_date, tmp_run.run_id)
    with pytest.raises(NotFound, match="no official run"):
        resolve_run(tmp_path, None)
    # a later date made official in a copy of the root becomes the default
    later = datetime.date(2026, 11, 13)
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    ctx = type(ctx)(ctx.profile, later, "20261113T000000Z-later", ctx.spec_hash, tmp_path)
    mark_official(ctx)
    mark_official(type(ctx)(ctx.profile, tmp_run.as_of_date, "older", ctx.spec_hash, tmp_path))
    assert resolve_run(tmp_path, None) == (later, "20261113T000000Z-later")
    assert resolve_run(tmp_path, tmp_run.as_of_date.isoformat()) == (tmp_run.as_of_date, "older")


def test_pinot_used_only_when_it_holds_exactly_the_official_run(tmp_run, monkeypatch):
    """Same run and the same row count as the run's Parquet partition, else Parquet."""
    svc = service_mod.RiskService(tmp_run.root, broker_url=DOWN, controller_url=DOWN)
    full = len(PandasExecutor(tmp_run.root)._dataset("pnl_scenarios").to_table().to_pandas())
    run, date = tmp_run.run_id, tmp_run.as_of_date

    def serving(loaded):
        monkeypatch.setattr(service_mod.pinot, "loaded_runs", lambda *a, **k: loaded)
        return svc.pinot_serves("pnl_scenarios", date, run)

    monkeypatch.setattr(service_mod.pinot, "is_up", lambda url: True)
    assert serving({run: full})
    assert not serving({"other-run": full})
    assert not serving({run: full, "old": 3})
    assert not serving({run: full - 1})  # partly loaded: same run, fewer rows
    assert not serving({run: full + 1})
    assert not serving({})
    assert not svc.pinot_serves("pnl_scenarios", datetime.date(2001, 1, 1), run)
    monkeypatch.setattr(
        service_mod.pinot,
        "loaded_runs",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("refused")),
    )
    assert not svc.pinot_serves("pnl_scenarios", date, run)
