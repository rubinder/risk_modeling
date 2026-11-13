"""Against a real Pinot (``docker compose up -d``); skipped when the controller is unreachable.

Isolation: the tests never touch the tables ``make demo`` loads. They use a copy of the semantic
model whose tables are named ``riskagg_test_<table>``, served from a root whose curated table
directories of those names link to the tiny ``tmp_run``. The test tables are dropped before
loading (a crashed earlier run may have left them) and again on teardown.
"""

import dataclasses
import os
import shutil
import time

import httpx
import pandas as pd
import pytest

from reference.models import prefixed
from riskagg.context import mark_official, run_context
from riskagg.semantic.model import SemanticModel
from riskagg.semantic.query import Query, build_drilldown, build_query, finish
from riskagg.serving import pinot
from riskagg.serving.executors import PandasExecutor, PinotExecutor
from riskagg.serving.service import RiskService

BROKER = pinot.DEFAULT_BROKER
SQL = f"{BROKER}/query/sql"
PREFIX = "riskagg_test_"
METRICS, VECTORS = PREFIX + "risk_metrics", PREFIX + "pnl_scenarios"
STRESS = PREFIX + "stress_results"


def _count(table: str) -> int:
    r = httpx.post(SQL, json={"sql": f"SELECT COUNT(*) FROM {table}"}, timeout=30)
    body = r.json()
    assert not body.get("exceptions"), body["exceptions"]
    return body["resultTable"]["rows"][0][0]


def _drop(controller: str, tables) -> None:
    for t in tables:
        httpx.delete(f"{controller}/tables/{t}", params={"type": "offline"}, timeout=60)
        httpx.delete(f"{controller}/schemas/{t}", timeout=60)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        listed = httpx.get(f"{controller}/tables", timeout=10).json()["tables"]
        if not set(tables) & set(listed):
            return
        time.sleep(1)
    raise AssertionError(f"tables {tables} still present after drop")


@dataclasses.dataclass(frozen=True)
class Served:
    root: object
    model: SemanticModel
    as_of_date: object
    run_id: str
    counts: dict


def _parquet_rows(served, table: str) -> int:
    return len(PandasExecutor(served.root)._dataset(table).to_table().to_pandas())


@pytest.fixture(scope="module")
def served(tmp_run, pinot_controller, tmp_path_factory):
    model = prefixed(tmp_run.model, PREFIX)
    assert set(model.tables) == {METRICS, VECTORS, STRESS} and model.vector_table == VECTORS
    root = tmp_path_factory.mktemp("pinot_root")
    (root / "curated").mkdir()
    for table in tmp_run.model.tables:
        os.symlink(tmp_run.root / "curated" / table, root / "curated" / (PREFIX + table))
    shutil.copytree(tmp_run.root / "runs", root / "runs")
    _drop(pinot_controller, list(model.tables))
    try:
        counts = pinot.load(run_context(root, tmp_run.run_id), model, pinot_controller)
        yield Served(root, model, tmp_run.as_of_date, tmp_run.run_id, counts)
    finally:
        _drop(pinot_controller, list(model.tables))


def _both(served, query):
    pin = PinotExecutor(BROKER).execute(query)
    par = PandasExecutor(served.root).execute(query)
    return pin, par


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, keys) -> None:
    a = a.sort_values(keys).reset_index(drop=True) if keys else a
    b = b.sort_values(keys).reset_index(drop=True) if keys else b
    assert list(a.columns) == list(b.columns) and len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-12, atol=1e-6)


@pytest.mark.pinot
def test_load_counts_match_parquet(served):
    for table in served.model.tables:
        assert served.counts[table] == _parquet_rows(served, table) == _count(table)


@pytest.mark.pinot
def test_demo_tables_are_not_touched(served, pinot_controller):
    """Loading the test tables leaves whatever ``make demo`` loaded exactly as it was."""

    def snapshot():
        listed = httpx.get(f"{pinot_controller}/tables", timeout=10).json()["tables"]
        out = {}
        for table in ("risk_metrics", "pnl_scenarios", "stress_results"):
            if table in listed:
                segs = httpx.get(f"{pinot_controller}/segments/{table}", timeout=10).json()
                out[table] = (sorted(s for e in segs for s in e.get("OFFLINE", [])), _count(table))
        return out

    before = snapshot()
    pinot.load(run_context(served.root, served.run_id), served.model, pinot_controller)
    assert snapshot() == before


@pytest.mark.parametrize(
    "measure,group_by,filters",
    [
        ("var_99_1d", ["desk_id"], {}),
        ("var_99_1d", [], {}),
        ("es_975_1d", ["region", "asset_class"], {"business_line_id": ["BL01"]}),
        ("svar_99_1d", ["book_id"], {}),
    ],
)
@pytest.mark.pinot
def test_pinot_and_parquet_agree(served, measure, group_by, filters):
    q = build_query(served.model, measure, group_by, filters, served.as_of_date, served.run_id)
    assert q.table == VECTORS
    pin, par = _both(served, q)
    _assert_same(finish(q, pin), finish(q, par), group_by)


@pytest.mark.parametrize(
    "group_by,filters",
    [(["desk_id", "scenario"], {}), ([], {"scenario": ["GFC_2008"]}), (["region"], {})],
)
@pytest.mark.pinot
def test_stress_pnl_pinot_and_parquet_agree(served, group_by, filters):
    q = build_query(served.model, "stress_pnl", group_by, filters, served.as_of_date, served.run_id)
    assert q.table == STRESS
    pin, par = _both(served, q)
    _assert_same(finish(q, pin), finish(q, par), group_by)


@pytest.mark.pinot
def test_lookup_and_total_sql_run_on_pinot(served):
    pinned = {"as_of_date": [served.as_of_date.isoformat()], "run_id": [served.run_id]}
    lookup = Query(
        METRICS, ("firm_id",), None,
        {**pinned, "measure": ["var_99_1d"], "node_level": ["firm_id"]}, "lookup", None,
    )  # fmt: skip
    pin, par = _both(served, lookup)
    _assert_same(pin, par, ["firm_id"])
    # A scenario-bearing lookup returns the stored scenario name beside the value.
    worst = build_query(
        served.model, "worst_stress_loss", ["firm_id"], {}, served.as_of_date, served.run_id
    )
    pin, par = _both(served, worst)
    assert list(pin.columns) == ["firm_id", "scenario", "value"]
    _assert_same(pin, par, ["firm_id"])
    # An additive total exercises SUM(...) with COUNT(*) AS "row_count": every P&L cell summed.
    total = Query(VECTORS, (), ("SUM", "pnl"), pinned, "additive", None)
    pin, par = _both(served, total)
    _assert_same(pin, par, [])
    empty = Query(VECTORS, (), ("SUM", "pnl"), {**pinned, "region": ["Atlantis"]},
                  "additive", None)  # fmt: skip
    pin, par = _both(served, empty)
    assert pin.empty and par.empty and list(pin.columns) == list(par.columns)


@pytest.mark.pinot
def test_drilldown_on_pinot(served):
    d = build_drilldown(
        served.model, METRICS, {"measure": ["var_99_1d"]}, served.as_of_date, served.run_id, 1000
    )
    keys = ["node_level", "firm_id", "business_line_id", "desk_id", "book_id"]
    pin = PinotExecutor(BROKER).fetch(d)
    par = PandasExecutor(served.root).fetch(d)
    _assert_same(pin, par, keys)


@pytest.mark.pinot
def test_service_uses_pinot_when_it_holds_the_official_run(served):
    pinned = RiskService(served.root, served.model).metric("var_99_1d", ["desk_id"])
    assert pinned.source == "pinot" and pinned.run_id == served.run_id
    parquet = RiskService(served.root, served.model, broker_url="http://127.0.0.1:9").metric(
        "var_99_1d", ["desk_id"]
    )
    assert parquet.source == "parquet"
    _assert_same(pinned.rows, parquet.rows, ["desk_id"])
    drill = RiskService(served.root, served.model).drilldown(METRICS, limit=5)
    assert drill.source == "pinot" and len(drill.rows) == 5


@pytest.mark.pinot
def test_loading_the_same_date_twice_does_not_change_counts(served, pinot_controller):
    before = {t: _count(t) for t in served.model.tables}
    pinot.load(run_context(served.root, served.run_id), served.model, pinot_controller)
    assert {t: _count(t) for t in served.model.tables} == before


@pytest.mark.pinot
def test_partly_loaded_run_is_not_served(served, pinot_controller, monkeypatch):
    """Pinot holds only the official run, but fewer rows than its Parquet: answer from Parquet.

    The vector table is loaded in 500-row segments and one is deleted, as if an upload had
    failed half way. VaR from the remaining rows would still have every scenario per group,
    just smaller sums, so only the row count shows the problem.
    """
    ctx = run_context(served.root, served.run_id)
    full = served.counts[VECTORS]
    assert full > 1000  # three segments of at most 500 rows
    monkeypatch.setattr(pinot, "SEGMENT_ROWS", 500)
    date = served.as_of_date.isoformat()
    try:
        pinot.load(ctx, served.model, pinot_controller)
        last = f"{pinot.segment_prefix(VECTORS, date)}{served.run_id}__2"
        r = httpx.delete(f"{pinot_controller}/segments/{VECTORS}_OFFLINE/{last}", timeout=60)
        assert r.status_code == 200, r.text
        deadline = time.monotonic() + 60
        while pinot.loaded_runs(SQL, VECTORS, date) == {served.run_id: full}:
            assert time.monotonic() < deadline, "segment delete never took effect"
            time.sleep(0.5)
        partial = pinot.loaded_runs(SQL, VECTORS, date)
        assert set(partial) == {served.run_id} and partial[served.run_id] == 1000
        svc = RiskService(served.root, served.model)
        assert not svc.pinot_serves(VECTORS, served.as_of_date, served.run_id)
        answer = svc.metric("var_99_1d", ["desk_id"])
        assert answer.source == "parquet"
        # the metrics table is still complete, so a drill-down there may use Pinot
        assert svc.drilldown(METRICS, limit=5).source == "pinot"
    finally:
        monkeypatch.undo()
        pinot.load(ctx, served.model, pinot_controller)
    assert RiskService(served.root, served.model).metric("var_99_1d").source == "pinot"


@pytest.mark.pinot
def test_new_official_run_replaces_the_date(served, pinot_controller, tmp_path):
    """A second official run for the same date replaces the first; Pinot then holds only it."""
    root = tmp_path / "root"
    shutil.copytree(served.root / "curated", root / "curated")  # copies the linked data
    shutil.copytree(served.root / "runs", root / "runs")
    new_id = served.run_id + "-b"
    date_dir = f"as_of_date={served.as_of_date.isoformat()}"
    for table in served.model.tables:
        src = root / "curated" / table / date_dir / f"run_id={served.run_id}"
        shutil.copytree(src, src.with_name(f"run_id={new_id}"))
    old = run_context(root, served.run_id)
    new = type(old)(old.profile, old.as_of_date, new_id, old.spec_hash, root)
    shutil.copytree(old.run_dir, new.run_dir)
    with pytest.raises(pinot.PinotLoadError, match="not the official run"):
        pinot.load(new, served.model, pinot_controller)
    mark_official(new)
    try:
        counts = pinot.load(new, served.model, pinot_controller)
        for table in served.model.tables:
            loaded = pinot.loaded_runs(SQL, table, served.as_of_date.isoformat())
            assert loaded == {new_id: counts[table]}
        assert RiskService(root, served.model).metric("var_99_1d").source == "pinot"
        assert RiskService(served.root, served.model).metric("var_99_1d").source == "parquet"
    finally:
        pinot.load(run_context(served.root, served.run_id), served.model, pinot_controller)


def test_is_up_false_when_unreachable():
    assert pinot.is_up("http://127.0.0.1:9") is False


def test_wait_times_out_with_what_it_saw(monkeypatch):
    monkeypatch.setattr(pinot, "loaded_runs", lambda *a, **k: {"r1": 5, "old": 3})
    with pytest.raises(pinot.PinotLoadError, match=r"did not serve \{'r1': 5\}.*'old': 3"):
        pinot._wait("http://controller", "t", "2026-11-12", {"r1": 5}, timeout=0)
