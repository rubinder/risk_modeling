"""Against a real Pinot (``docker compose up -d``); skipped when the controller is unreachable.

The module drops and recreates the tables it uses, then loads the tiny ``tmp_run``, so data
from an earlier demo cannot leak into the results.
"""

import shutil
import time

import httpx
import pandas as pd
import pytest

from riskagg.context import mark_official, run_context
from riskagg.semantic.query import Query, build_drilldown, build_query, finish
from riskagg.serving import pinot
from riskagg.serving.executors import PandasExecutor, PinotExecutor
from riskagg.serving.service import RiskService

BROKER = pinot.DEFAULT_BROKER
SQL = f"{BROKER}/query/sql"


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


def _parquet_rows(tmp_run, table: str) -> int:
    return len(PandasExecutor(tmp_run.root)._dataset(table).to_table().to_pandas())


@pytest.fixture(scope="module")
def loaded(tmp_run, pinot_controller):
    _drop(pinot_controller, list(tmp_run.model.tables))
    ctx = run_context(tmp_run.root, tmp_run.run_id)
    counts = pinot.load(ctx, tmp_run.model, pinot_controller)
    return ctx, counts


def _both(tmp_run, query):
    pin = PinotExecutor(BROKER).execute(query)
    par = PandasExecutor(tmp_run.root).execute(query)
    return pin, par


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, keys) -> None:
    a = a.sort_values(keys).reset_index(drop=True) if keys else a
    b = b.sort_values(keys).reset_index(drop=True) if keys else b
    assert list(a.columns) == list(b.columns) and len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-12, atol=1e-6)


@pytest.mark.pinot
def test_load_counts_match_parquet(loaded, tmp_run):
    _, counts = loaded
    for table in tmp_run.model.tables:
        assert counts[table] == _parquet_rows(tmp_run, table) == _count(table)


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
def test_pinot_and_parquet_agree(loaded, tmp_run, measure, group_by, filters):
    q = build_query(tmp_run.model, measure, group_by, filters, tmp_run.as_of_date, tmp_run.run_id)
    pin, par = _both(tmp_run, q)
    _assert_same(finish(q, pin), finish(q, par), group_by)


@pytest.mark.pinot
def test_lookup_and_total_sql_run_on_pinot(loaded, tmp_run):
    pinned = {"as_of_date": [tmp_run.as_of_date.isoformat()], "run_id": [tmp_run.run_id]}
    lookup = Query(
        "risk_metrics", ("firm_id",), None,
        {**pinned, "measure": ["var_99_1d"], "node_level": ["firm_id"]}, "lookup", None,
    )  # fmt: skip
    pin, par = _both(tmp_run, lookup)
    _assert_same(pin, par, ["firm_id"])
    # An additive total exercises SUM(...) with COUNT(*) AS "row_count": every P&L cell summed.
    total = Query("pnl_scenarios", (), ("SUM", "pnl"), pinned, "additive", None)
    pin, par = _both(tmp_run, total)
    _assert_same(pin, par, [])
    empty = Query("pnl_scenarios", (), ("SUM", "pnl"), {**pinned, "region": ["Atlantis"]},
                  "additive", None)  # fmt: skip
    pin, par = _both(tmp_run, empty)
    assert pin.empty and par.empty and list(pin.columns) == list(par.columns)


@pytest.mark.pinot
def test_drilldown_on_pinot(loaded, tmp_run):
    d = build_drilldown(
        tmp_run.model, "risk_metrics", {"measure": ["var_99_1d"]},
        tmp_run.as_of_date, tmp_run.run_id, 1000,
    )  # fmt: skip
    keys = ["node_level", "firm_id", "business_line_id", "desk_id", "book_id"]
    pin = PinotExecutor(BROKER).fetch(d)
    par = PandasExecutor(tmp_run.root).fetch(d)
    _assert_same(pin, par, keys)


@pytest.mark.pinot
def test_service_uses_pinot_when_it_holds_the_official_run(loaded, tmp_run):
    svc = RiskService(tmp_run.root)
    pinned = svc.metric("var_99_1d", ["desk_id"])
    assert pinned.source == "pinot" and pinned.run_id == tmp_run.run_id
    parquet = RiskService(tmp_run.root, broker_url="http://127.0.0.1:9").metric(
        "var_99_1d", ["desk_id"]
    )
    assert parquet.source == "parquet"
    _assert_same(pinned.rows, parquet.rows, ["desk_id"])


@pytest.mark.pinot
def test_loading_the_same_date_twice_does_not_change_counts(loaded, tmp_run, pinot_controller):
    ctx, _ = loaded
    before = {t: _count(t) for t in tmp_run.model.tables}
    pinot.load(ctx, tmp_run.model, pinot_controller)
    assert {t: _count(t) for t in tmp_run.model.tables} == before


@pytest.mark.pinot
def test_new_official_run_replaces_the_date(loaded, tmp_run, pinot_controller, tmp_path):
    """A second official run for the same date replaces the first; Pinot then holds only it."""
    root = tmp_path / "root"
    shutil.copytree(tmp_run.root / "curated", root / "curated")
    shutil.copytree(tmp_run.root / "runs", root / "runs")
    new_id = tmp_run.run_id + "-b"
    date_dir = f"as_of_date={tmp_run.as_of_date.isoformat()}"
    for table in tmp_run.model.tables:
        src = root / "curated" / table / date_dir / f"run_id={tmp_run.run_id}"
        shutil.copytree(src, src.with_name(f"run_id={new_id}"))
    old = run_context(root, tmp_run.run_id)
    new = type(old)(old.profile, old.as_of_date, new_id, old.spec_hash, root)
    shutil.copytree(old.run_dir, new.run_dir)
    with pytest.raises(pinot.PinotLoadError, match="not the official run"):
        pinot.load(new, tmp_run.model, pinot_controller)
    mark_official(new)
    try:
        counts = pinot.load(new, tmp_run.model, pinot_controller)
        for table in tmp_run.model.tables:
            served = pinot.loaded_runs(SQL, table, tmp_run.as_of_date.isoformat())
            assert served == {new_id: counts[table]}
        assert RiskService(root).metric("var_99_1d").source == "pinot"
        assert RiskService(tmp_run.root).metric("var_99_1d").source == "parquet"
    finally:
        pinot.load(run_context(tmp_run.root, tmp_run.run_id), tmp_run.model, pinot_controller)


def test_is_up_false_when_unreachable():
    assert pinot.is_up("http://127.0.0.1:9") is False


def test_wait_times_out_with_what_it_saw(monkeypatch):
    monkeypatch.setattr(pinot, "loaded_runs", lambda *a, **k: {"r1": 5, "old": 3})
    with pytest.raises(pinot.PinotLoadError, match=r"did not serve \{'r1': 5\}.*'old': 3"):
        pinot._wait("http://controller", "t", "2026-11-12", {"r1": 5}, timeout=0)
