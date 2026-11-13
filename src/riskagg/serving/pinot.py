"""Loading the official run's curated tables into Apache Pinot.

Mechanism: for every table in the semantic model, the schema and OFFLINE table config are
generated from the model and created or updated over the controller's REST API. The run's
partition is read with pyarrow, the hive partition keys (``as_of_date``, ``run_id``) are added
back as columns, and the rows are uploaded to ``/ingestFromFile`` in chunks; the controller
builds one segment per chunk. Segments are named ``<table>__<as_of_date>__<run_id>__<n>``, so

* re-loading the same run uploads segments with the same names, which Pinot replaces;
* after the upload every other segment of that as-of date (an earlier run, or extra chunks) is
  deleted, so the date holds exactly the official run's rows.

Pinot applies uploads and deletes asynchronously; ``load`` waits until the broker serves
exactly the expected row count for the date.
"""

import json
import logging
import tempfile
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from riskagg.context import RunContext, official_run
from riskagg.semantic.model import SemanticModel
from riskagg.semantic.pinot_schema import pinot_schema, pinot_table_config

DEFAULT_CONTROLLER = "http://localhost:9000"
DEFAULT_BROKER = "http://localhost:8000"
SEGMENT_ROWS = 1_000_000  # rows per uploaded segment
PARTITION_COLUMNS = ("as_of_date", "run_id")

log = logging.getLogger(__name__)


class PinotLoadError(RuntimeError):
    """The controller refused a request, or the loaded rows did not become queryable."""


def is_up(controller_url: str = DEFAULT_CONTROLLER, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(controller_url.rstrip("/") + "/health", timeout=timeout)
    except httpx.HTTPError:
        return False
    return r.status_code == 200


def segment_prefix(table: str, as_of_date: str) -> str:
    return f"{table}__{as_of_date}__"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def loaded_runs(sql_url: str, table: str, as_of_date: str, timeout: float = 10.0) -> dict:
    """``{run_id: row count}`` that Pinot serves for ``table`` on ``as_of_date``.

    ``sql_url`` is a broker's ``/query/sql`` or the controller's ``/sql`` endpoint. A table
    Pinot does not know yields ``{}``.
    """
    sql = (
        f"SELECT run_id, COUNT(*) FROM {table} WHERE as_of_date = {_quote(as_of_date)} "
        "GROUP BY run_id LIMIT 1000"
    )
    r = httpx.post(sql_url, json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if body.get("exceptions"):
        messages = "; ".join(str(e.get("message", e)) for e in body["exceptions"])
        if "TableDoesNotExist" in messages or "Unknown table" in messages:
            return {}
        raise PinotLoadError(f"Pinot query failed: {messages}")
    return {row[0]: int(row[1]) for row in body["resultTable"]["rows"]}


class _Controller:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=600.0)

    def _check(self, r: httpx.Response, what: str) -> httpx.Response:
        if r.status_code >= 400:
            raise PinotLoadError(f"{what}: HTTP {r.status_code}: {r.text}")
        return r

    def put_schema(self, schema: dict) -> None:
        r = self.http.post(f"{self.url}/schemas", params={"override": "true"}, json=schema)
        self._check(r, f"schema {schema['schemaName']}")

    def put_table(self, config: dict) -> None:
        name = config["tableName"]
        r = self.http.post(f"{self.url}/tables", json=config)
        if r.status_code == 409:  # exists: update in place
            r = self.http.put(f"{self.url}/tables/{name}", json=config)
        self._check(r, f"table {name}")

    def segments(self, table: str) -> list[str]:
        r = self._check(
            self.http.get(f"{self.url}/segments/{table}", params={"type": "OFFLINE"}),
            f"list segments of {table}",
        )
        return [s for entry in r.json() for s in entry.get("OFFLINE", [])]

    def ingest(self, table: str, path: Path, segment: str) -> None:
        batch = {
            "inputFormat": "parquet",
            "segmentNameGenerator.type": "fixed",
            "segmentNameGenerator.configs.segment.name": segment,
        }
        with path.open("rb") as f:
            r = self.http.post(
                f"{self.url}/ingestFromFile",
                params={
                    "tableNameWithType": f"{table}_OFFLINE",
                    "batchConfigMapStr": json.dumps(batch),
                },
                files={"file": (path.name, f, "application/octet-stream")},
            )
        self._check(r, f"ingest {segment}")

    def delete_segment(self, table: str, segment: str) -> None:
        r = self.http.delete(f"{self.url}/segments/{table}_OFFLINE/{segment}")
        self._check(r, f"delete {segment}")


def _run_rows(ctx: RunContext, model: SemanticModel, table: str) -> pa.Table:
    """The run's partition of ``table`` with the partition keys as columns, in model order."""
    data = ds.dataset(ctx.curated(table), format="parquet").to_table()
    n = data.num_rows
    keys = {"as_of_date": ctx.as_of_date.isoformat(), "run_id": ctx.run_id}
    columns = {}
    for name in model.tables[table]["columns"]:
        if name in data.column_names:
            columns[name] = data.column(name)
        elif name in keys:
            columns[name] = pa.array([keys[name]] * n, pa.string())
        else:
            raise PinotLoadError(f"{table}: column '{name}' is missing from {ctx.curated(table)}")
    return pa.table(columns)


def _wait(controller: str, table: str, as_of: str, expected: dict, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    seen: dict | str = {}
    while True:
        try:
            seen = loaded_runs(f"{controller}/sql", table, as_of)
            if seen == expected:
                return
        except (httpx.HTTPError, PinotLoadError) as e:
            seen = f"error: {e}"
        if time.monotonic() > deadline:
            raise PinotLoadError(
                f"{table} on {as_of}: Pinot did not serve {expected} within {timeout:.0f}s "
                f"(last saw {seen})"
            )
        time.sleep(1.0)


def load(
    ctx: RunContext,
    model: SemanticModel,
    controller_url: str = DEFAULT_CONTROLLER,
    timeout: float = 300.0,
) -> dict[str, int]:
    """Load ``ctx``'s curated tables into Pinot, replacing that as-of date. Returns row counts.

    ``ctx`` must be the official run of its as-of date.
    """
    if official_run(ctx.root, ctx.as_of_date) != ctx.run_id:
        raise PinotLoadError(
            f"run {ctx.run_id} is not the official run for {ctx.as_of_date}; refusing to load"
        )
    controller = _Controller(controller_url)
    as_of = ctx.as_of_date.isoformat()
    counts: dict[str, int] = {}
    for table in model.tables:
        controller.put_schema(pinot_schema(model, table))
        controller.put_table(pinot_table_config(model, table))
        rows = _run_rows(ctx, model, table)
        prefix = segment_prefix(table, as_of)
        names = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, start in enumerate(range(0, max(rows.num_rows, 1), SEGMENT_ROWS)):
                chunk = rows.slice(start, SEGMENT_ROWS)
                if chunk.num_rows == 0:
                    break
                name = f"{prefix}{ctx.run_id}__{i}"
                path = Path(tmp) / f"{name}.parquet"
                pq.write_table(chunk, path)
                controller.ingest(table, path, name)
                names.append(name)
        for stale in controller.segments(table):
            if stale.startswith(prefix) and stale not in names:
                controller.delete_segment(table, stale)
        expected = {ctx.run_id: rows.num_rows} if rows.num_rows else {}
        _wait(controller.url, table, as_of, expected, timeout)
        counts[table] = rows.num_rows
        log.info("loaded %s rows of %s for %s run %s", rows.num_rows, table, as_of, ctx.run_id)
    return counts
