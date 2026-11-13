"""Answering metric and drill-down requests: which run, and which backend.

Shared by the HTTP API and the in-process ``RiskClient``, so both pick the same source:

* the as-of date defaults to the latest one with an official run; the run is always that
  date's official run (``runs/official.json``);
* Pinot answers when its controller is up and the broker serves exactly that run for the
  date in the table the request reads, with as many rows as the run's Parquet partition;
  otherwise the curated Parquet files answer;
* a Pinot failure mid-request (error, incomplete result, connection) falls back to Parquet.
"""

import datetime
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.dataset as ds

from riskagg.context import DEFAULT_ROOT, official_runs
from riskagg.engine.spec import AggregationSpec, load_specs, stored_by
from riskagg.semantic.model import SemanticModel, derived_rows, load_model
from riskagg.semantic.query import (
    Drilldown,
    Query,
    QueryError,
    build_drilldown,
    build_query,
    finish,
)
from riskagg.serving import pinot
from riskagg.serving.executors import PandasExecutor, PinotError, PinotExecutor

log = logging.getLogger(__name__)

DEFAULT_DRILLDOWN_LIMIT = 1000


class NotFound(LookupError):
    """No such measure or table, or no official run for the requested date."""


@dataclass(frozen=True)
class Answer:
    rows: pd.DataFrame
    as_of_date: datetime.date
    run_id: str
    source: str  # "pinot" or "parquet"


def resolve_run(root: str | Path, as_of: datetime.date | str | None) -> tuple[datetime.date, str]:
    """``(as_of_date, official run_id)``; the latest official date when ``as_of`` is None."""
    runs = official_runs(root)
    if as_of is None:
        if not runs:
            raise NotFound(f"no official run under {root}")
        date = max(runs)
        return date, runs[date]
    if isinstance(as_of, str):
        try:
            as_of = datetime.date.fromisoformat(as_of)
        except ValueError:
            raise QueryError(f"invalid as-of date '{as_of}', expected YYYY-MM-DD") from None
    if as_of not in runs:
        raise NotFound(f"no official run for as-of date {as_of.isoformat()} under {root}")
    return as_of, runs[as_of]


def catalog(model: SemanticModel, specs: Sequence[AggregationSpec] | None = None) -> dict:
    """Measures with the slices each can answer and where each is precomputed (the shipped
    specifications unless ``specs`` is given), the derived rows stored beside a vector
    measure, dimensions, hierarchies and tables."""
    specs = load_specs(None, model) if specs is None else specs

    def slices(m) -> dict:
        if m.measure_class == "structured":
            return {"one_of": [list(level) for level in m.computed_at]}
        dims = model.grain if m.measure_class == "vector" else m.dimensions
        return {"any_subset_of": list(dims)}

    return {
        "measures": {
            name: {
                "label": m.label,
                "description": m.description,
                "unit": m.unit,
                "class": m.measure_class,
                "params": m.params,
                "group_by": slices(m),
                "stored_in_risk_metrics": stored_by(specs, name),
            }
            for name, m in model.measures.items()
        },
        "derived_rows": derived_rows(model),
        "dimensions": model.dimensions,
        "hierarchies": {k: list(v) for k, v in model.hierarchies.items()},
        "tables": sorted(model.tables),
    }


class RiskService:
    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        model: SemanticModel | None = None,
        broker_url: str = pinot.DEFAULT_BROKER,
        controller_url: str = pinot.DEFAULT_CONTROLLER,
    ):
        self.root = Path(root)
        self.model = model or load_model()
        self.broker_url = broker_url.rstrip("/")
        self.controller_url = controller_url
        self.parquet = PandasExecutor(self.root)
        self.pinot = PinotExecutor(self.broker_url)

    def catalog(self) -> dict:
        return catalog(self.model)

    def metric(
        self,
        measure: str,
        group_by: Sequence[str] = (),
        filters: Mapping[str, Sequence[str]] | None = None,
        as_of: datetime.date | str | None = None,
    ) -> Answer:
        if measure not in self.model.measures:
            raise NotFound(f"unknown measure '{measure}'; known: {sorted(self.model.measures)}")
        date, run_id = resolve_run(self.root, as_of)
        query = build_query(self.model, measure, group_by, filters or {}, date, run_id)
        rows, source = self._run(query, date, run_id)
        return Answer(finish(query, rows), date, run_id, source)

    def drilldown(
        self,
        table: str,
        filters: Mapping[str, Sequence[str]] | None = None,
        as_of: datetime.date | str | None = None,
        limit: int = DEFAULT_DRILLDOWN_LIMIT,
    ) -> Answer:
        if table not in self.model.tables:
            raise NotFound(f"unknown table '{table}'; known: {sorted(self.model.tables)}")
        date, run_id = resolve_run(self.root, as_of)
        drill = build_drilldown(self.model, table, filters or {}, date, run_id, limit)
        rows, source = self._run(drill, date, run_id)
        return Answer(rows, date, run_id, source)

    def run_rows(self, table: str, as_of_date: datetime.date, run_id: str) -> int:
        """Rows in the run's curated Parquet partition of ``table`` (0 when there is none)."""
        path = (
            self.root / "curated" / table / f"as_of_date={as_of_date.isoformat()}"
            / f"run_id={run_id}"
        )  # fmt: skip
        return ds.dataset(path, format="parquet").count_rows() if path.is_dir() else 0

    def pinot_serves(self, table: str, as_of_date: datetime.date, run_id: str) -> bool:
        """Pinot is up and holds, for the date in ``table``, exactly ``run_id``'s rows.

        The row count matters: during a load, or after a failed one, Pinot can hold only this
        run but not all of it. VaR over a subset of the grain rows still has every scenario
        per group, so nothing downstream would notice the smaller sums.
        """
        if not pinot.is_up(self.controller_url):
            return False
        try:
            loaded = pinot.loaded_runs(
                f"{self.broker_url}/query/sql", table, as_of_date.isoformat()
            )
        except (httpx.HTTPError, pinot.PinotLoadError) as e:
            log.warning("Pinot broker at %s is not answering (%s)", self.broker_url, e)
            return False
        expected = self.run_rows(table, as_of_date, run_id)
        if expected and loaded == {run_id: expected}:
            return True
        if set(loaded) == {run_id}:
            log.warning(
                "Pinot holds %s of %s rows of %s for %s run %s (load incomplete)",
                loaded[run_id], expected, table, as_of_date, run_id,
            )  # fmt: skip
        return False

    def _run(
        self, request: Query | Drilldown, date: datetime.date, run_id: str
    ) -> tuple[pd.DataFrame, str]:
        def parquet() -> pd.DataFrame:
            if isinstance(request, Query):
                return self.parquet.execute(request)
            return self.parquet.fetch(request)

        if not self.pinot_serves(request.table, date, run_id):
            log.warning(
                "Pinot is not serving %s for %s run %s; answering from Parquet",
                request.table,
                date,
                run_id,
            )
            return parquet(), "parquet"
        try:
            if isinstance(request, Query):
                return self.pinot.execute(request), "pinot"
            return self.pinot.fetch(request), "pinot"
        except (PinotError, httpx.HTTPError) as e:
            log.warning("Pinot query failed (%s); answering from Parquet", e)
            return parquet(), "parquet"
