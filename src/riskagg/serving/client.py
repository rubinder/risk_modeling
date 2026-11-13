"""Python client: the HTTP API when ``base_url`` is given, otherwise in-process.

In-process it uses ``RiskService``, so the source rule is the API's: Pinot when it is up and
serving the official run, otherwise the curated Parquet files (with a logged notice). After
each call ``.source`` says which one answered, and ``.as_of_date``/``.run_id`` which run.
"""

import datetime
from collections.abc import Sequence
from pathlib import Path

import httpx
import pandas as pd

from riskagg.context import DEFAULT_ROOT
from riskagg.semantic.query import QueryError, SliceNotAvailable
from riskagg.serving import pinot
from riskagg.serving.service import DEFAULT_DRILLDOWN_LIMIT, Answer, NotFound, RiskService

Values = str | Sequence[str]


def _list(values: Values) -> list[str]:
    return [values] if isinstance(values, str) else list(values)


class RiskClient:
    def __init__(
        self,
        base_url: str | None = None,
        root: str | Path = DEFAULT_ROOT,
        broker_url: str = pinot.DEFAULT_BROKER,
        controller_url: str = pinot.DEFAULT_CONTROLLER,
        http: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.http = http or (httpx.Client(timeout=120.0) if base_url else None)
        self.service = (
            None if base_url else RiskService(root, None, broker_url, controller_url)
        )
        self.source: str | None = None
        self.as_of_date: datetime.date | None = None
        self.run_id: str | None = None

    def catalog(self) -> dict:
        if self.service:
            return self.service.catalog()
        return self._get("/catalog", {})

    def metric(
        self,
        measure: str,
        group_by: Values = (),
        as_of: datetime.date | str | None = None,
        **filters: Values,
    ) -> pd.DataFrame:
        """Group-by dimensions plus ``value``; filters are ``dimension=value(s)``."""
        group_by = _list(group_by)
        filters = {k: _list(v) for k, v in filters.items()}
        if self.service:
            return self._keep(self.service.metric(measure, group_by, filters, as_of))
        params = self._params(filters, as_of)
        if group_by:
            params["group_by"] = ",".join(group_by)
        body = self._get(f"/metrics/{measure}", params)
        return self._keep_body(body, [*group_by, "value"])

    def drilldown(
        self,
        table: str,
        as_of: datetime.date | str | None = None,
        limit: int = DEFAULT_DRILLDOWN_LIMIT,
        **filters: Values,
    ) -> pd.DataFrame:
        filters = {k: _list(v) for k, v in filters.items()}
        if self.service:
            return self._keep(self.service.drilldown(table, filters, as_of, limit))
        params = {**self._params(filters, as_of), "limit": limit}
        return self._keep_body(self._get(f"/drilldown/{table}", params), None)

    def _keep(self, answer: Answer) -> pd.DataFrame:
        self.source, self.as_of_date, self.run_id = answer.source, answer.as_of_date, answer.run_id
        return answer.rows

    def _keep_body(self, body: dict, columns: list[str] | None) -> pd.DataFrame:
        self.source = body["source"]
        self.as_of_date = datetime.date.fromisoformat(body["as_of_date"])
        self.run_id = body["run_id"]
        if not body["rows"] and columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame.from_records(body["rows"], columns=columns)

    @staticmethod
    def _params(filters: dict[str, list[str]], as_of) -> dict:
        params: dict = {k: ",".join(v) for k, v in filters.items()}
        if as_of is not None:
            params["as_of"] = as_of.isoformat() if hasattr(as_of, "isoformat") else as_of
        return params

    def _get(self, path: str, params: dict) -> dict:
        r = self.http.get(self.base_url + path, params=params)
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        errors = {404: NotFound, 422: SliceNotAvailable, 400: QueryError}
        if r.status_code in errors:
            raise errors[r.status_code](detail)
        r.raise_for_status()
        raise RuntimeError(f"unexpected HTTP {r.status_code}: {r.text}")
