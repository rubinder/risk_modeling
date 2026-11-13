"""HTTP API over ``RiskService``: catalogue, metric query and drill-down.

Query parameters other than ``group_by``, ``as_of`` and ``limit`` are filters; a filter may be
repeated or comma-separated (``desk_id=DESK001,DESK002``).
Drill-down returns at most ``limit`` raw rows in no particular order.

Errors: unknown measure or table, or no official run -> 404; a slice the measure cannot
answer -> 422 with the reason; any other malformed request -> 400.
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from riskagg.context import DEFAULT_ROOT
from riskagg.semantic.model import SemanticModel, SpecError
from riskagg.semantic.query import SliceNotAvailable
from riskagg.serving import pinot
from riskagg.serving.service import DEFAULT_DRILLDOWN_LIMIT, Answer, NotFound, RiskService

RESERVED_PARAMS = {"group_by", "as_of", "limit"}


def _split(raw: str | None) -> list[str]:
    return [v for v in (raw or "").split(",") if v]


def _filters(request: Request) -> dict[str, list[str]]:
    params = request.query_params
    return {
        k: [v for raw in params.getlist(k) for v in _split(raw)]
        for k in params
        if k not in RESERVED_PARAMS
    }


def records(df: pd.DataFrame) -> list[dict]:
    """Rows as JSON-ready dicts with full float precision (Python scalars, NaN as null)."""
    plain = df.astype(object).where(df.notna(), None)
    return plain.to_dict(orient="records")


def _body(answer: Answer, **head) -> dict:
    return {
        **head,
        "as_of_date": answer.as_of_date.isoformat(),
        "run_id": answer.run_id,
        "source": answer.source,
        "rows": records(answer.rows),
    }


def create_app(
    root: str | Path = DEFAULT_ROOT,
    model: SemanticModel | None = None,
    broker_url: str = pinot.DEFAULT_BROKER,
    controller_url: str = pinot.DEFAULT_CONTROLLER,
) -> FastAPI:
    service = RiskService(root, model, broker_url, controller_url)
    app = FastAPI(title="riskagg", description="Risk metrics from the official run")

    @app.exception_handler(NotFound)
    def _not_found(request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(SliceNotAvailable)
    def _refused(request: Request, exc: SliceNotAvailable) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(SpecError)
    def _bad_request(request: Request, exc: SpecError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/catalog")
    def get_catalog() -> dict:
        return service.catalog()

    @app.get("/metrics/{measure}")
    def get_metric(
        measure: str, request: Request, group_by: str | None = None, as_of: str | None = None
    ) -> dict:
        answer = service.metric(measure, _split(group_by), _filters(request), as_of)
        return _body(answer, measure=measure)

    @app.get("/drilldown/{table}")
    def get_drilldown(
        table: str,
        request: Request,
        as_of: str | None = None,
        limit: int = DEFAULT_DRILLDOWN_LIMIT,
    ) -> dict:
        answer = service.drilldown(table, _filters(request), as_of, limit)
        return _body(answer, table=table)

    return app
