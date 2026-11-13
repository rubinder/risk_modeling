"""Semantic query layer: turns (measure, group-by, filters) into a physical query.

The rule that matters: a vector measure (VaR, ES) is never aggregated by SQL. The query
sums P&L per scenario for each requested group; ``finish`` then applies the quantile
convention in Python, once, whichever executor produced the rows.
"""

import datetime
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from riskagg import conventions
from riskagg.semantic.model import Measure, SemanticModel, SpecError

Kind = Literal["additive", "vector_scenarios", "lookup"]

VECTOR_TABLE = "pnl_scenarios"  # per-scenario P&L sums, at the grain
MAX_ROWS = 10_000_000  # Pinot truncates to 10 rows unless a LIMIT is given
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED = {"window", "value"}  # reserved words in Pinot's SQL dialect: always double-quoted
_VECTOR_FUNCTIONS = {"var": conventions.var_from_pnl, "es": conventions.es_from_pnl}


class QueryError(SpecError):
    """The request names something the semantic model does not know, or is malformed."""


class SliceNotAvailable(QueryError):
    """The measure cannot be sliced that way without producing a wrong number."""


@dataclass(frozen=True)
class Query:
    """A physical query over one table.

    ``filters`` maps a column to its allowed values (rendered as ``IN``); it carries the
    as-of date, run id and any fixed filter (window, measure, node_level) as well as the
    user's slice. ``aggregate`` is ``(function, column)`` or None for a plain lookup.
    """

    table: str
    select_dims: tuple[str, ...]
    aggregate: tuple[str, str] | None
    filters: Mapping[str, Sequence[str]]
    kind: Kind
    post: Measure | None

    @property
    def value_column(self) -> str:
        return "pnl" if self.kind == "vector_scenarios" else "value"

    @property
    def columns(self) -> list[str]:
        """Columns of the rows an executor returns, in order."""
        extra = ["scenario_idx"] if self.kind == "vector_scenarios" else []
        return [*self.select_dims, *extra, self.value_column]

    def to_pinot_sql(self) -> str:
        group = [*self.select_dims, *(["scenario_idx"] if self.kind == "vector_scenarios" else [])]
        if self.aggregate:
            func, col = self.aggregate
            if func.upper() != "SUM":
                raise QueryError(f"unsupported aggregate function '{func}'")
            measure_sql = f"SUM({_ident(col)}) AS {_ident(self.value_column, quote=True)}"
        else:
            measure_sql = _ident("value", quote=True)
        select = ", ".join([*(_ident(d) for d in group), measure_sql])
        where = " AND ".join(
            f"{_ident(col)} IN ({', '.join(_literal(v) for v in values)})"
            for col, values in self.filters.items()
        )
        sql = f"SELECT {select} FROM {_ident(self.table)} WHERE {where}"
        if self.aggregate and group:
            sql += " GROUP BY " + ", ".join(_ident(d) for d in group)
        return f"{sql} LIMIT {MAX_ROWS}"


def _ident(name: str, quote: bool = False) -> str:
    if not _IDENT.fullmatch(name):
        raise QueryError(f"refusing to render invalid identifier {name!r}")
    return f'"{name}"' if name in _RESERVED else name


def _literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _check_filters(model: SemanticModel, filters: Mapping[str, Sequence[str]]) -> None:
    for dim, values in filters.items():
        _known(model, dim)
        if isinstance(values, str) or not all(isinstance(v, str) for v in values) or not values:
            raise QueryError(f"filter '{dim}' must be a non-empty list of strings, got {values!r}")


def _known(model: SemanticModel, dim: str) -> None:
    if dim not in model.dimensions:
        raise QueryError(f"unknown dimension '{dim}'; known: {sorted(model.dimensions)}")


def _require_valid(measure: Measure, valid: Sequence[str], used: Sequence[str], why: str) -> None:
    for dim in used:
        if dim not in valid:
            raise SliceNotAvailable(f"{measure.name} cannot be sliced by '{dim}': {why}")


def build_query(
    model: SemanticModel,
    measure: str,
    group_by: Sequence[str],
    filters: Mapping[str, Sequence[str]],
    as_of_date: datetime.date | str,
    run_id: str,
) -> Query:
    m = model.measure(measure)
    group_by = tuple(group_by)
    if len(set(group_by)) != len(group_by):
        raise QueryError(f"duplicate dimension in group_by {list(group_by)}")
    for dim in group_by:
        _known(model, dim)
    _check_filters(model, filters)
    used = [*group_by, *filters]
    pinned = {
        "as_of_date": [as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else as_of_date],
        "run_id": [run_id],
    }
    user = {d: list(v) for d, v in filters.items()}

    if m.measure_class == "additive":
        _require_valid(m, m.dimensions, used, f"it is additive over {list(m.dimensions)} only")
        metric = _metric_column(model, m)
        return Query(m.table, group_by, ("SUM", metric), {**pinned, **user}, "additive", None)

    if m.measure_class == "vector":
        _require_valid(
            m,
            model.grain,
            used,
            "the measure is non-additive and P&L vectors are not stored at that grain "
            f"(available: {list(model.grain)})",
        )
        fixed = {**pinned, "window": [m.params["window"]], **user}
        return Query(VECTOR_TABLE, group_by, ("SUM", "pnl"), fixed, "vector_scenarios", m)

    # structured: only ever looked up at a level the spec precomputed
    wanted = frozenset(group_by)
    level = next((c for c in m.computed_at if frozenset(c) == wanted), None)
    if level is None:
        raise SliceNotAvailable(
            f"{m.name} is only precomputed at {[list(c) for c in m.computed_at]}; "
            f"group_by {list(group_by)} is not one of them and a structured measure cannot be "
            "rebuilt from other slices"
        )
    _require_valid(m, group_by, filters, "a lookup can only filter on the grouped dimensions")
    fixed = {**pinned, "measure": [m.name], "node_level": ["|".join(level)], **user}
    return Query(m.table, tuple(level), None, fixed, "lookup", m)


def _metric_column(model: SemanticModel, m: Measure) -> str:
    metrics = [c for c, s in model.tables[m.table]["columns"].items() if s.get("role") == "metric"]
    if len(metrics) != 1:
        raise QueryError(f"table '{m.table}' must have exactly one metric column, has {metrics}")
    return metrics[0]


def finish(query: Query, rows: pd.DataFrame) -> pd.DataFrame:
    """The final answer: dims plus ``value``.

    For ``vector_scenarios`` the per-scenario sums of each group go through the quantile
    convention here (never in SQL) and are scaled by sqrt(horizon_days).
    """
    out_cols = [*query.select_dims, "value"]
    if query.kind != "vector_scenarios":
        return rows[out_cols].reset_index(drop=True)
    measure = query.post
    fn = _VECTOR_FUNCTIONS[measure.function]
    level = float(measure.params["level"])
    # SIMPLIFICATION: multi-day horizons scale the one-day figure by sqrt(horizon_days)
    # instead of using overlapping multi-day P&L.
    scale = math.sqrt(float(measure.params.get("horizon_days", 1)))
    if rows.empty:
        return pd.DataFrame(columns=out_cols)
    dims = list(query.select_dims)
    groups = rows.groupby(dims, sort=True) if dims else [((), rows)]
    records = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        pnl = g.sort_values("scenario_idx")["pnl"].to_numpy(dtype=float)
        records.append((*key, fn(pnl, level) * scale))
    return pd.DataFrame.from_records(records, columns=out_cols)
