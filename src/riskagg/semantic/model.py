"""Semantic model: dimensions, hierarchies, grain, measures and physical tables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from riskagg.paths import semantic_model_path

MEASURE_CLASSES = ("additive", "vector", "structured")
DEFAULT_VECTOR_TABLE = "pnl_scenarios"


class SpecError(ValueError):
    """Invalid semantic model or aggregation specification."""


@dataclass(frozen=True)
class Measure:
    name: str
    label: str
    description: str
    unit: str
    measure_class: str
    function: str | None
    params: dict[str, Any]
    table: str
    dimensions: tuple[str, ...]
    computed_at: tuple[tuple[str, ...], ...] = ()
    # A structured measure whose stored rows name a scenario; lookups return it beside the value.
    carries_scenario: bool = False
    # Dimensions the measure is never sliced by because its quantity (capital) is not allocated
    # to them; a query naming one is refused with that reason.
    not_allocated_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticModel:
    dimensions: dict[str, dict[str, Any]]
    hierarchies: dict[str, tuple[str, ...]]
    grain: tuple[str, ...]
    measures: dict[str, Measure]
    tables: dict[str, dict[str, Any]]
    # The table of per-scenario P&L at the grain that vector measures are answered from.
    vector_table: str = DEFAULT_VECTOR_TABLE

    def measure(self, name: str) -> Measure:
        try:
            return self.measures[name]
        except KeyError:
            raise SpecError(f"unknown measure '{name}'") from None


def load_model(path: str | Path | None = None) -> SemanticModel:
    path = semantic_model_path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    dimensions = dict(raw.get("dimensions") or {})
    hierarchies = {k: tuple(v) for k, v in (raw.get("hierarchies") or {}).items()}
    grain = tuple(raw.get("grain") or ())
    for d in grain:
        if d not in dimensions:
            raise SpecError(f"grain: unknown dimension '{d}'")
    for h, levels in hierarchies.items():
        for d in levels:
            if d not in dimensions:
                raise SpecError(f"hierarchy '{h}': unknown dimension '{d}'")
    tables = dict(raw.get("tables") or {})
    measures: dict[str, Measure] = {}
    for name, m in (raw.get("measures") or {}).items():
        cls = m.get("class")
        if cls not in MEASURE_CLASSES:
            raise SpecError(f"measure '{name}': class must be one of {MEASURE_CLASSES}")
        dims = tuple(m.get("dimensions") or grain)
        for d in dims:
            if d not in dimensions:
                raise SpecError(f"measure '{name}': unknown dimension '{d}'")
        computed_at = tuple(tuple(level) for level in m.get("computed_at") or ())
        if cls == "structured" and not computed_at:
            raise SpecError(f"measure '{name}': structured measure needs computed_at")
        if cls != "structured" and computed_at:
            raise SpecError(f"measure '{name}': computed_at only applies to structured measures")
        if m.get("carries_scenario") and cls != "structured":
            raise SpecError(f"measure '{name}': carries_scenario only applies to structured measures")
        not_allocated = tuple(m.get("not_allocated_to") or ())
        for d in not_allocated:
            if d not in dimensions:
                raise SpecError(f"measure '{name}': not_allocated_to: unknown dimension '{d}'")
        if not_allocated and cls != "structured":
            raise SpecError(
                f"measure '{name}': not_allocated_to only applies to structured measures"
            )
        for level in computed_at:
            if set(level) & set(not_allocated):
                raise SpecError(
                    f"measure '{name}': computed_at {list(level)} uses a dimension it is "
                    "not allocated to"
                )
        table = m.get("table")
        if table not in tables:
            raise SpecError(f"measure '{name}': unknown table '{table}'")
        measures[name] = Measure(
            name=name,
            label=m.get("label", name),
            description=m.get("description", ""),
            unit=m.get("unit", ""),
            measure_class=cls,
            function=m.get("function"),
            params=dict(m.get("params") or {}),
            table=table,
            dimensions=dims,
            computed_at=computed_at,
            carries_scenario=bool(m.get("carries_scenario", False)),
            not_allocated_to=not_allocated,
        )
    vector_table = raw.get("vector_table", DEFAULT_VECTOR_TABLE)
    if any(m.measure_class == "vector" for m in measures.values()) and vector_table not in tables:
        raise SpecError(f"vector_table '{vector_table}' is not one of the tables {sorted(tables)}")
    return SemanticModel(dimensions, hierarchies, grain, measures, tables, vector_table)


def derived_rows(model: SemanticModel) -> dict[str, str]:
    """``{suffix: description}`` of the ``<measure>__<suffix>`` rows a table declares under
    ``derived_rows`` (``risk_metrics``: diversification rows of vector measures)."""
    return {
        suffix: " ".join(str(text).split())
        for spec in model.tables.values()
        for suffix, text in (spec.get("derived_rows") or {}).items()
    }
