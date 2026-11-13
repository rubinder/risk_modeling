"""The data dictionary: the semantic model rendered as Markdown for people.

Per measure: business name, description, unit, aggregation class, the slices it may be
queried at and the rule for combining it. Per dimension: where it can be used. Per table: what
a row is, how its columns may be combined, and its columns. Everything comes from
``config/semantic_model.yaml``; ``riskagg dictionary`` writes ``docs/data-dictionary.md``.
"""

from pathlib import Path

from riskagg.semantic.model import Measure, SemanticModel

HEADER = """# Data dictionary

Generated from `config/semantic_model.yaml` by `riskagg dictionary`. Do not edit by hand.

Every measure has one of three aggregation classes, which decide how it may be combined:

- **additive**: values sum over any of the measure's dimensions.
- **vector**: a VaR or ES. Never summed. P&L is summed scenario by scenario over the requested
  slice, and the quantile convention (`src/riskagg/conventions.py`) is applied once to each
  group's vector.
- **structured**: a capital charge, ratio or worst case recomputed at each node from that
  node's own inputs. Never summed, averaged or rebuilt from other slices; only looked up at the
  levels where it was computed.

A query for a slice the class does not allow is refused with the reason (HTTP 422). In
`risk_metrics`, a dimension that a node does not group by holds the literal `*`, never null.
"""


_NAMES = {"var": "VaR", "es": "ES"}


def _dims(dims) -> str:
    return ", ".join(f"`{d}`" for d in dims)


def _slices(model: SemanticModel, m: Measure) -> str:
    if m.measure_class == "structured":
        levels = "; ".join(f"({_dims(level)})" for level in m.computed_at)
        return f"exactly one of: {levels}"
    dims = model.grain if m.measure_class == "vector" else m.dimensions
    return f"any subset of: {_dims(dims)}"


def _rule(m: Measure) -> str:
    if m.measure_class == "additive":
        rule = "Additive: sum the values over any of its dimensions."
        if "scenario" in m.dimensions:
            rule += " Values under different scenarios are different quantities: slice by "
            rule += "`scenario`, then sum over anything else."
        return rule
    if m.measure_class == "vector":
        rule = (
            "Never summed. The P&L of the slice is summed scenario by scenario from "
            f"`pnl_scenarios` ({m.params.get('window')} window), then the {_NAMES[m.function]} "
            f"at {m.params.get('level')} is taken of each group's vector "
            "(`src/riskagg/conventions.py`)."
        )
        if float(m.params.get("horizon_days", 1)) != 1:
            rule += f" Scaled by sqrt({m.params['horizon_days']}) to the horizon."
        return rule
    rule = (
        "Never summed, averaged or rebuilt from other slices: only looked up at a level where "
        "it was computed."
    )
    if m.not_allocated_to:
        rule += f" Not allocated to {_dims(m.not_allocated_to)}: a query naming one is refused."
    if m.carries_scenario:
        rule += " Each row names its scenario in the `scenario` column."
    return rule


def _row(label: str, value: str) -> str:
    return f"| {label} | {' '.join(str(value).split())} |"


def _measure(model: SemanticModel, m: Measure) -> list[str]:
    rows = [
        _row("Business name", m.label),
        _row("Description", m.description),
        _row("Unit", m.unit),
        _row("Aggregation class", m.measure_class),
    ]
    if m.params:
        rows.append(_row("Parameters", ", ".join(f"{k} {v}" for k, v in m.params.items())))
    if m.measure_class == "vector":
        source = (
            f"queries: `{model.vector_table}`; aggregations declared in "
            f"`config/aggregations.yaml` also store rows in `{m.table}`"
        )
    else:
        source = f"`{m.table}`"
    rows += [
        _row("Answered from", source),
        _row("May be queried by", _slices(model, m)),
        _row("Rule for combining", _rule(m)),
    ]
    return [f"### `{m.name}`", "", "| | |", "|---|---|", *rows, ""]


def _dimension(model: SemanticModel, name: str, spec: dict) -> list[str]:
    hierarchies = [
        f"{h} level {levels.index(name) + 1} of {len(levels)}"
        for h, levels in model.hierarchies.items()
        if name in levels
    ]
    usable = []
    for m in model.measures.values():
        if m.measure_class == "vector":
            ok = name in model.grain
        elif m.measure_class == "additive":
            ok = name in m.dimensions
        else:
            ok = any(name in level for level in m.computed_at)
        if ok:
            usable.append(m.name)
    rows = [
        _row("Business name", spec.get("label", name)),
        _row("Description", spec.get("description", "")),
        _row("In the reporting grain", "yes" if name in model.grain else "no"),
        _row("Hierarchy", "; ".join(hierarchies) or "none"),
        _row("Measures with a slice that includes it", _dims(usable) or "none"),
    ]
    return [f"### `{name}`", "", "| | |", "|---|---|", *rows, ""]


def _table(model: SemanticModel, name: str, spec: dict) -> list[str]:
    stored = [m.name for m in model.measures.values() if m.table == name]
    vectors = [m.name for m in model.measures.values() if m.measure_class == "vector"]
    lines = [f"### `{name}`", "", " ".join(spec.get("description", "").split()), ""]
    if stored:
        lines += [f"Measures stored in it: {_dims(stored)}.", ""]
    if name == model.vector_table and vectors:
        lines += [f"Every query of {_dims(vectors)} is answered from it.", ""]
    if not stored and name != model.vector_table:
        lines += ["No measure is stored in it: read it with the drill-down.", ""]
    lines += ["| Column | Type | Role | Nullable |", "|---|---|---|---|"]
    for col, c in spec["columns"].items():
        nullable = "yes" if c.get("nullable") else "no"
        lines.append(f"| `{col}` | {c['type']} | {c['role']} | {nullable} |")
    return [*lines, ""]


def render_dictionary(model: SemanticModel) -> str:
    lines = [HEADER, "## Measures", ""]
    for m in model.measures.values():
        lines += _measure(model, m)
    lines += ["## Dimensions", ""]
    lines += [f"Reporting grain: {_dims(model.grain)}.", ""]
    for name, spec in model.dimensions.items():
        lines += _dimension(model, name, spec)
    lines += ["## Tables", ""]
    for name, spec in model.tables.items():
        lines += _table(model, name, spec)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_dictionary(model: SemanticModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dictionary(model))
    return path
