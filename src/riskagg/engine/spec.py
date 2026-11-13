"""Aggregation specifications: loading, validation against the semantic model, hashing."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from riskagg.paths import aggregations_path
from riskagg.semantic.model import SemanticModel, SpecError

__all__ = [
    "DERIVED_SUFFIXES",
    "AggregationSpec",
    "SpecError",
    "load_specs",
    "spec_hash",
    "stored_by",
]

# Rows a vector spec with ``diversification: true`` stores as ``<measure>__<suffix>``
# (``measures/var.py``); described under ``derived_rows`` of ``risk_metrics`` in the model.
DERIVED_SUFFIXES = ("sum_children", "diversification", "incremental")


@dataclass(frozen=True)
class AggregationSpec:
    name: str
    measure: str
    grouping_sets: tuple[tuple[str, ...], ...]
    filters: dict[str, tuple[str, ...]]
    diversification: bool


def _validate(spec: AggregationSpec, model: SemanticModel) -> None:
    def fail(msg: str) -> SpecError:
        return SpecError(f"spec '{spec.name}': {msg}")

    measure = model.measure(spec.measure) if spec.measure in model.measures else None
    if measure is None:
        raise fail(f"unknown measure '{spec.measure}'")
    if not spec.grouping_sets:
        raise fail("no grouping_sets")
    allowed = set(measure.dimensions)
    cls = measure.measure_class

    def check_dim(d: str) -> None:
        if d not in model.dimensions:
            raise fail(f"unknown dimension '{d}'")
        if d not in allowed:
            raise fail(f"dimension '{d}' is not available for {cls} measure '{measure.name}'")

    for gs in spec.grouping_sets:
        for d in gs:
            check_dim(d)
        if cls == "structured" and tuple(gs) not in {tuple(c) for c in measure.computed_at}:
            raise fail(
                f"structured measure '{measure.name}' is computed only at "
                f"{[list(c) for c in measure.computed_at]}, not {list(gs)}"
            )
    for d in spec.filters:
        check_dim(d)
    if spec.diversification:
        org = model.hierarchies.get("org", ())
        for gs in spec.grouping_sets:
            if tuple(gs) != org[: len(gs)] or not gs:
                raise fail(
                    f"diversification requires org hierarchy sets; {list(gs)} is not a "
                    f"prefix of {list(org)}"
                )
        if not any(len(gs) >= 2 for gs in spec.grouping_sets):
            raise fail("diversification requires at least two consecutive org levels")


def load_specs(path: str | Path | None, model: SemanticModel) -> list[AggregationSpec]:
    raw = yaml.safe_load(aggregations_path(path).read_text()) or []
    specs: list[AggregationSpec] = []
    seen: set[str] = set()
    for r in raw:
        name = r["name"]
        if name in seen:
            raise SpecError(f"spec '{name}': duplicate name")
        seen.add(name)
        filters = r.get("filters") or {}
        for d, vs in filters.items():
            if not isinstance(vs, list) or not all(isinstance(v, str) for v in vs):
                raise SpecError(f"spec '{name}': filter '{d}' must be a list of strings")
        spec = AggregationSpec(
            name=name,
            measure=r["measure"],
            grouping_sets=tuple(tuple(gs) for gs in r.get("grouping_sets") or ()),
            filters={d: tuple(vs) for d, vs in filters.items()},
            diversification=bool(r.get("diversification", False)),
        )
        _validate(spec, model)
        specs.append(spec)
    return specs


def stored_by(specs: Iterable[AggregationSpec], measure: str) -> list[dict]:
    """Where ``measure`` is precomputed: per spec, its node levels and derived measures."""
    out = []
    for s in specs:
        if s.measure != measure:
            continue
        seen: set[frozenset[str]] = set()
        levels = []
        for gs in s.grouping_sets:  # one node level per set of dimensions, first order wins
            if frozenset(gs) not in seen:
                seen.add(frozenset(gs))
                levels.append("|".join(gs))
        derived = [f"{measure}__{d}" for d in DERIVED_SUFFIXES] if s.diversification else []
        out.append({"aggregation": s.name, "node_levels": levels, "derived_measures": derived})
    return out


def spec_hash(specs: Iterable[AggregationSpec], reference_paths: Iterable[str | Path] = ()) -> str:
    """sha256 over canonical spec content and reference file bytes; first 12 hex chars."""
    h = hashlib.sha256()
    canonical = [asdict(s) for s in sorted(specs, key=lambda s: s.name)]
    h.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())
    for p in sorted(Path(p) for p in reference_paths):
        h.update(b"\0" + p.name.encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()[:12]
