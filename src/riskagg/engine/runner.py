"""run_all: every specification from one shared pass per filter set.

Vector specs with the same filters are answered from one ``aggregate_vectors`` call over the
union of their grouping sets: one scan of the grain, one grouping-sets aggregation. Other
measure classes are dispatched through ``MEASURE_REGISTRY``.
"""

from collections.abc import Callable, Sequence

from pyspark.sql import DataFrame

from riskagg.engine.spec import AggregationSpec
from riskagg.engine.vectors import METRIC_COLUMNS, aggregate_vectors, node_level, unique_sets
from riskagg.measures import var
from riskagg.semantic.model import SemanticModel, SpecError

MeasureFn = Callable[[AggregationSpec, SemanticModel, DataFrame], DataFrame]

# Measure name -> fn(spec, model, grain) returning rows in METRIC_COLUMNS. Later tasks register
# additive and structured measures here; vector measures are handled by the engine itself.
MEASURE_REGISTRY: dict[str, MeasureFn] = {}

FilterKey = tuple[tuple[str, tuple[str, ...]], ...]


def _filter_key(spec: AggregationSpec) -> FilterKey:
    return tuple(sorted((d, tuple(sorted(vs))) for d, vs in spec.filters.items()))


def _is_vector(spec: AggregationSpec, model: SemanticModel) -> bool:
    return model.measure(spec.measure).measure_class == "vector"


def node_vectors(
    specs: Sequence[AggregationSpec], model: SemanticModel, grain: DataFrame, n_scenarios: int
) -> DataFrame:
    """Node vectors for vector specs sharing one filter set: a single grouping-sets pass.

    ``node_level`` follows the first spec that declares each set of dimensions.
    """
    if not specs:
        raise ValueError("node_vectors: no specs")
    keys = {_filter_key(s) for s in specs}
    if len(keys) != 1:
        raise ValueError("node_vectors: specs must share the same filters")
    for s in specs:
        if not _is_vector(s, model):
            raise ValueError(f"node_vectors: spec '{s.name}' is not a vector measure")
    sets = [gs for s in specs for gs in s.grouping_sets]
    return aggregate_vectors(grain, sets, n_scenarios, dict(specs[0].filters))


def _vector_group_metrics(
    group: Sequence[AggregationSpec], model: SemanticModel, nodes: DataFrame
) -> DataFrame:
    """Every metric of ``group`` derived from its shared node frame in one linear pass."""
    canonical = unique_sets([gs for s in group for gs in s.grouping_sets])
    plan = var.MetricPlan(labels={frozenset(gs): node_level(gs) for gs in canonical})
    for spec in group:
        measure = model.measure(spec.measure)
        var.plan_metrics(plan, spec, measure)
        if spec.diversification:
            var.plan_diversification(plan, spec, measure, model.hierarchies["org"])
    return plan.apply(nodes)


def _empty(grain: DataFrame) -> DataFrame:
    schema = ", ".join(f"`{c}` {'double' if c == 'value' else 'string'}" for c in METRIC_COLUMNS)
    return grain.sparkSession.createDataFrame([], schema)


def run_all(
    specs: Sequence[AggregationSpec], model: SemanticModel, grain: DataFrame, n_scenarios: int
) -> DataFrame:
    """All metrics of ``specs`` in ``METRIC_COLUMNS``.

    Vector specs are grouped by filters; each group shares one ``node_vectors`` pass. A
    non-vector spec runs its ``MEASURE_REGISTRY`` entry, and an unregistered one is an error.
    """
    groups: dict[FilterKey, list[AggregationSpec]] = {}
    others: list[AggregationSpec] = []
    for spec in specs:
        if _is_vector(spec, model):
            groups.setdefault(_filter_key(spec), []).append(spec)
        else:
            measure = model.measure(spec.measure)
            if spec.measure not in MEASURE_REGISTRY:
                raise SpecError(
                    f"spec '{spec.name}': {measure.measure_class} measure '{spec.measure}' "
                    "has no implementation in MEASURE_REGISTRY"
                )
            others.append(spec)

    frames: list[DataFrame] = []
    for group in groups.values():
        nodes = node_vectors(group, model, grain, n_scenarios)
        frames.append(_vector_group_metrics(group, model, nodes))
    for spec in others:
        frames.append(MEASURE_REGISTRY[spec.measure](spec, model, grain).select(*METRIC_COLUMNS))

    if not frames:
        return _empty(grain)
    result = frames[0]
    for df in frames[1:]:
        result = result.unionByName(df)
    return result
