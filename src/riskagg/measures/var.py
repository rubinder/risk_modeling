"""VaR and expected shortfall from node P&L vectors, with diversification and incremental VaR.

Every value is a function of a summed vector (``riskagg.conventions``); a node's VaR is never
the sum of its children's VaRs. ``__sum_children`` is the one deliberate sum of standalone
values, reported beside the node's own figure to show the diversification benefit.

Metrics are derived from the node frame in one linear stream, never by branching it: a
parent's vector reaches its children through a window over the parent's dimensions, and each
row emits its metrics as an array of structs that is exploded once. Branching (a self-join or
a union of filtered copies) would let the optimizer push each branch's filters below the
grouping-sets aggregate and re-run it once per branch.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from riskagg.conventions import es_col, var_col
from riskagg.engine.spec import AggregationSpec
from riskagg.engine.vectors import GRAIN_DIMS, METRIC_COLUMNS, ROLLED_UP, node_level, unique_sets
from riskagg.semantic.model import Measure, SpecError

FUNCTIONS: dict[str, Callable[[Column, float], Column]] = {"var": var_col, "es": es_col}


def _value_fn(measure: Measure) -> Callable[[Column], Column]:
    try:
        f = FUNCTIONS[measure.function]
    except KeyError:
        raise SpecError(
            f"measure '{measure.name}': vector function must be one of {sorted(FUNCTIONS)}, "
            f"not {measure.function!r}"
        ) from None
    level = float(measure.params["level"])
    # SIMPLIFICATION: multi-day horizons scale the one-day figure by sqrt(horizon_days)
    # instead of using overlapping multi-day P&L.
    scale = math.sqrt(float(measure.params.get("horizon_days", 1)))
    return lambda vec: f(vec, level) * F.lit(scale)


@dataclass
class MetricPlan:
    """Window columns to add to the node frame and the metric structs each row may emit.

    ``labels`` maps a set of dimensions to the ``node_level`` string the node frame carries
    for it (the frame may come from a shared pass that ordered the dimensions differently).
    """

    labels: Mapping[frozenset[str], str] | None = None
    columns: dict[str, Column] = field(default_factory=dict)
    metrics: list[Column] = field(default_factory=list)
    _parent_cols: dict[str, str] = field(default_factory=dict)

    def label(self, gs: Sequence[str]) -> str:
        return self.labels[frozenset(gs)] if self.labels else node_level(gs)

    def is_node(self, gs: Sequence[str], measure: Measure) -> Column:
        return (F.col("window") == measure.params["window"]) & (
            F.col("node_level") == self.label(gs)
        )

    def emit(self, when: Column, spec: AggregationSpec, name: str, gs, value: Column) -> None:
        row = F.struct(
            F.lit(spec.name).alias("aggregation"),
            F.lit(name).alias("measure"),
            F.lit(node_level(gs)).alias("node_level"),
            value.cast("double").alias("value"),
        )
        self.metrics.append(F.when(when, row))

    def parent_pnl(self, parent: Sequence[str]) -> str:
        """Column holding, on every row sharing ``parent``'s dims, that parent's vector."""
        label = self.label(parent)
        if label not in self._parent_cols:
            name = f"_parent_pnl_{len(self._parent_cols)}"
            over = Window.partitionBy("window", *parent)
            pnl = F.when(F.col("node_level") == label, F.col("pnl"))
            self.columns[name] = F.max(pnl).over(over)  # exactly one parent row per partition
            self._parent_cols[label] = name
        return self._parent_cols[label]

    def add_column(self, prefix: str, col: Column) -> str:
        name = f"_{prefix}_{len(self.columns)}"
        self.columns[name] = col
        return name

    def apply(self, nodes: DataFrame) -> DataFrame:
        """One pass: add the window columns, explode the emitted metrics."""
        metrics = self.metrics or [F.when(F.lit(False), _placeholder())]
        df = nodes.withColumns(self.columns) if self.columns else nodes
        exploded = df.select(
            *GRAIN_DIMS, F.explode(F.filter(F.array(*metrics), lambda m: m.isNotNull())).alias("m")
        )
        return exploded.select(
            "m.aggregation",
            "m.measure",
            "m.node_level",
            *GRAIN_DIMS,
            F.lit(ROLLED_UP).alias("scenario"),
            "m.value",
        ).select(*METRIC_COLUMNS)


def _placeholder() -> Column:
    return F.struct(
        F.lit("").alias("aggregation"),
        F.lit("").alias("measure"),
        F.lit("").alias("node_level"),
        F.lit(0.0).alias("value"),
    )


def plan_metrics(plan: MetricPlan, spec: AggregationSpec, measure: Measure) -> None:
    """The measure at every node of ``spec``'s grouping sets."""
    f = _value_fn(measure)
    for gs in unique_sets(spec.grouping_sets):
        plan.emit(plan.is_node(gs, measure), spec, measure.name, gs, f(F.col("pnl")))


def _org_levels(spec: AggregationSpec, hierarchy_levels: Sequence[str]) -> list[tuple[str, ...]]:
    """The spec's grouping sets that are org-hierarchy prefixes, shortest first."""
    org = tuple(hierarchy_levels)
    by_len = {len(gs): tuple(gs) for gs in spec.grouping_sets if gs and tuple(gs) == org[: len(gs)]}
    return [by_len[k] for k in sorted(by_len)]


def plan_diversification(
    plan: MetricPlan, spec: AggregationSpec, measure: Measure, hierarchy_levels: Sequence[str]
) -> None:
    """Sum of children, diversification and incremental rows between consecutive org levels."""
    f = _value_fn(measure)
    levels = _org_levels(spec, hierarchy_levels)
    pnl = F.col("pnl")
    for parent, child in itertools.pairwise(levels):
        parent_pnl = F.col(plan.parent_pnl(parent))
        child_value = F.when(F.col("node_level") == plan.label(child), f(pnl))
        sum_children = F.col(
            plan.add_column(
                "sum_children", F.sum(child_value).over(Window.partitionBy("window", *parent))
            )
        )
        at_parent = plan.is_node(parent, measure)
        plan.emit(at_parent, spec, f"{measure.name}__sum_children", parent, sum_children)
        plan.emit(
            at_parent, spec, f"{measure.name}__diversification", parent, sum_children - f(pnl)
        )
        without_child = F.zip_with(parent_pnl, pnl, lambda p, c: p - c)
        plan.emit(
            plan.is_node(child, measure),
            spec,
            f"{measure.name}__incremental",
            child,
            f(parent_pnl) - f(without_child),
        )


def metrics_from_vectors(nodes: DataFrame, spec: AggregationSpec, measure: Measure) -> DataFrame:
    """The measure at every node of ``spec``'s grouping sets, in ``METRIC_COLUMNS``.

    Uses the measure's ``window``; ``horizon_days`` scales by its square root.
    """
    plan = MetricPlan()
    plan_metrics(plan, spec, measure)
    return plan.apply(nodes)


def diversification(
    nodes: DataFrame,
    spec: AggregationSpec,
    measure: Measure,
    hierarchy_levels: Sequence[str],
) -> DataFrame:
    """Diversification rows between consecutive org levels of ``spec``, in ``METRIC_COLUMNS``.

    At each parent node: ``<m>__sum_children`` (sum of the children's standalone values) and
    ``<m>__diversification`` (``sum_children - parent``). At each child node:
    ``<m>__incremental`` = ``parent - f(parent vector - child vector)``, with the same
    function, level, window and horizon scaling as the measure. A single child therefore
    gives zero diversification and an incremental equal to the parent value. The top level
    of the spec has no incremental row.
    """
    plan = MetricPlan()
    plan_diversification(plan, spec, measure, hierarchy_levels)
    return plan.apply(nodes)
