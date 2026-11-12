"""Element-wise vector sums and the single grouping-sets pass over the reporting grain."""

from collections.abc import Mapping, Sequence

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from riskagg.semantic.model import load_model

GRAIN_DIMS: tuple[str, ...] = load_model().grain
METRIC_COLUMNS = ["aggregation", "measure", "node_level", *GRAIN_DIMS, "scenario", "value"]
ROLLED_UP = "*"


def sum_vectors(col: str, n: int) -> Column:
    """Element-wise sum of an ``array<double>`` column: ``array(sum(col[0]), ...)``.

    Native aggregate expressions only, so rows never multiply by the scenario count.
    """
    c = F.col(col)
    return F.array(*[F.sum(c[i]) for i in range(n)])


def node_level(grouping_set: Sequence[str]) -> str:
    return "|".join(grouping_set)


def unique_sets(grouping_sets: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    """Grouping sets with duplicates (same dims in any order) removed; first order wins."""
    seen: set[frozenset[str]] = set()
    out: list[tuple[str, ...]] = []
    for gs in grouping_sets:
        key = frozenset(gs)
        if key not in seen:
            seen.add(key)
            out.append(tuple(gs))
    return out


def aggregate_vectors(
    grain: DataFrame,
    grouping_sets: Sequence[Sequence[str]],
    n_scenarios: int,
    filters: Mapping[str, Sequence[str]] | None = None,
) -> DataFrame:
    """Every grouping set in one grouping-sets aggregation, per ``window``.

    Sums over ``liquidity_horizon`` and ``in_reduced_set``. Columns: ``GRAIN_DIMS`` (``*``
    where rolled up), ``node_level``, ``window``, ``pnl``. ``filters`` maps a dimension to
    its allowed values and is applied before aggregating.
    """
    sets = unique_sets(grouping_sets)
    if not sets:
        raise ValueError("aggregate_vectors: no grouping sets")
    df = grain
    for dim, values in (filters or {}).items():
        df = df.filter(F.col(dim).isin(list(values)))
    cols = [d for d in GRAIN_DIMS if any(d in gs for gs in sets)]
    keyed = [[*gs, "window"] for gs in sets]
    agg = df.groupingSets(keyed, *cols, "window").agg(
        sum_vectors("pnl", n_scenarios).alias("pnl"), F.grouping_id().alias("_gid")
    )
    # grouping_id(): one bit per grouping column, first column most significant; 1 = rolled up.
    width = len(cols) + 1

    def bit(i: int) -> int:
        return 1 << (width - 1 - i)

    gid = F.col("_gid")
    level = None
    for gs in sets:
        mask = sum(bit(i) for i, d in enumerate(cols) if d not in gs)
        cond = gid == mask
        level = F.when(cond, node_level(gs)) if level is None else level.when(cond, node_level(gs))
    dims = []
    for d in GRAIN_DIMS:
        if d in cols:
            rolled = (gid.bitwiseAND(bit(cols.index(d)))) != 0
            dims.append(F.when(rolled, ROLLED_UP).otherwise(F.col(d)).alias(d))
        else:
            dims.append(F.lit(ROLLED_UP).alias(d))
    return agg.select(*dims, level.alias("node_level"), "window", "pnl")
