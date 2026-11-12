"""NumPy / pandas reference for node VaR, ES and diversification.

Computes straight from raw ``pnl_vectors`` and ``hierarchy`` (as pandas frames) without any
``riskagg.engine`` or ``riskagg.measures`` code; only the quantile convention is shared.
Results are keyed by ``(node_level, node)`` where ``node`` is a tuple over ``dims`` with
``"*"`` for rolled-up dimensions.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from riskagg.conventions import es_from_pnl, var_from_pnl

FUNCTIONS = {"var": var_from_pnl, "es": es_from_pnl}

Key = tuple[str, tuple[str, ...]]


def node_vectors(
    pnl_vectors: pd.DataFrame,
    hierarchy: pd.DataFrame,
    dims: Sequence[str],
    grouping_set: Sequence[str],
    window: str,
) -> dict[Key, np.ndarray]:
    df = pnl_vectors[pnl_vectors["window"] == window].merge(hierarchy, on="book_id")
    level = "|".join(grouping_set)
    out: dict[Key, np.ndarray] = {}
    for _, row in df.iterrows():
        node = tuple(row[d] if d in grouping_set else "*" for d in dims)
        vec = np.asarray(row["pnl"], dtype=float)
        key = (level, node)
        out[key] = out[key] + vec if key in out else vec.copy()
    return out


def node_metrics(
    pnl_vectors: pd.DataFrame,
    hierarchy: pd.DataFrame,
    dims: Sequence[str],
    grouping_sets: Sequence[Sequence[str]],
    window: str,
    function: str,
    level: float,
    horizon_days: int = 1,
) -> dict[Key, float]:
    f = FUNCTIONS[function]
    scale = math.sqrt(horizon_days)
    out: dict[Key, float] = {}
    for gs in grouping_sets:
        for key, vec in node_vectors(pnl_vectors, hierarchy, dims, gs, window).items():
            out[key] = f(vec, level) * scale
    return out


def diversification(
    pnl_vectors: pd.DataFrame,
    hierarchy: pd.DataFrame,
    dims: Sequence[str],
    org_levels: Sequence[Sequence[str]],
    window: str,
    function: str,
    level: float,
) -> dict[tuple[str, str, tuple[str, ...]], float]:
    """``{(suffix, node_level, node): value}`` for sum_children, diversification, incremental.

    ``org_levels`` are consecutive org prefixes, shortest first.
    """
    f = FUNCTIONS[function]
    out: dict[tuple[str, str, tuple[str, ...]], float] = {}
    for parent_set, child_set in itertools.pairwise(org_levels):
        parents = node_vectors(pnl_vectors, hierarchy, dims, parent_set, window)
        children = node_vectors(pnl_vectors, hierarchy, dims, child_set, window)
        pos = [dims.index(d) for d in parent_set]
        for (p_level, p_node), p_vec in parents.items():
            p_val = f(p_vec, level)
            total = 0.0
            for (c_level, c_node), c_vec in children.items():
                if all(c_node[i] == p_node[i] for i in pos):
                    total += f(c_vec, level)
                    out[("incremental", c_level, c_node)] = p_val - f(p_vec - c_vec, level)
            out[("sum_children", p_level, p_node)] = total
            out[("diversification", p_level, p_node)] = total - p_val
    return out
