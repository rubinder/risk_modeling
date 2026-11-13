"""NumPy reference for the FRTB IMA expected shortfall and IMCC (Basel MAR33).

Computes from raw ``pnl_vectors`` and ``hierarchy`` (pandas frames) with loops and NumPy. Imports
only ``riskagg.conventions`` (the quantile convention) from the package; the liquidity-horizon
cascade, the stressed scaling and the IMCC are written out again here.

Results are keyed by ``(node, risk_class)`` with ``node`` a tuple over ``node_cols`` and
``risk_class`` ``"*"`` for all risk classes together.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from riskagg.conventions import es_from_pnl

LEVEL = 0.975
HORIZONS = (10, 20, 40, 60, 120)
BASE = 10
RHO = 0.5
# (window, reduced set only) per ES set: full-current, reduced-current, reduced-stressed.
SETS = {"fc": ("CURRENT", False), "rc": ("CURRENT", True), "rs": ("STRESSED", True)}


def cascade(es_by_lh: dict[int, float]) -> float:
    """MAR33.4: sqrt(ES_10^2 + sum_{j>=2} (ES_j sqrt((LH_j - LH_{j-1}) / 10))^2)."""
    total = es_by_lh.get(HORIZONS[0], 0.0) ** 2
    for prev, lh in itertools.pairwise(HORIZONS):
        total += (es_by_lh.get(lh, 0.0) * math.sqrt((lh - prev) / BASE)) ** 2
    return math.sqrt(total)


def ima_es(
    pnl_vectors: pd.DataFrame, hierarchy: pd.DataFrame, node_cols: Sequence[str]
) -> dict[tuple, dict[str, float]]:
    """``{(node, risk_class): {es_fc, es_rc, es_rs, es}}``."""
    hier_cols = [c for c in node_cols if c not in pnl_vectors.columns]
    df = pnl_vectors.merge(hierarchy[["book_id", *hier_cols]], on="book_id")
    n = len(df["pnl"].iloc[0])
    # vectors[(node, rc, set, lh_j)] = summed one-day P&L of factors with horizon >= lh_j
    vectors: dict[tuple, np.ndarray] = {}
    for row in df.to_dict("records"):
        node = tuple(row[c] for c in node_cols)
        vec = np.asarray(row["pnl"], dtype=float)
        for name, (window, reduced_only) in SETS.items():
            if row["window"] != window or (reduced_only and not row["in_reduced_set"]):
                continue
            for lh in HORIZONS:
                if row["liquidity_horizon"] < lh:
                    continue
                for rc in (row["risk_class"], "*"):
                    key = (node, rc, name, lh)
                    vectors[key] = vectors.get(key, np.zeros(n)) + vec
    keys = {(node, rc) for node, rc, _, _ in vectors}
    out: dict[tuple, dict[str, float]] = {}
    for node, rc in keys:
        values = {}
        for name in SETS:
            es_by_lh = {
                lh: es_from_pnl(vectors[(node, rc, name, lh)], LEVEL) * math.sqrt(BASE)
                for lh in HORIZONS
                if (node, rc, name, lh) in vectors
            }
            values[f"es_{name}"] = cascade(es_by_lh)
        ratio = 1.0 if values["es_rc"] == 0 else values["es_fc"] / values["es_rc"]
        values["es"] = values["es_rs"] * max(ratio, 1.0)  # MAR33.6: ratio floored at 1
        out[(node, rc)] = values
    return out


def imcc(es: dict[tuple, dict[str, float]]) -> dict[tuple, dict[str, float]]:
    """``{node: {imcc_unconstrained, imcc_constrained_sum, imcc}}`` (MAR33.15)."""
    nodes = {node for node, _ in es}
    out = {}
    for node in nodes:
        unconstrained = es[(node, "*")]["es"]
        constrained = sum(v["es"] for (n, rc), v in es.items() if n == node and rc != "*")
        out[node] = {
            "imcc_unconstrained": unconstrained,
            "imcc_constrained_sum": constrained,
            "imcc": RHO * unconstrained + (1 - RHO) * constrained,
        }
    return out
