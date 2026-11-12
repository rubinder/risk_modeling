"""The single home of the VaR / expected-shortfall quantile convention.

P&L is positive for gains; VaR and ES are reported as positive numbers for losses and are
never clipped (a vector with only gains gives a negative VaR).

With n scenarios and confidence ``level``, the tail holds ``k = n * (1 - level)`` scenarios
(rounded to 9 decimals to absorb floating-point noise):

* VaR is the ``max(1, ceil(k))``-th largest loss.
* ES averages the worst ``floor(k)`` losses plus the fractional share of the next one,
  divided by ``k``.
"""

import math

import numpy as np
from pyspark.sql import Column
from pyspark.sql import functions as F


def _tail_size(n: int, level: float) -> float:
    return round(n * (1 - level), 9)


def var_rank(n: int, level: float) -> int:
    return max(1, math.ceil(_tail_size(n, level)))


def var_from_pnl(pnl: np.ndarray, level: float) -> float:
    losses = np.sort(-np.asarray(pnl, dtype=float))[::-1]
    return float(losses[var_rank(len(losses), level) - 1])


def es_from_pnl(pnl: np.ndarray, level: float) -> float:
    losses = np.sort(-np.asarray(pnl, dtype=float))[::-1]
    k = _tail_size(len(losses), level)
    whole = math.floor(k)
    total = float(losses[:whole].sum())
    if k > whole:
        total += (k - whole) * float(losses[whole])
    return total / k


def _tail_size_col(vec: Column, level: float) -> Column:
    return F.round(F.size(vec) * F.lit(1.0 - level), 9)


def var_col(vec: Column, level: float) -> Column:
    """VaR over an ``array<double>`` of P&L, as a native Spark expression."""
    rank = F.greatest(F.lit(1), F.ceil(_tail_size_col(vec, level))).cast("int")
    return -F.element_at(F.sort_array(vec), rank)


def es_col(vec: Column, level: float) -> Column:
    """Expected shortfall over an ``array<double>`` of P&L, as a native Spark expression."""
    k = _tail_size_col(vec, level)
    whole = F.floor(k).cast("int")
    worst_first = F.sort_array(vec)  # ascending P&L == descending loss
    whole_sum = F.aggregate(F.slice(worst_first, F.lit(1), whole), F.lit(0.0), lambda a, x: a - x)
    fraction = k - whole
    partial = F.when(fraction > 0, -fraction * F.element_at(worst_first, whole + 1)).otherwise(0.0)
    return (whole_sum + partial) / k
