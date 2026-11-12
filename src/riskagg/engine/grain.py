"""Reporting grain: trade P&L vectors reduced once to the semantic model's grain.

The grain keeps ``liquidity_horizon``, ``in_reduced_set`` and ``window`` beside the grain
dimensions, because FRTB IMA needs them; VaR and ES sum over the first two.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from riskagg.engine.vectors import GRAIN_DIMS, sum_vectors

__all__ = ["GRAIN_DIMS", "GRAIN_EXTRA", "build_reporting_grain"]

GRAIN_EXTRA = ("liquidity_horizon", "in_reduced_set", "window")


def _validate(pnl_vectors: DataFrame, books: DataFrame, n_scenarios: int) -> None:
    """One aggregate action: array lengths, nulls, and books missing from the hierarchy."""
    pnl = F.col("pnl")
    known = F.broadcast(books.select("book_id", F.lit(True).alias("_known")))
    stats = (
        pnl_vectors.join(known, "book_id", "left")
        .agg(
            F.sort_array(F.collect_set(F.size(pnl))).alias("lengths"),
            F.sum(pnl.isNull().cast("long")).alias("null_arrays"),
            F.sum(F.size(F.filter(pnl, lambda x: x.isNull()))).alias("null_elements"),
            F.sum(F.col("_known").isNull().cast("long")).alias("unknown_books"),
        )
        .first()
    )
    lengths = [n for n in stats["lengths"] if n is not None and n >= 0]
    null_arrays = stats["null_arrays"] or 0
    null_elements = stats["null_elements"] or 0
    if any(n != n_scenarios for n in lengths):
        raise ValueError(
            f"pnl_vectors: every pnl array must have length {n_scenarios}; found lengths {lengths}"
        )
    if null_arrays or null_elements:
        raise ValueError(
            f"pnl_vectors: {null_arrays} null pnl arrays and {null_elements} null pnl elements "
            f"(expected length {n_scenarios}, found lengths {lengths})"
        )
    if stats["unknown_books"]:
        raise ValueError(
            f"pnl_vectors: {stats['unknown_books']} rows have a book_id missing from hierarchy"
        )


def build_reporting_grain(
    pnl_vectors: DataFrame, hierarchy: DataFrame, n_scenarios: int
) -> DataFrame:
    """Sum trade vectors to ``GRAIN_DIMS`` + ``GRAIN_EXTRA``; the hierarchy is broadcast.

    Validates array lengths, nulls and book coverage first (one aggregate action) so a
    malformed input fails here, naming the lengths found, instead of producing a wrong
    number. The result is lazy; the caller decides whether to cache it.
    """
    hier_cols = [c for c in GRAIN_DIMS if c not in pnl_vectors.columns]
    books = hierarchy.select("book_id", *hier_cols)
    _validate(pnl_vectors, books, n_scenarios)
    org = F.broadcast(books)
    keys = [*GRAIN_DIMS, *GRAIN_EXTRA]
    return (
        pnl_vectors.join(org, "book_id")
        .groupBy(*keys)
        .agg(sum_vectors("pnl", n_scenarios).alias("pnl"))
        .select(*keys, "pnl")
    )
