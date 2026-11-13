"""Curated outputs: ``risk_metrics``, ``pnl_scenarios``, ``stress_results``, ``frtb_sa_detail``,
``frtb_ima_detail`` and ``ccar_projection``, partitioned by as-of date and run.

All are written with ``as_of_date`` and ``run_id`` as partition directories, and dynamic
partition overwrite, so a re-run never touches another run's partition. Reading the table
root exposes both partition keys as columns. Every row also carries ``spec_hash``, the hash of
the specifications and reference data that produced the run.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from riskagg.context import RunContext
from riskagg.engine.vectors import GRAIN_DIMS, sum_vectors
from riskagg.semantic.model import load_model

PARTITION_COLS = ["as_of_date", "run_id"]
RUN_COLS = [*PARTITION_COLS, "spec_hash"]  # added to every table by ``_write``


def _data_columns(table: str) -> list[str]:
    """The model's columns of ``table`` that a writer supplies (all but ``RUN_COLS``)."""
    return [c for c in load_model().tables[table]["columns"] if c not in RUN_COLS]


def _write(df: DataFrame, ctx: RunContext, table: str) -> None:
    df = df.select(*_data_columns(table)).withColumns(
        {
            "spec_hash": F.lit(ctx.spec_hash),
            "as_of_date": F.lit(ctx.as_of_date.isoformat()).cast("date"),
            "run_id": F.lit(ctx.run_id),
        }
    )
    (
        df.write.mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .partitionBy(*PARTITION_COLS)
        .parquet(str(ctx.curated_root(table)))
    )


def write_risk_metrics(df: DataFrame, ctx: RunContext) -> None:
    """``df`` in ``METRIC_COLUMNS``."""
    _write(df, ctx, "risk_metrics")


def write_pnl_scenarios(grain: DataFrame, ctx: RunContext, n_scenarios: int) -> None:
    """The grain summed over ``liquidity_horizon`` and ``in_reduced_set``, one row per scenario."""
    summed = grain.groupBy(*GRAIN_DIMS, "window").agg(sum_vectors("pnl", n_scenarios).alias("pnl"))
    exploded = summed.select(
        *GRAIN_DIMS, "window", F.posexplode("pnl").alias("scenario_idx", "pnl")
    )
    _write(exploded, ctx, "pnl_scenarios")


def write_stress_results(stress_grain: DataFrame, ctx: RunContext) -> None:
    """Stress P&L at ``GRAIN_DIMS`` x scenario (columns of the model's ``stress_results``)."""
    _write(stress_grain, ctx, "stress_results")


def write_frtb_sa_detail(detail: DataFrame, ctx: RunContext) -> None:
    """FRTB bucket detail (K_b, S_b) per node, risk class, component, bucket and scenario.

    Not additive: K_b values must never be summed. Columns of the model's ``frtb_sa_detail``.
    """
    _write(detail, ctx, "frtb_sa_detail")


def write_frtb_ima_detail(detail: DataFrame, ctx: RunContext) -> None:
    """FRTB IMA ES components per node and risk class (``*`` = all classes together).

    Not additive: expected shortfalls must never be summed. Columns of ``frtb_ima_detail``.
    """
    _write(detail, ctx, "frtb_ima_detail")


def write_ccar_projection(projection: DataFrame, ctx: RunContext) -> None:
    """The CCAR path per scenario, quarter and node (columns of the model's ``ccar_projection``).

    Rows of different ``node_level`` must never be summed together; capital columns are null
    for business lines.
    """
    _write(projection, ctx, "ccar_projection")
