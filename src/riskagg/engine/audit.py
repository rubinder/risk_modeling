"""Audit: the run manifest and the reconciliation of curated outputs against raw inputs.

``reconcile`` reads what was written to disk, never in-memory frames. ``CHECKS`` is the
extension point: later tasks append ``(ctx, spark) -> Check`` functions for stress, FRTB, CCAR.
"""

import json
import math
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from riskagg.context import RunContext
from riskagg.conventions import es_from_pnl, var_from_pnl
from riskagg.engine.vectors import GRAIN_DIMS
from riskagg.paths import REPO_ROOT
from riskagg.semantic.model import load_model
from riskagg.spark import SESSION_CONF

TOLERANCE = 1e-6


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


CheckFn = Callable[[RunContext, SparkSession], Check]


def content_checksum(df: DataFrame) -> tuple[int, str]:
    """``(row count, checksum)``; the checksum is order independent.

    It is the exact sum of ``xxhash64`` over all columns, computed in Spark as a decimal.
    """
    row = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.xxhash64(*df.columns).cast("decimal(38,0)")).alias("checksum"),
    ).first()
    return int(row["rows"]), str(row["checksum"] if row["checksum"] is not None else 0)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(
    ctx: RunContext,
    inputs: Mapping[str, DataFrame],
    spark: SparkSession,
    n_scenarios: int | None = None,
) -> dict:
    """Write ``runs/<run_id>/manifest.json`` and return its content."""
    datasets = {}
    for name, df in inputs.items():
        rows, checksum = content_checksum(df)
        datasets[name] = {"rows": rows, "checksum": checksum}
    manifest = {
        "run_id": ctx.run_id,
        "profile": ctx.profile,
        "as_of_date": ctx.as_of_date.isoformat(),
        "spec_hash": ctx.spec_hash,
        "n_scenarios": n_scenarios,
        "spark_version": spark.version,
        "spark_conf": {k: spark.conf.get(k, "") for k in SESSION_CONF},
        "git_commit": git_commit(),
        "inputs": datasets,
        "reconciliation": None,
    }
    _save(ctx, manifest)
    return manifest


def _save(ctx: RunContext, manifest: dict) -> None:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    ctx.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def record_reconciliation(ctx: RunContext, checks: list[Check]) -> None:
    manifest = json.loads(ctx.manifest_path.read_text())
    manifest["reconciliation"] = {
        "passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
    _save(ctx, manifest)


def _within(diff: str, ref: str) -> "F.Column":
    return F.abs(F.col(diff)) <= TOLERANCE * F.greatest(F.lit(1.0), F.abs(F.col(ref)))


def _raw_firm_vector(ctx: RunContext, spark: SparkSession) -> DataFrame:
    raw = spark.read.parquet(str(ctx.raw("pnl_vectors")))
    return (
        raw.select("window", F.posexplode("pnl").alias("scenario_idx", "pnl"))
        .groupBy("window", "scenario_idx")
        .agg(F.sum("pnl").alias("raw_pnl"))
    )


def check_firm_vector_is_sum_of_book_vectors(ctx: RunContext, spark: SparkSession) -> Check:
    """Raw firm vector (all trades) equals the sum of the curated book vectors, per scenario."""
    name = "firm_vector_equals_sum_of_book_vectors"
    curated = spark.read.parquet(str(ctx.curated("pnl_scenarios")))
    books = (
        curated.groupBy("book_id", "window", "scenario_idx")
        .agg(F.sum("pnl").alias("pnl"))
        .groupBy("window", "scenario_idx")
        .agg(F.sum("pnl").alias("books_pnl"))
    )
    joined = _raw_firm_vector(ctx, spark).join(books, ["window", "scenario_idx"], "full_outer")
    joined = joined.withColumn("diff", F.col("raw_pnl") - F.col("books_pnl"))
    stats = joined.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum((F.col("raw_pnl").isNull() | F.col("books_pnl").isNull()).cast("int")).alias("miss"),
        F.sum((~_within("diff", "raw_pnl")).cast("int")).alias("bad"),
        F.max(F.abs("diff")).alias("max_diff"),
    ).first()
    missing, bad, n = stats["miss"] or 0, stats["bad"] or 0, stats["n"]
    ok = n > 0 and missing == 0 and bad == 0
    return Check(
        name, ok, f"{n} (window, scenario) points, {missing} missing, {bad} outside tolerance, "
        f"max abs diff {stats['max_diff']}"
    )  # fmt: skip


def check_firm_var_from_pnl_scenarios(ctx: RunContext, spark: SparkSession) -> Check:
    """Firm-level vector measures in ``risk_metrics`` equal the measure of the curated firm vector.

    SIMPLIFICATION: assumes the firm-node row of a measure comes from an unfiltered
    specification; a filtered firm node would need the filter applied here too.
    """
    name = "pnl_scenarios_firm_sum_matches_risk_metrics"
    model = load_model()
    metrics = (
        spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where((F.col("node_level") == "firm_id") & ~F.col("measure").contains("__"))
        .select("measure", "value")
        .collect()
    )
    firm = (
        spark.read.parquet(str(ctx.curated("pnl_scenarios")))
        .groupBy("window", "scenario_idx")
        .agg(F.sum("pnl").alias("pnl"))
        .orderBy("window", "scenario_idx")
        .collect()
    )
    vectors: dict[str, list[float]] = {}
    for r in firm:
        vectors.setdefault(r["window"], []).append(r["pnl"])
    problems, checked = [], 0
    for r in metrics:
        measure = model.measures.get(r["measure"])
        if measure is None or measure.measure_class != "vector":
            continue
        pnl = np.array(vectors.get(measure.params["window"], []))
        if pnl.size == 0:
            problems.append(f"{measure.name}: no curated vector for {measure.params['window']}")
            continue
        fn = var_from_pnl if measure.function == "var" else es_from_pnl
        expected = fn(pnl, float(measure.params["level"])) * math.sqrt(
            float(measure.params.get("horizon_days", 1))
        )
        checked += 1
        if abs(r["value"] - expected) > TOLERANCE * max(1.0, abs(expected)):
            problems.append(f"{measure.name}: stored {r['value']} vs recomputed {expected}")
    ok = checked > 0 and not problems
    detail = f"{checked} firm measures recomputed" + (
        f"; {'; '.join(problems)}" if problems else ""
    )
    return Check(name, ok, detail)


def check_no_duplicate_nodes(ctx: RunContext, spark: SparkSession) -> Check:
    """No ``(node, measure)`` appears twice in ``risk_metrics``."""
    name = "no_duplicate_node_measure"
    df = spark.read.parquet(str(ctx.curated("risk_metrics")))
    key = ["measure", "node_level", *GRAIN_DIMS, "scenario"]
    dupes = df.groupBy(*key).count().where(F.col("count") > 1).count()
    return Check(name, dupes == 0, f"{dupes} duplicated (node, measure) keys")


CHECKS: list[CheckFn] = [
    check_firm_vector_is_sum_of_book_vectors,
    check_firm_var_from_pnl_scenarios,
    check_no_duplicate_nodes,
]


def reconcile(ctx: RunContext, spark: SparkSession) -> list[Check]:
    """Run every registered check; an exception inside a check is a failed check."""
    results = []
    for fn in CHECKS:
        try:
            results.append(fn(ctx, spark))
        except Exception as exc:  # noqa: BLE001 - a crashing check must fail, not abort the audit
            results.append(Check(fn.__name__, False, f"check raised {type(exc).__name__}: {exc}"))
    return results
