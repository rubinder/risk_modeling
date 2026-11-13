"""Command line: ``riskagg generate`` and ``riskagg run``."""

import datetime
from pathlib import Path

import typer
from pyspark.sql import SparkSession

from riskagg.context import DEFAULT_ROOT, mark_official, new_run, raw_path
from riskagg.data.market import make_exposures, make_pnl_vectors, make_sensitivities
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine import audit
from riskagg.engine.grain import build_reporting_grain
from riskagg.engine.runner import run_all
from riskagg.engine.spec import load_specs, spec_hash
from riskagg.paths import CONFIG_DIR, semantic_model_path
from riskagg.semantic.model import load_model
from riskagg.serving import curate
from riskagg.spark import get_spark

app = typer.Typer(no_args_is_help=True, add_completion=False)

ProfileOpt = typer.Option("--profile", help=f"One of {', '.join(PROFILES)}")
AsOfOpt = typer.Option("--as-of", help="As-of date, YYYY-MM-DD (default: today)")


def _profile(name: str):
    if name not in PROFILES:
        raise typer.BadParameter(f"unknown profile '{name}'; choose from {sorted(PROFILES)}")
    return PROFILES[name]


def _as_of(value: str | None) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value) if value else datetime.date.today()
    except ValueError:
        raise typer.BadParameter(f"invalid date '{value}', expected YYYY-MM-DD") from None


def _session() -> tuple[SparkSession, bool]:
    """The active session if there is one (and then it is not ours to stop), else a new one."""
    existing = SparkSession.getActiveSession() is not None
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    return spark, not existing


def reference_paths() -> list[Path]:
    ref = CONFIG_DIR / "reference"
    files = sorted(p for p in ref.rglob("*") if p.is_file()) if ref.is_dir() else []
    return [semantic_model_path(), *files]


@app.command()
def generate(
    profile: str = ProfileOpt,
    as_of: str | None = AsOfOpt,
    root: Path = DEFAULT_ROOT,
) -> None:
    """Write the raw datasets of a profile as Parquet (overwriting)."""
    p = _profile(profile)
    as_of_date = _as_of(as_of)
    spark, owned = _session()
    try:
        hierarchy = make_hierarchy(spark, p)
        risk_factors = make_risk_factors(spark, p)
        trades = make_trades(spark, p, as_of_date).cache()
        exposures = make_exposures(spark, p, trades, risk_factors).cache()
        datasets = {
            "hierarchy": hierarchy,
            "risk_factors": risk_factors,
            "trades": trades,
            "exposures": exposures,
            "pnl_vectors": make_pnl_vectors(spark, p, trades, exposures, risk_factors),
            "sensitivities": make_sensitivities(spark, p, exposures, risk_factors),
        }
        for name, df in datasets.items():
            df.write.mode("overwrite").parquet(str(raw_path(root, profile, as_of_date, name)))
        trades.unpersist()
        exposures.unpersist()
        typer.echo(f"generated {len(datasets)} datasets for {profile} {as_of_date} under {root}")
    finally:
        if owned:
            spark.stop()


@app.command()
def run(
    profile: str = ProfileOpt,
    as_of: str | None = AsOfOpt,
    root: Path = DEFAULT_ROOT,
) -> None:
    """Aggregate, curate, reconcile; mark the run official only if every check passes."""
    p = _profile(profile)
    as_of_date = _as_of(as_of)
    model = load_model()
    specs = load_specs(None, model)
    ctx = new_run(profile, as_of_date, root, spec_hash(specs, reference_paths()))
    spark, owned = _session()
    grain = None
    try:
        hierarchy = spark.read.parquet(str(ctx.raw("hierarchy")))
        pnl_vectors = spark.read.parquet(str(ctx.raw("pnl_vectors")))
        grain = build_reporting_grain(pnl_vectors, hierarchy, p.n_scenarios).cache()
        metrics = run_all(specs, model, grain, p.n_scenarios)
        curate.write_risk_metrics(metrics, ctx)
        curate.write_pnl_scenarios(grain, ctx, p.n_scenarios)
        audit.write_manifest(
            ctx, {"hierarchy": hierarchy, "pnl_vectors": pnl_vectors}, spark, p.n_scenarios
        )
        checks = audit.reconcile(ctx, spark)
        audit.record_reconciliation(ctx, checks)
        for c in checks:
            typer.echo(f"  [{'ok' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        firm = _firm_var(spark, ctx)
        typer.echo(f"run {ctx.run_id} spec_hash {ctx.spec_hash} firm var_99_1d {firm}")
        if not all(c.passed for c in checks):
            typer.echo("reconciliation failed: run not marked official", err=True)
            raise typer.Exit(1)
        mark_official(ctx)
        typer.echo(f"run {ctx.run_id} is official for {as_of_date}")
    finally:
        if grain is not None:
            grain.unpersist()
        if owned:
            spark.stop()


def _firm_var(spark: SparkSession, ctx) -> str:
    rows = (
        spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where("measure = 'var_99_1d' and node_level = 'firm_id'")
        .select("value")
        .collect()
    )
    return f"{rows[0]['value']:,.2f}" if rows else "n/a"
