"""Command line: ``generate``, ``run``, ``load-pinot``, ``serve`` and ``query``."""

import datetime
from pathlib import Path

import typer
from pyspark.sql import SparkSession

from riskagg.context import (
    DEFAULT_ROOT,
    mark_official,
    new_run,
    official_runs,
    raw_path,
    run_context,
)
from riskagg.data.banking import (
    make_banking_book,
    make_capital_start,
    make_default_risk_positions,
    make_macro_scenarios,
)
from riskagg.data.market import make_exposures, make_pnl_vectors, make_sensitivities
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.engine import audit
from riskagg.engine.grain import build_reporting_grain
from riskagg.engine.inputs import RunInputs
from riskagg.engine.runner import run_all
from riskagg.engine.spec import load_specs, spec_hash
from riskagg.measures.frtb_drc import sa_detail
from riskagg.measures.frtb_ima import ima_detail
from riskagg.measures.stress import load_scenarios, shock_table
from riskagg.paths import CONFIG_DIR, semantic_model_path
from riskagg.semantic.model import load_model
from riskagg.serving import curate, pinot
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
            "default_risk_positions": make_default_risk_positions(spark, p, trades),
            "macro_scenarios": make_macro_scenarios(spark, p),
            "banking_book": make_banking_book(spark, p, hierarchy),
            "capital_start": make_capital_start(spark, p, hierarchy),
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
    grain = stress_grain = projection = inputs = None
    try:
        hierarchy = spark.read.parquet(str(ctx.raw("hierarchy")))
        pnl_vectors = spark.read.parquet(str(ctx.raw("pnl_vectors")))
        grain = build_reporting_grain(pnl_vectors, hierarchy, p.n_scenarios).cache()
        inputs = RunInputs(
            grain=grain,
            n_scenarios=p.n_scenarios,
            sensitivities=spark.read.parquet(str(ctx.raw("sensitivities"))),
            trades=spark.read.parquet(str(ctx.raw("trades"))),
            hierarchy=hierarchy,
            default_risk_positions=spark.read.parquet(str(ctx.raw("default_risk_positions"))),
            shocks=shock_table(
                spark, load_scenarios(), spark.read.parquet(str(ctx.raw("risk_factors")))
            ),
            banking_book=spark.read.parquet(str(ctx.raw("banking_book"))),
            macro_scenarios=spark.read.parquet(str(ctx.raw("macro_scenarios"))),
            capital_start=spark.read.parquet(str(ctx.raw("capital_start"))),
        )
        stress_grain = inputs.stress_grain().cache()
        projection = inputs.ccar_projection().cache()
        metrics = run_all(specs, model, grain, p.n_scenarios, inputs)
        curate.write_risk_metrics(metrics, ctx)
        curate.write_pnl_scenarios(grain, ctx, p.n_scenarios)
        curate.write_stress_results(stress_grain, ctx)
        levels = [list(c) for c in model.measure("frtb_sa_charge").computed_at]
        curate.write_frtb_sa_detail(sa_detail(inputs, levels), ctx)
        ima_levels = [list(c) for c in model.measure("frtb_imcc").computed_at]
        curate.write_frtb_ima_detail(ima_detail(inputs, ima_levels), ctx)
        curate.write_ccar_projection(projection, ctx)
        audit.write_manifest(
            ctx,
            {
                "hierarchy": hierarchy,
                "pnl_vectors": pnl_vectors,
                "default_risk_positions": inputs.default_risk_positions,
                "banking_book": inputs.banking_book,
                "macro_scenarios": inputs.macro_scenarios,
                "capital_start": inputs.capital_start,
            },
            spark,
            p.n_scenarios,
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
        if stress_grain is not None:
            stress_grain.unpersist()
        if projection is not None:
            projection.unpersist()
        if inputs is not None:
            inputs.release()
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


@app.command("load-pinot")
def load_pinot(
    as_of: str | None = typer.Option(None, "--as-of", help="Only this date (default: all)"),
    root: Path = DEFAULT_ROOT,
    controller: str = typer.Option(pinot.DEFAULT_CONTROLLER, "--controller"),
) -> None:
    """Load the official run of every as-of date (or one) into Pinot, replacing that date."""
    runs = official_runs(root)
    if as_of is not None:
        date = _as_of(as_of)
        if date not in runs:
            raise typer.BadParameter(f"no official run for {date} under {root}")
        runs = {date: runs[date]}
    if not runs:
        typer.echo(f"no official run under {root}: nothing to load", err=True)
        raise typer.Exit(1)
    if not pinot.is_up(controller):
        typer.echo(f"Pinot controller at {controller} is not reachable", err=True)
        raise typer.Exit(1)
    model = load_model()
    for date, run_id in sorted(runs.items()):
        counts = pinot.load(run_context(root, run_id), model, controller)
        loaded = ", ".join(f"{t} {n:,} rows" for t, n in counts.items())
        typer.echo(f"loaded {date} run {run_id}: {loaded}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    root: Path = DEFAULT_ROOT,
    broker: str = typer.Option(pinot.DEFAULT_BROKER, "--broker"),
    controller: str = typer.Option(pinot.DEFAULT_CONTROLLER, "--controller"),
) -> None:
    """Serve the HTTP API (catalogue, metrics, drill-down)."""
    import uvicorn

    from riskagg.serving.api import create_app

    uvicorn.run(create_app(root, None, broker, controller), host=host, port=port)


@app.command()
def query(
    measure: str,
    group_by: str = typer.Option("", "--group-by", help="Comma-separated dimensions"),
    as_of: str | None = typer.Option(None, "--as-of"),
    top: int = typer.Option(0, "--top", help="Show only the N largest values"),
    root: Path = DEFAULT_ROOT,
    base_url: str | None = typer.Option(None, "--url", help="API base URL (default: in-process)"),
) -> None:
    """Answer one metric query through the client and say which source answered."""
    from riskagg.serving.client import RiskClient

    client = RiskClient(base_url=base_url, root=root)
    dims = [d for d in group_by.split(",") if d]
    df = client.metric(measure, dims, as_of=as_of)
    if top:
        df = df.sort_values("value", ascending=False).head(top)
    label = f"{measure} by {', '.join(dims)}" if dims else measure
    typer.echo(f"{label} for {client.as_of_date} run {client.run_id} (source: {client.source})")
    typer.echo(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
