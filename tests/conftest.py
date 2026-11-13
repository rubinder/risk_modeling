import datetime
from dataclasses import dataclass
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from typer.testing import CliRunner

from riskagg.cli import app
from riskagg.context import official_run
from riskagg.data.market import make_exposures, make_pnl_vectors
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.semantic.model import SemanticModel, load_model


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("riskagg-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def tiny_market(spark):
    """Tiny-profile ``(pnl_vectors, hierarchy, n_scenarios)``, cached for the session."""
    p = PROFILES["tiny"]
    trades = make_trades(spark, p, datetime.date(2026, 11, 12))
    rf = make_risk_factors(spark, p)
    expo = make_exposures(spark, p, trades, rf)
    pnl = make_pnl_vectors(spark, p, trades, expo, rf).cache()
    hierarchy = make_hierarchy(spark, p).cache()
    yield pnl, hierarchy, p.n_scenarios
    pnl.unpersist()
    hierarchy.unpersist()


@dataclass(frozen=True)
class TinyRun:
    root: Path
    as_of_date: datetime.date
    run_id: str
    model: SemanticModel


@pytest.fixture(scope="session")
def tmp_run(spark, tmp_path_factory):
    """Generate and run the tiny profile once into a temp root; the official run is served."""
    root = tmp_path_factory.mktemp("served")
    as_of = datetime.date(2026, 11, 12)
    runner = CliRunner()
    for cmd in ("generate", "run"):
        args = [cmd, "--profile", "tiny", "--as-of", as_of.isoformat(), "--root", str(root)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    run_id = official_run(root, as_of)
    assert run_id is not None
    return TinyRun(root, as_of, run_id, load_model())
