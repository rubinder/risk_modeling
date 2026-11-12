import datetime

import pytest
from pyspark.sql import SparkSession

from riskagg.data.market import make_exposures, make_pnl_vectors
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES


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
