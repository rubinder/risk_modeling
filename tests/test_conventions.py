import numpy as np
import pytest
from pyspark.sql import functions as F

from riskagg.conventions import es_col, es_from_pnl, var_col, var_from_pnl, var_rank


def test_var_and_es_20_scenarios():
    pnl = np.arange(-10, 10, dtype=float)  # losses 10, 9, 8, 7, ...
    assert var_from_pnl(pnl, 0.90) == 9.0  # 2nd worst
    assert es_from_pnl(pnl, 0.80) == 8.5  # mean of worst 4


def test_headline_levels_250_scenarios():
    pnl = np.arange(250, dtype=float) - 125  # losses 125, 124, 123, ...
    assert var_rank(250, 0.99) == 3
    assert var_from_pnl(pnl, 0.99) == 123.0
    assert es_from_pnl(pnl, 0.975) == pytest.approx(122.36)  # k = 6.25


def test_all_gains_gives_negative_var():
    assert var_from_pnl(np.array([1.0, 2.0, 3.0, 4.0]), 0.75) == -1.0


def test_spark_matches_numpy(spark):
    rng = np.random.default_rng(7)
    vectors = [rng.normal(0, 1000, 250).tolist() for _ in range(20)]
    df = spark.createDataFrame([(i, v) for i, v in enumerate(vectors)], "id int, pnl array<double>")
    df = df.orderBy(F.rand(3))
    rows = df.select(
        "id",
        var_col(F.col("pnl"), 0.99).alias("v99"),
        es_col(F.col("pnl"), 0.99).alias("e99"),
        var_col(F.col("pnl"), 0.975).alias("v975"),
        es_col(F.col("pnl"), 0.975).alias("e975"),
    ).collect()
    assert len(rows) == 20
    for r in rows:
        pnl = np.array(vectors[r["id"]])
        assert r["v99"] == pytest.approx(var_from_pnl(pnl, 0.99), abs=1e-9)
        assert r["e99"] == pytest.approx(es_from_pnl(pnl, 0.99), abs=1e-9)
        assert r["v975"] == pytest.approx(var_from_pnl(pnl, 0.975), abs=1e-9)
        assert r["e975"] == pytest.approx(es_from_pnl(pnl, 0.975), abs=1e-9)
