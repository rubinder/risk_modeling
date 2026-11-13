"""Benchmark: every group's variants agree, and the report has a measured row per variant.

Runs on the bench profile cut to 5,000 trades and 50 scenarios, so the skewed desk is present
but the suite stays fast (the generated code for 250 element sums is slow to compile).
"""

import dataclasses
import datetime
import re

import numpy as np
import pytest

from riskagg.bench.runner import assert_group_agrees, run_benchmark
from riskagg.bench.variants import Inputs, variants
from riskagg.data.market import make_exposures, make_pnl_vectors
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES

PROFILE = dataclasses.replace(PROFILES["bench"], n_trades=5_000, n_scenarios=50)
AS_OF = datetime.date(2026, 11, 12)


@pytest.fixture(scope="module")
def bench_inputs(spark):
    trades = make_trades(spark, PROFILE, AS_OF)
    rf = make_risk_factors(spark, PROFILE)
    pnl = make_pnl_vectors(spark, PROFILE, trades, make_exposures(spark, PROFILE, trades, rf), rf)
    pnl = pnl.cache()
    hierarchy = make_hierarchy(spark, PROFILE).cache()
    yield Inputs(pnl, hierarchy, PROFILE.n_scenarios)
    pnl.unpersist()
    hierarchy.unpersist()


def _conf_snapshot(spark) -> dict[str, str]:
    return dict(spark.conf.getAll)


def test_every_group_agrees_to_1e_6(spark, bench_inputs):
    before = _conf_snapshot(spark)
    by_group: dict[str, list] = {}
    for v in variants(spark):
        by_group.setdefault(v.group, []).append(v)
    assert set(by_group) == {
        "vector_sum", "rollup", "skew_aggregate", "skew_join", "join", "shuffle_partitions"
    }  # fmt: skip
    assert all(len(vs) >= 2 for vs in by_group.values())
    for group in by_group.values():
        assert_group_agrees(spark, group, bench_inputs, tol=1e-6)
    assert _conf_snapshot(spark) == before


def test_rollup_firm_vector_matches_numpy(spark, bench_inputs):
    """The variants agreeing is not enough: the firm node equals the NumPy sum of all trades."""
    expected = np.sum([r.pnl for r in bench_inputs.pnl_vectors.select("pnl").collect()], axis=0)
    for v in variants(spark):
        if v.group != "rollup":
            continue
        firm = v.build(spark, bench_inputs).where("node_level = 'firm_id'").collect()
        assert len(firm) == 1
        np.testing.assert_allclose(firm[0].pnl, expected, rtol=1e-9, atol=1e-6)


def test_variants_differ_in_the_plan_where_they_claim_to(spark, bench_inputs):
    plans = {}
    for v in variants(spark):
        with_conf = v.with_conf(spark)
        with with_conf:
            df = v.build(spark, bench_inputs)
            plans[v.name] = df._jdf.queryExecution().executedPlan().toString()
    assert "Generate posexplode" in plans["explode_regroup"]
    assert "posexplode" not in plans["array_index_sum"]
    assert "Expand" in plans["grain_then_grouping_sets"]
    assert "Expand" not in plans["per_level_from_trades"]
    assert "SortMergeJoin" in plans["sort_merge"] and "BroadcastHashJoin" in plans["broadcast"]
    assert "SortMergeJoin" in plans["skew_join_plain"]


def test_report_has_a_measured_row_per_variant(spark, tmp_path):
    before = _conf_snapshot(spark)
    out = tmp_path / "report.md"
    run_benchmark(
        PROFILE,
        out,
        repeats=1,
        spark=spark,
        work_dir=tmp_path / "work",
        warmup_rounds=0,
        cleaner_wait_s=0,
    )
    assert _conf_snapshot(spark) == before
    text = out.read_text()
    names = [v.name for v in variants(spark)]
    for name in names:
        rows = [ln for ln in text.splitlines() if ln.startswith(f"| `{name}` |")]
        assert len(rows) == 1, name
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        # variant | median s | min-max s | shuffle read | shuffle write | peak task mem | operators
        median, spread, read, write = cells[1:5]
        assert re.fullmatch(r"\d+\.\d+", median), rows[0]
        assert re.fullmatch(r"\d+\.\d+-\d+\.\d+", spread), rows[0]
        assert re.fullmatch(r"[\d,]+\.\d+", read) and re.fullmatch(r"[\d,]+\.\d+", write)
    assert "Apple" in text or "CPU" in text
    assert f"{PROFILE.n_trades:,}" in text

