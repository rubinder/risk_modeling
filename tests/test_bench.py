"""Benchmark: every group's variants agree, and the report has a measured row per variant.

Runs on the bench profile cut to 5,000 trades and 50 scenarios, so the skewed desk is present
but the suite stays fast (with 250 element sums every small query took far longer).
"""

import dataclasses
import datetime
import re

import httpx
import numpy as np
import pytest

from riskagg.bench.runner import _run_stats, assert_group_agrees, run_benchmark
from riskagg.bench.variants import Inputs, variants
from riskagg.data.market import make_exposures, make_pnl_vectors
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import PROFILES
from riskagg.paths import REPO_ROOT

PROFILE = dataclasses.replace(PROFILES["bench"], n_trades=5_000, n_scenarios=50)
AS_OF = datetime.date(2026, 11, 12)
STUDY_PAGE = REPO_ROOT / "docs/study/05-aggregation-engineering.md"
REPORT = REPO_ROOT / "docs/benchmarks/report.md"


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
    cpu = re.search(r"^\| CPU \| (.*) \|$", text, re.MULTILINE)
    assert cpu and cpu.group(1).strip() not in ("", "unknown"), cpu
    assert f"{PROFILE.n_trades:,}" in text


class _FakeApi:
    """The REST calls ``_run_stats`` makes; stage 2 is missing from the status store."""

    def executions(self, tag):
        return [{"id": 0, "status": "COMPLETED", "failedJobIds": [], "successJobIds": [0]}]

    def get(self, path, **params):
        if path == "/jobs/0":
            return {"stageIds": [1, 2]}
        if path == "/stages/2":
            request = httpx.Request("GET", "http://127.0.0.1/stages/2")
            raise httpx.HTTPStatusError("404", request=request, response=httpx.Response(404))
        if path == "/stages/1":
            return [{
                "status": "COMPLETE", "attemptId": 0, "shuffleReadBytes": 1,
                "shuffleWriteBytes": 1, "shuffleWriteRecords": 1, "numTasks": 1,
            }]  # fmt: skip
        if path.endswith("/taskSummary"):
            return {"peakExecutionMemory": [1.0, 1.0], "executorRunTime": [1.0, 1.0]}
        return {"planDescription": "== Physical Plan ==\n"}


def test_a_stage_missing_from_the_rest_api_fails_the_run():
    with pytest.raises(RuntimeError, match=r"no data for stages \[2\]"):
        _run_stats(_FakeApi(), "bench x", 1.0)


_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
# Sections of the study page that quote the report.
QUOTING_SECTIONS = ("## 6. What the benchmark measured", "## 7. What does not transfer")


def _section(page: str, heading: str) -> str:
    return page.split(heading, 1)[1].split("\n## ", 1)[0]


def _table(report: str) -> dict[str, list[str]]:
    """Report table cells by variant name."""
    rows = {}
    for ln in report.splitlines():
        m = re.match(r"\| `(\w+)` \|", ln)
        if m:
            rows[m.group(1)] = [c.strip() for c in ln.strip().strip("|").split("|")]
    return rows


def test_study_page_quotes_only_numbers_from_the_report():
    """Every number in the sections that quote the report appears in the report."""
    page = STUDY_PAGE.read_text()
    report_numbers = set(_NUMBER.findall(REPORT.read_text()))
    for heading in QUOTING_SECTIONS:
        quoted = set(_NUMBER.findall(_section(page, heading)))
        assert quoted, f"{heading} quotes no numbers"
        assert quoted <= report_numbers, (heading, sorted(quoted - report_numbers))


def test_study_page_quotes_the_right_variant():
    """Key figures sit next to the variant they belong to, as in the report's tables."""
    page = STUDY_PAGE.read_text()
    t = _table(REPORT.read_text())

    def med(v: str) -> str:
        return t[v][1]

    def write(v: str) -> str:
        return t[v][4]

    expected = [
        f"{med('explode_regroup')} s against {med('array_index_sum')} s",
        f"{med('per_level_from_trades')} s against {med('grain_then_grouping_sets')} s",
        (
            f"Plain {med('skew_agg_plain')} s, AQE settings {med('skew_agg_aqe_skew_join')} s, "
            f"salted {med('skew_agg_salted')} s"
        ),
        f"Plain {med('skew_join_plain')} s; AQE forced {med('skew_join_aqe_skew_join')} s",
        (
            f"shuffle write {write('skew_join_aqe_skew_join')} MiB\nagainst "
            f"{write('skew_join_plain')} MiB"
        ),
        f"{med('sort_merge')} s against {med('broadcast')} s",
        f"{med('static_200')} s at 200 static partitions, {med('sized_to_data')} s sized",
        f"shuffled {write('sort_merge')} MiB",
    ]
    flat = " ".join(page.split())
    for phrase in expected:
        assert " ".join(phrase.split()) in flat, phrase
