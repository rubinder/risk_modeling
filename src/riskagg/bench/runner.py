"""run_benchmark: time every variant, read its metrics from the Spark REST status API, report.

Each variant is checked against the others of its group on a sample, then run ``repeats``
times, all variants once per round in an order shuffled per round (seeded), so drift and the
variant that ran just before do not favour one of them.
A run is a ``noop`` write of the variant's DataFrame (every row computed, nothing written),
tagged with a job description. The metrics of that run come from the REST status API of the
driver's UI, bound to the loopback address:

- ``GET /sql`` (``description`` is the tag) gives the SQL execution and its job ids;
  ``GET /sql/{id}?planDescription=true`` gives the executed (final adaptive) plan.
- ``GET /jobs/{id}`` gives the stage ids; ``GET /stages/{id}`` gives each attempt's
  ``shuffleReadBytes``, ``shuffleWriteBytes``, ``shuffleWriteRecords`` and ``numTasks``
  (summed over the completed attempts).
- ``GET /stages/{id}/{attempt}/taskSummary?quantiles=0.5,1.0`` gives the largest per-task
  ``peakExecutionMemory`` (the maximum over the run's stages), and the longest
  ``executorRunTime`` with the median task of the same stage (how uneven that stage was).
"""

import datetime
import gc
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from riskagg.bench.variants import AQE_SKEW_CONF, SALT_BUCKETS, Inputs, Variant, variants
from riskagg.data.market import make_exposures, make_pnl_vectors
from riskagg.data.org import make_hierarchy, make_risk_factors, make_trades
from riskagg.data.profiles import Profile
from riskagg.spark import get_spark

AS_OF = datetime.date(2026, 11, 12)
TOLERANCE = 1e-6
SAMPLE_EVERY_NTH_BOOK = 20
MIB = 2**20
# After each run: a JVM GC, then this pause, so the ContextCleaner's deletion of the run's
# shuffle files (and the disk traffic it causes) does not overlap the next run. A shorter
# pause made the run after a large shuffle measurably slower.
CLEANER_WAIT_S = 5.0
UNEVEN_MIN_MS = 100  # stages faster than this are scheduling noise, not skew


# --- agreement ------------------------------------------------------------------------------


def _keyed(rows) -> dict[tuple, np.ndarray]:
    out = {}
    for r in rows:
        d = r.asDict()
        out[tuple(sorted((k, v) for k, v in d.items() if k != "pnl"))] = np.asarray(d["pnl"])
    if len(out) != len(rows):
        raise AssertionError("duplicate keys in a variant's result")
    return out


def assert_group_agrees(
    spark: SparkSession, group: Sequence[Variant], inputs: Inputs, tol: float = TOLERANCE
) -> int:
    """Every variant of ``group`` returns the same rows; vectors agree to ``tol`` relative to
    their magnitude (``|a - b| <= tol * max(1, |b|)``). Returns the number of rows compared."""
    results = {}
    for v in group:
        with v.with_conf(spark):
            results[v.name] = _keyed(v.build(spark, inputs).collect())
    ref_name, ref = next(iter(results.items()))
    for name, got in results.items():
        if got.keys() != ref.keys():
            raise AssertionError(f"{name} and {ref_name} return different keys")
        for k, b in ref.items():
            a = got[k]
            if a.shape != b.shape or np.any(np.abs(a - b) > tol * np.maximum(1.0, np.abs(b))):
                raise AssertionError(f"{name} and {ref_name} differ at {dict(k)}")
    return len(ref)


# --- REST status API ------------------------------------------------------------------------


class StatusApi:
    """The driver's REST status API, read on 127.0.0.1."""

    def __init__(self, spark: SparkSession):
        url = spark.sparkContext.uiWebUrl
        if not url:
            raise RuntimeError(
                "the benchmark reads the Spark REST status API, which needs spark.ui.enabled=true"
            )
        app = spark.sparkContext.applicationId
        self.base = f"http://127.0.0.1:{urlparse(url).port}/api/v1/applications/{app}"
        self.client = httpx.Client(timeout=60)

    def get(self, path: str, **params):
        r = self.client.get(self.base + path, params=params)
        r.raise_for_status()
        return r.json()

    def executions(self, tag: str, timeout: float = 120.0) -> list[dict]:
        """The finished SQL executions described by ``tag`` (the listener bus is asynchronous)."""
        deadline = time.monotonic() + timeout
        while True:
            found = [
                e for e in self.get("/sql", details="false", length=1_000_000)
                if e["description"] == tag
            ]  # fmt: skip
            if found and all(e["status"] != "RUNNING" for e in found):
                return found
            if time.monotonic() > deadline:
                raise TimeoutError(f"no finished SQL execution for {tag}")
            time.sleep(0.05)


@dataclass
class Run:
    seconds: float
    shuffle_read: int
    shuffle_write: int
    shuffle_write_records: int
    tasks: int
    peak_task_memory: int | None
    longest_task_ms: int | None
    longest_stage_median_ms: int | None
    # The most uneven stage (largest longest/median task ratio) among stages of two or more
    # tasks whose longest task took at least UNEVEN_MIN_MS: (longest task, median task) in ms.
    uneven_stage_ms: tuple[int, int] | None
    plan: str


def _run_stats(api: StatusApi, tag: str, seconds: float) -> Run:
    execs = api.executions(tag)
    failed = [e["id"] for e in execs if e["status"] != "COMPLETED" or e["failedJobIds"]]
    if failed:
        raise RuntimeError(f"{tag}: SQL executions {failed} failed")
    stage_ids = sorted(
        {s for e in execs for j in e["successJobIds"] for s in api.get(f"/jobs/{j}")["stageIds"]}
    )
    read = write = records = tasks = 0
    peaks: list[float] = []
    longest: list[tuple[float, float]] = []  # (max, median) task run time per stage
    uneven: list[tuple[float, float]] = []  # the same, for stages of two or more tasks
    missing: list[int] = []
    for sid in stage_ids:
        try:
            attempts = api.get(f"/stages/{sid}")
        except httpx.HTTPStatusError:
            missing.append(sid)  # e.g. evicted from the status store (spark.ui.retainedStages)
            continue
        for a in attempts:
            if a["status"] != "COMPLETE":
                continue
            read += a["shuffleReadBytes"]
            write += a["shuffleWriteBytes"]
            records += a["shuffleWriteRecords"]
            tasks += a["numTasks"]
            ts = api.get(f"/stages/{sid}/{a['attemptId']}/taskSummary", quantiles="0.5,1.0")
            if ts.get("peakExecutionMemory"):
                peaks.append(ts["peakExecutionMemory"][-1])
            if ts.get("executorRunTime"):
                pair = (ts["executorRunTime"][-1], ts["executorRunTime"][0])
                longest.append(pair)
                if a["numTasks"] >= 2 and pair[0] >= UNEVEN_MIN_MS:
                    uneven.append(pair)
    if missing:
        raise RuntimeError(
            f"{tag}: the REST status API returned no data for stages {missing}; the run's "
            "shuffle, task and memory totals would be undercounted"
        )
    root = min(e["id"] for e in execs)
    plan = api.get(f"/sql/{root}", details="false", planDescription="true")["planDescription"]
    return Run(
        seconds, read, write, records, tasks,
        int(max(peaks)) if peaks else None,
        int(max(longest)[0]) if longest else None,
        int(max(longest)[1]) if longest else None,
        _most_uneven(uneven),
        plan,
    )  # fmt: skip


def _most_uneven(stages: list[tuple[float, float]]) -> tuple[int, int] | None:
    if not stages:
        return None
    hi, med = max(stages, key=lambda p: p[0] / max(p[1], 1.0))
    return int(hi), int(med)


def _measure_once(
    spark: SparkSession, api: StatusApi, v: Variant, inputs: Inputs, cleaner_wait_s: float
) -> Run:
    sc = spark.sparkContext
    tag = f"bench {v.name} {uuid.uuid4().hex[:12]}"
    sc.setJobGroup(tag, tag)
    try:
        with v.with_conf(spark):
            df = v.build(spark, inputs)
            start = time.perf_counter()
            df.write.format("noop").mode("overwrite").save()
            seconds = time.perf_counter() - start
    finally:
        sc.setLocalProperty("spark.jobGroup.id", None)
        sc.setLocalProperty("spark.job.description", None)
    run = _run_stats(api, tag, seconds)
    # Let the JVM drop this run's shuffle files before the next run (ContextCleaner).
    del df
    gc.collect()
    spark._jvm.System.gc()
    time.sleep(cleaner_wait_s)
    return run


# --- plans ----------------------------------------------------------------------------------

_SKIP_OPS = {
    "ColumnarToRow", "InputAdapter", "WholeStageCodegen", "ShuffleQueryStage",
    "BroadcastQueryStage", "ResultQueryStage", "TableCacheQueryStage", "Project", "Filter",
}  # fmt: skip


def final_plan(plan_description: str) -> list[str]:
    """The executed plan's operator tree: the adaptive final plan, ids and statistics removed.

    From the node details it adds each shuffle ``Exchange``'s partitioning, the directory each
    ``Scan parquet`` reads and the function of each ``Generate``.
    """
    head, _, details = plan_description.partition("\n\n\n")
    lines = head.splitlines()
    if any("== Final Plan ==" in ln for ln in lines):
        start = next(i for i, ln in enumerate(lines) if "== Final Plan ==" in ln) + 1
        end = next((i for i, ln in enumerate(lines) if "== Initial Plan ==" in ln), len(lines))
        lines = lines[start:end]
    else:
        lines = lines[1:]  # drop "== Physical Plan =="
    args = {}
    for block in details.split("\n\n"):
        m = re.match(r"\((\d+)\) (\S+)", block.strip())
        a = re.search(r"^Arguments: (.*)$", block, re.MULTILINE)
        loc = re.search(r"^Location: \S+ \[(.*?)\]", block, re.MULTILINE)
        if not m:
            continue
        if a and m.group(2) == "Exchange":
            args[m.group(1)] = re.sub(r"#\d+L?", "", a.group(1).split(", ENSURE")[0])
        elif a and m.group(2) == "Generate":
            args[m.group(1)] = a.group(1).split("(", 1)[0]
        elif loc and m.group(2) == "Scan":
            args[m.group(1)] = Path(loc.group(1).split(",")[0]).name
    if lines:
        indent = len(lines[0]) - len(lines[0].lstrip())
        lines = [ln[indent:] for ln in lines]
    out = []
    for ln in lines:
        m = re.search(r" \((\d+)\)", ln)
        clean = re.sub(r", Statistics\(.*?\)$", "", ln)
        clean = re.sub(r" \(\d+\)", "", clean).rstrip()
        if m and m.group(1) in args:
            clean += f" {args[m.group(1)]}"
        out.append(clean.rstrip())
    return out


def key_operators(tree: list[str]) -> str:
    """Counted operators of the final plan, in plan order, without the bookkeeping nodes."""
    counts: dict[str, int] = {}
    for ln in tree:
        body = re.sub(r"^[\s:|+\-*]*", "", ln)
        m = re.match(r"([A-Za-z]+(?:\(skew=true\))?)(?: parquet)?(,\s*(\w+))?", body)
        if not m:
            continue
        name = m.group(1)
        if name == "Scan":
            name = "Scan parquet " + body.split()[-1]
        elif name == "Generate":
            name = "Generate " + body.split()[-1]
        if name in _SKIP_OPS:
            continue
        if name == "AQEShuffleRead" and m.group(3):
            name = f"AQEShuffleRead({m.group(3)})"
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{k} ×{n}" if n > 1 else k for k, n in counts.items())


# --- data, machine and session --------------------------------------------------------------


def bench_session() -> SparkSession:
    """A local session with the UI (and so the REST status API) on 127.0.0.1, random port.

    The UI's bind address comes from ``SPARK_LOCAL_IP`` (read when the JVM starts); the driver
    is bound to loopback too.
    """
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    spark = get_spark(
        "riskagg-bench",
        spark__driver__bindAddress="127.0.0.1",
        spark__driver__host="127.0.0.1",
        spark__ui__enabled="true",
        spark__ui__port="0",
        spark__ui__showConsoleProgress="false",
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.parquet"))


def _prepare(spark: SparkSession, p: Profile, work: Path) -> tuple[Inputs, dict]:
    """Generate the profile's P&L vectors and hierarchy once, as Parquet, and read them back."""
    trades = make_trades(spark, p, AS_OF)
    rf = make_risk_factors(spark, p)
    pnl = make_pnl_vectors(spark, p, trades, make_exposures(spark, p, trades, rf), rf)
    pnl.write.mode("overwrite").parquet(str(work / "pnl_vectors"))
    make_hierarchy(spark, p).write.mode("overwrite").parquet(str(work / "hierarchy"))
    pnl_vectors = spark.read.parquet(str(work / "pnl_vectors"))
    hierarchy = spark.read.parquet(str(work / "hierarchy"))
    facts = {
        "pnl_rows": pnl_vectors.count(),
        "pnl_bytes": _dir_bytes(work / "pnl_vectors"),
        "pnl_files": len(list((work / "pnl_vectors").glob("*.parquet"))),
    }
    return Inputs(pnl_vectors, hierarchy, p.n_scenarios), facts


def _sample(spark: SparkSession, inputs: Inputs, work: Path) -> Inputs:
    """Every ``SAMPLE_EVERY_NTH_BOOK``-th book (BOOK0001 is in the oversized desk), as its own
    Parquet copy: a variant's plan on the sample is then the same as on the full data, so the
    agreement check also compiles (warms) the generated code the timed runs use."""
    book_no = F.substring("book_id", 5, 4).cast("int")
    keep = (book_no - 1) % SAMPLE_EVERY_NTH_BOOK == 0
    path = str(work / "sample_pnl_vectors")
    inputs.pnl_vectors.where(keep).write.mode("overwrite").parquet(path)
    return Inputs(spark.read.parquet(path), inputs.hierarchy, inputs.n_scenarios)


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str:
    model = _sysctl("machdep.cpu.brand_string")
    if model:
        return model
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    value = _sysctl("hw.memsize")
    if value:
        return int(value)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def machine_facts(spark: SparkSession) -> dict[str, str]:
    jvm = spark._jvm.System
    ram = _ram_bytes()
    conf = spark.sparkContext.getConf()
    return {
        "CPU": _cpu_model(),
        "Logical cores": str(os.cpu_count()),
        "RAM": f"{ram / 2**30:.0f} GiB" if ram else "unknown",
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "Java": f"{jvm.getProperty('java.version')} ({jvm.getProperty('java.vm.name')})",
        "Spark": spark.version,
        "Master": spark.sparkContext.master,
        "Driver memory": conf.get("spark.driver.memory", "1g (Spark default)"),
        "Default parallelism": str(spark.sparkContext.defaultParallelism),
        "Session shuffle partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "AQE": spark.conf.get("spark.sql.adaptive.enabled"),
    }


# --- report ---------------------------------------------------------------------------------


@dataclass
class Measurement:
    variant: Variant
    runs: list[Run] = field(default_factory=list)

    @property
    def times(self) -> list[float]:
        return [r.seconds for r in self.runs]

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def spread(self) -> float:
        return max(self.times) - min(self.times)

    @property
    def typical(self) -> Run:
        """The run whose time is the median (the lower middle for an even count)."""
        return sorted(self.runs, key=lambda r: r.seconds)[(len(self.runs) - 1) // 2]

    @property
    def peak_task_memory(self) -> int | None:
        peaks = [r.peak_task_memory for r in self.runs if r.peak_task_memory is not None]
        return max(peaks) if peaks else None

    @property
    def tree(self) -> list[str]:
        return final_plan(self.typical.plan)


def _mib(n: int | None) -> str:
    return "not available" if n is None else f"{n / MIB:,.1f}"


def _secs(ms: int | None) -> str:
    return "not available" if ms is None else f"{ms / 1000:.2f}"


def _row(m: Measurement) -> str:
    t = m.typical
    cells = [
        f"`{m.variant.name}`",
        f"{m.median:.2f}",
        f"{min(m.times):.2f}-{max(m.times):.2f}",
        _mib(t.shuffle_read),
        _mib(t.shuffle_write),
        _mib(m.peak_task_memory),
        f"{_secs(t.longest_task_ms)} ({_secs(t.longest_stage_median_ms)})",
        f"{t.tasks:,}",
        key_operators(m.tree),
    ]
    return "| " + " | ".join(cells) + " |"


def compare(base: Measurement, other: Measurement) -> str:
    """One sentence comparing ``other`` with ``base``; within noise if the spread covers it."""
    a, b = base.variant.name, other.variant.name
    diff = other.median - base.median
    noise = max(base.spread, other.spread)
    head = f"`{b}` {other.median:.2f} s against `{a}` {base.median:.2f} s"
    if len(base.runs) > 1 and abs(diff) <= noise:
        return (
            f"{head}: the difference ({abs(diff):.2f} s) is within the run-to-run spread "
            f"({noise:.2f} s), so no difference is measured."
        )
    if diff < 0:
        return f"{head}: {base.median / other.median:.2f}× faster."
    return f"{head}: {other.median / base.median:.2f}× slower."


def noise_controls(measurements: list[Measurement]) -> list[str]:
    """Variants that ran the same code to the same executed plan, compared with each other."""
    same: dict[tuple, list[Measurement]] = {}
    for m in measurements:
        same.setdefault((m.variant.build, tuple(m.tree)), []).append(m)
    out = []
    for ms in same.values():
        if len(ms) < 2:
            continue
        medians = [m.median for m in ms]
        out.append(
            ", ".join(f"`{m.variant.name}`" for m in ms)
            + " ran the same query to the same executed plan: medians "
            + ", ".join(f"{x:.2f}" for x in medians)
            + f" s, {max(medians) - min(medians):.2f} s apart. A difference of that size between"
            " other variants is not evidence of a difference either."
        )
    return out


def _has(m: Measurement, op: str) -> bool:
    return any(op in ln for ln in m.tree)


def _count(m: Measurement, op: str) -> int:
    return _n(m.tree, op)


def _above(m: Measurement, op: str) -> list[str] | None:
    """The plan lines between the root and the first line containing ``op``."""
    for i, ln in enumerate(m.tree):
        if op in ln:
            return m.tree[:i]
    return None


def _n(lines: list[str], op: str) -> int:
    return sum(1 for ln in lines if re.search(rf"(^|[\s\-*]){re.escape(op)}\b", ln))


def _uneven(m: Measurement) -> str:
    u = m.typical.uneven_stage_ms
    if u is None:
        return "not available"
    return f"{_secs(u[0])} s against a median task of {_secs(u[1])} s"


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def _findings(group: str, ms: dict[str, Measurement]) -> list[str]:
    """Plain statements generated from the measurements and the executed plans."""
    out: list[str] = []
    if group == "vector_sum":
        e, a = ms["explode_regroup"], ms["array_index_sum"]
        out.append(
            f"The exploded form wrote {e.typical.shuffle_write_records:,} shuffle records "
            f"({_mib(e.typical.shuffle_write)} MiB) against {a.typical.shuffle_write_records:,} "
            f"({_mib(a.typical.shuffle_write)} MiB): `Generate posexplode` multiplies the rows by "
            "the scenario count before the first aggregate, and the second aggregate "
            f"(`collect_list`) runs as {'ObjectHashAggregate' if _has(e, 'ObjectHashAggregate') else 'a non-hash aggregate'}"
            " because its buffer is not fixed-width."
        )  # fmt: skip
    elif group == "rollup":
        p, g = ms["per_level_from_trades"], ms["grain_then_grouping_sets"]
        out.append(
            f"Per-level reads the trade vectors {_times(_count(p, 'Scan parquet pnl_vectors'))} "
            f"and shuffles {_mib(p.typical.shuffle_write)} MiB; the engine reads them "
            f"{_times(_count(g, 'Scan parquet pnl_vectors'))} and shuffles "
            f"{_mib(g.typical.shuffle_write)} MiB, "
            f"with {_count(g, 'Expand')} `Expand` over the grain rows. "
            "The grain is not cached here, so the engine's variant pays for building it."
        )
    elif group == "skew_aggregate":
        p, q, s = ms["skew_agg_plain"], ms["skew_agg_aqe_skew_join"], ms["skew_agg_salted"]
        same = key_operators(p.tree) == key_operators(q.tree)
        out.append(
            "The AQE skew settings left the plan "
            + ("unchanged" if same else "changed")
            + (": AQE's skew handling rewrites joins, and this query has no shuffled join." if same else ".")
        )  # fmt: skip
        out.append(
            f"The plain plan aggregates partially before its shuffle (`HashAggregate` ×"
            f"{_count(p, 'HashAggregate')} around one `Exchange`), so each map task sends at most "
            f"one row per desk: {p.typical.shuffle_write_records:,} shuffle records "
            f"({_mib(p.typical.shuffle_write)} MiB). The oversized desk never reaches the "
            f"reducer as raw rows. Salting raised that to {s.typical.shuffle_write_records:,} "
            f"records and added an aggregation pass."
        )
    elif group == "skew_join":
        p, q, s = ms["skew_join_plain"], ms["skew_join_aqe_skew_join"], ms["skew_join_salted"]
        for m in (p, q, s):
            skew = _has(m, "skew=true")
            out.append(
                f"`{m.variant.name}`: "
                + ("the sort-merge join was split for skew (`SortMergeJoin(skew=true)`)" if skew
                   else "no skew split in the plan")
                + f"; most uneven stage: longest task {_uneven(m)}; "
                f"shuffle write {_mib(m.typical.shuffle_write)} MiB."
            )  # fmt: skip
        out.append(
            "The join's shuffle carries whole trade vectors (no partial aggregation can happen "
            "before a join), so the oversized desk lands in one reduce partition."
        )
        above = _above(q, "SortMergeJoin(skew=true)")
        if above is not None and _n(above, "Exchange") and _n(above, "HashAggregate") == 1:
            out.append(
                "Forcing the skew split cost a shuffle: in the plain plan the aggregate sits on "
                "the join's output with no `Exchange` between them, because the join already "
                "partitions by desk. Splitting the skewed partition breaks that partitioning, so "
                "AQE inserted an `Exchange` above `SortMergeJoin(skew=true)`, and the single "
                "`HashAggregate` above it has no partial stage below the new shuffle: the joined "
                f"vectors were shuffled twice ({_mib(q.typical.shuffle_write)} MiB written "
                f"against {_mib(p.typical.shuffle_write)} MiB)."
            )
    elif group == "join":
        smj, bhj = ms["sort_merge"], ms["broadcast"]
        out.append(
            f"The sort-merge join shuffled {_mib(smj.typical.shuffle_write)} MiB "
            f"({smj.typical.shuffle_write_records:,} records), the trade vectors themselves; the "
            f"broadcast join shuffled {_mib(bhj.typical.shuffle_write)} MiB "
            f"({bhj.typical.shuffle_write_records:,} records), only the partially aggregated grain."
        )
    elif group == "shuffle_partitions":
        for m in ms.values():
            out.append(
                f"`{m.variant.name}`: {m.typical.tasks:,} tasks, shuffle read "
                f"{_mib(m.typical.shuffle_read)} MiB, operators {key_operators(m.tree)}."
            )
    return out


GROUP_TITLES = {
    "vector_sum": "Element-wise vector sum",
    "rollup": "Roll-up of the org hierarchy",
    "skew_aggregate": "Key skew in a hash aggregate",
    "skew_join": "Key skew in a sort-merge join",
    "join": "Hierarchy join",
    "shuffle_partitions": "Shuffle-partition sizing",
}


def render_report(
    p: Profile, facts: dict, machine: dict, measurements: list[Measurement], repeats: int,
    sample_rows: dict[str, int], warmup_rounds: int, cleaner_wait_s: float,
) -> str:  # fmt: skip
    lines = [
        "# Benchmark report",
        "",
        (
            f"Generated by `riskagg bench --profile {p.name}` on {datetime.date.today().isoformat()}."
            " Every number below was produced by `src/riskagg/bench/runner.py` on the machine"
            " described; nothing is written by hand. This is Spark in local mode on one machine:"
            " the driver is the only executor, and every shuffle goes through the local disk, not"
            " the network."
        ),
        "",
        "## Machine and session",
        "",
        "| Item | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in machine.items()],
        "",
        "## Data",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Profile | `{p.name}` |",
        f"| Trades | {p.n_trades:,} |",
        f"| Books / desks / business lines | {p.n_books:,} / {p.n_desks:,} / {p.n_business_lines:,} |",
        f"| Share of trades in the oversized desk | {p.skew_desk_share:.0%} |",
        f"| Scenarios per vector | {p.n_scenarios:,} |",
        f"| `pnl_vectors` rows | {facts['pnl_rows']:,} |",
        f"| `pnl_vectors` on disk | {_mib(facts['pnl_bytes'])} MiB in {facts['pnl_files']:,} Parquet files |",
        (
            f"| Timed runs per variant | {repeats}, after {warmup_rounds} untimed warm-up"
            " round(s); each round runs every variant once, in a seeded random order |"
        ),
        "",
        "## Method",
        "",
        (
            '- Each run is `df.write.format("noop").mode("overwrite").save()`: every row is'
            " computed and nothing is written. Wall time is measured around that call. Inputs are"
            " read from Parquet in every run; nothing is cached."
        ),
        (
            "- The median of the runs is reported, with the fastest and slowest. A difference"
            " smaller than the larger spread of the two variants is reported as within noise."
        ),
        (
            "- Shuffle bytes, shuffle records and task counts are summed over the run's completed"
            " stages (`GET /api/v1/applications/{app}/stages/{stage}`: `shuffleReadBytes`,"
            " `shuffleWriteBytes`, `shuffleWriteRecords`, `numTasks`), from the median run. Peak"
            " task execution memory is the largest `peakExecutionMemory` of any task of any stage"
            " of any run (`GET .../stages/{stage}/{attempt}/taskSummary?quantiles=0.5,1.0`). The"
            " longest task is the largest `executorRunTime` of the median run, shown with the"
            " median task of the same stage, which says how uneven that stage was. The skew"
            " findings name the most uneven stage: the largest ratio of longest to median task"
            " among stages of two or more tasks whose longest task took at least"
            f" {UNEVEN_MIN_MS} ms. Peak memory"
            " is Spark's execution memory (aggregation hash maps, sort buffers), not the JVM heap."
        ),
        (
            f"- After every run: a JVM garbage collection and a {cleaner_wait_s:g} s pause"
            " (not timed), so Spark's deletion of that run's shuffle files does not overlap the"
            " next run. One untimed warm-up round runs first, so code generation and the page"
            " cache are warm."
        ),
        (
            "- The plan is the executed plan of the median run, the adaptive final plan"
            " (`GET .../sql/{execution}?planDescription=true`)."
        ),
        "- Before timing, the variants of each group were run on a sample (every"
        f" {SAMPLE_EVERY_NTH_BOOK}th book, including one book of the oversized desk) and their"
        f" rows compared: the same keys, vectors equal to {TOLERANCE:g} relative to their size. "
        + ", ".join(f"`{g}` {n:,} rows" for g, n in sample_rows.items())
        + ".",
        f"- Salted variants use {SALT_BUCKETS} salt values. The AQE skew variants set "
        + ", ".join(f"`{k}={v}`" for k, v in AQE_SKEW_CONF.items())
        + ".",
        "",
        (
            "Columns: median and min-max wall time in seconds; shuffle read and write in MiB; peak"
            " task execution memory in MiB; longest task (and its stage's median task) in"
            " seconds; tasks; key operators of the"
            " executed plan (bookkeeping nodes such as `Project` and query stages left out)."
        ),
        "",
    ]
    groups: dict[str, list[Measurement]] = {}
    for m in measurements:
        groups.setdefault(m.variant.group, []).append(m)
    for group, ms in groups.items():
        lines += [f"## `{group}`: {GROUP_TITLES.get(group, group)}", ""]
        lines += [f"- `{m.variant.name}`: {m.variant.description}"
                  + (f" (conf: {', '.join(f'`{k}={v}`' for k, v in m.variant.conf.items())})"
                     if m.variant.conf else "")
                  for m in ms]  # fmt: skip
        lines += [
            "",
            (
                "| Variant | Median s | Min-max s | Shuffle read MiB | Shuffle write MiB "
                "| Peak task exec. memory MiB | Longest task s (stage median) | Tasks | Key operators |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            *[_row(m) for m in ms],
            "",
        ]
        lines += [f"- {compare(ms[0], m)}" for m in ms[1:]]
        lines += [f"- {s}" for s in _findings(group, {m.variant.name: m for m in ms})]
        lines.append("")
    lines += ["## Noise controls", ""]
    lines += [f"- {c}" for c in noise_controls(measurements)] or ["- none"]
    lines += ["", "## Executed plans", ""]
    for m in measurements:
        lines += [f"### `{m.variant.name}`", "", "```", *m.tree, "```", ""]
    return "\n".join(lines)


def run_benchmark(
    profile: Profile,
    out_path: Path,
    repeats: int = 3,
    spark: SparkSession | None = None,
    work_dir: Path | None = None,
    warmup_rounds: int = 1,
    cleaner_wait_s: float = CLEANER_WAIT_S,
) -> list[Measurement]:
    """Generate ``profile``'s data, check and time every variant, write the report."""
    owned = spark is None
    spark = spark or bench_session()
    work = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="riskagg-"))
    try:
        api = StatusApi(spark)
        inputs, facts = _prepare(spark, profile, work)
        all_variants = variants(spark)
        groups: dict[str, list[Variant]] = {}
        for v in all_variants:
            groups.setdefault(v.group, []).append(v)
        sample = _sample(spark, inputs, work)
        sample_rows = {g: assert_group_agrees(spark, vs, sample) for g, vs in groups.items()}
        measurements = [Measurement(v) for v in all_variants]
        for round_no in range(warmup_rounds + repeats):
            order = list(measurements)
            random.Random(round_no).shuffle(order)  # a different, reproducible order per round
            for m in order:
                run = _measure_once(spark, api, m.variant, inputs, cleaner_wait_s)
                if round_no >= warmup_rounds:
                    m.runs.append(run)
        report = render_report(
            profile, facts, machine_facts(spark), measurements, repeats, sample_rows,
            warmup_rounds, cleaner_wait_s,
        )  # fmt: skip
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n")
        return measurements
    finally:
        if work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
        if owned:
            spark.stop()
