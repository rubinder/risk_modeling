"""Audit: the run manifest and the reconciliation of curated outputs against raw inputs.

``reconcile`` reads what was written to disk, never in-memory frames. ``CHECKS`` is the
extension point: later tasks append ``(ctx, spark) -> Check`` functions for stress, FRTB, CCAR.
"""

import itertools
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from riskagg.context import RunContext
from riskagg.conventions import es_from_pnl, var_from_pnl
from riskagg.engine.spec import AggregationSpec, load_specs
from riskagg.engine.vectors import GRAIN_DIMS, node_level, unique_sets
from riskagg.measures import ccar, frtb_ima
from riskagg.measures.stress import load_scenarios
from riskagg.paths import REPO_ROOT
from riskagg.semantic.model import Measure, load_model
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


def _grain_cells(df: DataFrame, value: str) -> DataFrame:
    """``df``'s P&L summed per reporting-grain cell, window and scenario."""
    return df.groupBy(*GRAIN_DIMS, "window", "scenario_idx").agg(F.sum("pnl").alias(value))


def check_pnl_scenarios_match_raw_per_grain_cell(ctx: RunContext, spark: SparkSession) -> Check:
    """Every curated ``pnl_scenarios`` cell equals the raw trade vectors of that cell.

    A cell is a reporting-grain node (book, its org and booking attributes, asset class,
    currency, risk class), a window and a scenario. The raw side joins the raw trade vectors
    to the raw hierarchy itself, so P&L moved between books, or a row relabelled to another
    book or desk, fails here even when the firm total is unchanged. The firm vector equalling
    the sum of the book vectors follows.
    """
    name = "pnl_scenarios_equal_raw_per_grain_cell"
    raw = spark.read.parquet(str(ctx.raw("pnl_vectors")))
    hier_cols = [c for c in GRAIN_DIMS if c not in raw.columns]
    hierarchy = spark.read.parquet(str(ctx.raw("hierarchy"))).select("book_id", *hier_cols)
    exploded = raw.join(F.broadcast(hierarchy), "book_id", "left").select(
        *GRAIN_DIMS, "window", F.posexplode("pnl").alias("scenario_idx", "pnl")
    )
    curated = spark.read.parquet(str(ctx.curated("pnl_scenarios")))
    keys = [*GRAIN_DIMS, "window", "scenario_idx"]
    joined = _grain_cells(exploded, "raw_pnl").join(
        _grain_cells(curated, "curated_pnl"), keys, "full_outer"
    )
    joined = joined.withColumn("diff", F.col("raw_pnl") - F.col("curated_pnl"))
    stats = joined.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum((F.col("raw_pnl").isNull() | F.col("curated_pnl").isNull()).cast("int")).alias("miss"),
        F.sum((~_within("diff", "raw_pnl")).cast("int")).alias("bad"),
        F.max(F.abs("diff")).alias("max_diff"),
    ).first()
    missing, bad, n = stats["miss"] or 0, stats["bad"] or 0, stats["n"]
    ok = n > 0 and missing == 0 and bad == 0
    return Check(
        name, ok, f"{n} (grain cell, window, scenario) points, {missing} on one side only, "
        f"{bad} outside tolerance, max abs diff {stats['max_diff']}"
    )  # fmt: skip


_DERIVED = ("sum_children", "diversification", "incremental")


def _vector_value(measure: Measure, pnl: np.ndarray) -> float:
    fn = var_from_pnl if measure.function == "var" else es_from_pnl
    horizon = math.sqrt(float(measure.params.get("horizon_days", 1)))
    return fn(pnl, float(measure.params["level"])) * horizon


def _node_vectors(
    pnl: DataFrame, dims: list[str], filters: Mapping[str, Sequence[str]]
) -> dict[tuple, np.ndarray]:
    """``{(window, *dim values): vector}`` from one aggregation of curated ``pnl_scenarios``."""
    for d, values in filters.items():
        pnl = pnl.where(F.col(d).isin(list(values)))
    rows = (
        pnl.groupBy("window", *dims, "scenario_idx").agg(F.sum("pnl").alias("pnl")).collect()
    )
    n = 1 + max((r["scenario_idx"] for r in rows), default=-1)
    out: dict[tuple, np.ndarray] = {}
    for r in rows:
        key = (r["window"], *(r[d] for d in dims))
        out.setdefault(key, np.full(n, np.nan))[r["scenario_idx"]] = r["pnl"]
    return out


def _expected_vector_rows(
    measure: Measure, spec: AggregationSpec, org: tuple[str, ...], vectors: dict
) -> dict[tuple, float]:
    """``{(measure name, node_level, node key): value}`` a spec should have stored.

    ``vectors[node_level]`` holds that level's node vectors; ``node key`` is the tuple of the
    level's dimension values. Diversification rows pair consecutive org levels of the spec,
    shortest first, as the engine does.
    """
    window = measure.params["window"]
    out: dict[tuple, float] = {}
    for gs in unique_sets(spec.grouping_sets):
        level = node_level(gs)
        for (w, *key), vec in vectors[level].items():
            if w == window:
                out[(measure.name, level, tuple(key))] = _vector_value(measure, vec)
    if not spec.diversification:
        return out
    by_len = {len(gs): tuple(gs) for gs in spec.grouping_sets if tuple(gs) == org[: len(gs)]}
    levels = [by_len[k] for k in sorted(by_len)]
    for parent, child in itertools.pairwise(levels):
        p_level, c_level = node_level(parent), node_level(child)
        parents = {tuple(k): v for (w, *k), v in vectors[p_level].items() if w == window}
        children = {tuple(k): v for (w, *k), v in vectors[c_level].items() if w == window}
        for pkey, pvec in parents.items():
            standalone = sum(
                _vector_value(measure, c) for ckey, c in children.items()
                if ckey[: len(pkey)] == pkey
            )  # fmt: skip
            out[(f"{measure.name}__sum_children", p_level, pkey)] = standalone
            out[(f"{measure.name}__diversification", p_level, pkey)] = (
                standalone - _vector_value(measure, pvec)
            )
        for ckey, cvec in children.items():
            pvec = parents[ckey[: len(parent)]]
            out[(f"{measure.name}__incremental", c_level, ckey)] = _vector_value(
                measure, pvec
            ) - _vector_value(measure, pvec - cvec)
    return out


def check_vector_measures_from_pnl_scenarios(ctx: RunContext, spark: SparkSession) -> Check:
    """Every vector-measure row in ``risk_metrics`` equals a recomputation from ``pnl_scenarios``.

    That covers each measure at every node of its specification, and the ``__sum_children``,
    ``__diversification`` and ``__incremental`` rows. The curated P&L is summed in Spark once
    per (node level, filter set), with the specification's filters applied, and the quantile
    convention of ``conventions.py`` is applied in NumPy; the diversification rows are rebuilt
    here, not by the engine's code. Rows missing from or extra to ``risk_metrics`` fail too.
    """
    name = "vector_measures_match_pnl_scenarios"
    model = load_model()
    specs = {s.name: s for s in load_specs(None, model)}
    vector = {m for m, v in model.measures.items() if v.measure_class == "vector"}
    names = [*vector, *(f"{m}__{d}" for m in vector for d in _DERIVED)]
    stored_rows = (
        spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where(F.col("measure").isin(names))
        .collect()
    )
    stored = {
        (r["aggregation"], r["measure"], r["node_level"],
         tuple(r[d] for d in r["node_level"].split("|"))): r["value"]
        for r in stored_rows
    }  # fmt: skip
    problems: list[str] = []
    expected: dict[tuple, float] = {}
    pnl = spark.read.parquet(str(ctx.curated("pnl_scenarios")))
    cache: dict[tuple, dict] = {}
    for agg in sorted({k[0] for k in stored}):
        spec = specs.get(agg)
        if spec is None or spec.measure not in vector:
            problems.append(f"aggregation '{agg}' is not a vector specification")
            continue
        fkey = tuple(sorted(spec.filters.items()))
        vectors = {}
        for gs in unique_sets(spec.grouping_sets):
            if (fkey, gs) not in cache:
                cache[(fkey, gs)] = _node_vectors(pnl, list(gs), spec.filters)
            vectors[node_level(gs)] = cache[(fkey, gs)]
        rows = _expected_vector_rows(
            model.measure(spec.measure), spec, model.hierarchies["org"], vectors
        )
        expected |= {(agg, *k): v for k, v in rows.items()}
    for key in sorted(set(stored) - set(expected)):
        problems.append(f"{key}: stored but not expected")
    for key in sorted(set(expected) - set(stored)):
        problems.append(f"{key}: expected but not stored")
    for key, exp in expected.items():
        got = stored.get(key)
        if got is not None and not abs(got - exp) <= TOLERANCE * max(1.0, abs(exp)):
            problems.append(f"{key}: stored {got} vs recomputed {exp}")
    detail = f"{len(expected)} vector-measure rows recomputed"
    if problems:
        detail += f"; {len(problems)} problems: " + "; ".join(problems[:5])
    return Check(name, bool(expected) and not problems, detail)


def check_no_duplicate_nodes(ctx: RunContext, spark: SparkSession) -> Check:
    """No ``(node, measure)`` appears twice in ``risk_metrics``."""
    name = "no_duplicate_node_measure"
    df = spark.read.parquet(str(ctx.curated("risk_metrics")))
    key = ["measure", "node_level", *GRAIN_DIMS, "scenario"]
    dupes = df.groupBy(*key).count().where(F.col("count") > 1).count()
    return Check(name, dupes == 0, f"{dupes} duplicated (node, measure) keys")


def check_stress_firm_pnl_is_sum_of_trades(ctx: RunContext, spark: SparkSession) -> Check:
    """Firm stress P&L per scenario in ``stress_results`` equals the sum over the raw trades.

    The raw side sums the raw sensitivities per risk factor (a few dozen rows, not a trade
    count) and applies each scenario's shocks in NumPy. P&L is linear in delta, gamma and vega,
    so the per-factor sums give the same total as summing trade by trade. It also checks the
    firm ``worst_stress_loss`` row against the worst of those per-scenario totals.
    """
    name = "stress_firm_pnl_equals_sum_of_trades"
    scenarios = load_scenarios()
    by_factor = {
        r["risk_factor_id"]: r
        for r in spark.read.parquet(str(ctx.raw("sensitivities")))
        .groupBy("risk_factor_id")
        .agg(F.sum("delta").alias("delta"), F.sum("gamma").alias("gamma"), F.sum("vega").alias("vega"))
        .collect()
    }  # fmt: skip
    factors = spark.read.parquet(str(ctx.raw("risk_factors"))).collect()
    unknown = sorted(set(by_factor) - {f["risk_factor_id"] for f in factors})
    expected: dict[str, float] = {}
    for sc in scenarios:
        total = 0.0
        for f in factors:
            agg = by_factor.get(f["risk_factor_id"])
            if agg is None:
                continue
            x = sc.shock_for(f["risk_class"], f["factor_type"], f["bucket"], f["currency"])
            if f["factor_type"] == "DELTA":
                total += agg["delta"] * x + 0.5 * agg["gamma"] * x * x
            else:
                total += agg["vega"] * x
        expected[sc.name] = total
    stored = {
        r["scenario"]: r["pnl"]
        for r in spark.read.parquet(str(ctx.curated("stress_results")))
        .groupBy("scenario")
        .agg(F.sum("pnl").alias("pnl"))
        .collect()
    }
    problems = []
    if unknown:
        problems.append(f"sensitivities for unknown risk factors {unknown}")
    if set(stored) != set(expected):
        problems.append(f"scenarios {sorted(stored)} stored vs {sorted(expected)} defined")
    for sc, exp in expected.items():
        got = stored.get(sc)
        if got is not None and abs(got - exp) > TOLERANCE * max(1.0, abs(exp)):
            problems.append(f"{sc}: stored {got} vs raw {exp}")
    worst = (
        spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where((F.col("measure") == "worst_stress_loss") & (F.col("node_level") == "firm_id"))
        .select("scenario", "value")
        .collect()
    )
    if len(worst) != 1:
        problems.append(f"{len(worst)} firm worst_stress_loss rows, expected 1")
    elif expected:
        scen, loss = min(
            ((n, -v) for n, v in sorted(expected.items())), key=lambda t: (-t[1], t[0])
        )
        if worst[0]["scenario"] != scen or abs(worst[0]["value"] - loss) > TOLERANCE * max(
            1.0, abs(loss)
        ):
            problems.append(
                f"firm worst {worst[0]['scenario']} {worst[0]['value']} vs raw {scen} {loss}"
            )
    detail = f"{len(expected)} scenarios compared" + (
        f"; {'; '.join(problems)}" if problems else ""
    )
    return Check(name, bool(expected) and not problems, detail)


FRTB_COMPONENTS = ("frtb_sbm_charge", "frtb_drc_charge", "frtb_rrao_charge")


def check_frtb_sa_is_sum_of_components(ctx: RunContext, spark: SparkSession) -> Check:
    """At every node, ``frtb_sa_charge`` = SBM + default risk + residual add-on charges.

    Deliberately not checked: that a parent's SBM charge is at most the sum of its children's.
    It is not guaranteed (the binding correlation scenario can differ between nodes).
    """
    name = "frtb_sa_equals_sum_of_components"
    metrics = (
        spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where(F.col("measure").isin("frtb_sa_charge", *FRTB_COMPONENTS))
        .groupBy("node_level", *GRAIN_DIMS)
        .agg(
            F.max(F.when(F.col("measure") == "frtb_sa_charge", F.col("value"))).alias("sa"),
            F.sum(F.when(F.col("measure").isin(*FRTB_COMPONENTS), F.col("value"))).alias("parts"),
            F.count(F.lit(1)).alias("n"),
        )
    )
    stats = metrics.agg(
        F.count(F.lit(1)).alias("nodes"),
        F.sum((F.col("n") != 1 + len(FRTB_COMPONENTS)).cast("int")).alias("incomplete"),
        F.sum((F.abs(F.col("sa") - F.col("parts")) > TOLERANCE * F.greatest(F.lit(1.0), F.abs("sa"))).cast("int")).alias("bad"),
    ).first()  # fmt: skip
    nodes, incomplete, bad = stats["nodes"], stats["incomplete"] or 0, stats["bad"] or 0
    return Check(
        name,
        nodes > 0 and incomplete == 0 and bad == 0,
        f"{nodes} nodes, {incomplete} without all four rows, {bad} where the total differs",
    )


def _firm_ima_from_raw(ctx: RunContext, spark: SparkSession) -> dict[str, float] | None:
    """Firm IMCC from the raw trade vectors, summed per scenario in Spark and finished in NumPy.

    Shares only the pure formulas of ``frtb_ima`` (cascade, stressed ratio) with the run; the
    grouping into sets and horizons is redone here from the raw columns. ``None`` when the raw
    data has no stressed window (no IMCC is reported then).
    """
    raw = spark.read.parquet(str(ctx.raw("pnl_vectors")))
    keys = ["risk_class", "window", "in_reduced_set", "liquidity_horizon"]
    rows = (
        raw.select(*keys, F.posexplode("pnl").alias("i", "pnl"))
        .groupBy(*keys, "i")
        .agg(F.sum("pnl").alias("pnl"))
        .collect()
    )
    n = 1 + max((r["i"] for r in rows), default=-1)
    parts: dict[tuple, np.ndarray] = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        parts.setdefault(key, np.zeros(n))[r["i"]] = r["pnl"]
    if not any(k[1] == frtb_ima.STRESSED for k in parts):
        return None
    ten_day = math.sqrt(frtb_ima.BASE_HORIZON)
    scopes = sorted({k[0] for k in parts}) + [frtb_ima.ALL_CLASSES]
    es: dict[str, float] = {}
    for scope in scopes:
        by_set = {}
        for name, (window, reduced_only) in frtb_ima.ES_SETS.items():
            by_lh = {}
            for lh in frtb_ima.LIQUIDITY_HORIZONS:
                members = [
                    v for (rc, w, red, h), v in parts.items()
                    if scope in (rc, frtb_ima.ALL_CLASSES) and w == window
                    and (red or not reduced_only) and h >= lh
                ]  # fmt: skip
                if members:
                    by_lh[lh] = es_from_pnl(sum(members), frtb_ima.LEVEL) * ten_day
            by_set[name] = frtb_ima.lh_adjusted_es(by_lh)
        es[scope] = frtb_ima.scaled_es(by_set["rs"], by_set["fc"], by_set["rc"])
    unconstrained = es.pop(frtb_ima.ALL_CLASSES)
    constrained = sum(es.values())
    return {
        "frtb_imcc_unconstrained": unconstrained,
        "frtb_imcc_constrained_sum": constrained,
        "frtb_imcc": frtb_ima.RHO * unconstrained + (1 - frtb_ima.RHO) * constrained,
    }


def check_frtb_imcc_from_raw(ctx: RunContext, spark: SparkSession) -> Check:
    """Firm ``frtb_imcc`` (and its two parts) in ``risk_metrics`` equal a recomputation from raw.

    SIMPLIFICATION: assumes the firm rows come from unfiltered specifications.
    """
    name = "frtb_imcc_firm_matches_raw_vectors"
    stored = {
        r["measure"]: r["value"]
        for r in spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where(F.col("measure").isin(*frtb_ima.IMA_MEASURES) & (F.col("node_level") == "firm_id"))
        .collect()
    }
    expected = _firm_ima_from_raw(ctx, spark)
    if expected is None:
        return Check(name, not stored, f"no stressed window in raw data; {len(stored)} firm rows")
    problems = []
    for measure, exp in expected.items():
        got = stored.get(measure)
        if got is None:
            problems.append(f"{measure}: no firm row")
        elif abs(got - exp) > TOLERANCE * max(1.0, abs(exp)):
            problems.append(f"{measure}: stored {got} vs raw {exp}")
    detail = f"{len(expected)} firm IMA measures recomputed" + (
        f"; {'; '.join(problems)}" if problems else ""
    )
    return Check(name, not problems, detail)


def _ccar_from_raw(ctx: RunContext, spark: SparkSession) -> tuple[dict, dict]:
    """The CCAR projection and capital measures rebuilt in plain Python from files on disk.

    Inputs: the raw banking book, macro scenarios and starting capital, and the curated
    ``stress_results`` (already reconciled to the raw sensitivities by its own check) for the
    global-market-shock loss. Shares only the parameters file with the run, not its code.
    Returns ``(rows, metrics)``: ``rows[(scenario, quarter, node_level, node_id)]`` is a dict of
    projection columns, ``metrics[(measure, node_level, node_id, scenario)]`` a value.
    """
    params = ccar.load_params()
    book = [r.asDict() for r in spark.read.parquet(str(ctx.raw("banking_book"))).collect()]
    capital = {
        r["legal_entity"]: r.asDict()
        for r in spark.read.parquet(str(ctx.raw("capital_start"))).collect()
    }
    macro = {
        (r["scenario"], r["quarter"]): (r["unemployment_rate"], r["gdp_growth"])
        for r in spark.read.parquet(str(ctx.raw("macro_scenarios"))).collect()
    }
    gms = ccar.global_market_shock(load_scenarios())
    trading = {
        (r["legal_entity"], r["business_line_id"]): -r["pnl"]
        for r in spark.read.parquet(str(ctx.curated("stress_results")))
        .where(F.col("scenario") == gms)
        .groupBy("legal_entity", "business_line_id")
        .agg(F.sum("pnl").alias("pnl"))
        .collect()
    }
    flows = ("ppnr", "credit_loss", "trading_loss", "net_income")
    pairs = sorted({(b["legal_entity"], b["business_line_id"]) for b in book} | set(trading))
    rows: dict[tuple, dict] = {}

    def add(key: tuple, values: dict) -> None:
        acc = rows.setdefault(key, dict.fromkeys(flows, 0.0))
        for c in flows:
            acc[c] += values[c]

    for scenario in ccar.SCENARIOS:
        u0 = macro[(scenario, 0)][0]
        for q in range(1, ccar.N_QUARTERS + 1):
            u, g = macro[(scenario, q)]
            for le, bl in pairs:
                ppnr = credit = 0.0
                for b in book:
                    if (b["legal_entity"], b["business_line_id"]) != (le, bl):
                        continue
                    stress = 1 + params.pd_unemployment_beta * max(0.0, u - u0)
                    pd_q = min(1.0, b["pd_base"] / 4 * stress)
                    credit += b["ead"] * pd_q * b["lgd"]
                    revenue = b["revenue_rate_q"] * (1 + params.revenue_gdp_beta * g)
                    ppnr += b["ead"] * (revenue - b["expense_rate_q"])
                shocked = scenario == ccar.SEVERELY_ADVERSE and q == 1
                tl = trading.get((le, bl), 0.0) if shocked else 0.0
                values = {"ppnr": ppnr, "credit_loss": credit, "trading_loss": tl,
                          "net_income": (ppnr - credit - tl) * (1 - params.tax_rate)}
                add((scenario, q, ccar.BUSINESS_LINE, bl), values)
                add((scenario, q, ccar.LEGAL_ENTITY, le), values)
                add((scenario, q, ccar.FIRM, ccar.FIRM_ID), values)  # fmt: skip
    rwa = {le: c["rwa_trading"] for le, c in capital.items()}
    for b in book:
        rwa[b["legal_entity"]] = rwa.get(b["legal_entity"], 0.0) + b["ead"] * b["risk_weight"]
    nodes = {
        le: (ccar.LEGAL_ENTITY, c["cet1_start"], rwa[le], c["dividends_q"])
        for le, c in capital.items()
    }
    nodes[ccar.FIRM_ID] = (
        ccar.FIRM,
        sum(c["cet1_start"] for c in capital.values()),
        sum(rwa.values()),
        sum(c["dividends_q"] for c in capital.values()),
    )  # fmt: skip
    metrics: dict[tuple, float] = {}
    first, last = params.scb_dividend_quarters
    for node_id, (level, cet1_0, node_rwa, dividends) in nodes.items():
        start = cet1_0 / node_rwa
        metrics[(ccar.START_RATIO, level, node_id, "*")] = start
        for scenario in ccar.SCENARIOS:
            cet1 = cet1_0
            path = []
            for q in range(1, ccar.N_QUARTERS + 1):
                r = rows[(scenario, q, level, node_id)]
                cet1 += r["net_income"]
                r.update(dividends=dividends, cet1=cet1, rwa=node_rwa, cet1_ratio=cet1 / node_rwa)
                path.append(cet1 / node_rwa)
            metrics[(ccar.MIN_RATIO, level, node_id, scenario)] = min(path)
        decline = start - metrics[(ccar.MIN_RATIO, level, node_id, ccar.SEVERELY_ADVERSE)]
        add_on = (last - first + 1) * dividends / node_rwa
        metrics[(ccar.SCB, level, node_id, "*")] = max(params.scb_floor, decline + add_on)
    for key, r in rows.items():
        if key[2] == ccar.BUSINESS_LINE:
            r.update(dividends=None, cet1=None, rwa=None, cet1_ratio=None)
    return rows, metrics


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOLERANCE * max(1.0, abs(b))


def check_ccar_from_raw(ctx: RunContext, spark: SparkSession) -> Check:
    """Every ``ccar_projection`` row and every CCAR capital measure in ``risk_metrics`` equals a
    plain-Python recomputation from the raw banking book, macro scenarios, starting capital and
    the curated global-market-shock P&L; and stored firm losses equal the sum over business
    lines and over legal entities, per scenario and quarter.
    """
    name = "ccar_projection_matches_raw"
    expected_rows, expected_metrics = _ccar_from_raw(ctx, spark)
    stored = {
        (r["scenario"], r["quarter"], r["node_level"], r["node_id"]): r.asDict()
        for r in spark.read.parquet(str(ctx.curated("ccar_projection"))).collect()
    }
    problems = []
    if set(stored) != set(expected_rows):
        problems.append(f"{len(stored)} stored rows vs {len(expected_rows)} expected (keys differ)")
    columns = (*ccar.LOSS_COLUMNS, *ccar.CAPITAL_COLUMNS)
    for key, exp in expected_rows.items():
        got = stored.get(key)
        if got is None:
            continue
        for c in columns:
            if not _close(got[c], exp[c]):
                problems.append(f"{key} {c}: stored {got[c]} vs raw {exp[c]}")
    for (scenario, q) in {(k[0], k[1]) for k in stored}:
        for c in ("ppnr", "credit_loss", "trading_loss", "net_income"):
            by_level = {
                level: sum(v[c] for k, v in stored.items() if k[:3] == (scenario, q, level))
                for level in (ccar.BUSINESS_LINE, ccar.LEGAL_ENTITY, ccar.FIRM)
            }
            firm = by_level[ccar.FIRM]
            if not all(_close(v, firm) for v in by_level.values()):
                problems.append(f"{scenario} q{q} {c}: levels do not reconcile {by_level}")
    metric_rows = (
        spark.read.parquet(str(ctx.curated("risk_metrics")))
        .where(F.col("measure").isin(*ccar.CCAR_MEASURE_NAMES))
        .collect()
    )
    got_metrics = {}
    for r in metric_rows:
        level, node = (
            (ccar.FIRM, ccar.FIRM_ID) if r["node_level"] == "firm_id"
            else (ccar.LEGAL_ENTITY, r["legal_entity"])
        )  # fmt: skip
        got_metrics[(r["measure"], level, node, r["scenario"])] = r["value"]
    if set(got_metrics) != set(expected_metrics):
        problems.append(f"capital measures {sorted(got_metrics)} vs {sorted(expected_metrics)}")
    for key, exp in expected_metrics.items():
        if key in got_metrics and not _close(got_metrics[key], exp):
            problems.append(f"{key}: stored {got_metrics[key]} vs raw {exp}")
    detail = (
        f"{len(expected_rows)} projection rows, {len(expected_metrics)} capital measures recomputed"
    )
    if problems:
        detail += f"; {len(problems)} problems: " + "; ".join(problems[:5])
    return Check(name, bool(expected_rows) and not problems, detail)


CHECKS: list[CheckFn] = [
    check_pnl_scenarios_match_raw_per_grain_cell,
    check_vector_measures_from_pnl_scenarios,
    check_no_duplicate_nodes,
    check_stress_firm_pnl_is_sum_of_trades,
    check_frtb_sa_is_sum_of_components,
    check_frtb_imcc_from_raw,
    check_ccar_from_raw,
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
