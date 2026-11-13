import datetime
import json
import re

import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F
from typer.testing import CliRunner

from riskagg.cli import app
from riskagg.context import RunContext, mark_official, new_run, official_run
from riskagg.engine import audit
from riskagg.semantic.model import load_model
from riskagg.serving import curate

AS_OF = "2026-11-12"
ARGS = ["--profile", "tiny", "--as-of", AS_OF]
runner = CliRunner()


def _invoke(cmd, root):
    return runner.invoke(app, [cmd, *ARGS, "--root", str(root)])


@pytest.fixture(scope="module")
def generated(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    result = _invoke("generate", root)
    assert result.exit_code == 0, result.output
    return root


def _fresh_root(generated, tmp_path):
    """A private copy of the generated raw data, so a test's runs do not leak into others."""
    import shutil

    shutil.copytree(generated / "raw", tmp_path / "raw")
    return tmp_path


def _run_dirs(root, table="risk_metrics"):
    return sorted(
        p.name for p in (root / "curated" / table / f"as_of_date={AS_OF}").glob("run_id=*")
    )


def test_run_id_format_and_context(tmp_path):
    ctx = new_run("tiny", datetime.date(2026, 11, 12), tmp_path, "abc")
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", ctx.run_id)
    assert ctx.raw("trades") == tmp_path / "raw/tiny/as_of_date=2026-11-12/trades"
    assert ctx.curated("risk_metrics") == (
        tmp_path / f"curated/risk_metrics/as_of_date=2026-11-12/run_id={ctx.run_id}"
    )
    assert ctx.manifest_path == tmp_path / "runs" / ctx.run_id / "manifest.json"
    assert official_run(tmp_path, ctx.as_of_date) is None
    mark_official(ctx)
    assert official_run(tmp_path, ctx.as_of_date) == ctx.run_id
    assert json.loads((tmp_path / "runs/official.json").read_text()) == {AS_OF: ctx.run_id}


def test_generate_writes_raw_datasets(generated):
    base = generated / "raw/tiny" / f"as_of_date={AS_OF}"
    names = {
        "hierarchy", "risk_factors", "trades", "exposures", "pnl_vectors", "sensitivities",
        "default_risk_positions", "macro_scenarios", "banking_book", "capital_start",
    }  # fmt: skip
    assert {p.name for p in base.iterdir()} == names
    assert all(list((base / n).glob("*.parquet")) for n in names)


def test_run_curates_and_writes_manifest(spark, generated, tmp_path):
    root = _fresh_root(generated, tmp_path)
    result = _invoke("run", root)
    assert result.exit_code == 0, result.output
    assert "var_99_1d" in result.output
    (run_id,) = [d.split("=", 1)[1] for d in _run_dirs(root)]
    assert official_run(root, datetime.date(2026, 11, 12)) == run_id

    model = load_model()
    manifest = json.loads((root / "runs" / run_id / "manifest.json").read_text())
    assert len(model.tables) == 6
    for table in model.tables:  # every curated table: run id, date and spec hash on every row
        df = spark.read.parquet(str(root / "curated" / table))
        assert set(df.columns) == set(model.tables[table]["columns"]), table
        assert {"as_of_date", "run_id", "spec_hash"} <= set(df.columns), table
        assert [tuple(r) for r in df.select("run_id", "spec_hash").distinct().collect()] == [
            (run_id, manifest["spec_hash"])
        ], table
        assert df.where(F.col("spec_hash").isNull()).count() == 0, table
    metrics = spark.read.parquet(str(root / "curated/risk_metrics"))
    assert dict(metrics.dtypes)["as_of_date"] == "date"

    assert manifest["run_id"] == run_id and manifest["profile"] == "tiny"
    assert manifest["as_of_date"] == AS_OF and manifest["n_scenarios"] == 20
    assert manifest["spark_version"] == spark.version and manifest["git_commit"]
    assert manifest["spark_conf"]["spark.sql.session.timeZone"] == "UTC"
    # every raw dataset a measure reads (only `exposures`, an input of the generator, is not)
    assert set(manifest["inputs"]) == {
        "hierarchy", "pnl_vectors", "sensitivities", "trades", "risk_factors",
        "default_risk_positions", "banking_book", "macro_scenarios", "capital_start",
    }  # fmt: skip
    assert manifest["inputs"]["hierarchy"]["rows"] == 6
    assert manifest["inputs"]["trades"]["rows"] == 20
    for name, entry in manifest["inputs"].items():
        assert entry["rows"] > 0 and entry["checksum"] not in ("", "0"), name
    assert manifest["reconciliation"]["passed"] is True
    assert len(manifest["reconciliation"]["checks"]) == len(audit.CHECKS)


def test_checksum_is_order_independent_and_content_sensitive(spark):
    df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], "x int, y string")
    same = df.orderBy(F.col("x").desc())
    changed = spark.createDataFrame([(1, "a"), (2, "b"), (3, "d")], "x int, y string")
    assert audit.content_checksum(df) == audit.content_checksum(same)
    assert audit.content_checksum(df) != audit.content_checksum(changed)
    assert audit.content_checksum(df)[0] == 3


def test_second_run_is_official_and_both_directories_remain(spark, generated, tmp_path):
    root = _fresh_root(generated, tmp_path)
    ids = []
    for _ in range(2):
        result = _invoke("run", root)
        assert result.exit_code == 0, result.output
        ids.append(re.search(r"run (\S+) is official for", result.output).group(1))
        if len(ids) == 1:
            assert official_run(root, datetime.date(2026, 11, 12)) == ids[0]
    first, second = ids
    assert first != second
    for table in ("risk_metrics", "pnl_scenarios"):
        assert _run_dirs(root, table) == sorted(f"run_id={i}" for i in ids)
    assert (root / "runs" / first / "manifest.json").exists()
    assert (root / "runs" / second / "manifest.json").exists()
    official = official_run(root, datetime.date(2026, 11, 12))
    assert official == second
    assert official != first

    metrics = spark.read.parquet(str(root / "curated/risk_metrics"))
    both = metrics.select("run_id").distinct().count()
    assert both == 2
    scoped = metrics.where(F.col("run_id") == official)
    keys = ["measure", "node_level", "firm_id", "business_line_id", "desk_id", "book_id",
            "legal_entity", "region", "asset_class", "currency", "risk_class", "scenario"]  # fmt: skip
    assert scoped.count() > 0
    assert scoped.groupBy(*keys).count().where("count > 1").count() == 0
    # Reading without the run filter sees every node twice: the filter is what makes it unique.
    assert metrics.groupBy(*keys).count().where("count != 2").count() == 0


def test_pnl_scenarios_shape(spark, generated, tmp_path):
    root = _fresh_root(generated, tmp_path)
    result = _invoke("run", root)
    assert result.exit_code == 0, result.output
    df = spark.read.parquet(str(root / "curated/pnl_scenarios"))
    assert dict(df.dtypes)["scenario_idx"] == "int"
    stats = df.agg(F.min("scenario_idx"), F.max("scenario_idx")).first()
    assert (stats[0], stats[1]) == (0, 19)
    per_node = df.groupBy("book_id", "asset_class", "currency", "risk_class", "window").count()
    assert per_node.where("count != 20").count() == 0


def _corrupt_one_value(root, table="pnl_scenarios"):
    leaf = next((root / "curated" / table).glob("as_of_date=*/run_id=*"))
    _corrupt_leaf(leaf)


def _corrupt_leaf(leaf):
    """+1000 on the pnl of one row of ``leaf``, chosen deterministically: the smallest
    ``GRAIN_KEY`` over all its Parquet files. Only that row's file is rewritten."""
    best = None
    for path in sorted(leaf.glob("part-*.parquet")):
        t = pq.read_table(path, columns=GRAIN_KEY).to_pylist()
        for i, row in enumerate(t):
            key = tuple(row[k] for k in GRAIN_KEY)
            if best is None or key < best[0]:
                best = (key, path, i)
    assert best is not None, f"no rows under {leaf}"
    _, path, i = best
    t = pq.read_table(path)
    pnl = t.column("pnl").to_pylist()
    pnl[i] += 1000.0
    pq.write_table(t.set_column(t.schema.get_field_index("pnl"), "pnl", [pnl]), path)
    path.with_name(f".{path.name}.crc").unlink(missing_ok=True)  # Hadoop rejects a stale checksum


@pytest.fixture(scope="module")
def official(spark, generated, tmp_path_factory):
    """One official tiny run; corruption tests work on copies of its root."""
    root = _fresh_root(generated, tmp_path_factory.mktemp("official"))
    result = _invoke("run", root)
    assert result.exit_code == 0, result.output
    return root


def _copy_run(official, tmp_path):
    import shutil

    root = tmp_path / "root"
    shutil.copytree(official, root)
    as_of = datetime.date.fromisoformat(AS_OF)
    return RunContext("tiny", as_of, official_run(root, as_of), "x", root)


def _rewrite_leaf(leaf, change):
    """Rewrite a run partition as one Parquet file holding ``change(rows)`` (a pandas frame)."""
    import pyarrow as pa

    files = sorted(leaf.glob("part-*.parquet"))
    table = pa.concat_tables([pq.read_table(p) for p in files])
    rows = change(table.to_pandas())
    for p in files:
        p.unlink()
        p.with_name(f".{p.name}.crc").unlink(missing_ok=True)
    out = pa.Table.from_pandas(rows, schema=table.schema, preserve_index=False)
    pq.write_table(out, leaf / "part-00000-rewritten.parquet")


def _failed(checks):
    return {c.name for c in checks if not c.passed}


GRAIN_KEY = ["book_id", "asset_class", "currency", "risk_class", "window", "scenario_idx"]


def test_reconcile_detects_corrupted_grain(spark, official, tmp_path):
    ctx = _copy_run(official, tmp_path)
    assert _failed(audit.reconcile(ctx, spark)) == set()
    _corrupt_one_value(ctx.root)
    assert "pnl_scenarios_equal_raw_per_grain_cell" in _failed(audit.reconcile(ctx, spark))


def _move_pnl_between_books(rows):
    """+1000 on one book, -1000 on another, same window and scenario: the firm total is kept."""
    cell = rows[(rows.window == "CURRENT") & (rows.scenario_idx == 0)].sort_values(GRAIN_KEY)
    first = cell.index[0]
    other = cell[cell.book_id != cell.loc[first, "book_id"]].index[0]
    rows.loc[first, "pnl"] += 1000.0
    rows.loc[other, "pnl"] -= 1000.0
    return rows


def _relabel_desk(rows):
    """One row's desk_id changed to another desk's: every total is kept."""
    first = rows.sort_values(GRAIN_KEY).index[0]
    rows.loc[first, "desk_id"] = min(set(rows.desk_id) - {rows.loc[first, "desk_id"]})
    return rows


@pytest.mark.parametrize("change", [_move_pnl_between_books, _relabel_desk])
def test_reconcile_detects_pnl_moved_within_the_firm(spark, official, tmp_path, change):
    ctx = _copy_run(official, tmp_path)
    _rewrite_leaf(ctx.curated("pnl_scenarios"), change)
    firm = (
        spark.read.parquet(str(ctx.curated("pnl_scenarios")))
        .groupBy("window", "scenario_idx").agg(F.sum("pnl").alias("pnl"))
    )  # fmt: skip
    raw = (
        spark.read.parquet(str(ctx.raw("pnl_vectors")))
        .select("window", F.posexplode("pnl").alias("scenario_idx", "pnl"))
        .groupBy("window", "scenario_idx").agg(F.sum("pnl").alias("raw"))
    )  # fmt: skip
    worst = firm.join(raw, ["window", "scenario_idx"]).agg(F.max(F.abs(F.col("pnl") - F.col("raw"))))
    assert worst.first()[0] < 1e-6  # the firm vector is unchanged
    assert "pnl_scenarios_equal_raw_per_grain_cell" in _failed(audit.reconcile(ctx, spark))


DESK = "firm_id|business_line_id|desk_id"
BOOK = f"{DESK}|book_id"


def _bump(measure, level):
    def change(rows):
        hit = rows[(rows.measure == measure) & (rows.node_level == level)].sort_values("desk_id")
        rows.loc[hit.index[0], "value"] += 1000.0
        return rows

    return change


def _drop(measure, level):
    def change(rows):
        hit = rows[(rows.measure == measure) & (rows.node_level == level)].sort_values("desk_id")
        return rows.drop(index=hit.index[0])

    return change


VECTOR_ROW_CORRUPTIONS = {
    "desk": _bump("var_99_1d", DESK),
    "desk_asset_class": _bump("var_99_1d", "desk_id|asset_class"),
    "sum_children": _bump("var_99_1d__sum_children", "firm_id"),
    "diversification": _bump("var_99_1d__diversification", "firm_id|business_line_id"),
    "incremental": _bump("var_99_1d__incremental", BOOK),
    "missing": _drop("var_99_1d", DESK),
    "es_desk": _bump("es_975_1d", DESK),
    "svar_incremental": _bump("svar_99_1d__incremental", BOOK),
    "var_10d_firm": _bump("var_99_10d", "firm_id"),
}


@pytest.mark.parametrize(
    "change", list(VECTOR_ROW_CORRUPTIONS.values()), ids=list(VECTOR_ROW_CORRUPTIONS)
)
def test_reconcile_recomputes_every_stored_vector_measure(spark, official, tmp_path, change):
    ctx = _copy_run(official, tmp_path)
    _rewrite_leaf(ctx.curated("risk_metrics"), change)
    assert _failed(audit.reconcile(ctx, spark)) == {"vector_measures_match_pnl_scenarios"}


def _bump_firm_component_and_total(component):
    """+1000 on the firm row of ``component`` and of ``frtb_sa_charge``: total = sum still holds."""

    def change(rows):
        firm = rows.node_level == "firm_id"
        for measure in (component, "frtb_sa_charge"):
            (i,) = rows[firm & (rows.measure == measure)].index
            rows.loc[i, "value"] += 1000.0
        return rows

    return change


@pytest.mark.parametrize("component", ["frtb_drc_charge", "frtb_rrao_charge"])
def test_reconcile_recomputes_firm_drc_and_rrao_from_raw(spark, official, tmp_path, component):
    ctx = _copy_run(official, tmp_path)
    _rewrite_leaf(ctx.curated("risk_metrics"), _bump_firm_component_and_total(component))
    checks = {c.name: c for c in audit.reconcile(ctx, spark)}
    assert _failed(checks.values()) == {"frtb_drc_rrao_firm_match_raw"}
    assert component in checks["frtb_drc_rrao_firm_match_raw"].detail


def test_firm_drc_and_rrao_check_passes_with_its_own_numbers(spark, official, tmp_path):
    ctx = _copy_run(official, tmp_path)
    check = audit.check_frtb_drc_rrao_from_raw(ctx, spark)
    assert check.passed, check.detail
    assert "2 firm FRTB components recomputed" in check.detail


def test_run_exits_nonzero_and_keeps_official_when_reconcile_fails(
    spark, generated, tmp_path, monkeypatch
):
    root = _fresh_root(generated, tmp_path)
    result = _invoke("run", root)
    assert result.exit_code == 0, result.output
    official_file = root / "runs" / "official.json"
    before = official_file.read_text()

    real_write = curate.write_pnl_scenarios

    def corrupting_write(grain, ctx, n_scenarios):
        real_write(grain, ctx, n_scenarios)
        _corrupt_leaf(ctx.curated("pnl_scenarios"))

    monkeypatch.setattr(curate, "write_pnl_scenarios", corrupting_write)
    result = _invoke("run", root)
    assert result.exit_code != 0, result.output
    assert "[FAIL] pnl_scenarios_equal_raw_per_grain_cell" in result.output, result.output
    assert official_file.read_text() == before
    assert len(_run_dirs(root)) == 2
    failed_id = next(d.split("=", 1)[1] for d in _run_dirs(root) if d.split("=", 1)[1]
                     not in before)  # fmt: skip
    manifest = json.loads((root / "runs" / failed_id / "manifest.json").read_text())
    assert manifest["reconciliation"]["passed"] is False


def test_failing_registered_check_blocks_official(generated, tmp_path, monkeypatch):
    root = _fresh_root(generated, tmp_path)

    def failing(ctx, spark):
        return audit.Check("always_fails", False, "injected")

    monkeypatch.setattr(audit, "CHECKS", [*audit.CHECKS, failing])
    result = _invoke("run", root)
    assert result.exit_code != 0, result.output
    assert "[FAIL] always_fails: injected" in result.output, result.output
    assert official_run(root, datetime.date(2026, 11, 12)) is None


def test_cli_leaves_an_active_sessions_conf_alone(spark, generated, tmp_path):
    """The CLI uses an active session as it is: its own settings would otherwise leak into the
    test session (shuffle partitions 4 -> 8, ...) for every later test."""
    from riskagg.spark import SESSION_CONF

    keys = [k for k in SESSION_CONF if not k.startswith("spark.driver.")]
    before = {k: spark.conf.get(k, None) for k in keys}
    assert before["spark.sql.shuffle.partitions"] == "4"
    result = _invoke("run", _fresh_root(generated, tmp_path))
    assert result.exit_code == 0, result.output
    assert {k: spark.conf.get(k, None) for k in keys} == before
    assert not spark.sparkContext._jsc.sc().isStopped()


def test_spec_hash_depends_on_reference_files(tmp_path):
    from riskagg.engine.spec import load_specs, spec_hash

    ref = tmp_path / "ref.csv"
    ref.write_text("a,1\n")
    specs = load_specs(None, load_model())
    h1 = spec_hash(specs, [ref])
    ref.write_text("a,2\n")
    assert spec_hash(specs, [ref]) != h1
