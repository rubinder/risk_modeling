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
    names = {"hierarchy", "risk_factors", "trades", "exposures", "pnl_vectors", "sensitivities"}
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
    for table in ("risk_metrics", "pnl_scenarios"):
        df = spark.read.parquet(str(root / "curated" / table))
        assert set(df.columns) == set(model.tables[table]["columns"])
        assert df.select("run_id").distinct().collect()[0].run_id == run_id
    metrics = spark.read.parquet(str(root / "curated/risk_metrics"))
    assert metrics.select("spec_hash").distinct().count() == 1
    assert dict(metrics.dtypes)["as_of_date"] == "date"

    manifest = json.loads((root / "runs" / run_id / "manifest.json").read_text())
    assert manifest["run_id"] == run_id and manifest["profile"] == "tiny"
    assert manifest["as_of_date"] == AS_OF and manifest["n_scenarios"] == 20
    assert manifest["spec_hash"] == metrics.select("spec_hash").first().spec_hash
    assert manifest["spark_version"] == spark.version and manifest["git_commit"]
    assert manifest["spark_conf"]["spark.sql.session.timeZone"] == "UTC"
    assert set(manifest["inputs"]) == {"hierarchy", "pnl_vectors"}
    assert manifest["inputs"]["hierarchy"]["rows"] == 6
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
    assert _invoke("run", root).exit_code == 0
    assert _invoke("run", root).exit_code == 0
    for table in ("risk_metrics", "pnl_scenarios"):
        assert len(_run_dirs(root, table)) == 2
    first, second = (d.split("=", 1)[1] for d in _run_dirs(root))
    assert (root / "runs" / first / "manifest.json").exists()
    assert (root / "runs" / second / "manifest.json").exists()
    official = official_run(root, datetime.date(2026, 11, 12))
    assert official in (first, second)

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
    assert _invoke("run", root).exit_code == 0
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
    """Alter one pnl value in the first Parquet file of ``leaf``."""
    path = min(leaf.glob("part-*.parquet"))
    t = pq.read_table(path)
    pnl = t.column("pnl").to_pylist()
    pnl[0] += 1000.0
    pq.write_table(t.set_column(t.schema.get_field_index("pnl"), "pnl", [pnl]), path)
    path.with_name(f".{path.name}.crc").unlink(missing_ok=True)  # Hadoop rejects a stale checksum


def test_reconcile_detects_corrupted_grain(spark, generated, tmp_path):
    root = _fresh_root(generated, tmp_path)
    assert _invoke("run", root).exit_code == 0
    run_id = official_run(root, datetime.date(2026, 11, 12))
    ctx = RunContext("tiny", datetime.date(2026, 11, 12), run_id, "x", root)
    assert all(c.passed for c in audit.reconcile(ctx, spark))
    _corrupt_one_value(root)
    failed = {c.name for c in audit.reconcile(ctx, spark) if not c.passed}
    assert "firm_vector_equals_sum_of_book_vectors" in failed


def test_run_exits_nonzero_and_keeps_official_when_reconcile_fails(
    spark, generated, tmp_path, monkeypatch
):
    root = _fresh_root(generated, tmp_path)
    assert _invoke("run", root).exit_code == 0
    official_file = root / "runs" / "official.json"
    before = official_file.read_text()

    real_write = curate.write_pnl_scenarios

    def corrupting_write(grain, ctx, n_scenarios):
        real_write(grain, ctx, n_scenarios)
        _corrupt_leaf(ctx.curated("pnl_scenarios"))

    monkeypatch.setattr(curate, "write_pnl_scenarios", corrupting_write)
    result = _invoke("run", root)
    assert result.exit_code != 0
    assert "FAIL" in result.output
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
    assert result.exit_code != 0
    assert official_run(root, datetime.date(2026, 11, 12)) is None


def test_spec_hash_depends_on_reference_files(tmp_path):
    from riskagg.engine.spec import load_specs, spec_hash

    ref = tmp_path / "ref.csv"
    ref.write_text("a,1\n")
    specs = load_specs(None, load_model())
    h1 = spec_hash(specs, [ref])
    ref.write_text("a,2\n")
    assert spec_hash(specs, [ref]) != h1
