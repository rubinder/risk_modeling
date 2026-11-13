"""The documents a reader relies on: data dictionary, README example, notebook, simplifications.

* the data dictionary lists every measure, dimension and table, and the committed copy is
  current;
* the README's "add an aggregation" spec, appended to a copy of ``config/aggregations.yaml``,
  gives ``var_99_1d`` rows at ``node_level == "region"`` through ``riskagg run`` with no
  Python change;
* the replication notebook runs top to bottom on the tiny profile, without Pinot or a server;
* every ``SIMPLIFICATION:`` marker under ``src/`` is listed on a study page;
* every number the README quotes from the benchmark is in the generated report.
"""

import json
import re
import shutil
import subprocess
import sys

import pyarrow.dataset as ds
import pytest
import yaml
from typer.testing import CliRunner

from riskagg import paths
from riskagg.cli import app
from riskagg.context import official_run
from riskagg.paths import REPO_ROOT
from riskagg.semantic.dictionary import render_dictionary, write_dictionary
from riskagg.semantic.model import load_model
from riskagg.serving.client import RiskClient

README = REPO_ROOT / "README.md"
DICTIONARY = REPO_ROOT / "docs/data-dictionary.md"
NOTEBOOK = REPO_ROOT / "notebooks/replicate.ipynb"
REPORT = REPO_ROOT / "docs/benchmarks/report.md"
STUDY = REPO_ROOT / "docs/study"
DOWN = "http://127.0.0.1:9"  # nothing listens on the discard port
runner = CliRunner()


def _section(text: str, start: str, end: str) -> str:
    """The text between two HTML comment markers of the README."""
    begin = text.index(f"<!-- {start} -->") + len(f"<!-- {start} -->")
    return text[begin : text.index(f"<!-- {end} -->", begin)]


# --- data dictionary -------------------------------------------------------------------------


def test_dictionary_lists_every_measure_dimension_and_table(tmp_path):
    model = load_model()
    out = tmp_path / "dictionary.md"
    write_dictionary(model, out)
    text = out.read_text()
    for name, m in model.measures.items():
        assert f"### `{name}`" in text
        assert m.label in text
    for name in model.dimensions:
        assert f"### `{name}`" in text
    for name in model.tables:
        assert f"### `{name}`" in text
    var = text[text.index("### `var_99_1d`") :].split("\n### ", 1)[0]
    assert "vector" in var and "USD" in var and "never summed" in var.lower()
    scb = text[text.index("### `ccar_scb`") :].split("\n### ", 1)[0]
    assert "`firm_id`, `legal_entity`" in scb and "not allocated" in scb.lower()


def test_committed_dictionary_is_current():
    assert DICTIONARY.read_text() == render_dictionary(load_model()), (
        "docs/data-dictionary.md is stale: run `uv run riskagg dictionary`"
    )


def test_cli_dictionary(tmp_path):
    out = tmp_path / "dd.md"
    result = runner.invoke(app, ["dictionary", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text() == render_dictionary(load_model())


# --- README: add an aggregation by editing YAML ----------------------------------------------


def _readme_example_spec() -> list[dict]:
    block = _section(README.read_text(), "example-spec:start", "example-spec:end")
    (yaml_text,) = re.findall(r"```yaml\n(.*?)```", block, flags=re.DOTALL)
    return yaml.safe_load(yaml_text)


def test_readme_example_adds_var_by_region_without_code(tmp_run, tmp_path, monkeypatch):
    spec = _readme_example_spec()
    assert [s["measure"] for s in spec] == ["var_99_1d"]
    config = tmp_path / "config"
    shutil.copytree(paths.CONFIG_DIR, config)
    aggregations = config / "aggregations.yaml"
    aggregations.write_text(aggregations.read_text() + "\n" + yaml.safe_dump(spec, sort_keys=False))
    monkeypatch.setattr(paths, "CONFIG_DIR", config)

    root = tmp_path / "data"
    shutil.copytree(tmp_run.root / "raw", root / "raw")
    args = ["run", "--profile", "tiny", "--as-of", tmp_run.as_of_date.isoformat()]
    result = runner.invoke(app, [*args, "--root", str(root)])
    assert result.exit_code == 0, result.output
    run_id = official_run(root, tmp_run.as_of_date)

    metrics = ds.dataset(root / "curated/risk_metrics", partitioning="hive").to_table().to_pandas()
    rows = metrics[(metrics.measure == "var_99_1d") & (metrics.node_level == "region")]
    assert set(rows.run_id) == {run_id}
    assert set(rows.aggregation) == {spec[0]["name"]}
    hierarchy = ds.dataset(root / "raw/tiny" / f"as_of_date={tmp_run.as_of_date}/hierarchy")
    regions = set(hierarchy.to_table().column("region").to_pylist())
    assert set(rows.region) == regions and len(rows) == len(regions)
    others = [d for d in tmp_run.model.grain if d != "region"]
    assert (rows[others] == "*").all().all()

    # the stored rows equal the ad-hoc answer the API gives from the curated P&L vectors
    client = RiskClient(root=root, broker_url=DOWN, controller_url=DOWN)
    ad_hoc = client.metric("var_99_1d", ["region"]).set_index("region")["value"]
    stored = rows.set_index("region")["value"].sort_index()
    assert stored.to_dict() == pytest.approx(ad_hoc.sort_index().to_dict(), abs=1e-6)


# --- notebook ----------------------------------------------------------------------------------


def test_notebook_is_committed_without_outputs():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_notebook_executes_on_tiny(tmp_path):
    """Top to bottom in a fresh kernel: it generates and runs tiny itself, Pinot not needed."""
    cmd = [
        sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=900", "--output-dir", str(tmp_path), str(NOTEBOOK),
    ]  # fmt: skip
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, result.stderr[-4000:]
    executed = json.loads((tmp_path / NOTEBOOK.name).read_text())
    text = "".join(
        "".join(o.get("text", "")) for c in executed["cells"] for o in c.get("outputs", [])
    )
    assert "firm var_99_1d reproduced" in text
    assert "K_b reproduced" in text


# --- simplifications -------------------------------------------------------------------------

# Each SIMPLIFICATION marker in src/ -> (file, words from the marker's first line) and the
# study page whose simplifications list carries it, with words from that list item.
MARKER_PAGES: list[tuple[str, str, str, str]] = [
    ("semantic/query.py", "multi-day horizons scale", "01-var-es.md", "Square-root-of-time"),
    ("measures/var.py", "multi-day horizons scale", "01-var-es.md", "Square-root-of-time"),
    ("data/market.py", "one common factor per risk class", "01-var-es.md", "Factor model"),
    ("data/market.py", "the stressed window is", "01-var-es.md", "Stressed window"),
    ("data/market.py", "scenario P&L is linear", "01-var-es.md", "Linear P&L"),
    ("data/market.py", "curvature uses a symmetric", "04-frtb.md", "Curvature up equals"),
    ("engine/audit.py", "assumes the firm-node row", "01-var-es.md", "Firm reconciliation"),
    ("engine/audit.py", "assumes the firm rows come", "04-frtb.md", "IMA reconciliation"),
    ("measures/stress.py", "scenarios are hypothetical", "02-stress.md", "Hypothetical"),
    ("measures/stress.py", "second-order Taylor", "02-stress.md", "Sensitivity-based"),
    ("measures/stress.py", "the shock is instantaneous", "02-stress.md", "Instantaneous shock"),
    ("data/banking.py", "two hand-set paths", "03-ccar.md", "Hand-set macro paths"),
    ("data/banking.py", "three stylised portfolios", "03-ccar.md", "Three stylised loan"),
    ("measures/ccar.py", "tax is applied symmetrically", "03-ccar.md", "Tax applied"),
    ("measures/ccar.py", "constant balance sheet", "03-ccar.md", "Constant balance sheet"),
    ("measures/ccar.py", "no preferred stock", "03-ccar.md", "No preferred stock"),
    ("measures/ccar.py", "no counterparty default", "03-ccar.md", "No counterparty default"),
    ("measures/ccar.py", "the stress capital buffer", "03-ccar.md", "legal entities as well"),
    ("measures/frtb_sa.py", "one curve per currency", "04-frtb.md", "One curve per currency"),
    ("measures/frtb_sa.py", "equity spot only", "04-frtb.md", "Equity spot only"),
    ("measures/frtb_sa.py", "one vega factor", "04-frtb.md", "One vega factor per bucket"),
    ("measures/frtb_sa.py", "one curvature factor", "04-frtb.md", "One curvature factor"),
    ("measures/frtb_sa.py", "sensitivities are taken", "04-frtb.md", "Sensitivities taken"),
    ("measures/frtb_drc.py", "gross JTD", "04-frtb.md", "Gross JTD"),
    ("measures/frtb_drc.py", "no maturity scaling", "04-frtb.md", "No maturity scaling"),
    ("measures/frtb_drc.py", "long and short positions", "04-frtb.md", "ignores seniority"),
    ("measures/frtb_drc.py", "two buckets", "04-frtb.md", "Two DRC buckets"),
    ("measures/frtb_drc.py", "the residual risk add-on", "04-frtb.md", "RRAO at the 1.0%"),
    ("measures/frtb_ima.py", "the 10-day ES", "04-frtb.md", "scaled by √10"),
    ("measures/frtb_ima.py", "the reduced set is not", "04-frtb.md", "never tested against"),
    ("measures/frtb_ima.py", "liquidity horizons are", "04-frtb.md", "MAR33.12 floors"),
    ("measures/frtb_ima.py", "the capital reported", "04-frtb.md", "IMCC only"),
    ("data/org.py", "bucket 12 (an index)", "04-frtb.md", "bucket 12 (an index)"),
    ("data/org.py", "each risk factor joins", "04-frtb.md", "drawn at random"),
]


def _markers() -> list[tuple[str, str]]:
    src = REPO_ROOT / "src/riskagg"
    found = []
    for path in sorted(src.rglob("*.py")):
        for line in path.read_text().splitlines():
            m = re.search(r"SIMPLIFICATION:\s*(.*)", line)
            if m:
                found.append((path.relative_to(src).as_posix(), m.group(1).strip()))
    return found


def _simplifications_list(page: str) -> str:
    text = (STUDY / page).read_text()
    m = re.search(
        r"^## \d+\. What this project simplifies.*?(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert m, f"{page} has no simplifications section"
    return " ".join(m.group(0).split())


def test_every_simplification_marker_is_listed_on_a_study_page():
    markers = _markers()
    assert markers, "the scan found no SIMPLIFICATION markers"
    unmatched, used = [], set()
    for file, text in markers:
        entries = [e for e in MARKER_PAGES if e[0] == file and e[1].lower() in text.lower()]
        if len(entries) != 1:
            unmatched.append(f"src/riskagg/{file}: {text}")
            continue
        used.add(entries[0])
        _, _, page, phrase = entries[0]
        assert phrase in _simplifications_list(page), (
            f"src/riskagg/{file} marker '{text}' is not in the simplifications list of {page}"
        )
    assert not unmatched, "markers with no study-page entry in MARKER_PAGES:\n" + "\n".join(
        unmatched
    )
    assert used == set(MARKER_PAGES), f"stale entries: {set(MARKER_PAGES) - used}"


# --- README facts ------------------------------------------------------------------------------


def test_readme_benchmark_numbers_are_in_the_report():
    quoted = _section(README.read_text(), "benchmark:start", "benchmark:end")
    report = REPORT.read_text()
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", quoted)
    assert numbers
    for n in numbers:
        assert re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d])", report), (
            f"README quotes {n}, which is not in docs/benchmarks/report.md"
        )


def test_readme_links_resolve():
    text = README.read_text()
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if "://" in target:
            continue
        assert (REPO_ROOT / target).exists(), f"README links to missing {target}"
    assert "job-description" not in text and "job description" not in text.lower()
