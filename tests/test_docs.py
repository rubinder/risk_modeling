"""The documents a reader relies on: data dictionary, README example, notebook, simplifications.

* the data dictionary lists every measure, dimension and table, and the committed copy is
  current;
* the README's "add an aggregation" spec, appended to a copy of ``config/aggregations.yaml``,
  gives ``var_99_1d`` rows at ``node_level == "region"`` through ``riskagg run`` with no
  Python change;
* the replication notebook runs top to bottom on the tiny profile, without Pinot or a server;
* every ``SIMPLIFICATION:`` marker under ``src/`` and ``config/`` is listed on a study page;
* every number the README quotes from the benchmark is tied to its sentence in the report;
* ``make demo`` goes on to the queries when Pinot cannot be started or loaded;
* README links and anchors resolve, and no tracked file names the private role text.
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
from riskagg.engine.spec import load_specs
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


def _section_of(text: str, heading: str) -> str:
    return text[text.index(heading) :].split("\n### ", 1)[0]


def test_dictionary_lists_a_vector_measure_as_stored_only_when_a_spec_declares_it():
    model = load_model()
    specs = load_specs(None, model)
    full = render_dictionary(model, specs)
    stored = _section_of(full, "### `risk_metrics`")
    for m in ("var_99_1d", "var_99_10d", "es_975_1d", "svar_99_1d"):
        assert f"`{m}`" in stored and f"`{m}__diversification`" in stored, m
    assert "`var_by_org`" in _section_of(full, "### `var_99_1d`")
    # derived rows are documented, not only named
    for suffix in ("sum_children", "diversification", "incremental"):
        assert f"`<measure>__{suffix}`" in full, suffix

    without = render_dictionary(model, [s for s in specs if s.measure != "var_99_10d"])
    stored = _section_of(without, "### `risk_metrics`")
    assert "`var_99_10d`" not in stored and "`var_99_10d__" not in stored
    assert "`var_99_1d`" in stored
    measure = _section_of(without, "### `var_99_10d`")
    assert "no specification" in measure and "risk_metrics" in measure


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


def _readme_example_yaml() -> str:
    block = _section(README.read_text(), "example-spec:start", "example-spec:end")
    (yaml_text,) = re.findall(r"```yaml\n(.*?)```", block, flags=re.DOTALL)
    return yaml_text


def test_readme_example_adds_var_by_region_without_code(tmp_run, tmp_path, monkeypatch):
    yaml_text = _readme_example_yaml()
    spec = yaml.safe_load(yaml_text)
    assert [s["measure"] for s in spec] == ["var_99_1d"]
    config = tmp_path / "config"
    shutil.copytree(paths.CONFIG_DIR, config)
    aggregations = config / "aggregations.yaml"
    # exactly what the README says: append the block's text to the file
    existing = aggregations.read_text()
    aggregations.write_text(existing + ("" if existing.endswith("\n") else "\n") + yaml_text)
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

# Each SIMPLIFICATION marker in src/ or config/ -> (file, words from the marker's first line) and the
# study page whose simplifications list carries it, with words from that list item.
MARKER_PAGES: list[tuple[str, str, str, str]] = [
    ("src/riskagg/semantic/query.py", "multi-day horizons scale", "01-var-es.md", "Square-root-of-time"),
    ("src/riskagg/measures/var.py", "multi-day horizons scale", "01-var-es.md", "Square-root-of-time"),
    ("src/riskagg/data/market.py", "one common factor per risk class", "01-var-es.md", "Factor model"),
    ("src/riskagg/data/market.py", "the stressed window is", "01-var-es.md", "Stressed window"),
    ("src/riskagg/data/market.py", "scenario P&L is linear", "01-var-es.md", "Linear P&L"),
    ("src/riskagg/data/market.py", "curvature uses a symmetric", "04-frtb.md", "Curvature up equals"),
    ("src/riskagg/engine/audit.py", "assumes the firm rows come", "04-frtb.md", "IMA reconciliation"),
    ("src/riskagg/measures/stress.py", "scenarios are hypothetical", "02-stress.md", "Hypothetical"),
    ("src/riskagg/measures/stress.py", "second-order Taylor", "02-stress.md", "Sensitivity-based"),
    ("src/riskagg/measures/stress.py", "the shock is instantaneous", "02-stress.md", "Instantaneous shock"),
    ("src/riskagg/data/banking.py", "two hand-set paths", "03-ccar.md", "Hand-set macro paths"),
    ("src/riskagg/data/banking.py", "three stylised portfolios", "03-ccar.md", "Three stylised loan"),
    ("src/riskagg/measures/ccar.py", "tax is applied symmetrically", "03-ccar.md", "Tax applied"),
    ("src/riskagg/measures/ccar.py", "constant balance sheet", "03-ccar.md", "Constant balance sheet"),
    ("src/riskagg/measures/ccar.py", "no preferred stock", "03-ccar.md", "No preferred stock"),
    ("src/riskagg/measures/ccar.py", "no counterparty default", "03-ccar.md", "No counterparty default"),
    ("src/riskagg/measures/ccar.py", "the stress capital buffer", "03-ccar.md", "legal entities as well"),
    ("src/riskagg/measures/frtb_sa.py", "one curve per currency", "04-frtb.md", "One curve per currency"),
    ("src/riskagg/measures/frtb_sa.py", "equity spot only", "04-frtb.md", "Equity spot only"),
    ("src/riskagg/measures/frtb_sa.py", "one vega factor", "04-frtb.md", "One vega factor per bucket"),
    ("src/riskagg/measures/frtb_sa.py", "one curvature factor", "04-frtb.md", "One curvature factor"),
    ("src/riskagg/measures/frtb_sa.py", "sensitivities are taken", "04-frtb.md", "Sensitivities taken"),
    ("src/riskagg/measures/frtb_drc.py", "gross JTD", "04-frtb.md", "Gross JTD"),
    ("src/riskagg/measures/frtb_drc.py", "no maturity scaling", "04-frtb.md", "No maturity scaling"),
    ("src/riskagg/measures/frtb_drc.py", "long and short positions", "04-frtb.md", "ignores seniority"),
    ("src/riskagg/measures/frtb_drc.py", "two buckets", "04-frtb.md", "Two DRC buckets"),
    ("src/riskagg/measures/frtb_drc.py", "the residual risk add-on", "04-frtb.md", "RRAO at the 1.0%"),
    ("src/riskagg/measures/frtb_ima.py", "the 10-day ES", "04-frtb.md", "scaled by √10"),
    ("src/riskagg/measures/frtb_ima.py", "the reduced set is not", "04-frtb.md", "never tested against"),
    ("src/riskagg/measures/frtb_ima.py", "liquidity horizons are", "04-frtb.md", "MAR33.12 floors"),
    ("src/riskagg/measures/frtb_ima.py", "the capital reported", "04-frtb.md", "IMCC only"),
    ("src/riskagg/data/org.py", "bucket 12 (an index)", "04-frtb.md", "bucket 12 (an index)"),
    ("src/riskagg/data/org.py", "each risk factor joins", "04-frtb.md", "drawn at random"),
    ("config/reference/frtb_sa.yaml", "only buckets 1, 5, 8 and 12", "04-frtb.md", "buckets 1, 5, 8 and 12 only"),
]


def _markers() -> list[tuple[str, str]]:
    files = [*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "config").rglob("*.yaml")]
    found = []
    for path in sorted(files):
        for line in path.read_text().splitlines():
            m = re.search(r"SIMPLIFICATION:\s*(.*)", line)
            if m:
                found.append((path.relative_to(REPO_ROOT).as_posix(), m.group(1).strip()))
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
            unmatched.append(f"{file}: {text}")
            continue
        used.add(entries[0])
        _, _, page, phrase = entries[0]
        assert phrase in _simplifications_list(page), (
            f"{file} marker '{text}' is not in the simplifications list of {page}"
        )
    assert not unmatched, "markers with no study-page entry in MARKER_PAGES:\n" + "\n".join(
        unmatched
    )
    assert used == set(MARKER_PAGES), f"stale entries: {set(MARKER_PAGES) - used}"


# --- README facts ------------------------------------------------------------------------------


# Each README benchmark claim (its numbers, as the README words them) and the report lines that
# state it. Every number in the README's benchmark section must belong to one of these claims.
BENCHMARK_CLAIMS: list[tuple[str, tuple[str, ...]]] = [
    ("Apple M3", ("| CPU | Apple M3 |",)),
    ("8 logical cores", ("| Logical cores | 8 |",)),
    ("16 GiB", ("| RAM | 16 GiB |",)),
    ("100,000 trades", ("| Trades | 100,000 |",)),
    ("250 scenarios per vector", ("| Scenarios per vector | 250 |",)),
    ("356.5 MiB of P&L vectors", ("| `pnl_vectors` on disk | 356.5 MiB",)),
    (
        "took 0.96 s against 12.23 s for explode-and-regroup, 12.76× faster",
        ("`array_index_sum` 0.96 s against `explode_regroup` 12.23 s: 12.76× faster.",),
    ),
    (
        "took 1.85 s against 5.00 s for one aggregation per level from trades, 2.70× faster",
        ("`grain_then_grouping_sets` 1.85 s against `per_level_from_trades` 5.00 s: 2.70× faster.",),
    ),
    (
        (
            "Shuffle partitions: 8 (one per core) took 0.92 s against 3.51 s with 200 static "
            "partitions, 3.80× faster"
        ),
        (
            "`sized_to_data`: 8 shuffle partitions (one per core)",
            "`static_200`: 200 shuffle partitions",
            "`sized_to_data` 0.92 s against `static_200` 3.51 s: 3.80× faster.",
        ),
    ),
]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_readme_benchmark_claims_match_the_report():
    quoted = _flat(_section(README.read_text(), "benchmark:start", "benchmark:end"))
    report = _flat(REPORT.read_text())
    rest = quoted
    for claim, sources in BENCHMARK_CLAIMS:
        assert claim in quoted, f"README benchmark section no longer says {claim!r}"
        for source in sources:
            assert source in report, f"report does not state {source!r} (README: {claim!r})"
        rest = rest.replace(claim, "")
    assert not re.findall(r"\d", rest), f"README numbers not tied to a report line: {rest!r}"


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path) -> set[str]:
    return {_slug(h) for h in re.findall(r"^#+ (.+)$", path.read_text(), flags=re.MULTILINE)}


def test_readme_links_and_anchors_resolve():
    text = README.read_text()
    links = re.findall(r"\]\(([^)\s]+)\)", text)
    assert links
    for link in links:
        if "://" in link:
            continue
        target, _, anchor = link.partition("#")
        path = REPO_ROOT / target if target else README
        assert path.exists(), f"README links to missing {target}"
        if anchor:
            assert anchor in _anchors(path), f"README links to missing anchor {link}"


# Built from parts so this file does not match its own search.
_ROLE_TEXT = ["job" + " description", "job" + "-description", "lead" + " engineer"]
# The one allowed hit: Spark's own term for a job label, in the benchmark runner.
_ALLOWED = {"src/riskagg/bench/runner.py": "tagged with a " + _ROLE_TEXT[0] + "."}


def test_no_tracked_file_refers_to_the_role_text():
    pattern = "\\|".join(_ROLE_TEXT)
    out = subprocess.run(
        ["git", "grep", "-n", "-i", pattern],
        cwd=REPO_ROOT, check=False, capture_output=True, text=True,
    )  # fmt: skip
    assert out.returncode in (0, 1), out.stderr
    hits = [
        line for line in out.stdout.splitlines()
        if not any(line.startswith(f"{f}:") and ok in line for f, ok in _ALLOWED.items())
    ]  # fmt: skip
    assert not hits, "role text found:\n" + "\n".join(hits)


# --- make demo without a working Pinot -------------------------------------------------------


def _stub(path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(
    ("docker_body", "riskagg_fails_on", "message"),
    [
        ('[ "$1" = info ] && exit 1\nexit 0\n', None, "Docker is not running"),
        ('[ "$1" = info ] && exit 0\nexit 1\n', None, "Pinot could not be started"),
        ('exit 0\n', "load-pinot", "Loading Pinot failed"),
    ],
)  # fmt: skip
def test_make_demo_continues_to_the_queries_without_pinot(
    tmp_path, docker_body, riskagg_fails_on, message
):
    """Docker down, Pinot not starting, or loading failing: the demo still runs its queries."""
    log = tmp_path / "calls.log"
    docker = _stub(tmp_path / "docker", f'echo "docker $*" >> {log}\n{docker_body}')
    fail = f'[ "$1" = "{riskagg_fails_on}" ] && exit 1\n' if riskagg_fails_on else ""
    riskagg = _stub(tmp_path / "riskagg", f'echo "riskagg $*" >> {log}\n{fail}exit 0\n')
    result = subprocess.run(
        ["make", "--no-print-directory", "demo", f"DOCKER={docker}", f"RISKAGG={riskagg}",
         "AS_OF=2026-11-12", f"ROOT={tmp_path / 'data'}"],
        cwd=REPO_ROOT, check=False, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert message in result.stdout and "answer from Parquet" in result.stdout
    lines = log.read_text().splitlines()
    riskagg_cmds = [line.split()[1] for line in lines if line.startswith("riskagg ")]
    assert riskagg_cmds[:2] == ["generate", "run"]
    assert riskagg_cmds[-2:] == ["query", "query"]
