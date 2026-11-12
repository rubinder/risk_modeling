import textwrap

import pytest

from riskagg.engine.spec import SpecError, load_specs, spec_hash
from riskagg.paths import aggregations_path, semantic_model_path
from riskagg.semantic.model import load_model

ORG = ["firm_id", "business_line_id", "desk_id", "book_id"]


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def _specs(tmp_path, body, model=None):
    model = model or load_model()
    return load_specs(_write(tmp_path, "agg.yaml", body), model)


def test_shipped_config_loads():
    model = load_model(semantic_model_path())
    specs = load_specs(aggregations_path(), model)
    assert [s.name for s in specs] == ["var_by_org", "var_by_desk_asset_class"]
    assert model.grain == (
        "firm_id", "business_line_id", "desk_id", "book_id", "legal_entity",
        "region", "asset_class", "currency", "risk_class",
    )  # fmt: skip
    assert model.hierarchies["org"] == tuple(ORG)
    assert model.measure("svar_99_1d").params["window"] == "STRESSED"
    assert model.measure("es_975_1d").function == "es"
    assert specs[0].diversification and len(specs[0].grouping_sets) == 4


def test_unknown_dimension_rejected(tmp_path):
    with pytest.raises(SpecError, match="unknown dimension 'trader'"):
        _specs(tmp_path, "- {name: s, measure: var_99_1d, grouping_sets: [[trader]]}")


def test_vector_measure_outside_grain_rejected(tmp_path):
    with pytest.raises(SpecError, match="not available for vector"):
        _specs(tmp_path, "- {name: s, measure: var_99_1d, grouping_sets: [[counterparty_id]]}")
    with pytest.raises(SpecError, match="not available for vector"):
        _specs(
            tmp_path,
            "- {name: s, measure: var_99_1d, grouping_sets: [[desk_id]],"
            " filters: {counterparty_id: [C1]}}",
        )


def _structured_model(tmp_path):
    text = (
        semantic_model_path()
        .read_text()
        .replace(
            "\ntables:",
            """
  cet1_charge:
    label: Charge
    unit: USD
    class: structured
    computed_at: [[firm_id], [firm_id, desk_id]]
    table: risk_metrics

tables:""",
            1,
        )
    )
    return load_model(_write(tmp_path, "model.yaml", text))


def test_structured_measure_only_at_computed_levels(tmp_path):
    model = _structured_model(tmp_path)
    ok = _specs(tmp_path, "- {name: s, measure: cet1_charge, grouping_sets: [[firm_id]]}", model)
    assert ok[0].measure == "cet1_charge"
    with pytest.raises(SpecError, match="computed only at"):
        _specs(tmp_path, "- {name: s, measure: cet1_charge, grouping_sets: [[desk_id]]}", model)


def test_diversification_requires_org_hierarchy_sets(tmp_path):
    ok = _specs(
        tmp_path,
        "- {name: s, measure: var_99_1d, diversification: true,"
        " grouping_sets: [[firm_id], [firm_id, business_line_id]]}",
    )
    assert ok[0].diversification
    for sets in ("[[firm_id]]", "[[desk_id, asset_class]]", "[[firm_id, desk_id]]"):
        with pytest.raises(SpecError, match="diversification requires"):
            _specs(
                tmp_path,
                f"- {{name: s, measure: var_99_1d, diversification: true, grouping_sets: {sets}}}",
            )


def test_spec_hash_changes_with_reference_data(tmp_path):
    specs = load_specs(aggregations_path(), load_model())
    ref = tmp_path / "ref.csv"
    ref.write_text("a,1\n")
    h1 = spec_hash(specs, [ref])
    assert len(h1) == 12 and h1 == spec_hash(specs, [ref])
    ref.write_text("a,2\n")
    assert spec_hash(specs, [ref]) != h1
    assert spec_hash(specs[:1], [ref]) != spec_hash(specs, [ref])


@pytest.mark.parametrize("value", ["D1", "", "[1]", "[NO]", "[[D1]]", "{a: b}"])
def test_filter_values_must_be_list_of_strings(tmp_path, value):
    body = f"- {{name: x, measure: var_99_1d, grouping_sets: [[desk_id]], filters: {{desk_id: {value}}}}}"
    with pytest.raises(SpecError, match="spec 'x': filter 'desk_id' must be a list of strings"):
        _specs(tmp_path, body)


def test_filter_list_of_strings_accepted(tmp_path):
    body = (
        "- {name: x, measure: var_99_1d, grouping_sets: [[desk_id]],"
        " filters: {desk_id: ['D1', 'NO']}}"
    )
    assert _specs(tmp_path, body)[0].filters == {"desk_id": ("D1", "NO")}
