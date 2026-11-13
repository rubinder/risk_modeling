"""Tolerances of the reconciliation checks at demo-scale magnitudes, without Spark.

A diversification or incremental row is a difference of two numbers that can be of order
1e10 while the row itself is about 0; floating-point noise in it is then about 1e-5. The
check must scale its tolerance with those operands, and still catch a real error.
"""

import numpy as np

from riskagg.engine import audit
from riskagg.engine.spec import AggregationSpec
from riskagg.semantic.model import load_model

ORG = ("firm_id", "business_line_id")
N = 250


def _rows():
    """Two business lines with comonotonic losses of order 1e10: diversification is 0."""
    model = load_model()
    measure = model.measure("svar_99_1d")
    spec = AggregationSpec("svar_by_org", "svar_99_1d", (ORG[:1], ORG), {}, True)
    rng = np.random.default_rng(0)
    shape = rng.standard_normal(N) * 5e9
    bl1, bl2 = shape, 1.25 * shape
    vectors = {
        "firm_id": {("STRESSED", "FIRM"): bl1 + bl2},
        "firm_id|business_line_id": {
            ("STRESSED", "FIRM", "BL01"): bl1,
            ("STRESSED", "FIRM", "BL02"): bl2,
        },
    }
    return audit._expected_vector_rows(measure, spec, model.hierarchies["org"], vectors)


def test_derived_rows_carry_the_scale_of_their_operands():
    rows = _rows()
    div_value, div_scale = rows[("svar_99_1d__diversification", "firm_id", ("FIRM",))]
    assert abs(div_value) < 1e-3  # comonotonic: VaR adds
    assert div_scale > 1e10  # the operands, not the near-zero result


def test_noise_on_a_near_zero_diversification_passes_and_a_real_error_fails():
    expected = {("svar_by_org", *k): v for k, v in _rows().items()}
    key = ("svar_by_org", "svar_99_1d__diversification", "firm_id", ("FIRM",))
    exact = {k: v for k, (v, _) in expected.items()}

    noisy = {**exact, key: exact[key] + 1.1444091796875e-05}  # seen on the demo profile
    assert audit._vector_row_problems(noisy, expected) == []

    wrong = {**exact, key: exact[key] + 1e3}
    (problem,) = audit._vector_row_problems(wrong, expected)
    assert "svar_99_1d__diversification" in problem

    # the same holds for an incremental row and for the measure itself
    inc = ("svar_by_org", "svar_99_1d__incremental", "firm_id|business_line_id", ("FIRM", "BL01"))
    assert audit._vector_row_problems({**exact, inc: exact[inc] + 1e-5}, expected) == []
    assert len(audit._vector_row_problems({**exact, inc: exact[inc] + 1e3}, expected)) == 1
    firm = ("svar_by_org", "svar_99_1d", "firm_id", ("FIRM",))
    assert len(audit._vector_row_problems({**exact, firm: exact[firm] + 1e3}, expected)) == 1


def test_close_scales_with_the_operands_not_the_result():
    assert audit._close(1e-5, 0.0, scale=2e10)
    assert not audit._close(1e3, 0.0, scale=2e10, rel=audit.VECTOR_TOLERANCE)
    assert not audit._close(1e-5, 0.0)  # no operands given: the value itself is the scale
