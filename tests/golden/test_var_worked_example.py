"""Golden values for the worked example in ``docs/study/01-var-es.md``.

Every number the worked example prints is a constant here. Each one is recomputed from the
tiny profile through the real engine (reporting grain, grouping-sets node vectors, the metric
plan with diversification) and compared, and each is looked up in the study page, so the page
and the code cannot drift apart. Money is printed rounded to whole dollars.
"""

import dataclasses

import numpy as np
import pytest

from riskagg import conventions
from riskagg.engine.grain import GRAIN_DIMS, build_reporting_grain
from riskagg.engine.runner import node_vectors, run_all
from riskagg.engine.spec import AggregationSpec
from riskagg.paths import REPO_ROOT
from riskagg.semantic.model import Measure, load_model

STUDY_PAGE = REPO_ROOT / "docs/study/01-var-es.md"
ORG = ("firm_id", "business_line_id", "desk_id", "book_id")
DESK_LEVEL = "|".join(ORG[:3])
BOOK_LEVEL = "|".join(ORG)
DESK, BOOKS = "DESK001", ("BOOK0001", "BOOK0002")
N_SCENARIOS = 20
VAR_LEVEL, ES_LEVEL = 0.90, 0.80
VAR_RANK, ES_TAIL = 2, 4

# One-day P&L per scenario (index 0..19), CURRENT window. Gains positive.
PNL = {
    "BOOK0001": [
        104289, -5409, -111142, 81996, 47287, -13372, -55747, -56930, 194775, -26043,
        -32101, 21081, -103175, -135523, -51017, 204318, -43169, -97927, -74629, 114096,
    ],
    "BOOK0002": [
        -31306, 257451, 379576, 85816, -438084, 51407, -34537, -17868, -14037, -352777,
        -95446, -109196, 260169, 303314, -57773, -434430, -140589, -117470, -18394, -159607,
    ],
    "DESK001": [
        72983, 252041, 268434, 167812, -390797, 38034, -90283, -74798, 180738, -378820,
        -127547, -88115, 156994, 167791, -108790, -230112, -183757, -215397, -93023, -45511,
    ],
}  # fmt: skip
# Losses (= -P&L), worst first, as (scenario, loss).
DESK_LOSSES = [
    (4, 390797), (9, 378820), (15, 230112), (17, 215397), (16, 183757),
    (10, 127547), (14, 108790), (18, 93023), (6, 90283), (11, 88115),
    (7, 74798), (19, 45511), (5, -38034), (0, -72983), (12, -156994),
    (13, -167791), (3, -167812), (8, -180738), (1, -252041), (2, -268434),
]  # fmt: skip
WORST_FOUR = {
    "BOOK0001": [(13, 135523), (2, 111142), (12, 103175), (17, 97927)],
    "BOOK0002": [(4, 438084), (15, 434430), (9, 352777), (19, 159607)],
    "DESK001": DESK_LOSSES[:4],
}
VAR = {"BOOK0001": 111142, "BOOK0002": 434430, "DESK001": 378820}
ES = {"BOOK0001": 111942, "BOOK0002": 346224, "DESK001": 303781}
VAR_SUM_OF_BOOKS, VAR_DIVERSIFICATION = 545572, 166752
ES_SUM_OF_BOOKS, ES_DIVERSIFICATION = 458166, 154385
INCREMENTAL = {"BOOK0001": -55610, "BOOK0002": 267678}
INCREMENTAL_SUM = 212068  # not the desk VaR


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _measure(name: str, function: str, level: float) -> Measure:
    params = {"level": level, "window": "CURRENT", "horizon_days": 1}
    return Measure(name, name, "", "USD", "vector", function, params, "risk_metrics", GRAIN_DIMS)


@pytest.fixture(scope="module")
def engine(tiny_market):
    """Node vectors and metrics for the org hierarchy, at 90% VaR and 80% ES."""
    pnl, hierarchy, n = tiny_market
    assert n == N_SCENARIOS
    var90, es80 = _measure("var_90", "var", VAR_LEVEL), _measure("es_80", "es", ES_LEVEL)
    base = load_model()
    model = dataclasses.replace(base, measures={**base.measures, "var_90": var90, "es_80": es80})
    sets = tuple(ORG[: i + 1] for i in range(4))
    specs = [AggregationSpec(f"worked_{m}", m, sets, {}, True) for m in ("var_90", "es_80")]
    grain = build_reporting_grain(pnl, hierarchy, n)
    vectors = {}
    for r in node_vectors(specs, model, grain, n).where("window = 'CURRENT'").collect():
        if r.node_level == DESK_LEVEL and r.desk_id == DESK:
            vectors[DESK] = np.array(r.pnl)
        elif r.node_level == BOOK_LEVEL and r.book_id in BOOKS:
            vectors[r.book_id] = np.array(r.pnl)
    metrics = {}
    for r in run_all(specs, model, grain, n).collect():
        node = r.book_id if r.node_level == BOOK_LEVEL else r.desk_id
        if r.node_level in (DESK_LEVEL, BOOK_LEVEL) and node in (DESK, *BOOKS):
            metrics[(r.measure, node)] = r.value
    return vectors, metrics


def test_convention_ranks():
    assert conventions.var_rank(N_SCENARIOS, VAR_LEVEL) == VAR_RANK
    assert conventions._tail_size(N_SCENARIOS, ES_LEVEL) == ES_TAIL
    # The production levels quoted on the page: 3rd-worst of 250 at 99%; ES tail 6.25.
    assert conventions.var_rank(250, 0.99) == 3
    assert conventions._tail_size(250, 0.975) == 6.25


def test_pnl_vectors(engine):
    vectors, _ = engine
    assert set(vectors) == {DESK, *BOOKS}
    for node, expected in PNL.items():
        assert [_money(x) for x in vectors[node]] == [_money(x) for x in expected], node
    # The desk vector is the element-wise sum of its books' vectors.
    np.testing.assert_allclose(vectors[DESK], vectors[BOOKS[0]] + vectors[BOOKS[1]], atol=1e-6)


def test_sorted_losses(engine):
    vectors, _ = engine
    for node, worst in WORST_FOUR.items():
        order = np.argsort(vectors[node], kind="stable")  # most negative P&L first
        got = [(int(s), _money(-vectors[node][s])) for s in order]
        assert got[:4] == [(s, _money(x)) for s, x in worst], node
        if node == DESK:
            assert got == [(s, _money(x)) for s, x in DESK_LOSSES]


def test_var_es_diversification_incremental(engine):
    _, m = engine
    for node in (DESK, *BOOKS):
        assert _money(m[("var_90", node)]) == _money(VAR[node]), node
        assert _money(m[("es_80", node)]) == _money(ES[node]), node
        # VaR is the VAR_RANK-th worst loss; ES the mean of the ES_TAIL worst.
        assert m[("var_90", node)] == pytest.approx(
            -np.sort(PNL[node])[VAR_RANK - 1], abs=1.0
        )
    assert _money(m[("var_90__sum_children", DESK)]) == _money(VAR_SUM_OF_BOOKS)
    assert _money(m[("var_90__diversification", DESK)]) == _money(VAR_DIVERSIFICATION)
    assert _money(m[("es_80__sum_children", DESK)]) == _money(ES_SUM_OF_BOOKS)
    assert _money(m[("es_80__diversification", DESK)]) == _money(ES_DIVERSIFICATION)
    for book, other in (BOOKS, BOOKS[::-1]):
        inc = m[("var_90__incremental", book)]
        assert _money(inc) == _money(INCREMENTAL[book])
        # Removing one of two books leaves the other: incremental = desk VaR - other's VaR.
        assert inc == pytest.approx(m[("var_90", DESK)] - m[("var_90", other)], abs=1e-6)
    total = sum(m[("var_90__incremental", b)] for b in BOOKS)
    assert _money(total) == _money(INCREMENTAL_SUM) != _money(m[("var_90", DESK)])


def test_non_subadditive_example():
    """The page's two-book example where VaR breaks subadditivity and ES does not."""
    x = np.zeros(N_SCENARIOS)
    y = np.zeros(N_SCENARIOS)
    x[0], y[1] = -100.0, -100.0
    var = conventions.var_from_pnl
    es = conventions.es_from_pnl
    assert var(x, VAR_LEVEL) == var(y, VAR_LEVEL) == 0.0
    assert var(x + y, VAR_LEVEL) == 100.0
    assert es(x, ES_LEVEL) == es(y, ES_LEVEL) == 25.0
    assert es(x + y, ES_LEVEL) == 50.0


def test_study_page_prints_these_numbers():
    text = STUDY_PAGE.read_text()
    a, b, d = PNL["BOOK0001"], PNL["BOOK0002"], PNL["DESK001"]
    for s in range(N_SCENARIOS):
        row = f"| {s} | {_money(a[s])} | {_money(b[s])} | {_money(d[s])} |"
        assert row in text, row
    for rank, (s, loss) in enumerate(DESK_LOSSES, start=1):
        row = f"| {rank} | {s} | {_money(loss)} |"
        assert row in text, row
    for node, worst in WORST_FOUR.items():
        line = f"{node} worst four: " + ", ".join(f"{_money(x)} (s{s})" for s, x in worst)
        assert line in text, line
    for value in (
        *VAR.values(),
        *ES.values(),
        VAR_SUM_OF_BOOKS,
        VAR_DIVERSIFICATION,
        ES_SUM_OF_BOOKS,
        ES_DIVERSIFICATION,
        *INCREMENTAL.values(),
        INCREMENTAL_SUM,
    ):
        assert _money(value) in text, value
