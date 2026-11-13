# Value at Risk and Expected Shortfall

This page covers what the two measures are, why a regulator asks for them, how they
aggregate (and how that shapes the Spark engine and the OLAP layer), a worked example you
can check by hand, what this project simplifies, and the questions an interviewer is likely
to ask. Every number in the worked example is asserted by
`tests/golden/test_var_worked_example.py`.

## 1. What the measures are and why regulators ask for them

**P&L (profit and loss).** The change in value of a position over a period. In this project
P&L is positive for a gain and negative for a loss.

**Scenario.** One possible move of the market over one day: a value for every risk factor
(rate, equity price, FX rate). **Historical simulation** takes the scenarios from real past
days. For example, scenario 17 is "what the market did on the 17th day of the window".
The **window** is the set of days used. This project has two windows: `CURRENT`, standing for
the most recent year, and `STRESSED`, standing for a period of market turmoil.

**P&L vector.** For one position, its P&L under each scenario, in scenario order:
`[pnl_0, pnl_1, ..., pnl_{N-1}]`. With 250 scenarios a vector holds 250 numbers. Every risk
number on this page is a function of one such vector.

**Loss.** Minus the P&L. A P&L of -390,797 is a loss of 390,797.

**Value at Risk (VaR) at level α** (for example α = 99%). A loss that you exceed on only
(1 − α) of days. A one-day 99% VaR of 10m means: on 99 days out of 100, you expect to lose
less than 10m. With historical simulation, VaR is read straight off the sorted losses.

**Expected Shortfall (ES) at level α**, also called conditional VaR. The average loss on
the days in the worst (1 − α) tail. VaR says where the tail starts. ES says how bad the tail
is on average.

**The quantile convention.** With N scenarios the tail holds `k = N × (1 − α)` scenarios.
This project, in `src/riskagg/conventions.py`, uses:

- VaR is the loss ranked `max(1, ceil(k))` from the worst. No interpolation. With 250
  scenarios at 99%, k = 2.5, so VaR is the 3rd-worst loss.
- ES is the sum of the `floor(k)` worst losses, plus the fraction `k − floor(k)` of the next
  one, divided by k. With 250 scenarios at 97.5%, k = 6.25: the 6 worst losses, plus a
  quarter of the 7th, all divided by 6.25.

Banks differ here: some interpolate between ranks, some take the 2nd or 3rd worst. Two
systems that disagree on the convention will not reconcile, so the convention lives in one
module and nothing else ranks losses.

**Why regulators ask for them.** A bank's trading book needs capital to absorb the losses
the market can inflict. A regulator needs a number that is comparable across banks and
checkable after the fact.

- The Basel Committee's 1996 Market Risk Amendment let banks use internal models for
  market-risk capital: VaR at 99%, one-tailed, over a 10-day holding period, from at least
  one year of history. It allowed a 10-day figure to be scaled from a one-day figure by the
  square root of 10. Capital was the larger of yesterday's VaR and a multiplier (at least 3)
  times the 60-day average VaR.
- **Backtesting** checks the model. Each day's actual loss is compared with the previous
  day's one-day 99% VaR; a loss above it is an **exception**. Over 250 days, 0 to 4
  exceptions is the green zone, 5 to 9 amber (the multiplier is raised), 10 or more red.
- **Basel 2.5** (2009) added **stressed VaR**: the same 99% VaR, but computed on a
  12-month window of significant stress. The 2008 crisis had shown that VaR computed on a
  calm year is far too low when markets turn.
- The **Fundamental Review of the Trading Book** (FRTB: Basel standard of January 2016,
  revised January 2019) replaced VaR and stressed VaR in the internal-models capital charge
  with ES at 97.5%. The ES is calibrated to a stress period and scaled from a 10-day base to
  liquidity horizons of 10, 20, 40, 60 or 120 days. VaR remains for backtesting: each
  trading desk is backtested on one-day VaR at both 97.5% and 99%.

Why ES at 97.5% replaced VaR at 99%:

1. VaR ignores the size of losses beyond the quantile. Two portfolios can have the same
   VaR while one of them loses ten times more on its worst day. ES averages those losses, so
   it sees them.
2. VaR is not **subadditive**: the VaR of a combined portfolio can be larger than the sum
   of the parts' VaRs (section 2 has an example). ES is always subadditive, so merging
   two books can never raise ES above the sum of their ES values. For a normal
   distribution, 97.5% ES is about 2.34 standard deviations and 99% VaR about 2.33, which
   is why that pair of levels was chosen: similar capital in calm conditions, but ES also
   sees fat tails.

## 2. How the measures aggregate, and why

**Additive** means the parent's value is the sum of the children's values. Notional and
P&L in one scenario are additive. VaR and ES are not. Each child's VaR comes from that
child's own worst days, and those are different days from the parent's worst days. Losses
on one book are offset by gains on another in the same scenario. Netting happens scenario by
scenario, before ranking, and ranking does not commute with summing.

The only correct way to get a parent's VaR or ES is therefore:

1. sum the children's P&L vectors **element by element** (scenario 0 with scenario 0,
   scenario 1 with scenario 1, and so on), which gives the parent's P&L vector;
2. apply VaR or ES to that one vector.

This project calls such a measure a **vector** measure. A vector measure is never produced
by summing metric values: not in Spark, not in Pinot SQL, not in pandas.

**Diversification benefit** is the sum of the children's standalone values minus the
parent's value. For VaR it is usually positive but can be negative, because VaR is not
subadditive. Example with 20 scenarios at 90% (so VaR is the 2nd-worst loss): book X loses
100 in scenario 0 and nothing otherwise; book Y loses 100 in scenario 1 and nothing
otherwise. Each book's 2nd-worst loss is 0, so each VaR is 0. The combined book loses 100
in two scenarios, so its VaR is 100, more than 0 + 0. At 80% ES (the mean of the 4 worst
losses) each book has ES 25 and the combined book 50, which does not exceed 25 + 25.

**Diversification on the demo profile is almost nil, and that is the data, not the
engine.** On the `demo` profile (seed 42) the firm's stressed 99% VaR, 21.07 billion, equals
the sum of its four business lines' stressed VaRs, 21.07 billion, to within float noise (a
benefit of 0.00%); in the current window the benefit is 0.34% of the sum (firm 6.04 billion,
business lines 6.06 billion). The cause is in the generator. Every book draws its trades from
the same mix of asset classes and currencies over the same 54 risk factors, with 70% of trades
long (`make_trades` in `src/riskagg/data/org.py`, `LONG_SHARE` in
`src/riskagg/data/market.py`), and holds about 500 trades. So books are net long in 84% of
(book, risk factor) pairs, with a median net exposure of 44% of gross, and
every business line's P&L vector is close to a scaled copy of the firm's (correlation above
0.999). Their losses are **comonotonic**: the same scenario is the ranked loss for every
child. In the stressed window scenario 64 is the third-worst loss (the 99% VaR of 250
scenarios) of the firm and of all four business lines, and when the same scenario is the
ranked loss of every child, VaR adds exactly. In the current window the firm's third-worst is
scenario 84, as for three business lines, but BL01's is scenario 78; that one difference is
the whole benefit. A real bank's desks hold different and often offsetting positions (a desk
hedging another, market makers both long and short), so the benefit between desks is usually
a substantial share of their summed VaR. The tiny example in section 3, whose books offset
each other, shows that case. Pinned by `tests/golden/test_demo_diversification.py`.

**Incremental VaR** of a child is the parent's VaR minus the parent's VaR without that
child: `VaR(parent) − VaR(parent vector − child vector)`. It answers "how much does this
book add to the desk's risk?". It is not the child's standalone VaR, and it is negative when
the child hedges the rest of the parent. Incremental values of the children do not sum to
the parent either.

**What this does to the Spark design** (`src/riskagg/engine/`):

- P&L vectors stay as `array<double>` columns, one row per position, and are summed element
  by element inside the aggregation (`sum_vectors` in `engine/vectors.py`: one native
  `sum(pnl[i])` per scenario, no Python UDF). Exploding to one row per scenario would
  multiply the row count by 250 before the shuffle.
- Trades are reduced once to a **reporting grain**: one vector per book × legal entity ×
  region × asset class × currency × risk class × liquidity horizon × reduced-set flag ×
  window (`build_reporting_grain` in `engine/grain.py`; FRTB IMA needs the liquidity horizon
  and the reduced-set flag, and VaR and ES sum over them). Every hierarchy node is built
  from that small result, not from trades.
- All the nodes the specifications ask for are built in one grouping-sets pass
  (`aggregate_vectors`); only declared groupings are computed, never the full cube.
- VaR, ES, sum of children, diversification and incremental VaR are then derived from the
  node vectors in one pass (`measures/var.py`). A parent's vector reaches its children
  through a window over the parent's dimensions, so the incremental `parent − child` is an
  element-wise `zip_with`.

**What this does to the OLAP layer** (`src/riskagg/semantic/`, `src/riskagg/serving/`):

- `risk_metrics` holds VaR and ES only at the nodes the specifications computed. The
  shipped ones store `var_99_1d`, `var_99_10d`, `es_975_1d` and `svar_99_1d` at the firm and
  every business line, desk and book, each with its sum-of-children, diversification and
  incremental rows (`<measure>__sum_children`, `__diversification`, `__incremental`), plus
  `var_99_1d` by desk and asset class (`config/aggregations.yaml`).
- `pnl_scenarios` holds the reporting-grain vectors exploded to one row per scenario. Within
  one scenario P&L is additive, so the query layer can answer VaR for any combination of the
  grain dimensions. The SQL only sums P&L per scenario (`GROUP BY <dims>, scenario_idx`).
  The quantile is applied afterwards in Python by the same convention functions, for Pinot
  and Parquet alike (`build_query` and `finish` in `semantic/query.py`).
- A slice outside the grain (counterparty, for example) is refused with an explanation,
  because the vectors needed are not stored. Summing the stored VaRs would give a wrong
  number, so the query layer does not try.
- No star-tree index (Pinot's pre-aggregation) is built on a table that serves a vector
  measure, because pre-aggregating the metric values would mean summing them
  (`_star_tree` in `semantic/pinot_schema.py`).
- A truncated Pinot result would silently move the quantile, so the executor refuses
  partial results and `finish` checks that every group has every scenario exactly once.

## 3. Worked example (tiny profile, 90% VaR and 80% ES)

The `tiny` profile has 20 scenarios, 3 desks and 6 books. At these sizes the regulatory
levels are degenerate: 99% VaR of 20 scenarios is just the worst loss. So the example uses
**90% VaR**, where k = 20 × 0.10 = 2 and VaR is the **2nd-worst loss**, and **80% ES**, where
k = 20 × 0.20 = 4 and ES is the **mean of the 4 worst losses**. The ranking, the averaging
and the diversification rows are those of 99% VaR and 97.5% ES on 250 scenarios, with one
difference: both k here are whole numbers, so the fractional weight that 97.5% ES on 250
scenarios puts on its 7th-worst loss (k = 6.25) is not exercised.

Desk DESK001 holds two books, BOOK0001 and BOOK0002. Their one-day P&L vectors (window
`CURRENT`, summed over everything below the book) come from `riskagg generate --profile
tiny` followed by the engine. Money is rounded to whole dollars here and computed
unrounded, so a column may differ by 1 from the sum of the rounded values. The desk column
is the element-wise sum of the two book columns.

| Scenario | BOOK0001 | BOOK0002 | DESK001 |
|---:|---:|---:|---:|
| 0 | 104,289 | -31,306 | 72,983 |
| 1 | -5,409 | 257,451 | 252,041 |
| 2 | -111,142 | 379,576 | 268,434 |
| 3 | 81,996 | 85,816 | 167,812 |
| 4 | 47,287 | -438,084 | -390,797 |
| 5 | -13,372 | 51,407 | 38,034 |
| 6 | -55,747 | -34,537 | -90,283 |
| 7 | -56,930 | -17,868 | -74,798 |
| 8 | 194,775 | -14,037 | 180,738 |
| 9 | -26,043 | -352,777 | -378,820 |
| 10 | -32,101 | -95,446 | -127,547 |
| 11 | 21,081 | -109,196 | -88,115 |
| 12 | -103,175 | 260,169 | 156,994 |
| 13 | -135,523 | 303,314 | 167,791 |
| 14 | -51,017 | -57,773 | -108,790 |
| 15 | 204,318 | -434,430 | -230,112 |
| 16 | -43,169 | -140,589 | -183,757 |
| 17 | -97,927 | -117,470 | -215,397 |
| 18 | -74,629 | -18,394 | -93,023 |
| 19 | 114,096 | -159,607 | -45,511 |

**Sorting the desk's losses.** Loss = −P&L, worst first. A negative loss is a gain.

| Rank | Scenario | Loss |
|---:|---:|---:|
| 1 | 4 | 390,797 |
| 2 | 9 | 378,820 |
| 3 | 15 | 230,112 |
| 4 | 17 | 215,397 |
| 5 | 16 | 183,757 |
| 6 | 10 | 127,547 |
| 7 | 14 | 108,790 |
| 8 | 18 | 93,023 |
| 9 | 6 | 90,283 |
| 10 | 11 | 88,115 |
| 11 | 7 | 74,798 |
| 12 | 19 | 45,511 |
| 13 | 5 | -38,034 |
| 14 | 0 | -72,983 |
| 15 | 12 | -156,994 |
| 16 | 13 | -167,791 |
| 17 | 3 | -167,812 |
| 18 | 8 | -180,738 |
| 19 | 1 | -252,041 |
| 20 | 2 | -268,434 |

**Which rank and why.** k = 2 is a whole number, so `ceil(k) = 2`: the desk's 90% VaR is
the rank-2 loss, **378,820** (scenario 9). Exactly 2 of the 20 scenarios (10%) lose at least
that much. The desk's 80% ES is the mean of ranks 1 to 4:
(390,797 + 378,820 + 230,112 + 215,397) / 4 = **303,781**.

The same steps for each vector (worst first; sN is scenario N):

- BOOK0001 worst four: 135,523 (s13), 111,142 (s2), 103,175 (s12), 97,927 (s17)
- BOOK0002 worst four: 438,084 (s4), 434,430 (s15), 352,777 (s9), 159,607 (s19)
- DESK001 worst four: 390,797 (s4), 378,820 (s9), 230,112 (s15), 215,397 (s17)

| Node | 90% VaR (rank 2) | 80% ES (mean of 4 worst) |
|---|---:|---:|
| BOOK0001 | 111,142 | 111,942 |
| BOOK0002 | 434,430 | 346,224 |
| DESK001 | 378,820 | 303,781 |

(ES here can be below VaR because the ES level, 80%, is lower than the VaR level, 90%.)

**Diversification.** The desk's VaR is not the sum of its books' VaRs:

- sum of the books' VaRs: 111,142 + 434,430 = **545,572**;
- desk VaR: **378,820**;
- diversification benefit: 545,572 − 378,820 = **166,752**.

The reason is visible in the table. The books' VaRs come from different days, scenario 2
for BOOK0001 and scenario 15 for BOOK0002. On scenario 15, BOOK0001 gained 204,318, which
offset most of BOOK0002's loss. The desk's VaR comes from yet another day, scenario 9. For
ES: sum of the books' ES 111,942 + 346,224 = **458,166**, desk ES 303,781, diversification
**154,385**. ES is subadditive, so this can never be negative.

**Incremental VaR.** Removing BOOK0001 from the desk leaves BOOK0002 alone, so the desk VaR
without BOOK0001 is BOOK0002's VaR, 434,430. BOOK0001's incremental VaR is
378,820 − 434,430 = **-55,610**: BOOK0001 reduces the desk's risk, because it tends to gain
when BOOK0002 loses. Its standalone VaR of 111,142 says nothing about that. Likewise
BOOK0002's incremental VaR is 378,820 − 111,142 = **267,678**. The two incrementals
(−55,610 + 267,678 = 212,068) do not add up to the desk's VaR either.

In the engine these are the rows `var_90`, `var_90__sum_children`, `var_90__diversification`
and `var_90__incremental` that `run_all` produces for an org-hierarchy specification at these
levels. The golden test declares that specification itself (no shipped one is at 90% or
80%), builds the rows from the generated tiny data through `build_reporting_grain`,
`node_vectors` and `run_all`, and checks every number above. The official run stores the same
kinds of rows for the production measures (`es_975_1d__diversification`, for example), where
the ES tail of 250 scenarios is fractional.

## 4. What this project simplifies compared with a real bank

Marked `SIMPLIFICATION:` in the code:

- **Square-root-of-time scaling.** Multi-day horizons scale the one-day figure by
  sqrt(horizon_days) instead of using overlapping multi-day P&L. This assumes independent,
  identically distributed daily P&L with zero mean and an unchanged portfolio. It is in
  `src/riskagg/measures/var.py` (`_value_fn`) and `src/riskagg/semantic/query.py`
  (`finish`), which must agree.
- **Factor model.** One common factor per risk class plus one market-wide factor, and a
  single correlation level per window, instead of an estimated correlation matrix
  (`src/riskagg/data/market.py`). Real P&L vectors come from real market history.
- **Stressed window.** The stressed window is the current factor model with volatility
  tripled and correlation raised, not a historical 12-month period of significant stress
  chosen by the bank (`src/riskagg/data/market.py`).
- **Linear P&L.** Scenario P&L is linear in the factor returns (exposure × return), not a
  full revaluation of each trade under each historical scenario
  (`make_pnl_vectors` in `src/riskagg/data/market.py`). Options in a bank are repriced.
- **Books are alike.** Every book draws its trades from the same mix of asset classes,
  currencies and risk factors, mostly long, so on the `demo` profile books, desks and
  business lines are nearly comonotonic and the diversification benefit between them is about
  zero (section 2). Real desks have mandates and hedge each other
  (`LONG_SHARE` in `src/riskagg/data/market.py`, `make_trades` in `src/riskagg/data/org.py`).

Also different from a bank, by the project's scope:

- 20 or 250 synthetic scenarios, equally weighted, rather than years of real history.
- No backtesting against actual P&L, no P&L attribution test, no multiplier.
- ES is computed one-day on the current window. FRTB's liquidity-horizon cascade and stress
  calibration are covered in the FRTB page.

## 5. Likely interview questions

1. **How do you compute historical-simulation VaR?** Build the portfolio's P&L vector over
   the scenarios, negate it to get losses, sort, and take the loss at rank
   `max(1, ceil(N × (1 − α)))`. Code: `var_from_pnl` and `var_col` in
   `src/riskagg/conventions.py`.
2. **Which loss is the 99% VaR of 250 scenarios, and why does it matter?** The 3rd worst,
   because 250 × 1% = 2.5 rounds up to 3, with no interpolation. Others take the 2nd or
   interpolate, and the systems then fail to reconcile. Code: `var_rank` in
   `src/riskagg/conventions.py`.
3. **Why can't you add up desk VaRs to get the firm's VaR?** Each desk's VaR comes from its
   own worst scenarios, and gains on one desk offset losses on another in the same scenario.
   Sum the vectors scenario by scenario, then take the quantile. Code: `aggregate_vectors`
   in `src/riskagg/engine/vectors.py`.
4. **How do you answer VaR for an ad-hoc slice, say region × asset class?** Store P&L per
   scenario at the reporting grain (`pnl_scenarios`). The query sums P&L by the slice's
   dimensions and scenario, and the quantile is applied to each group's vector afterwards.
   Code: `build_query` and `finish` in `src/riskagg/semantic/query.py`.
5. **Why is VaR by counterparty refused?** Counterparty is not in the reporting grain, so no
   vector exists at that level, and any number built from stored VaRs would be wrong. The
   request gets a 422 with the reason. Code: `_require_valid` and `SliceNotAvailable` in
   `src/riskagg/semantic/query.py`.
6. **What is Expected Shortfall and why did FRTB move to it?** The average loss beyond the
   quantile. It captures tail severity and is subadditive. VaR is neither. Basel moved the
   internal-models charge from 99% VaR to 97.5% ES. Code: `es_from_pnl` in
   `src/riskagg/conventions.py`, and `test_es_subadditive` in `tests/test_var.py`.
7. **What are the diversification benefit and incremental VaR?** Diversification is the sum
   of the children's VaRs minus the parent's VaR. A child's incremental VaR is the parent's
   VaR minus the parent's VaR without the child; it can be negative for a hedge. Code:
   `plan_diversification` in `src/riskagg/measures/var.py`.
8. **How do you get a 10-day VaR?** Here, by scaling one-day VaR by sqrt(10), which assumes
   i.i.d. daily P&L and a static portfolio (marked as a simplification). Overlapping 10-day
   P&L is the alternative. Code: `_value_fn` in `src/riskagg/measures/var.py`.
9. **What is stressed VaR?** The same VaR on a window of market stress (Basel 2.5). Here it is
   `svar_99_1d`, the `STRESSED` window. Code: `config/semantic_model.yaml` and
   `factor_returns` in `src/riskagg/data/market.py`.
10. **How do you keep the Spark job cheap with 250-element vectors?** Keep them as arrays and
    sum element by element with native aggregate expressions, so rows never multiply by the
    scenario count. Reduce to the reporting grain once, then compute only the declared
    groupings in a single grouping-sets pass. Code: `sum_vectors` and `aggregate_vectors` in
    `src/riskagg/engine/vectors.py`, and `build_reporting_grain` in
    `src/riskagg/engine/grain.py`.
11. **How does Pinot serve VaR without summing VaR?** Pinot only ever sums P&L within a
    scenario; the quantile runs afterwards in Python with the same convention. Partial or
    truncated results are refused, and no star-tree pre-aggregates a vector table. If Pinot
    is down or holds another run, the service answers from Parquet and says so. Code:
    `Query.to_pinot_sql` in `src/riskagg/semantic/query.py`, `PinotExecutor` in
    `src/riskagg/serving/executors.py`, `RiskService` in `src/riskagg/serving/service.py`.
12. **How do you prove a reported VaR is right and reproducible?** Every row carries the run
    id, as-of date and a hash of the specification and reference data. A manifest records
    input checksums. Reconciliation checks recompute every stored VaR and ES row (each node,
    with its diversification rows) from the curated vectors, and check every curated vector
    cell against the raw trades. Only a run that passes them is marked official and served.
    Code: `reconcile`, `check_vector_measures_from_pnl_scenarios` and
    `check_pnl_scenarios_match_raw_per_grain_cell` in `src/riskagg/engine/audit.py`,
    `mark_official` in `src/riskagg/context.py`.
