# Stress testing

This page covers what a stress test is and why regulators ask for one, how stress results
aggregate (one measure adds up, another does not), a worked example on the `tiny` profile,
what this project simplifies, and the questions an interviewer is likely to ask. Every
number in the worked example is asserted by `tests/golden/test_stress_worked_example.py`.

## 1. What stress testing is and why regulators ask for it

**Scenario.** A named, hand-written set of shocks to risk factors: "equities fall 30%",
"rates rise 200 basis points". Unlike a historical-simulation scenario (see the VaR page),
it is not drawn from past days. Someone decides it is plausible and severe, and writes it
down.

**Shock.** The size of the move of one risk factor. Units depend on the factor: equity and FX
shocks are relative returns (-0.30 = -30%), interest-rate (GIRR) shocks are absolute changes
in the rate (0.02 = +200 basis points), volatility (VEGA) shocks are absolute changes in
implied volatility (0.25 = 25 volatility points).

**Sensitivity.** How much a position's value changes for a small move in one risk factor.
**Delta** is the first-order effect, **gamma** the second-order (curvature) effect, **vega**
the effect of a change in implied volatility.

**Stress P&L.** The P&L of a position if a scenario happened to the market instantly. Here,
per trade and risk factor:

- on a DELTA factor: `delta × shock + ½ × gamma × shock²`;
- on a VEGA factor: `vega × shock`.

It is summed over the trade's risk factors. Gains are positive, losses negative (the same
convention as every P&L in this project). The **worst stress loss** of a node is the loss
under its worst scenario, reported as a positive number when it is a loss.

**Why regulators ask for it.** VaR and ES describe losses that the recent past says are
likely. They say nothing about what has not happened lately. A stress test asks a different
question: "if this specific bad thing happens, what do we lose, and can we survive it?"

- The Basel Committee's 1996 Market Risk Amendment already required banks using internal
  models to run a rigorous stress-testing programme beside VaR. Stress losses are a check on
  models that are calibrated to the recent past.
- After the 2007 to 2009 crisis the Basel Committee issued principles for sound stress-testing
  practices (2009, revised in 2018), and supervisors made stress testing a routine,
  forward-looking tool rather than an afterthought.
- In the United States the Federal Reserve's annual supervisory stress tests (CCAR and the
  Dodd-Frank Act stress tests) project capital under a prescribed severely adverse scenario.
  For firms with large trading operations that scenario includes a **global market shock**:
  an instantaneous, hypothetical set of market moves applied to the trading book. This
  project flags one scenario (`GFC_2008`) as the global market shock so that the CCAR page
  can reuse it.
- The FRTB internal-models approach prices the risk of factors that cannot be modelled from
  observable prices with a **stress scenario capital requirement**, which is itself a
  stress loss.
- **Reverse stress testing** inverts the question: find the scenario that breaks the bank.
  That needs a search over scenarios and is not built here.

## 2. How stress results aggregate, and why

Two different things are reported, and they aggregate differently.

**Stress P&L is additive within a scenario.** The P&L of a desk under `GFC_2008` is the sum
of its books' P&L under `GFC_2008`, which is the sum of its trades'. Nothing is ranked or
netted across scenarios; each scenario is a separate number. So for any slice of the grain
(desk, region, asset class, currency, risk class, or any combination) the answer is
`SUM(pnl) ... GROUP BY <slice>, scenario`. It is the easy case for an OLAP store: a plain
sum, correct at every level, with no vector to carry and no quantile to apply afterwards.

**The worst scenario is not additive.** The worst stress loss of a node is
`max over scenarios of (loss of the node under that scenario)`. A parent's loss is the
**sum over the children, scenario by scenario, and only then** the maximum is taken. Each
child may be hurt most by a different scenario, and in a scenario that hurts one child
another may be gaining. So the parent's worst loss can be much smaller than the sum of the
children's worst losses. The same reason VaR is not additive (section 2 of the VaR page), but
with a handful of scenarios instead of 250 it is easy to see by hand:

| Scenario | Desk D1 | Desk D2 | Firm |
|---|---:|---:|---:|
| A | -100 | 50 | -50 |
| B | 50 | -100 | -50 |

D1's worst loss is 100 (scenario A) and D2's is 100 (scenario B). The firm's worst loss is 50.
The sum of the desks' worst losses is 200. The worst-of measure also changes which
scenario is worst: here the firm ties between A and B and the tie goes to the name that sorts
first. A node whose scenarios all gain has a negative "worst loss" (the smallest gain);
it is never clipped to zero.

So the project treats `worst_stress_loss` as a **structured** measure
(`config/semantic_model.yaml`): computed once at the declared nodes (the four organisation
levels), stored in `risk_metrics` with the worst scenario's name in the `scenario` column, and
only looked up. A request for it at any other slice is refused, not approximated. `stress_pnl`
is an **additive** measure served from its own table, `stress_results`.

**What this does to the Spark design** (`src/riskagg/measures/stress.py`):

- The shocks are a small table, scenarios × risk factors (5 × 54 on tiny), built on the driver
  from the YAML rules (`shock_table`). It is independent of the trade count and is broadcast.
- Stress P&L per trade comes from one broadcast join of the sensitivities with the shock table
  and one `groupBy` (`stress_pnl`). No Python UDF.
- It is reduced once to the reporting-grain dimensions plus scenario (`stress_by_grain`):
  `asset_class` and `currency` from the trade, the organisation dimensions from the hierarchy,
  which is broadcast. That frame is cached and written as `stress_results`, and it is also what
  the worst-scenario figures are computed from, so the two cannot disagree.
- Per node, scenario P&L is summed first (`groupBy(<node dims>, scenario)`); the worst is then
  the first row of a window ordered by P&L ascending and scenario name ascending
  (`worst_stress_loss`). Ties therefore go to the scenario whose name sorts first, and the
  result does not depend on partitioning.
- The engine reaches the sensitivities, trades and hierarchy through a `RunInputs` object
  (`src/riskagg/engine/inputs.py`) that `MEASURE_REGISTRY` functions receive. Vector measures
  are untouched, so the one-grouping-sets-pass property still holds
  (`test_one_pass_for_shared_specs`).

**What this does to the OLAP layer:**

- `stress_pnl` is answered by `build_query` as a plain `SUM("pnl") ... GROUP BY ...` over
  `stress_results`, valid for any combination of the nine grain dimensions and `scenario`
  (`semantic/query.py`). Parquet and Pinot give the same rows.
- Because every measure served from `stress_results` is additive, `_star_tree` in
  `semantic/pinot_schema.py` gives that table a **star-tree index**. A star-tree
  pre-aggregates the `SUM` of `pnl` along a fixed order of dimensions
  (`dimensionsSplitOrder`), with a special "star" node meaning "all values" of a dimension at
  each level. A query that groups or filters by a prefix-friendly set of those dimensions
  reads a few pre-summed nodes instead of scanning and summing every row; `maxLeafRecords`
  caps how many raw rows a leaf may hold before it is split further. The price is build time
  and storage, and it only helps for the aggregate functions declared in
  `functionColumnPairs` (here `SUM__pnl`). It is safe here because pre-summing a sum is still
  the sum. It is never built for `risk_metrics` or `pnl_scenarios`, where a pre-aggregate of a
  VaR or of a mixed table would be wrong.

## 3. Worked example (tiny profile)

The `tiny` profile has 6 books and 3 desks. `riskagg run` computes the five scenarios of
`config/reference/stress_scenarios.yaml`. Money is rounded to whole dollars here and computed
unrounded, so a column may differ by 1 from the sum of the rounded values.

**One trade, by hand.** A DELTA factor with delta 100 and gamma 50 under a shock of -0.30
loses `100 × (-0.30) + ½ × 50 × 0.09 = -30 + 2.25 = -27.75`. Add a VEGA factor with vega 10
under a volatility shock of +0.20: +2.00. The trade's stress P&L is **-25.75**.

**Which shock applies.** A rule matches on risk class and factor type, and optionally on
bucket and currency; the rule with the most optional fields wins, and a factor that matches
no rule is not shocked (shock 0, never null). In `FX_CRISIS` the general FX delta rule is
-0.05 and the BRL and ZAR rules are -0.20, so the BRL factor gets **-0.20** and the EUR
factor **-0.05**. In `GFC_2008` the same two factors get -0.25 and -0.10.

**Aggregating up.** DESK002 holds BOOK0003 and BOOK0004. Stress P&L by scenario:

| Scenario | BOOK0003 | BOOK0004 | DESK002 |
|---|---:|---:|---:|
| EQUITY_DOWN_30 | 0 | 5,535,602 | 5,535,602 |
| FX_CRISIS | -215,491 | 8,788 | -206,703 |
| GFC_2008 | -505,741 | 6,856,323 | 6,350,582 |
| RATES_UP_200BP | 315,169 | 560,735 | 875,904 |
| VOL_SPIKE | 0 | 262,730 | 262,730 |

Each desk figure is the sum of its books' figures in the same row. The same holds one level
up. Firm stress P&L by scenario, with the three desks:

| Scenario | DESK001 | DESK002 | DESK003 | FIRM |
|---|---:|---:|---:|---:|
| EQUITY_DOWN_30 | -1,338,961 | 5,535,602 | 1,212,374 | 5,409,016 |
| FX_CRISIS | -1,089,169 | -206,703 | -4,191,554 | -5,487,426 |
| GFC_2008 | -4,485,980 | 6,350,582 | -11,111,876 | -9,247,274 |
| RATES_UP_200BP | 3,085,543 | 875,904 | 10,572,677 | 14,534,124 |
| VOL_SPIKE | 97,051 | 262,730 | 0 | 359,781 |

(In the first row the rounded desk figures add up to 5,409,015: the rounding difference.)

**The worst scenario.** Worst loss is minus the lowest P&L in a column:

- DESK002: the worst scenario is FX_CRISIS, loss **206,703**. Its two books are worst in
  different scenarios: BOOK0003 loses **505,741** in GFC_2008, and BOOK0004's worst scenario
  is FX_CRISIS, where it gains 8,788, a "worst loss" of **-8,788**. The books' worst losses
  add up to 505,741 − 8,788 = **496,953**, more than twice the desk's 206,703. In GFC_2008
  BOOK0004's gain of 6,856,323 more than covers BOOK0003's loss.
- Firm: the worst scenario is GFC_2008, loss **9,247,274**. The desks' worst losses are
  4,485,980 (DESK001, GFC_2008), 206,703 (DESK002, FX_CRISIS) and 11,111,876 (DESK003,
  GFC_2008); they add up to **15,804,559**, not 9,247,274.

The firm and desk rows are the `worst_stress_loss` rows that `run_all` writes to
`risk_metrics` through `worst_stress_by_org`; the golden test recomputes each figure from the
generated tiny data through `stress_pnl`, `stress_by_grain` and `run_all`.

## 4. What this project simplifies compared with a real bank

Marked `SIMPLIFICATION:` in the code:

- **Sensitivity-based, not full revaluation.** Stress P&L is a second-order expansion on
  delta and gamma plus linear vega (`stress_pnl` in `src/riskagg/measures/stress.py`). A bank
  reprices every trade, especially options, under the shocked market. The expansion is
  accurate for small shocks and drifts for large ones, and it misses cross-effects such as
  vega changing with spot.
- **Hypothetical, hand-set shocks.** The scenarios in `config/reference/stress_scenarios.yaml`
  are round numbers, not calibrated to a historical episode or to a regulator's published
  scenario (`load_scenarios`). Real scenario design is a governance process.
- **Instantaneous shock.** One move applied at once: no time path, no management actions,
  no liquidity horizons, no second-round effects in factors that no rule shocks
  (`stress_pnl`).
- **Generated sensitivities.** Gamma exists only for option trades, and delta and vega come
  from a synthetic exposure model, not from pricing models
  (`make_sensitivities` in `src/riskagg/data/market.py`, whose curvature figures belong to
  FRTB and are unused here).

Also different from a bank, by the project's scope:

- Five scenarios, with no credit-spread or commodity risk classes (the risk-factor universe
  has only GIRR, EQ and FX).
- No reverse stress testing, no scenario-to-scenario correlation or probability.
- The audit check that rebuilds firm stress P&L from the raw sensitivities uses the same
  shock-matching function as production (`Scenario.shock_for`); that function is covered by
  its own unit tests, but the check does not independently validate the rule semantics
  (`check_stress_firm_pnl_is_sum_of_trades` in `src/riskagg/engine/audit.py`).

## 5. Likely interview questions

1. **What is a stress test, and how is it different from VaR?** VaR reads a quantile off
   losses seen in recent history; a stress test applies a specific, severe, often
   hypothetical move and reports the loss. VaR answers "how bad on a bad day"; stress
   answers "what if this happens". Code: `stress_pnl` in `src/riskagg/measures/stress.py`.
2. **How do you compute stress P&L from sensitivities?** Per risk factor: `delta × shock +
   ½ × gamma × shock²` for DELTA factors, `vega × shock` for VEGA, summed over the trade's
   factors. It is second order, not a revaluation. Code: `stress_pnl`, and `test_formula` in
   `tests/test_stress.py`.
3. **What is wrong with sensitivity-based stress for large shocks?** A Taylor expansion is
   local. For a 30% equity fall or an option near a barrier, higher-order terms and
   cross-effects (vega depends on spot) matter, so banks fully revalue. The code marks this as
   a simplification. Code: the `SIMPLIFICATION:` comments above `stress_pnl`.
4. **How are shocks defined and which one applies to a factor?** Rules match on risk class
   and factor type, optionally bucket and currency; the most specific rule wins, no match
   means zero, and two equally specific matches are rejected as ambiguous. Code:
   `Scenario.shock_for`, `shock_table`, and `test_most_specific_shock_wins`.
5. **Why is stress P&L additive and VaR not?** Within one scenario the P&L of a parent is
   the sum of the children's, and nothing is ranked. VaR ranks the sum over scenarios, and
   ranking does not commute with summing. Code: `build_query` (additive branch) in
   `src/riskagg/semantic/query.py`, and `test_additive_across_hierarchy`.
6. **Is the worst scenario additive? Give an example.** No. If desk 1 loses 100 in A and gains
   50 in B, and desk 2 does the opposite, each desk's worst loss is 100 but the firm loses 50
   in either, so 50 < 200. Netting happens per scenario before taking the worst. Code:
   `worst_stress_loss`, and `test_worst_of_is_not_additive`.
7. **How do you serve the worst stress loss for a slice you did not precompute?** You cannot
   from other slices: it is a structured measure, only looked up at the declared nodes, and
   any other request is refused with an explanation. A new node needs a new grouping set in
   the specification. Code: the structured branch of `build_query`, and
   `test_worst_stress_loss_is_looked_up_not_rebuilt`.
8. **Why is a star-tree index right for `stress_results` and wrong for the VaR tables?**
   A star-tree pre-aggregates SUM along a dimension order; it is only correct when summing
   pre-summed values gives the right answer, true for stress P&L, false for a VaR or for
   mixed-measure tables. The index is enabled only when every measure the table serves is
   additive. Code: `_star_tree` in `src/riskagg/semantic/pinot_schema.py`, and
   `test_star_tree_applies_to_stress_results_only`.
9. **How do you keep the Spark job cheap when the shocks multiply the data?** The shock table
   is scenarios × risk factors (hundreds of rows), broadcast; the sensitivities are joined
   once and aggregated per trade and scenario, then reduced to the grain. Nothing proportional
   to the trade count reaches the driver. Code: `shock_table`, `stress_pnl`, `stress_by_grain`.
10. **How do you break ties between scenarios, and why does it matter?** Equal worst losses
    go to the scenario name that sorts first, so reruns and different partitionings give the
    same stored scenario; otherwise the published "worst scenario" could flip between runs.
    Code: the window ordering in `worst_stress_loss`, and
    `test_worst_tie_goes_to_first_name_and_gains_are_not_clipped`.
11. **What sign does a worst loss have when every scenario gains?** Negative: it is reported
    as minus the lowest P&L and never clipped, like VaR for a node with only gains. Clipping
    would hide that the node is a net beneficiary and would break reconciliation. Code:
    `worst_stress_loss`.
12. **How do you know the stress numbers reconcile?** The audit check rebuilds each
    scenario's firm P&L from the raw sensitivities (summed per risk factor, shocks applied in
    NumPy) and compares it to the sum of the curated `stress_results`, and checks the firm
    worst-loss row against the same totals. A run that fails is not marked official. Code:
    `check_stress_firm_pnl_is_sum_of_trades` in `src/riskagg/engine/audit.py`.
