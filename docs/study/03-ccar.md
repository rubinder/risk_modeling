# CCAR: projecting capital through a stress scenario

This page covers:

- what the Federal Reserve's supervisory stress test and capital planning rules (CCAR, DFAST)
  are, and who they apply to;
- the scenarios, the nine-quarter horizon and the building blocks of the projection;
- how CET1 capital moves quarter to quarter, and how the stress capital buffer is computed;
- why the aggregation is a path through time rather than a quantile or a cube roll-up, and why
  capital ratios cannot be cut by business line;
- a worked example on the `tiny` profile;
- what this project simplifies, and the questions an interviewer is likely to ask.

Every number in the worked example is asserted by
`tests/golden/test_ccar_worked_example.py`. Regulatory statements are cited to the rule text or
to the Federal Reserve's own documentation (sources at the end of section 1).

## 1. What CCAR is and why the Federal Reserve runs it

**The exercise.** The Federal Reserve Board runs an annual **supervisory stress test** of large
banks. Its own description is that the test "assesses whether banks are sufficiently
capitalized to absorb losses during stressful conditions while meeting obligations to creditors
and counterparties and continuing to be able to lend to households and businesses" [Fed-ST]. The
Board projects each bank's losses, revenue and capital through a hypothetical recession and
asks whether capital stays strong enough.

**CCAR and DFAST.** From 2013 the Federal Reserve's capital assessment of large banks had two
parts: "the Dodd-Frank Act Stress Test (stress test) and the Comprehensive Capital Analysis and
Review (CCAR)" [Fed-ST].

- **DFAST** is the quantitative projection: losses, revenue and capital ratios under the
  scenarios.
- **CCAR** added a judgement on each bank's capital plan: a quantitative pass/fail and a
  qualitative review of capital planning.
- Both parts have since been folded into other processes: "In 2019, the Board incorporated the
  qualitative evaluation into the standard, confidential supervisory process. In 2020, the
  Federal Reserve replaced the quantitative CCAR evaluation with the stress capital buffer"
  [Fed-ST].

So today the projection still runs every year, and its result sets each bank's **stress capital
buffer** (section 1.6). "CCAR" survives as the everyday name for the whole cycle, and it is the
name used here.

**Who it applies to.** Bank holding companies, savings and loan holding companies and US
intermediate holding companies of foreign banks with $100 billion or more in assets
[Fed-ST, footnote 1]. For bank holding companies and US intermediate holding companies the
stress test rule is 12 CFR part 252, subpart E (scope in 252.43(a)) and the capital plan rule is
12 CFR 225.8 (scope in 225.8(b)(1)); for savings and loan holding companies they are 12 CFR
part 238, subparts O and S [Fed-ST, footnote 1].

- The largest (global systemically important banks and Category II and III firms) are tested
  every year.
- Category IV firms are tested every other year, "in each year ending in an even number"
  [12 CFR 252.44(d)(1), Table 1].

**Why.** A bank's regular capital ratios describe the past. The stress test asks a
forward-looking question: if a severe recession and a market crash started today, would this
bank still have enough capital at the worst point? The rule frames it as an analysis "to
evaluate the ability of the covered company to absorb losses in specified economic and
financial conditions" [12 CFR 252.44(a)(1)].

### 1.1 The scenarios

The Board uses "a minimum of two different scenarios, including a baseline scenario and a
severely adverse scenario" [12 CFR 252.44(b)].

- **Baseline:** "a set of conditions that affect the U.S. economy or the financial condition of a
  covered company and that reflect the consensus views of the economic and financial outlook"
  [12 CFR 252.42].
- **Severely adverse:** conditions that "overall are significantly more severe than those
  associated with the baseline scenario and may include trading or other additional
  components" [12 CFR 252.42]. In practice it is a deep recession: unemployment rises sharply,
  output falls, asset prices drop.

The scenarios are paths of macroeconomic and financial variables, quarter by quarter. Here they
are two hand-set paths of two variables, the unemployment rate and annualised real GDP growth
(`MACRO_PATHS` in `src/riskagg/data/banking.py`). The severely adverse path takes unemployment
from 4.0% to a peak of 10.0%; the baseline path barely moves.

### 1.2 The nine-quarter horizon

Projections run over a **planning horizon**: "the period of at least nine consecutive quarters,
beginning on the first day of a stress test cycle over which the relevant projections extend"
[12 CFR 252.42; 225.8(d)(16) has the same nine-quarter minimum].

Quarter 0 here is the starting point, the last actual balance sheet. Quarters 1 to 9 are
projected. Nine quarters is long enough for a recession to work through credit losses, which
peak well after the shock.

### 1.3 The building blocks: PPNR, credit losses, trading losses

The Fed's retained-earnings model computes pre-tax net income per quarter as [Fed-AGG,
Equation C2]:

```
pre-tax net income = PPNR - provisions - trading and counterparty losses - other losses
```

- **PPNR** (pre-provision net revenue) is "the sum of net interest income and non-interest income
  less expenses before adjusting for loss provisions" [12 CFR 252.42]. It is what the bank earns
  from its business before credit losses. A recession squeezes it: less activity, lower fees,
  lower rates.
  - Here: `ppnr = ead × (revenue_rate_q × (1 + 2.0 × gdp_growth_q) − expense_rate_q)`. Revenue
    falls with GDP; expenses do not.
- **Credit losses** (provisions) are the losses on loans as borrowers default.
  - Here: `credit_loss = ead × pd_q × lgd`, where `ead` is exposure at default, `lgd` loss given
    default and `pd_q` the quarterly probability of default, stressed by unemployment:
    `pd_q = min(1, pd_base / 4 × (1 + 0.35 × max(0, unemployment_q − unemployment_0)))`, with
    unemployment in percentage points. The rise in unemployment is measured from quarter 0 of
    the same scenario, so PD only goes up.
- **Trading losses** come from the **global market shock** (next section).
- **Taxes.** The Fed's model applies a tax rate of 21% to pre-tax net income, with adjustments
  for deferred tax assets [Fed-AGG, Retained Earnings Model, footnote 18]. Here:
  `net_income = (ppnr − credit_loss − trading_loss) × (1 − 0.21)`, applied symmetrically, so a
  loss earns a full tax credit.

The beta of 0.35, the GDP sensitivity of 2.0 and the tax rate of 0.21 are in
`config/reference/ccar.yaml`. The first two are the project's stylised choices, not regulatory
values.

### 1.4 The global market shock, and who gets it

The severely adverse scenario of firms with large trading books adds a **global market shock**
(GMS): an instantaneous set of moves in market prices, applied to the bank's trading positions
on one as-of date. The Fed's documentation says [Fed-GMS, Statement of Purpose]:

- "The losses associated with the global market shock are recognized in the first quarter of
  the scenario horizon, and no changes to losses or gains are assumed in subsequent quarters of
  the scenario."
- Trading profits and losses are measured mark-to-market, while the banking book's revenue and
  losses are generally accrual-based. The macro scenario plays out over nine quarters, while the
  GMS hits at once, "based on a scenario assumption that market dislocations can happen rapidly
  and unpredictably at any time during the scenario horizon."

**Which firms.** Per the Fed's documentation, a firm is subject to the GMS if it is subject to
the stress test, "has aggregate trading assets and liabilities of $50 billion or more, or
aggregate trading assets and liabilities equal to ten percent or more of total consolidated
assets; and is not a Category IV firm" [Fed-MR, Trading P&L Model, footnote 177, citing 12 CFR
252.54(b)(2)(i)].

**The largest counterparty default component.** Firms with substantial trading or custodial
operations also take a **largest counterparty default** (LCPD) loss: "the loss each firm would
experience if its largest counterparty were to default under the hypothetical GMS component"
[Fed-MR, LCPD Model]. It is explained here but not built.

**Here.** The stress scenario `GFC_2008` is flagged `global_market_shock: true` in
`config/reference/stress_scenarios.yaml` (see the stress page).

- The run takes its stress P&L from the same stress engine, `RunInputs.stress_grain()`, and sums
  it by legal entity and business line. Nothing is re-shocked.
- `trading_loss = −P&L`, so a loss is positive.
- It is booked in quarter 1 of SEVERELY_ADVERSE only, and is zero in every other quarter and in
  BASELINE (`gms_loss_by_entity_bl` and `_pair_quarters` in `src/riskagg/measures/ccar.py`).
- **If GFC_2008 is a gain for a node, its trading loss is negative**, and the gain raises its
  capital. This is kept, not clipped, for the same reason a VaR is never clipped: clipping would
  overstate the loss and break the reconciliation. On tiny this happens to legal entity LE01
  (section 3).

### 1.5 How CET1 moves quarter to quarter

**CET1** (common equity tier 1 capital) is the highest-quality capital: common shares and
retained earnings, less regulatory deductions. The **CET1 ratio** is CET1 divided by
**risk-weighted assets** (RWA). RWA is exposures weighted by riskiness: a loan with risk weight
0.9 counts 90% of its exposure, a mortgage with 0.4 counts 40%. The minimum CET1 ratio is
4.5% [12 CFR 217.10(a)(1)(i)].

In the supervisory projection, retained earnings move each quarter by after-tax net income, less
preferred dividends and minority interests [Fed-AGG, Equation C1]. **Common dividends are not
deducted.** The rule says the Board assumes "the covered company will not pay any dividends on
any instruments that qualify as common equity tier 1 capital" [12 CFR 252.44(c)(1)]. The Fed's
model documentation adds that "four quarters of common dividends are included in the calculation
of each firm's stress capital buffer separately" [Fed-AGG, Retained Earnings Model].

The Board also assumes no redemption or repurchase of any regulatory capital instrument and no
issuance of common or preferred stock [12 CFR 252.44(c)(3)-(4)], and a balance sheet and RWA
held constant over the horizon [Fed-AGG, Balances Model]. It does **not** assume preferred
dividends stop: the company "will make payments on instruments that qualify as additional tier 1
capital or tier 2 capital equal to the stated dividend, interest, or principal due on such
instrument" [12 CFR 252.44(c)(2)], and those preferred dividends are deducted each quarter
[Fed-AGG, Equation C1]. This project has no preferred stock, so it deducts nothing (a
simplification, section 4). So, here:

```
cet1_q       = cet1_{q-1} + net_income_q            (cet1_0 = cet1_start)
rwa          = sum(ead × risk_weight) + rwa_trading (constant)
cet1_ratio_q = cet1_q / rwa
```

The `dividends` column of `ccar_projection` carries each quarter's **planned** common dividend.
It is reported and used by the stress capital buffer's dividend add-on, and never deducted from
the path.

**Why not deduct dividends from the path as well?** The buffer below is "decline in the ratio
plus four quarters of dividends". If the projected path had already paid those dividends, the
decline would include them and the add-on would count them a second time. The rule keeps the
path dividend-free so that each dollar of planned dividends is counted once, in the add-on.

### 1.6 The stress capital buffer

The rule in force until 30 November 2026 [12 CFR 225.8(f)(2), current text]: the requirement is
the greater of

- (A) the CET1 ratio "as of the final quarter of the previous capital plan cycle", minus
  (B) "the lowest projected ratio ... in any quarter of the planning horizon under a supervisory
  stress test", plus (C) the ratio of "the sum of the bank holding company's planned common
  stock dividends (expressed as a dollar amount) for each of the fourth through seventh
  quarters of the planning horizon" to "the risk-weighted assets ... in the quarter in which the
  bank holding company had its lowest projected ratio"; and
- "2.5 percent."

In words:

```
scb = max(2.5%, start ratio − lowest projected ratio + four quarters of planned dividends / RWA at the trough)
```

**The rule changes on 1 December 2026.** The Board's final rule "Modifications to the Capital
Plan Rule and Stress Capital Buffer Requirement" (91 FR 62636, 2 October 2026, FR Doc
2026-20246, effective 1 December 2026) does three things:

- **Moves the effective date.** The annual effective date of the buffer moves from 1 October to
  1 January.
- **Moves the dividend quarters.** It replaces the quarters with "the **fifth through eighth**
  quarters of the current planning horizon". The preamble says the shift is "consistent with the
  changes in the effective date". It applies to the requirements calculated after the 2027
  stress test.
- **Averages the decline.** It defines the start-to-trough drop as the **stress capital decline**
  and, for firms tested in the previous year, uses "the average of the stress capital decline of
  the current capital plan cycle and ... the stress capital decline of the capital plan cycle for
  the previous calendar year". Averaging "will begin starting with the stress capital buffer
  requirements effective on January 1, 2029". A firm with no prior-year test uses the
  single-cycle decline (new paragraph (f)(2)(ii)).

This project uses the amended quarters, `scb_dividend_quarters: [5, 8]` in
`config/reference/ccar.yaml`, with both versions cited there. Planned dividends are constant per
quarter here, so quarters 4–7 and 5–8 give the same number. Averaging over two cycles is **not
built**: the project runs one cycle, which is what (f)(2)(ii) prescribes for a firm without a
previous year.

**How the buffer feeds the capital requirement.** For a firm subject to the rule, the standardised
approach capital conservation buffer requirement equals "its stress capital buffer requirement
plus its applicable countercyclical capital buffer amount ... plus its applicable GSIB
surcharge" [12 CFR 217.11(c)(1)(iii)(A)].

- The firm's effective CET1 requirement is therefore the 4.5% minimum plus the SCB, plus any
  countercyclical buffer and G-SIB surcharge.
- A firm whose buffer falls below the requirement faces a maximum payout ratio on distributions
  [12 CFR 217.11(c)].
- The 2.5% floor matches the buffer of banks outside the rule: such a bank escapes payout
  limits only with a capital conservation buffer above 2.5% plus any countercyclical amount
  [12 CFR 217.11(a)(4)(ii)]. A firm subject to the rule that has not yet received an SCB is
  given 2.5% [12 CFR 217.11(a)(2)(vi)(B)].

**Sources** (all fetched 13 November 2026):

- [Fed-ST] Federal Reserve, "Stress Tests and Capital Planning",
  https://www.federalreserve.gov/supervisionreg/stress-tests-capital-planning.htm.
- 12 CFR 225.8, 252.42-252.44, 217.10 and 217.11: eCFR, text as of 10 November 2026.
- 91 FR 62636 (FR Doc 2026-20246):
  https://www.federalregister.gov/documents/2026/10/02/2026-20246/modifications-to-the-capital-plan-rule-and-stress-capital-buffer-requirement.
- Federal Reserve, supervisory stress test model documentation, Fall 2026, linked from
  https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-2027.htm:
  - [Fed-GMS] Global Market Shock Component;
  - [Fed-MR] Market Risk Models;
  - [Fed-AGG] Aggregation Models.

## 2. How CCAR aggregates, and why it is a different shape

**A path, not a quantile.** VaR and expected shortfall rank many scenarios and read off a tail.
FRTB nests correlations inside a regulatory formula. CCAR does neither.

- It takes **two** scenarios, each a single path through nine quarters.
- It adds up flows across risk types (revenue, credit losses, trading losses) into one number per
  quarter.
- It accumulates that number into a **stock**, capital, over time.
- The headline figures are a minimum over time (the trough ratio) and a buffer built from it.

The aggregation axes are therefore *time* and *risk type*, not *scenario rank*.

**What adds up and what does not.**

- **Flows add across nodes within one scenario and quarter.** PPNR, credit loss, trading loss
  and net income of the firm are the sum over business lines, and also the sum over legal
  entities. That is the reconciliation the audit checks (`check_ccar_from_raw` in
  `src/riskagg/engine/audit.py`) and `test_losses_reconcile_across_cuts`.
- **Flows add across quarters.** CET1 at quarter 9 is the starting CET1 plus the sum of the nine
  net incomes (`test_cumulative_path_is_running_sum`).
- **CET1 and RWA add across legal entities**, so the firm's CET1 is the sum of the entities'
  CET1 (a simplification: a real consolidation eliminates intra-group items).
- **Ratios do not add, and do not average simply.** The firm's ratio is the firm's CET1 over
  the firm's RWA, recomputed from the summed amounts.
- **The minimum and the buffer are not additive at all.** The firm's trough can fall in a
  different quarter from each entity's trough. An entity that gains in the market shock
  offsets one that loses. So the firm's buffer is neither the sum nor the RWA-weighted average
  of the entities' buffers (section 3 shows 2.9211% against a weighted average of 4.1945%).
  This is the same non-additivity as the worst stress loss on the stress page.

**Why capital ratios cannot be cut by business line.** Revenue and losses come from loans and
trades, which belong to a business line, so they can be reported by business line. Capital does
not belong to one.

- CET1 is held by legal entities: the holding company and its subsidiaries are the ones that
  issue shares, retain earnings and file regulatory capital reports.
- A business line's "capital" exists only if the bank chooses an allocation rule (by RWA, by
  stressed loss, by economic capital), and different rules give different answers.
- This project allocates no capital to business lines. The three capital measures
  (`ccar_start_cet1_ratio`, `ccar_min_cet1_ratio`, `ccar_scb`) declare
  `not_allocated_to: [business_line_id, desk_id, book_id]` in `config/semantic_model.yaml`.
- `build_query` refuses any such slice with `SliceNotAvailable` ("capital is not allocated to
  business lines"), and the API returns 422. In `ccar_projection` the business-line rows carry
  losses and revenue, with `cet1`, `rwa`, `cet1_ratio` and `dividends` null.

**What this does to the Spark design** (`src/riskagg/measures/ccar.py`):

- **Macro paths.** These are a 20-row table. Each projected quarter is joined to quarter 0 of
  the same scenario to get the unemployment rise, and the result is broadcast.
- **Banking book.** Segments × 18 scenario-quarters go through one cross join and one `groupBy`
  to (scenario, quarter, legal entity, business line). The PD stress, PPNR and credit loss are
  column expressions, with no UDF.
- **Trading loss.** It comes from the run's cached stress grain, filtered to the flagged scenario
  and summed by (legal entity, business line). The CCAR and stress pages therefore read the same
  numbers.
- **The path.** Business-line rows are a `groupBy` of the pair rows. Legal-entity rows add the
  constant RWA and a **running sum** of net income over a window partitioned by (scenario,
  legal entity) and ordered by quarter. The firm rows sum the legal entities.
- **The summary.** The minimum ratio is the first row of a window ordered by (ratio, quarter),
  so ties go to the earliest quarter. The buffer joins the start ratio, the stressed trough and
  the add-on dividends.
- **Size.** The projection is tiny (a few hundred rows even on demo) and is cached once per run.
  It is written as `ccar_projection` and also feeds the `risk_metrics` rows, so the two cannot
  disagree.

**What this does to the OLAP layer:**

- The three capital measures are **structured** and only looked up, at
  `[firm_id]` and `[firm_id, legal_entity]`. `ccar_min_cet1_ratio` carries the scenario
  (`carries_scenario: true`), so a lookup returns one row per scenario.
- `ccar_projection` is a detail table read through the drill-down. No measure is served from
  it, so no star-tree is built. Its rows at different `node_level`s must never be summed
  together, because each level already holds the whole firm.

## 3. Worked example (tiny profile)

The tiny profile has two legal entities (LE01, LE02) and two business lines (BL01, BL02).

- The banking book has three segments per (legal entity, business line) pair: CORPORATE,
  MORTGAGE and CARDS, so 12 in all.
- Money is rounded to whole dollars and ratios to four decimals of a percent here, but **every
  figure is computed from unrounded inputs**. A printed sum or product can therefore differ from
  the printed result in the last digit.

**One segment, one quarter.** Segment `LE02|BL02|CORPORATE` has EAD 27,692,533, annual PD
0.0203, LGD 0.3763, quarterly revenue rate 0.00656 and quarterly expense rate 0.00307. In
quarter 1 of SEVERELY_ADVERSE, unemployment is 5.5% against 4.0% at quarter 0 (a rise of 1.5
points) and GDP growth is −6% annualised.

- Quarterly PD: 0.0203 / 4 × (1 + 0.35 × 1.5) = **0.007739375**.
- Credit loss: 27,692,533 × 0.007739375 × 0.3763 = **80,650**.
- Revenue rate: 0.00656 × (1 + 2.0 × (−0.06)) = **0.0057728**.
- PPNR: 27,692,533 × (0.0057728 − 0.00307) = **74,847**.

The EAD printed here is rounded. The products use the stored EAD, 27,692,533.33.

**One legal entity, quarter 1.** LE02 sums its six segments, then adds the GFC_2008 loss of its
trading books:

| SEVERELY_ADVERSE, quarter 1 | LE02 |
|---|---:|
| PPNR | 561,291 |
| Credit loss | 525,500 |
| Trading loss (global market shock) | 10,682,964 |
| Pre-tax net income | −10,647,174 |
| Net income (× 0.79) | −8,411,267 |
| CET1, start | 18,681,691 |
| CET1, end of quarter 1 | 10,270,424 |
| RWA | 157,917,928 |
| CET1 ratio | 6.5036% |

The planned dividend of 125,167.33 a quarter is shown in `ccar_projection` but not deducted:
18,681,691 − 8,411,267 = 10,270,424.

**LE02's path.** The trading loss is a one-off in quarter 1. Credit losses keep exceeding PPNR as
unemployment climbs to 10% and stays high, so capital keeps falling:

| Quarter | Net income | CET1 | CET1 ratio |
|---:|---:|---:|---:|
| 1 | −8,411,267 | 10,270,424 | 6.5036% |
| 2 | −156,025 | 10,114,399 | 6.4048% |
| 3 | −217,817 | 9,896,582 | 6.2669% |
| 4 | −251,025 | 9,645,556 | 6.1080% |
| 5 | −276,340 | 9,369,216 | 5.9330% |
| 6 | −255,650 | 9,113,566 | 5.7711% |
| 7 | −206,376 | 8,907,190 | 5.6404% |
| 8 | −157,919 | 8,749,271 | 5.5404% |
| 9 | −119,807 | 8,629,464 | 5.4645% |

Under BASELINE, LE02's ratio only rises; its lowest is quarter 1, at 12.0359%.

**Losses by cut.** Summed over the nine SEVERELY_ADVERSE quarters:

| Node | PPNR | Credit loss | Trading loss | Net income |
|---|---:|---:|---:|---:|
| BL01 | 11,088,004 | 14,010,136 | −1,864,602 | −835,448 |
| BL02 | 5,130,949 | 6,031,534 | 11,111,876 | −9,489,844 |
| LE01 | 10,093,534 | 11,874,877 | −1,435,691 | −273,065 |
| LE02 | 6,125,419 | 8,166,793 | 10,682,964 | −10,052,227 |
| FIRM | 16,218,953 | 20,041,670 | 9,247,274 | −10,325,292 |

- Each column of the firm row is the sum of the two business lines and also of the two legal
  entities (to within the rounding of the printed figures).
- The firm's trading loss, 9,247,274, is the firm's GFC_2008 loss on the stress page.
- BL02 holds DESK003, whose GFC_2008 loss of 11,111,876 also appears there.
- **BL01 and LE01 have negative trading losses**: GFC_2008 is a net gain for their trading
  books. The gain is kept, and it raises LE01's capital in quarter 1.

**The stress capital buffer.** RWA is constant, so the trough RWA is the starting RWA. Every
node's SEVERELY_ADVERSE trough is quarter 9. The add-on is four quarters (5 to 8) of planned
dividends over that RWA:

| | LE01 | LE02 | FIRM |
|---|---:|---:|---:|
| Start ratio (`ccar_start_cet1_ratio`) | 11.4000% | 11.8300% | 11.5742% |
| Lowest SEVERELY_ADVERSE ratio (`ccar_min_cet1_ratio`) | 11.2822% | 5.4645% | 8.9253% |
| Decline | 0.1178% | 6.3655% | 2.6489% |
| Planned dividends, quarters 5-8 | 560,402 | 500,669 | 1,061,071 |
| RWA at the trough | 231,877,637 | 157,917,928 | 389,795,565 |
| Dividend add-on | 0.2417% | 0.3170% | 0.2722% |
| Decline + add-on | 0.3594% | 6.6825% | 2.9211% |
| `ccar_scb` (floored at 2.5%) | 2.5000% | 6.6825% | 2.9211% |

- **LE01 is at the floor.** Its market-shock gain nearly offsets its credit losses, so its
  decline is tiny and the 2.5% floor binds.
- **LE02 is well above it.** It carries most of the trading loss.
- **The firm is 2.9211%,** below LE02's figure, because LE01's gain cushions the firm.
  Weighting the entity buffers by RWA gives 4.1945%, and the sum would be meaningless; neither
  is the firm's buffer.
- The BASELINE minima are 11.6554% (LE01), 12.0359% (LE02) and 11.8096% (firm).

## 4. What this project simplifies compared with a real bank

Marked `SIMPLIFICATION:` in the code:

- **Hand-set macro paths.** Two variables over nine quarters (`MACRO_PATHS` in
  `src/riskagg/data/banking.py`). The Fed publishes many more domestic and international
  variables per scenario.
- **Three stylised loan portfolios** (CORPORATE, MORTGAGE, CARDS) per legal entity and business
  line, with uniform random parameters (`SEGMENT_TYPES` in `src/riskagg/data/banking.py`). The
  Fed's credit models project losses per product from loan-level data.
- **Constant balance sheet** (`project` in `src/riskagg/measures/ccar.py`). EAD, risk weights
  and RWA stay at their starting values for nine quarters. This matches the Fed's own assumption
  [Fed-AGG, Balances Model]; the simplification is that nothing else changes either (no
  maturities, no migration of risk weights, no regulatory-capital deductions).
- **Tax applied symmetrically** (`_pair_quarters`). A pre-tax loss earns a 21% tax credit at
  once, with no deferred-tax-asset limits or valuation allowance, which the Fed's model does
  apply.
- **No counterparty default component**, and no operational-risk losses or other trading
  effects beyond the one global market shock (`project`). The largest counterparty default
  loss, credit valuation adjustment losses and incremental default losses on trading positions
  are explained in section 1.4 but not built.
- **No preferred stock or other additional tier 1 / tier 2 instruments** (`project`). The
  supervisory path deducts their stated dividends and interest every quarter
  [12 CFR 252.44(c)(2); Fed-AGG, Equation C1]; here there are none, so CET1 moves by net
  income alone.
- **The stress capital buffer is computed for legal entities as well as the firm**, and from one
  stress test cycle (`summary`). The rule sets it only for the holding company, and from 2029
  averages the decline over two cycles.

Also different from a bank, by the project's scope:

- **No capital actions other than planned common dividends.** No buybacks or issuance, which
  the supervisory path assumes away anyway [12 CFR 252.44(c)(3)-(4)].
- **Dividends are a constant quarterly amount per legal entity**, so the choice between quarters
  4-7 and 5-8 changes no number here.
- **The firm's CET1 is the sum of the legal entities'**, with no consolidation adjustments.
- **The global market shock is one of the stress page's hand-set scenarios,** not the Fed's
  published GMS (which shocks thousands of risk factors). Its P&L is sensitivity-based and
  inherits the stress page's simplifications.
- **No qualitative assessment** of capital planning, no company-run stress test, and no
  BHC-designed scenario.

## 5. Likely interview questions

1. **What is CCAR, and how does it relate to DFAST and the stress capital buffer?** DFAST is
   the Fed's supervisory projection of losses, revenue and capital under its scenarios. CCAR
   was the Fed's assessment of capital plans built on it. Since 2020 the quantitative part of
   CCAR has been replaced by the stress capital buffer, which the stress test sets [Fed-ST].
   Code: `src/riskagg/measures/ccar.py`.
2. **Who has to do it, and how often?** US BHCs, SLHCs and IHCs with $100 billion or more in
   assets. The largest are tested annually; Category IV firms in even years
   [Fed-ST, footnote 1; 12 CFR 252.43, 252.44(d) for BHCs and IHCs; 12 CFR part 238,
   subpart O for SLHCs].
3. **What are the scenarios, and over what horizon?** At least a baseline and a severely
   adverse scenario, over a planning horizon of at least nine quarters [12 CFR 252.42,
   252.44(b)]. Code: `make_macro_scenarios`, and `test_baseline_has_no_trading_loss`.
4. **What is PPNR, and why is it in a stress test?** Net interest income plus non-interest income
   less expenses, before loss provisions [12 CFR 252.42]. It is the bank's first line of defence:
   earnings absorb losses before capital does, and a recession shrinks them. Code: the `ppnr`
   expression in `_pair_quarters`, and `test_one_entity_first_quarter`.
5. **How do you stress credit losses?** EAD × PD × LGD per segment and quarter, with PD rising
   with unemployment above its starting level, capped at 1. Code: `_pair_quarters`, and
   `test_pd_is_capped_at_one_and_never_falls_below_base`.
6. **What is the global market shock, which firms get it, and when is it booked?** An
   instantaneous market shock to trading positions on one as-of date. It applies to non-Category
   IV firms with trading assets and liabilities of $50 billion or more, or 10% or more of total
   assets. It is booked in full in quarter 1 of the severely adverse scenario
   [Fed-GMS; Fed-MR, footnote 177]. Code: `gms_loss_by_entity_bl`, and
   `test_trading_loss_is_the_gfc_stress_loss_and_gains_are_kept`.
7. **How does CET1 move each quarter? Are dividends deducted?** CET1 plus after-tax net income.
   Common dividends are not deducted in the supervisory projection: the Board assumes none
   [12 CFR 252.44(c)(1)]. Planned dividends enter the buffer once, through the add-on;
   deducting them in the path too would count them twice. Code: `project`, and
   `test_cumulative_path_is_running_sum`.
8. **How is the stress capital buffer calculated?** Max of 2.5% and (start ratio − lowest
   projected ratio + four quarters of planned dividends / RWA in the trough quarter). The
   dividend quarters are the fourth through seventh until 30 November 2026 and the fifth through
   eighth after, with two-year averaging of the decline from 2029 [12 CFR 225.8(f)(2);
   91 FR 62636]. Code: `summary`, `test_scb_above_floor_by_hand` and `test_scb_floor`.
9. **How does the buffer become a capital requirement?** The standardised capital conservation
   buffer requirement is the SCB plus any countercyclical buffer and G-SIB surcharge, on top of
   the 4.5% CET1 minimum. Falling short limits distributions through a maximum payout ratio
   [12 CFR 217.10, 217.11(c)].
10. **Why is this aggregation different from VaR's?** VaR ranks hundreds of scenarios and takes
    a quantile, so it is not additive. CCAR runs two scenarios as nine-quarter paths. Flows add
    across nodes and over time; capital is a running sum; the headline numbers are a minimum
    over time and a buffer derived from it, and those are not additive across entities. Code:
    the window in `project`, and `test_firm_ratio_is_not_an_average_of_entity_ratios`.
11. **Why can you report losses by business line but not a capital ratio?** Losses come from
    positions, which belong to a business line. Capital is held by legal entities and is not
    allocated to business lines here, so any business-line ratio would depend on an arbitrary
    allocation rule. The semantic model refuses the query. Code: `not_allocated_to` in
    `config/semantic_model.yaml`, the refusal in `build_query`, and
    `test_business_line_capital_ratio_refused`.
12. **The firm's buffer is lower than one entity's. Is that a bug?** No. The firm nets the
    entities, scenario by scenario and quarter by quarter, before taking the trough. One
    entity's market-shock gain cushions another's loss, as with the worst stress loss. On tiny
    the firm is at 2.9211% against LE02's 6.6825%. Code: `summary`, and the golden test.
13. **How do you know the projection is right?** The audit check rebuilds every row of
    `ccar_projection` and every capital measure in plain Python, from the raw banking book, macro
    paths and starting capital and the curated stress P&L. It also checks that firm losses equal
    the sum of business lines and of legal entities. A run that fails is not marked official.
    Code: `check_ccar_from_raw` in `src/riskagg/engine/audit.py`, and
    `test_audit_catches_a_wrong_stored_value`.
