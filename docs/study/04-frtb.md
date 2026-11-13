# FRTB: the Fundamental Review of the Trading Book

This page covers what FRTB is and why it replaced Basel 2.5, how the standardised approach
(SA) and the internal models approach (IMA) compute capital, how each aggregates and why
neither can be produced by rolling up a cube, worked examples on the `tiny` profile, what this
project simplifies, and the questions an interviewer is likely to ask. Every number in the
worked examples is asserted by `tests/golden/test_frtb_worked_example.py`, which recomputes it
through the production code.

Paragraph references (MAR21.4, RBC20.4, ...) are to the Basel Framework on bis.org: the market
risk chapters MAR11 to MAR33 in their versions effective 1 January 2023 (MAR21 last updated 23
March 2026), and the risk-based capital chapters RBC20 and RBC25. History and the problems
FRTB was meant to fix are from the Basel Committee's "The market risk framework: In brief"
(January 2019), also on bis.org.

## 1. What FRTB is and why it replaced Basel 2.5

**Trading book.** The positions a bank holds to trade (for short-term resale, to profit from
price moves, or to hedge such positions), as opposed to the **banking book** of loans and
investments held to maturity. Which book an instrument sits in decides which capital rules
apply to it.

**Market risk.** The risk of losses from moves in market prices: interest rates, credit
spreads, equity prices, exchange rates, commodity prices.

**Capital requirement.** The amount of the bank's own funds (equity, mainly) that must stand
behind a portfolio to absorb losses. Market risk capital is converted into **risk-weighted
assets** (RWA) by multiplying by 12.5 (MAR20.1 for SA, MAR33.46 for IMA), so it can be added
to credit and operational risk RWA in the capital ratio.

**Risk class, risk factor, bucket.** A **risk factor** is one market variable that moves a
position's value: the 5-year point of the USD interest rate curve, the EUR/USD exchange rate,
an equity price. Factors are grouped into **buckets** of similar factors (all tenors of one
currency's curve; equities of one size, region and sector) and buckets into **risk classes**
(MAR21.1 defines seven: general interest rate risk, three kinds of credit spread risk, equity,
commodity and FX). This project models three: general interest rate risk (GIRR), equity (EQ)
and FX.

**Why there was a review.** Trading-book losses in the 2008 crisis were far larger than the
capital held for them. In July 2009 the Basel Committee added a stop-gap, "Basel 2.5", that
raised capital mainly for credit-risky traded instruments, and in 2012 it started a
fundamental review of the trading book. The resulting standard was published in January 2016
and revised in January 2019; in the Basel Framework its chapters are effective as of 1 January
2023. The Committee's own summary lists four weaknesses of the Basel 2.5 framework and the FRTB
answer to each:

| Weakness in Basel 2.5 | FRTB answer |
|---|---|
| A weak trading/banking book boundary let banks move instruments between books to lower capital. | A stricter boundary (RBC25): lists of instruments that must go in each book, and switching is strictly limited (RBC25.14). A capital benefit from switching is never allowed; it is added back as a surcharge (RBC25.15). |
| Model approval was bank-wide, so supervisors could not take a model away from one badly performing desk. | Approval at the **trading desk** level (MAR12, MAR30.4): a desk is a group of traders with a defined strategy (MAR12.1), and each desk must keep passing backtesting and the P&L attribution test to keep its model. |
| Capital came mainly from VaR models, which measured tail risk and liquidity risk badly and allowed unrestrained diversification. | **Expected shortfall** at 97.5% replaces VaR (MAR33.3), with **liquidity horizons** of 10 to 120 days per risk factor (MAR33.4, MAR33.12), calibration to a period of stress (MAR33.5), limits on diversification across risk classes (MAR33.14-33.15), and a separate charge for risk factors without enough data to model (MAR33.16). |
| The standardised approach was outdated and not risk-sensitive enough to be a credible complement and fallback to models. | A new, sensitivity-based SA (MAR20-MAR23). Every bank computes it (MAR11.8); a desk that fails its model tests falls back to it (MAR30.4, MAR32.43). |

**Why ES instead of VaR.** VaR at 99% is the loss exceeded on 1% of days; it says nothing
about how bad those days are. Expected shortfall at 97.5% is the average loss on the worst
2.5% of days, so it grows when the tail gets fatter. It is also subadditive (the ES of a
portfolio is at most the sum of the ES of its parts), which VaR is not; the VaR page
(`01-var-es.md`) has a two-book example.

**Why liquidity horizons.** A 10-day horizon assumes any position can be sold or hedged in 10
days. That is true for USD rates, false for a small-cap equity or a high-yield credit spread.
FRTB gives each risk-factor category a horizon (MAR33.12) and scales the ES so that factors
that take longer to exit carry more capital.

**Why the SA is also a floor.** The SA is not only for banks without models:

- a bank using the IMA for any desk must still compute the SA for all desks together and for
  each IMA desk on its own, as if it were a standalone portfolio (MAR11.8), and report the SA
  monthly (MAR20.2);
- the total market risk requirement combines the IMA of eligible desks with the SA of all
  other desks, in a formula that also uses the SA of all desks (MAR33.43, section 3.5);
- bank-wide, RWA may not fall below 72.5% of the RWA that standardised approaches would give
  (the **output floor**, RBC20.4, subject to the transitional arrangements of RBC90).

So even a model bank computes the SA for the whole book and for each desk on its own, and
reports it at least monthly. This project computes it for every desk and for the firm on every
run.

## 2. The standardised approach

The SA capital requirement is the simple sum of three components (MAR20.4):

```
SA = sensitivities-based method (SBM) + default risk capital (DRC) + residual risk add-on (RRAO)
```

### 2.1 The sensitivities-based method (MAR21)

**Sensitivity.** The change in a position's value for a small move of one risk factor.
**Delta** is the first-order sensitivity to a price or rate; **vega** is the sensitivity to
an option's implied volatility; **curvature** captures what delta misses for options under a
large up or down shock (MAR20.4). In this project the sign is P&L per unit move (relative for
equity and FX, absolute for rates), so a long position has a positive delta.

The method, per risk class, for delta and for vega (MAR21.4):

1. **Net.** Sum the sensitivities of all instruments to the same risk factor: `s_k`. Opposite
   positions offset fully: two opposite swaps on the same curve have zero GIRR (MAR21.4(2)).
2. **Weight.** `WS_k = s_k × RW_k` (MAR21.4(3)). The risk weight is the size of a stressed
   move: for GIRR 1.7% at 0.25 years down to 1.1% at 5 years and beyond (MAR21.42 Table 1),
   divided by √2 for the specified currencies EUR, USD, GBP, AUD, JPY, SEK, CAD and the
   reporting currency (MAR21.44); for equity 55% for bucket 1, 30% for 5, 50% for 8, 15% for
   12 (MAR21.77 Table 10); for FX 15% (MAR21.87), divided by √2 for specified currency pairs
   (MAR21.88).
3. **Within a bucket.** `K_b = sqrt(max(0, Σ_k WS_k² + Σ_k Σ_{l≠k} ρ_kl WS_k WS_l))`
   (MAR21.4(4)), with a prescribed correlation `ρ` between factors of the same bucket. For two
   GIRR tenors of one curve `ρ = max(exp(-3% × |T_k - T_l| / min(T_k, T_l)), 40%)` (MAR21.46).
4. **Across buckets.** `Δ = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc S_b S_c)` with `S_b = Σ_k WS_k`
   and a cross-bucket correlation `γ` (50% between GIRR currencies, MAR21.50). If the sum
   under the root is negative, `S_b` is replaced by `max(min(S_b, K_b), -K_b)` (MAR21.4(5)).
5. **Curvature** uses an upward and a downward shock per factor; the loss beyond what delta
   predicts is the **CVR**, and buckets and classes aggregate with squared correlations
   (MAR21.5, MAR21.100-21.101).
6. **Three correlation scenarios** (MAR21.6). Correlations themselves are uncertain and change
   in a crisis, so steps 3 to 5 are repeated with every `ρ` and `γ`: as given (**medium**),
   `min(1.25 × x, 100%)` (**high**) and `max(2 × x - 100%, 75% × x)` (**low**).
7. **Largest total.** Per scenario, the delta, vega and curvature charges of all risk classes
   are summed; the SBM charge is the largest of the three totals (MAR21.7).

Two points matter for aggregation. First, hedges help through correlation: a long and a short
in two buckets offset only to the extent `γ` allows, and lower correlation means less offset,
which is why the low scenario is often the binding one for a hedged book. Second, every step
is non-linear (square roots, a maximum), so nothing computed for a desk can be added to get
the firm.

### 2.2 The default risk charge (MAR22)

Jump-to-default (JTD) risk is the loss if an issuer defaults overnight, which the spread
sensitivities of the SBM do not capture.

1. **Gross JTD per exposure** (MAR22.9, MAR22.11): `max(LGD × notional + P&L, 0)` for a long
   and `min(LGD × notional + P&L, 0)` for a short, where P&L is the mark-to-market loss or
   gain already taken. **LGD** (loss given default) is 100% for equity and non-senior debt,
   75% for senior debt, 25% for covered bonds (MAR22.12). A long is a position that loses on
   default (MAR22.10).
2. **Offset per obligor** (MAR22.19): longs and shorts to the same issuer net where the short
   has the same or lower seniority than the long (an equity short can offset a bond long,
   not the reverse).
3. **Buckets** (MAR22.22): corporates, sovereigns, local governments and municipalities.
4. **Hedge benefit ratio** per bucket (MAR22.23): `HBR = Σ net long JTD / (Σ net long JTD +
   Σ |net short JTD|)`, unweighted.
5. **Bucket charge** (MAR22.25): `max(Σ RW × net long JTD - HBR × Σ RW × |net short JTD|, 0)`,
   with risk weights by rating from 0.5% (AAA) to 100% (defaulted) (MAR22.24 Table 2).
6. **No hedging across buckets** (MAR22.26): the DRC is the sum of the bucket charges.

### 2.3 The residual risk add-on (MAR23)

A flat charge for risks the SBM and DRC do not see: 1.0% of gross notional for instruments
with an **exotic underlying** (longevity, weather, future realised volatility; MAR23.3) and
0.1% for other residual risks (MAR23.8). It is a simple sum, so it is the one additive
component.

## 3. The internal models approach

### 3.1 Expected shortfall on a ten-day base horizon

ES is computed daily for the bank and for each IMA desk (MAR33.2) at 97.5% one-tailed
(MAR33.3), on a **base horizon** of 10 days: from 10-day changes of risk factors, which may
overlap, and not scaled from a shorter horizon (MAR33.4). This project's P&L vectors are one
day, so it scales the one-day ES by √10 (a marked simplification, section 6).

### 3.2 The liquidity-horizon cascade (MAR33.4, MAR33.12)

Each risk factor gets a **liquidity horizon** `LH`, one of 10, 20, 40, 60 or 120 days, from
MAR33.12 Table 2. The ones this project uses:

| Risk factor category (MAR33.12 Table 2) | LH (days) | Factors here |
|---|---:|---|
| Interest rate: specified currencies (EUR, USD, GBP, AUD, JPY, SEK, CAD, domestic) | 10 | GIRR USD, EUR, GBP, JPY |
| Interest rate: unspecified currencies | 20 | GIRR BRL, ZAR |
| Interest rate: volatility | 60 | GIRR vega |
| Equity price (large cap) | 10 | EQ buckets 1, 5, 8, 12 |
| Equity price (large cap): volatility | 20 | EQ vega |
| FX rate: specified currency pairs | 10 | USD/EUR, USD/GBP, USD/JPY, USD/BRL, USD/ZAR |
| FX: volatility | 40 | FX vega |

The specified FX pairs in MAR33.12's footnote are the 19 pairs against USD that MAR21.88 also
specifies (USD/BRL and USD/ZAR included) plus EUR/JPY, EUR/GBP, EUR/CHF and JPY/AUD, and their
first-order crosses. So the generated data holds horizons 10, 20, 40 and 60; none is 120.

`Q(j)` is the set of factors whose horizon is at least `LH_j`. With `ES_T(P, j)` the 10-day ES
when only the factors in `Q(j)` move (all others held constant):

```
ES = sqrt( ES_T(P)² + Σ_{j≥2} ( ES_T(P, j) × sqrt((LH_j - LH_{j-1}) / 10) )² )
```

Read it as time slices. In the first 10 days every factor moves: `ES_T(P)`. From day 10 to
day 20 the factors with a 10-day horizon are assumed gone and the rest keep moving for 10 more
days: `ES_T(P, 2) × sqrt(10/10)`. From day 20 to 40, the factors with horizon at least 40 move
for 20 more days: `× sqrt(20/10)`. Slices are combined as independent (square, sum, root),
like variances of independent periods.

### 3.3 Calibration to stress: the reduced set (MAR33.5-33.7)

The ES must reflect a period of stress. Many risk factors lack reliable history back to 2007,
so the calibration is indirect (MAR33.5): the bank picks a **reduced set** of factors with long
history, finds the 12-month window over an observation horizon going back at least to 2007
in which the portfolio loses most (MAR33.7), and computes

- `ES_R,S`: reduced set, stressed period;
- `ES_R,C`: reduced set, current (most recent 12 months);
- `ES_F,C`: full set, current;

each liquidity-adjusted as above. Then (MAR33.6)

```
ES = ES_R,S × ES_F,C / ES_R,C,   the ratio floored at 1
```

The reduced set must explain at least 75% of the full model: the average of `ES_R,C / ES_F,C`
over the preceding 12 weeks must be at least 75% (MAR33.5 and its FAQ3), measured at the
level of all IMA desks together (FAQ2). The stressed period and reduced set are updated at
least quarterly (MAR33.7, MAR33.44). If a scope has no reduced-set factors at all, `ES_R,C`
and `ES_R,S` are both zero and the formula gives zero; the 75% test exists to stop that.

### 3.4 The IMCC: capping diversification across risk classes (MAR33.13-33.15)

`IMCC(C)` is the stress-calibrated ES of all modellable factors together, with correlations
across risk classes as the model sees them (**unconstrained**, MAR33.13). `IMCC(C_i)` is the
same with only the factors of broad risk class `i` moving (interest rate, equity, FX,
commodity, credit spread; MAR33.14); the sum of them assumes no diversification across
classes (**constrained**). The capital for modellable risk factors is (MAR33.15)

```
IMCC = ρ × IMCC(C) + (1 - ρ) × Σ_i IMCC(C_i),   ρ = 0.5
```

The same stress period serves the portfolio and every risk class (MAR33.15).

### 3.5 Explained but not built

**Non-modellable risk factors (NMRF).** A factor is modellable only if it passes the **risk
factor eligibility test**: at least 24 real price observations a year with no 90-day period
of fewer than four, or at least 100 over the previous 12 months (MAR31.12-31.13). A factor
that fails gets a **stress scenario capital requirement (SES)**, a loss calibrated to at least
the prudence of the ES at a liquidity horizon of at least 20 days (MAR33.16), and the SES
figures combine with a correlation of 0.6 for the general ones and zero for demonstrated
idiosyncratic credit and equity ones (MAR33.17).

**IMA default risk charge.** A separate default model: a one-year, 99.9% VaR with two types of
systematic factors, computed weekly; the capital is the larger of the 12-week average and the
latest figure (MAR33.18-33.22).

**Multiplier and averaging.** For approved and eligible desks (MAR33.41)

```
C_A = max( IMCC_{t-1} + SES_{t-1};  m_c × IMCC_avg + SES_avg )
```

with averages over the previous 60 days and `m_c` = 1.5 plus a backtesting add-on of 0 to 0.5
(MAR33.42). The market risk requirement is then `C_A + DRC` for eligible desks plus the SA of
the others (`C_U`), with a surcharge when a desk is in the PLA amber zone, limited by
(MAR33.43)

```
ACR_total = min( IMA_G,A + surcharge + C_U;  SA_all desks ) + max( 0;  IMA_G,A - SA_G,A )
```

where the surcharge is `k × max(0, SA_G,A - IMA_G,A)` and `k = 0.5 × (SA of amber desks) /
(SA of all eligible desks)` (MAR33.45).

**Backtesting** (MAR32.4-32.19). The one-day VaR is compared with actual and hypothetical
P&L over the last 12 months (250 days). Bank-wide, at 99%: 0 to 4 exceptions is green
(multiplier 1.50), 5 to 9 amber (1.70 to 1.92), 10 or more red (2.00) (MAR32.9 Table 1). At a
desk, at 97.5% and 99%: more than 12 exceptions at 99% or 30 at 97.5% sends the desk to the SA
(MAR32.19).

**P&L attribution test** (MAR32.20-32.44). Compares the desk's **risk-theoretical P&L**
(what the risk model's own valuation engine says the P&L was) with its **hypothetical P&L**
(front-office revaluation of yesterday's positions with today's market) over 250 days, with
two metrics: Spearman rank correlation and the Kolmogorov-Smirnov distance between the two
distributions (MAR32.34-32.41). Green: correlation above 0.80 and KS below 0.09; red:
correlation below 0.70 or KS above 0.12; amber otherwise (MAR32.42). Red desks go to the SA
(MAR32.43); amber desks stay on the IMA with the surcharge (MAR32.44).

**Desk eligibility** (MAR30.4). Qualitative approval of the bank's risk management, nomination
of desks in and out of scope (a desk may not be put out of scope because its SA is lower), and
ongoing backtesting and PLA per desk. At least 10% of the bank's market risk capital must come
from desks that qualify for the IMA for the bank to keep using it (MAR32.2).

## 4. How each aggregates, and why SA is not a cube roll-up

A cube roll-up assumes that a parent's number is a function of its children's numbers
(usually their sum). That holds for stress P&L and for the RRAO and fails for everything
else on this page:

- **Netting happens before weighting.** The SBM nets sensitivities per risk factor across
  every instrument of the node first (MAR21.4(2)). Desk A long 100 and desk B short 100 of
  the same factor are each charged; the firm, which nets to zero, is not. The DRC does the
  same per obligor (MAR22.19).
- **Correlation-weighted nesting.** Within-bucket and across-bucket square roots of
  correlation-weighted sums, then a maximum over three scenarios. `sqrt(a + b)` is not
  `sqrt(a) + sqrt(b)`, and the binding scenario can differ between a desk and its parent.
- **Recomputation per reporting node.** MAR11.8 and MAR21.7 require the SA for all desks
  together and for each IMA desk standalone, each with its own binding scenario. Every node
  is a separate computation from the netted positions at that node.
- **IMA is a non-linear function of node vectors.** ES needs the node's own summed P&L
  vector; so does each of the 15 cumulative vectors of the cascade (3 sets x 5 horizons) and
  each per-risk-class scope. The ratio and the IMCC are non-linear in them.

So **firm ≠ Σ desks** for every FRTB measure here except the RRAO (section 5.3 and 5.5 give
the tiny numbers).

**What this means for the Spark design.**

- SA (`src/riskagg/measures/frtb_sa.py`, `frtb_drc.py`): per node level, one `groupBy` nets
  sensitivities per node and factor; a self-join of the node's weighted sensitivities within
  each bucket, cross-joined with the three-row scenario table, gives `K_b`; a self-join of the
  node's buckets within each risk class gives the class charge; a window picks the binding
  scenario. `build_level` caches the bucket detail and the total once per level, and the four
  SA measures and the curated `frtb_sa_detail` table are all read from that cache.
- IMA (`src/riskagg/measures/frtb_ima.py`): per node level, **one aggregation over the
  reporting grain**. Each grain row is exploded into the (set, `LH_j`) pairs it belongs to
  (at most 15) by a native `filter` over a literal array of structs; one grouping-sets
  aggregation with element-wise vector sums produces the vector of every node, risk class
  (and `*` for all classes), set and horizon; `es_col` turns each into an ES; a scalar
  `groupBy` applies the cascade, the ratio and the IMCC. No Python UDF, no driver collect.
- Neither is the generic grouping-sets pass that serves VaR. They are their own plans, run
  once per reported level (`[firm_id]` and `[firm_id, business_line_id, desk_id]`), and
  never derived from a lower level.

**What this means for the OLAP layer.** All FRTB measures are **structured**
(`config/semantic_model.yaml`): computed at the declared nodes only, stored in `risk_metrics`,
and looked up. A request at another level (a book, a region) is refused, not approximated.
The intermediate figures are kept in two curated tables, `frtb_sa_detail` (`K_b`, `S_b` per
node, bucket and scenario) and `frtb_ima_detail` (`es_fc`, `es_rc`, `es_rs`, `es` per node and
risk-class scope). Neither may be summed, so neither gets a Pinot star-tree index; they are
read through the drill-down endpoint.

## 5. Worked examples (tiny profile)

The `tiny` profile has three desks: DESK001 and DESK002 in BL01, DESK003 in BL02. Money is in
USD rounded to whole dollars and computed unrounded, so a figure may differ by 1 from the sum
or product of rounded figures shown beside it.

### 5.1 One SBM bucket: DESK002, GIRR delta, USD

DESK002's net GIRR delta sensitivities (P&L per unit rate move, netted over its trades):

| Risk factor | Net sensitivity | Risk weight | Weighted sensitivity |
|---|---:|---:|---:|
| GIRR.USD.1Y | 15,758,470 | 1.1314% | 178,287 |
| GIRR.USD.2Y | 24,818,521 | 0.9192% | 228,141 |
| GIRR.GBP.1Y | 3,291,693 | 1.1314% | 37,241 |

The weights are MAR21.42's 1.6% (1 year) and 1.3% (2 years) divided by √2, because USD and
GBP are specified currencies (MAR21.44).

The USD bucket has two factors. Their tenor correlation is `exp(-3% × |2 - 1| / 1)`, 0.9704
(MAR21.46; the table there prints 97.0%), and the three scenarios give:

| Scenario | ρ(1Y, 2Y) | K_USD |
|---|---:|---:|
| LOW | 0.9409 | 400,469 |
| MEDIUM | 0.9704 | 403,460 |
| HIGH | 1.0000 | 406,428 |

`K_USD = sqrt(178,287² + 228,141² + 2 × ρ × 178,287 × 228,141)`. At HIGH the factors are
perfectly correlated and `K_USD` is just the sum `S_USD` = **406,428**. Lower correlation
gives a smaller `K_b` here because both sensitivities have the same sign; for opposite signs
it would be larger. The GBP bucket has one factor: `K_GBP = S_GBP` = **37,241**.

### 5.2 One cross-bucket step and the binding scenario

Across the two GIRR buckets, `γ` = 50% (MAR21.50), which becomes 0.375 (LOW), 0.500 (MEDIUM),
0.625 (HIGH):

```
GIRR delta = sqrt( K_USD² + K_GBP² + 2 × γ × S_USD × S_GBP )
           = 416,070 (LOW), 423,441 (MEDIUM), 430,686 (HIGH)
```

Both `S` are positive, so higher correlation means a larger charge. DESK002's FX delta goes
the other way, because it is short GBP and long JPY against USD: 1,689,667 (LOW), 1,455,020
(MEDIUM), 1,174,390 (HIGH). Summing the delta, vega and curvature charges of all classes per
scenario gives DESK002's SBM totals:

| Scenario | DESK002 SBM total |
|---|---:|
| LOW | 6,496,321 |
| MEDIUM | 6,342,869 |
| HIGH | 6,141,510 |

The SBM charge is the largest, **6,496,321** at LOW: the hedges in the FX book are worth less
when correlations fall.

### 5.3 SA totals: the firm is not the sum of the desks

| Node | SBM | DRC | RRAO | SA total |
|---|---:|---:|---:|---:|
| DESK001 | 7,363,134 | 0 | 0 | 7,363,134 |
| DESK002 | 6,496,321 | 879,046 | 0 | 7,375,367 |
| DESK003 | 10,194,253 | 1,207,208 | 140,010 | 11,541,471 |
| Sum of desks | 24,053,709 | 2,086,254 | 140,010 | 26,279,972 |
| FIRM (recomputed) | 18,751,325 | 599,295 | 140,010 | 19,490,629 |

Only the RRAO (1% of the one exotic trade's notional) adds up. The SBM and DRC at the firm
come from netting all desks' sensitivities and positions first.

### 5.4 One DRC bucket: DESK002, corporates

| Issuer | Rating | Seniority | Direction | Notional | JTD | Risk weight |
|---|---|---|---|---:|---:|---:|
| ISS009 | AAA | NON_SENIOR | LONG | 15,758,470 | 15,758,470 | 0.5% |
| ISS019 | BBB | EQUITY | LONG | 16,943,207 | 16,943,207 | 6% |
| ISS013 | BB | EQUITY | SHORT | 1,508,801 | -1,508,801 | 15% |

All three have LGD 100% (non-senior debt and equity, MAR22.12), so JTD equals notional.

- Net long JTD: 15,758,470 + 16,943,207 = **32,701,677**; net short: **1,508,801**.
- HBR = 32,701,677 / (32,701,677 + 1,508,801) = **0.9559**.
- Risk-weighted long: 0.5% × 15,758,470 + 6% × 16,943,207 = **1,095,385**; risk-weighted
  short: 15% × 1,508,801 = **226,320**.
- Charge = 1,095,385 - 0.9559 × 226,320 = **879,046**.

DESK002's sovereign bucket holds one short (ISS030, AAA senior, JTD 75% × 3,291,693 =
2,468,770). With no long, HBR is 0 and the charge is 0: a short alone never earns capital
relief.

At the firm, ISS013 also appears at DESK001 as a senior long with JTD 75% × 5,012,145 =
**3,759,109**. The firm nets the two: 3,759,109 - 1,508,801 = **2,250,308** net long. (An
equity short may offset a senior bond long under MAR22.19.) That is part of why the firm DRC,
599,295, is far below the desks' 2,086,254.

### 5.5 One IMA cascade and IMCC: DESK002

The 10-day ES (one-day ES × √10) of the cumulative vectors of DESK002, all risk classes
together:

| Set | LH ≥ 10 | LH ≥ 20 | LH ≥ 40 | LH ≥ 60 | Cascade |
|---|---:|---:|---:|---:|---:|
| Full, current (F,C) | 1,238,368 | 52,871 | 52,871 | 2,125 | 1,241,753 |
| Reduced, current (R,C) | 904,182 | 52,871 | 52,871 | 2,125 | 908,812 |
| Reduced, stressed (R,S) | 3,489,718 | 114,030 | 114,030 | 3,782 | 3,495,306 |

The "LH ≥ 20" and "LH ≥ 40" vectors are the same: DESK002's only factors with a horizon over
10 days are FX vega (40) and GIRR vega (60). "LH ≥ 60" is GIRR vega alone, and nothing has a
120-day horizon. For the full set:

```
ES_F,C = sqrt( 1,238,368² + (52,871 × √1)² + (52,871 × √2)² + (2,125 × √2)² + (0 × √6)² )
       = 1,241,753
```

The longer horizons add little here because the long-horizon factors are small; a desk
dominated by volatility or credit positions would look different.

Stress calibration: the ratio `ES_F,C / ES_R,C` = 1,241,753 / 908,812 = **1.3663** (above 1,
so the floor does not bind), and `ES` = 3,495,306 × 1.3663 = **4,775,800**.

Per risk class alone (each also cascaded and stress-scaled):

| Risk class | ES F,C | ES R,C | ES R,S | ES |
|---|---:|---:|---:|---:|
| EQ | 910,398 | 665,721 | 4,107,845 | 5,617,634 |
| FX | 722,776 | 608,532 | 1,324,056 | 1,572,629 |
| GIRR | 217,024 | 174,517 | 634,934 | 789,581 |

FX's ratio is 722,776 / 608,532, above 1; for GIRR it is 217,024 / 174,517. EQ alone
(5,617,634) is larger than all classes together (4,775,800): the classes diversify each other
in the model.

```
IMCC = 0.5 × 4,775,800 + 0.5 × (5,617,634 + 1,572,629 + 789,581)
     = 0.5 × 4,775,800 + 0.5 × 7,979,845
     = 6,377,822
```

Across the firm: IMCC DESK001 5,783,239, DESK002 6,377,822, DESK003 15,967,860, a sum of
28,128,922; the firm, recomputed from its own vectors, is **20,297,172**.

One more thing the tiny data shows: DESK003's only equity factor (EQ.8) is outside the
reduced set, so its equity `ES_R,C` and `ES_R,S` are 0 and its stress-calibrated equity ES is
0, though its current equity ES is not. In a real bank the 75% test of MAR33.5 would fail on
a reduced set like that; this project does not run the test.

## 6. What this project simplifies compared with a real bank

Marked `SIMPLIFICATION:` in the code.

Standardised approach (`src/riskagg/measures/frtb_sa.py`, `frtb_drc.py`,
`config/reference/frtb_sa.yaml`, `src/riskagg/data/market.py`):

- **One curve per currency for GIRR**: no inflation or cross-currency basis factors
  (MAR21.8), and no 99.90% different-curve correlation (MAR21.45, MAR21.47).
- **Equity spot only**: no repo-rate factor (MAR21.12).
- **Equity buckets 1, 5, 8 and 12 only**: bucket 11 (other sector, MAR21.79) is not supported.
- **One vega factor per bucket**: no option-maturity dimension, so the option-maturity
  correlation is 1 (MAR21.93-21.94).
- **One curvature factor per bucket**: the bucket's CVRs are summed.
- **Sensitivities taken as given**, not computed by the prescribed bumps (MAR21.18-21.26).
- **Curvature up equals curvature down in the generated data**: a symmetric quadratic
  repricing at one shock per risk class (`CURVATURE_SHOCK` in `market.py`), not full
  revaluation.
- **Gross JTD = LGD × notional**, with no P&L adjustment (MAR22.11).
- **No maturity scaling**: every position is treated as one year or longer (MAR22.19-22.20).
- **Netting per issuer ignores seniority**: MAR22.19 only allows a short to offset a long of
  the same or higher seniority.
- **Two DRC buckets**, corporates and sovereigns, with no local-government bucket
  (MAR22.22) and so no cross-bucket hedge question (MAR22.26 forbids it anyway).
- **RRAO at the 1.0% exotic rate only** (MAR23.8); the 0.1% rate for other residual risks is
  not applied.

Internal models approach (`src/riskagg/measures/frtb_ima.py`, `src/riskagg/data/org.py`,
`src/riskagg/data/market.py`):

- **One-day vectors scaled by √10** instead of ES on overlapping 10-day changes (MAR33.4).
- **The stressed window is synthetic**: the current factor model with volatility tripled and
  correlation raised, not the bank's worst 12 months since 2007 (MAR33.7).
- **The reduced set is drawn at random** (80% of factors, with at least one delta factor per
  class) and **never tested against the 75% requirement** (MAR33.5).
- **Liquidity horizons are the MAR33.12 floors** per factor, with no desk-level extension and
  no cap at instrument maturity; equity bucket 12 (an index) takes the large-cap horizon
  directly instead of the weighted average of its constituents (MAR33.12 FAQ5).
- **Scenario P&L is linear in factor returns**, not a full revaluation, so "holding the other
  risk classes constant" (MAR33.14) is exactly the class's own P&L.
- **IMCC only**: no SES for non-modellable factors, no IMA default risk charge, no 60-day
  average, no multiplier, every desk treated as eligible (no backtesting, PLA or desk
  approval).
- **IMA reconciliation check assumes unfiltered specifications**: the audit check that
  recomputes the firm's `frtb_imcc` and its two parts from the raw vectors assumes the firm rows
  come from a specification with no filters; a filtered firm node would need the filter applied
  in the check too (`check_frtb_imcc_from_raw` in `src/riskagg/engine/audit.py`).

## 7. Likely interview questions

1. **What are the three components of the FRTB standardised approach?** The SBM (delta,
   vega and curvature on prescribed risk factors), the default risk charge and the residual
   risk add-on, simply summed (MAR20.4). Code: `frtb_sa_total` in
   `src/riskagg/measures/frtb_drc.py`.
2. **Walk through the SBM for one risk class.** Net sensitivities per factor, multiply by the
   risk weight, combine within each bucket with `ρ` under a square root, combine buckets with
   `γ`, repeat for three correlation scenarios, sum classes per scenario, take the largest.
   Code: `weighted_sensitivities`, `bucket_detail`, `charges_from_detail`, `sbm_total` in
   `src/riskagg/measures/frtb_sa.py`.
3. **Why three correlation scenarios, and why does the low one often bind?** Correlations
   are uncertain and shift in a crisis. Hedges offset through correlation; lower correlation
   means less offset, so a hedged book is charged most in the low scenario (DESK002 above).
   Code: `_scenario` in `frtb_sa.py`, and `test_sa_totals_and_binding_scenario`.
4. **Why can't you sum desk SA charges to get the firm?** Netting happens before weighting,
   the formula is nested square roots and a maximum, and the binding scenario can differ by
   node. On tiny the desks sum to 26.3m and the firm is 19.5m. Code: `build_level` in
   `frtb_drc.py`, and `test_nodes_are_recomputed_not_rolled_up_through_the_run_path` in
   `tests/test_frtb_drc.py`.
5. **How does the DRC recognise hedges?** Per obligor, longs and shorts net (subject to
   seniority); per bucket, the hedge benefit ratio scales down the risk-weighted shorts; no
   offset across buckets. Code: `drc` in `frtb_drc.py`, and `test_hedge_benefit_and_charge`.
6. **Why did FRTB replace VaR with ES?** VaR ignores how bad losses beyond it are and is not
   subadditive; ES at 97.5% averages the tail. Code: `es_col` and `es_from_pnl` in
   `src/riskagg/conventions.py`.
7. **What is the liquidity-horizon cascade?** ES of all factors, plus the ES of the factors
   still held after each horizon, scaled by the square root of the extra time and combined in
   quadrature. It charges illiquid factors for longer. Code: `lh_adjusted_es` and
   `es_by_horizon` in `src/riskagg/measures/frtb_ima.py`, and `test_liquidity_horizon_cascade`.
8. **How is the ES calibrated to stress without full history?** A reduced set of factors with
   long history gives `ES_R,S` in the worst 12 months; it is scaled by `ES_F,C / ES_R,C`,
   floored at 1; the reduced set must explain 75% of the full ES. Code: `stressed_scaling` and
   `scaled_es` in `frtb_ima.py`, and `test_stressed_scaling_pure`.
9. **What is the IMCC and why ρ = 0.5?** Half the unconstrained ES (model correlations across
   risk classes) plus half the sum of per-class ES (no diversification across classes); the
   regulator only half-trusts the bank's cross-class correlations. Code: `imcc` in
   `frtb_ima.py`, and `test_imcc_weights_unconstrained_and_constrained_equally`.
10. **How would you compute 15 cumulative ES vectors per node efficiently in Spark?** Explode
    each grain row into the (set, horizon) pairs it belongs to with a native array `filter`,
    aggregate all of them in one grouping-sets pass with element-wise sums, apply the quantile
    with a native expression, then a scalar `groupBy`. One shuffle per level, no UDF. Code:
    `es_by_horizon` and `_targets` in `frtb_ima.py`.
11. **What decides whether a desk may use the IMA?** Desk-level approval, then ongoing
    backtesting (exception counts at 97.5% and 99%) and the P&L attribution test (Spearman and
    KS; red sends the desk to the SA, amber adds a surcharge), with non-modellable factors
    charged through SES. Not built here; explained in section 3.5. Code: the `SIMPLIFICATION:`
    comments at the top of `frtb_ima.py`.
12. **How do you know the FRTB numbers in the store are right?** Every run reconciles from
    disk: the SA total must equal its components at every node, and the firm IMCC is rebuilt
    from the raw trade vectors with different grouping code; a failing check stops the run
    being official. Golden tests tie this page to the code. Code:
    `check_frtb_sa_is_sum_of_components` and `check_frtb_imcc_from_raw` in
    `src/riskagg/engine/audit.py`.
