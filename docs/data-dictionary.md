# Data dictionary

Generated from `config/semantic_model.yaml` by `riskagg dictionary`. Do not edit by hand.

Every measure has one of three aggregation classes, which decide how it may be combined:

- **additive**: values sum over any of the measure's dimensions.
- **vector**: a VaR or ES. Never summed. P&L is summed scenario by scenario over the requested
  slice, and the quantile convention (`src/riskagg/conventions.py`) is applied once to each
  group's vector.
- **structured**: a capital charge, ratio or worst case recomputed at each node from that
  node's own inputs. Never summed, averaged or rebuilt from other slices; only looked up at the
  levels where it was computed.

A query for a slice the class does not allow is refused with the reason (HTTP 422). In
`risk_metrics`, a dimension that a node does not group by holds the literal `*`, never null.

## Measures

### `var_99_1d`

| | |
|---|---|
| Business name | VaR 99% 1-day |
| Description | Historical-simulation value at risk, 99%, 1-day horizon, current window |
| Unit | USD |
| Aggregation class | vector |
| Parameters | level 0.99, window CURRENT, horizon_days 1 |
| Answered from | queries: `pnl_scenarios`; aggregations declared in `config/aggregations.yaml` also store rows in `risk_metrics` |
| May be queried by | any subset of: `firm_id`, `business_line_id`, `desk_id`, `book_id`, `legal_entity`, `region`, `asset_class`, `currency`, `risk_class` |
| Rule for combining | Never summed. The P&L of the slice is summed scenario by scenario from `pnl_scenarios` (CURRENT window), then the VaR at 0.99 is taken of each group's vector (`src/riskagg/conventions.py`). |

### `var_99_10d`

| | |
|---|---|
| Business name | VaR 99% 10-day |
| Description | Historical-simulation value at risk, 99%, 10-day horizon, current window |
| Unit | USD |
| Aggregation class | vector |
| Parameters | level 0.99, window CURRENT, horizon_days 10 |
| Answered from | queries: `pnl_scenarios`; aggregations declared in `config/aggregations.yaml` also store rows in `risk_metrics` |
| May be queried by | any subset of: `firm_id`, `business_line_id`, `desk_id`, `book_id`, `legal_entity`, `region`, `asset_class`, `currency`, `risk_class` |
| Rule for combining | Never summed. The P&L of the slice is summed scenario by scenario from `pnl_scenarios` (CURRENT window), then the VaR at 0.99 is taken of each group's vector (`src/riskagg/conventions.py`). Scaled by sqrt(10) to the horizon. |

### `es_975_1d`

| | |
|---|---|
| Business name | ES 97.5% 1-day |
| Description | Expected shortfall, 97.5%, 1-day horizon, current window |
| Unit | USD |
| Aggregation class | vector |
| Parameters | level 0.975, window CURRENT, horizon_days 1 |
| Answered from | queries: `pnl_scenarios`; aggregations declared in `config/aggregations.yaml` also store rows in `risk_metrics` |
| May be queried by | any subset of: `firm_id`, `business_line_id`, `desk_id`, `book_id`, `legal_entity`, `region`, `asset_class`, `currency`, `risk_class` |
| Rule for combining | Never summed. The P&L of the slice is summed scenario by scenario from `pnl_scenarios` (CURRENT window), then the ES at 0.975 is taken of each group's vector (`src/riskagg/conventions.py`). |

### `svar_99_1d`

| | |
|---|---|
| Business name | Stressed VaR 99% 1-day |
| Description | Value at risk, 99%, 1-day horizon, stressed window |
| Unit | USD |
| Aggregation class | vector |
| Parameters | level 0.99, window STRESSED, horizon_days 1 |
| Answered from | queries: `pnl_scenarios`; aggregations declared in `config/aggregations.yaml` also store rows in `risk_metrics` |
| May be queried by | any subset of: `firm_id`, `business_line_id`, `desk_id`, `book_id`, `legal_entity`, `region`, `asset_class`, `currency`, `risk_class` |
| Rule for combining | Never summed. The P&L of the slice is summed scenario by scenario from `pnl_scenarios` (STRESSED window), then the VaR at 0.99 is taken of each group's vector (`src/riskagg/conventions.py`). |

### `stress_pnl`

| | |
|---|---|
| Business name | Stress P&L |
| Description | Sensitivity-based P&L of the book under a stress scenario. Negative is a loss. Additive within a scenario; slice by scenario, then sum over anything else. |
| Unit | USD |
| Aggregation class | additive |
| Answered from | `stress_results` |
| May be queried by | any subset of: `firm_id`, `business_line_id`, `desk_id`, `book_id`, `legal_entity`, `region`, `asset_class`, `currency`, `risk_class`, `scenario` |
| Rule for combining | Additive: sum the values over any of its dimensions. Values under different scenarios are different quantities: slice by `scenario`, then sum over anything else. |

### `worst_stress_loss`

| | |
|---|---|
| Business name | Worst stress loss |
| Description | Loss under the worst stress scenario of the node, as a positive number (negative if every scenario gains). The scenario is in the scenario column of risk_metrics. Not additive: only looked up at the nodes where it was computed. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`); (`firm_id`, `business_line_id`, `desk_id`); (`firm_id`, `business_line_id`, `desk_id`, `book_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. Each row names its scenario in the `scenario` column. |

### `frtb_sa_charge`

| | |
|---|---|
| Business name | FRTB SA capital charge |
| Description | Standardised approach market risk capital: sensitivities-based charge (binding correlation scenario) plus default risk charge plus residual risk add-on. A positive number is capital. Recomputed from the netted positions at each node, never a sum of the nodes below: only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `frtb_sbm_charge`

| | |
|---|---|
| Business name | FRTB SBM charge |
| Description | Sensitivities-based method charge at the binding correlation scenario (MAR21). A positive number is capital. Recomputed from the netted positions at each node, never a sum of the nodes below: only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `frtb_drc_charge`

| | |
|---|---|
| Business name | FRTB default risk charge |
| Description | Default risk charge for non-securitisations, summed over buckets (MAR22). A positive number is capital. Recomputed from the netted positions at each node, never a sum of the nodes below: only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `frtb_rrao_charge`

| | |
|---|---|
| Business name | FRTB residual risk add-on |
| Description | Residual risk add-on on exotic-underlying notional (MAR23). A positive number is capital. Recomputed from the netted positions at each node, never a sum of the nodes below: only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `frtb_imcc`

| | |
|---|---|
| Business name | FRTB IMA capital for modellable risk (IMCC) |
| Description | Internal models approach capital for modellable risk factors (MAR33.15): 0.5 x the unconstrained liquidity-adjusted, stress-calibrated 97.5% ES plus 0.5 x the sum of the per-risk-class ES. Excludes NMRF (SES), the IMA default risk charge, the multiplier and the 60-day average. A positive number is capital. Recomputed from each node's own vectors, never a sum of the nodes below: only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `frtb_imcc_unconstrained`

| | |
|---|---|
| Business name | FRTB IMCC(C), unconstrained |
| Description | Liquidity-adjusted, stress-calibrated 97.5% ES of all risk classes together, with correlations across risk classes as observed (MAR33.13, MAR33.15). Only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `frtb_imcc_constrained_sum`

| | |
|---|---|
| Business name | FRTB sum of IMCC(C_i), constrained |
| Description | Sum over risk classes of each risk class's liquidity-adjusted, stress-calibrated 97.5% ES computed alone (MAR33.14): no diversification across risk classes. Only looked up. |
| Unit | USD |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `business_line_id`, `desk_id`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. |

### `ccar_start_cet1_ratio`

| | |
|---|---|
| Business name | CCAR starting CET1 ratio |
| Description | Common equity tier 1 capital over risk-weighted assets at the start of the CCAR planning horizon (quarter 0), as a decimal. Scenario is *. A capital ratio: computed for legal entities and the firm only; capital is not allocated to business lines, desks or books. |
| Unit | ratio |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `legal_entity`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. Not allocated to `business_line_id`, `desk_id`, `book_id`: a query naming one is refused. |

### `ccar_min_cet1_ratio`

| | |
|---|---|
| Business name | CCAR minimum CET1 ratio |
| Description | Lowest projected CET1 ratio over the nine projected quarters of a CCAR scenario (scenario column: BASELINE or SEVERELY_ADVERSE), as a decimal. The firm's ratio is the firm's CET1 over the firm's RWA, not a sum or average of entity ratios. Legal entities and firm only. |
| Unit | ratio |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `legal_entity`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. Not allocated to `business_line_id`, `desk_id`, `book_id`: a query naming one is refused. Each row names its scenario in the `scenario` column. |

### `ccar_scb`

| | |
|---|---|
| Business name | Stress capital buffer (stylised) |
| Description | max(2.5%, starting CET1 ratio - minimum ratio under SEVERELY_ADVERSE + planned common dividends of the add-on quarters / RWA in the trough quarter), as a decimal (12 CFR 225.8(f)(2)). Scenario is *. Legal entities and firm only; never summed. |
| Unit | ratio |
| Aggregation class | structured |
| Answered from | `risk_metrics` |
| May be queried by | exactly one of: (`firm_id`); (`firm_id`, `legal_entity`) |
| Rule for combining | Never summed, averaged or rebuilt from other slices: only looked up at a level where it was computed. Not allocated to `business_line_id`, `desk_id`, `book_id`: a query naming one is refused. |

## Dimensions

Reporting grain: `firm_id`, `business_line_id`, `desk_id`, `book_id`, `legal_entity`, `region`, `asset_class`, `currency`, `risk_class`.

### `firm_id`

| | |
|---|---|
| Business name | Firm |
| Description | Top of the organisation hierarchy |
| In the reporting grain | yes |
| Hierarchy | org level 1 of 4 |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl`, `worst_stress_loss`, `frtb_sa_charge`, `frtb_sbm_charge`, `frtb_drc_charge`, `frtb_rrao_charge`, `frtb_imcc`, `frtb_imcc_unconstrained`, `frtb_imcc_constrained_sum`, `ccar_start_cet1_ratio`, `ccar_min_cet1_ratio`, `ccar_scb` |

### `business_line_id`

| | |
|---|---|
| Business name | Business line |
| Description | Business line within the firm |
| In the reporting grain | yes |
| Hierarchy | org level 2 of 4 |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl`, `worst_stress_loss`, `frtb_sa_charge`, `frtb_sbm_charge`, `frtb_drc_charge`, `frtb_rrao_charge`, `frtb_imcc`, `frtb_imcc_unconstrained`, `frtb_imcc_constrained_sum` |

### `desk_id`

| | |
|---|---|
| Business name | Desk |
| Description | Trading desk within the business line |
| In the reporting grain | yes |
| Hierarchy | org level 3 of 4 |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl`, `worst_stress_loss`, `frtb_sa_charge`, `frtb_sbm_charge`, `frtb_drc_charge`, `frtb_rrao_charge`, `frtb_imcc`, `frtb_imcc_unconstrained`, `frtb_imcc_constrained_sum` |

### `book_id`

| | |
|---|---|
| Business name | Book |
| Description | Trading book within the desk |
| In the reporting grain | yes |
| Hierarchy | org level 4 of 4 |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl`, `worst_stress_loss` |

### `legal_entity`

| | |
|---|---|
| Business name | Legal entity |
| Description | Booking legal entity |
| In the reporting grain | yes |
| Hierarchy | none |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl`, `ccar_start_cet1_ratio`, `ccar_min_cet1_ratio`, `ccar_scb` |

### `region`

| | |
|---|---|
| Business name | Region |
| Description | Booking region |
| In the reporting grain | yes |
| Hierarchy | none |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl` |

### `asset_class`

| | |
|---|---|
| Business name | Asset class |
| Description | Asset class of the position |
| In the reporting grain | yes |
| Hierarchy | none |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl` |

### `currency`

| | |
|---|---|
| Business name | Currency |
| Description | Trade currency |
| In the reporting grain | yes |
| Hierarchy | none |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl` |

### `risk_class`

| | |
|---|---|
| Business name | Risk class |
| Description | Regulatory risk class |
| In the reporting grain | yes |
| Hierarchy | none |
| Measures with a slice that includes it | `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `stress_pnl` |

### `scenario`

| | |
|---|---|
| Business name | Scenario |
| Description | Stress scenario name |
| In the reporting grain | no |
| Hierarchy | none |
| Measures with a slice that includes it | `stress_pnl` |

### `counterparty_id`

| | |
|---|---|
| Business name | Counterparty |
| Description | Trade counterparty (not in the vector grain) |
| In the reporting grain | no |
| Hierarchy | none |
| Measures with a slice that includes it | none |

## Tables

### `risk_metrics`

One row per aggregation, measure and node of the official run. node_level names the node's grouping set (dimensions joined by |); a dimension the node does not group by holds *. Every measure here is looked up, never summed across rows.

Measures stored in it: `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d`, `worst_stress_loss`, `frtb_sa_charge`, `frtb_sbm_charge`, `frtb_drc_charge`, `frtb_rrao_charge`, `frtb_imcc`, `frtb_imcc_unconstrained`, `frtb_imcc_constrained_sum`, `ccar_start_cet1_ratio`, `ccar_min_cet1_ratio`, `ccar_scb`.

| Column | Type | Role | Nullable |
|---|---|---|---|
| `as_of_date` | DATE | time | no |
| `run_id` | STRING | dimension | no |
| `spec_hash` | STRING | dimension | no |
| `aggregation` | STRING | dimension | no |
| `measure` | STRING | dimension | no |
| `node_level` | STRING | dimension | no |
| `firm_id` | STRING | dimension | no |
| `business_line_id` | STRING | dimension | no |
| `desk_id` | STRING | dimension | no |
| `book_id` | STRING | dimension | no |
| `legal_entity` | STRING | dimension | no |
| `region` | STRING | dimension | no |
| `asset_class` | STRING | dimension | no |
| `currency` | STRING | dimension | no |
| `risk_class` | STRING | dimension | no |
| `scenario` | STRING | dimension | no |
| `value` | DOUBLE | metric | no |

### `pnl_scenarios`

P&L per scenario (scenario_idx) and window at the reporting grain: the source of every VaR and ES query. Additive across rows of one window and one scenario_idx only; a VaR or ES is a quantile of a group's per-scenario sums.

Every query of `var_99_1d`, `var_99_10d`, `es_975_1d`, `svar_99_1d` is answered from it.

| Column | Type | Role | Nullable |
|---|---|---|---|
| `as_of_date` | DATE | time | no |
| `run_id` | STRING | dimension | no |
| `firm_id` | STRING | dimension | no |
| `business_line_id` | STRING | dimension | no |
| `desk_id` | STRING | dimension | no |
| `book_id` | STRING | dimension | no |
| `legal_entity` | STRING | dimension | no |
| `region` | STRING | dimension | no |
| `asset_class` | STRING | dimension | no |
| `currency` | STRING | dimension | no |
| `risk_class` | STRING | dimension | no |
| `window` | STRING | dimension | no |
| `scenario_idx` | INT | dimension | no |
| `pnl` | DOUBLE | metric | no |

### `stress_results`

Stress P&L per scenario at the reporting grain (gains positive). Additive across rows of one scenario; never summed across scenarios.

Measures stored in it: `stress_pnl`.

| Column | Type | Role | Nullable |
|---|---|---|---|
| `as_of_date` | DATE | time | no |
| `run_id` | STRING | dimension | no |
| `firm_id` | STRING | dimension | no |
| `business_line_id` | STRING | dimension | no |
| `desk_id` | STRING | dimension | no |
| `book_id` | STRING | dimension | no |
| `legal_entity` | STRING | dimension | no |
| `region` | STRING | dimension | no |
| `asset_class` | STRING | dimension | no |
| `currency` | STRING | dimension | no |
| `risk_class` | STRING | dimension | no |
| `scenario` | STRING | dimension | no |
| `pnl` | DOUBLE | metric | no |

### `frtb_sa_detail`

FRTB SA bucket detail per node, risk class, component (DELTA, VEGA, CURVATURE), bucket and correlation scenario: k_b is the bucket charge, s_b the bucket's signed sum of weighted sensitivities. Neither may be summed over buckets, nodes or scenarios.

No measure is stored in it: read it with the drill-down.

| Column | Type | Role | Nullable |
|---|---|---|---|
| `as_of_date` | DATE | time | no |
| `run_id` | STRING | dimension | no |
| `node_level` | STRING | dimension | no |
| `firm_id` | STRING | dimension | no |
| `business_line_id` | STRING | dimension | no |
| `desk_id` | STRING | dimension | no |
| `risk_class` | STRING | dimension | no |
| `component` | STRING | dimension | no |
| `bucket` | STRING | dimension | no |
| `corr_scenario` | STRING | dimension | no |
| `k_b` | DOUBLE | metric | no |
| `s_b` | DOUBLE | metric | no |

### `frtb_ima_detail`

FRTB IMA expected shortfalls per node and risk-class scope (risk_class * is all classes together), each liquidity-horizon adjusted: es_fc for the full set of risk factors on the current window, es_rc for the reduced set on the current window, es_rs for the reduced set on the stressed window, and es = es_rs x max(es_fc / es_rc, 1). None may be summed over nodes or risk classes.

No measure is stored in it: read it with the drill-down.

| Column | Type | Role | Nullable |
|---|---|---|---|
| `as_of_date` | DATE | time | no |
| `run_id` | STRING | dimension | no |
| `node_level` | STRING | dimension | no |
| `firm_id` | STRING | dimension | no |
| `business_line_id` | STRING | dimension | no |
| `desk_id` | STRING | dimension | no |
| `risk_class` | STRING | dimension | no |
| `es_fc` | DOUBLE | metric | no |
| `es_rc` | DOUBLE | metric | no |
| `es_rs` | DOUBLE | metric | no |
| `es` | DOUBLE | metric | no |

### `ccar_projection`

The CCAR path: one row per scenario, projected quarter (1 to 9) and node (node_level BUSINESS_LINE, LEGAL_ENTITY or FIRM). Loss and revenue columns add across nodes of one level within a scenario and quarter; cet1, rwa and cet1_ratio are null for business lines and are levels on a path, never summed over quarters or levels.

No measure is stored in it: read it with the drill-down.

| Column | Type | Role | Nullable |
|---|---|---|---|
| `as_of_date` | DATE | time | no |
| `run_id` | STRING | dimension | no |
| `scenario` | STRING | dimension | no |
| `quarter` | INT | dimension | no |
| `node_level` | STRING | dimension | no |
| `node_id` | STRING | dimension | no |
| `ppnr` | DOUBLE | metric | no |
| `credit_loss` | DOUBLE | metric | no |
| `trading_loss` | DOUBLE | metric | no |
| `net_income` | DOUBLE | metric | no |
| `dividends` | DOUBLE | metric | yes |
| `cet1` | DOUBLE | metric | yes |
| `rwa` | DOUBLE | metric | yes |
| `cet1_ratio` | DOUBLE | metric | yes |
