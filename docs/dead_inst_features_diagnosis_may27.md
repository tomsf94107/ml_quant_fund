# Dead inst_* Features — Root Cause Diagnosis

**Date:** May 27 2026 PM VN
**Context:** Yesterday's dead feature audit showed all 4 inst_* (dark pool)
features have 100% zero LGB importance across 30 sampled tickers.
Initial guess: data quality issue or model architecture mismatch.

## Diagnosis

**It's neither. It's TEMPORAL COVERAGE.**

| Feature | Coverage | Date range |
|---|---|---|
| inst_block_buy_sell_7d | 8-10% of training rows | Mar 20 - May 26 2026 |
| inst_signed_flow_30d | 8-10% | Mar 20 - May 26 2026 |
| inst_auction_imbal_5d | 8-9% | Mar 20 - May 22 2026 |
| inst_signed_flow_5d | 8-9% | Mar 20 - May 22 2026 |

Builder loads 2-year training window (Jan 2024 - May 2026).
UW Lee-Ready inst data only goes back ~10 weeks (Mar 20 2026).
Result: 91% of training rows have feature=NaN → filled with median/0.
Trees cannot find information gain in a feature that's constant in 91% of cases.

## Implications

**Adding more dark pool / Form 4 / darkpool_prints features WILL FAIL THE SAME WAY.**

Tested:
- form4_parsed: 35 tickers, 12 distinct days (May 1-26), even sparser than UW inst
- darkpool_prints (earnings_monitor.db): starts Apr 2 2026, similar coverage

## Options

A. Wait — coverage improves naturally. By Sept 2026 we'll have ~6 months,
   reaching ~25% coverage. By Mar 2027, ~50% coverage.

B. Shorten training window to 60 days. Drastic — loses long-term signal,
   would catastrophically reduce non-inst feature quality.

C. Reframe features as binary "event in last 7d" indicators. Less rich
   signal but coverage becomes ~100%.

D. Backfill historical UW inst data — depends on API support, unclear.

## Decision

**Do not add Form 4 dollar-value or block-size darkpool features now.**
They will be dead-on-arrival for the same reason as existing inst_* features.

Revisit options C and D in next session. For now, focus on A8 integration
(which has 5+ years of training data).

## Validation tests run

NVDA, AAPL, TSLA, EME, BYND inspected — all confirm Mar 20 2026 start.
form4_parsed top tickers (DDOG, DELL, ANET, SNOW) — all cluster on 1-3 days.
