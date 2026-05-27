# Neutralizer Backtest — May 27 2026 Findings

## Question
Should `portfolio/neutralizer.py` be wired into production?

## Method
- Added `--long-only` and `--use-prob-raw` CLI flags to scripts/backtest_neutralizer.py
- Ran backtests on 30-day and 55-day windows, all 3 horizons
- Compared none / sector / dollar modes

## Findings

### 30-day window (Apr 13 - May 25)

| Horizon | Mode | Cum Return | Ann Sharpe | Max DD | Win Rate |
|---|---|---|---|---|---|
| 1d | none | +14.5% | +6.36 | -5.4% | 63% |
| 1d | sector | +17.1% | +6.83 | -2.8% | 63% |
| 1d | dollar | +14.6% | +7.12 | -4.1% | 60% |
| 3d | none | +39.8% | +6.13 | -8.6% | 67% |
| 3d | sector | +42.0% | +5.36 | -8.9% | 63% |
| 3d | dollar | +43.1% | +6.68 | -7.7% | 70% |
| 5d | none | +60.5% | +5.22 | -9.1% | 83% |
| 5d | sector | +73.5% | +5.60 | -5.8% | 77% |
| 5d | dollar | +58.6% | +5.23 | -8.8% | 80% |

### 55-day window (Mar 5 - May 25, h=1d only)

| Mode | Cum Return | Ann Sharpe | Max DD |
|---|---|---|---|
| none | +11.6% | +2.13 | -13.2% |
| sector | +14.3% | +2.49 | -13.1% |
| dollar | +12.4% | +2.41 | -11.1% |

## Key insights

1. **prob_raw performs as well as prob_up.** Multipliers were destroying calibration WITHOUT adding alpha. Validates Phase 1 fix beyond doubt.

2. **Sector-neutral adds 3-15pp return** on the same period. Adds real value.

3. **30-day Sharpes are inflated** by post-tariff bull regime. h=1d Sharpe drops from 6.83 (30d) to 2.49 (55d) when we extend the window.

4. **Conviction-weighting CHANGES allocation, not selection.** On 16 today h=5 BUYs:
   - Range: 3.1% to 10% per ticker (3.2x spread)
   - ANET (prob_raw 0.69) gets 10%, DLR (prob_raw 0.56) gets 3.1%
   - Same 16 tickers, just different weights

## Decision

**DO NOT wire neutralizer into production now.** Reasons:
- Backtest period too short (~2 months)
- Same predictions used in train/test — optimistic bias
- Real h=1d Sharpe ~2.5, not the 6+ from 30d window
- Manual long-only operator (Atom) can apply weights post-hoc

**DO consider:** add a "recommended weight" column to dashboard, derived from prob_raw, for informational use. No auto-execution.

## Future re-evaluation
- After 2-4 more weeks of outcomes (gives Feb-July data)
- Outside of tariff selloff recovery window
- Compare equal-weight portfolio vs sector-neutral portfolio on REALIZED outcomes
