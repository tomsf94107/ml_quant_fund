# A8 Top-Decile Cross-Sectional: Interpretation & Action Plan

## What A8 is

Target = "is this ticker in the top 10% of universe today by 5-day forward return?"
OOS AUC = 0.677

This is the only target tested today that produces **macro-independent ticker-picking alpha**.

## Why A8 matters

By construction, top-decile membership cannot be predicted by macro features.
Every ticker on the same date sees the same VIX, yield curve, dollar strength.
So if model gets AUC 0.677 on top-decile, it found ticker-specific signal.

| Feature | A1_pct0 imp | A8 imp |
|---|---|---|
| volatility_10d | 0.040 | 0.183 |
| short_pct_float | 0.039 | 0.165 |
| vwap | 0.032 | 0.068 |
| rev_growth_yoy | 0.000 | 0.034 |
| 8-K total | 0.000 | 0.008 |
| MACRO total | 0.699 | **0.021** |

The macro share drops from 70% to 2%. Cross-sectional features get to shine.

## Signals A8 revealed

### Signal 1: short_pct_float (0.165 importance, rank 2)
Highly shorted names cluster in top-decile. Why: squeeze potential creates extreme moves.
Action: In production, weight predictions higher when short_pct_float > peer-median.

### Signal 2: Volatility-times-short interaction
volatility_10d and short_pct_float together = squeeze candidate.
Action: Engineer interaction feature: volatility_10d * short_pct_float.

### Signal 3: rev_growth_yoy (0.034 importance, rank 11)
Real fundamentals matter for top-decile membership.
Action: Filter BUY candidates by rev_growth_yoy > 0 if available.

### Signal 4: low_52w_ratio (0.041)
Stocks far from 52-week low have less snapback potential.
Action: Watch this in screening; mean-reversion candidates.

## Alternative uses

| # | Approach | Effort | Risk |
|---|---|---|---|
| A | A8 prob as new feature in main model | Medium | Low |
| B | A8 for position SIZING only (not signal) | Low | Low |
| C | A8 as Stage 1 screener, main model as Stage 2 | High | Medium |
| D | Take only feature-engineering lessons (don't ship A8) | Low | Low |
| E | A/B parallel tracking | Low | Low |
| F | Multi-horizon A8 (1d, 3d, 10d, 21d) | Medium | Low |
| G | Sector-conditional top-decile | Medium | Medium |
| H | A8 as scoring overlay on current BUYs | Low | Low |

## Recommended path

Combine B + E + G (low risk, high information, no architecture change):
1. **E**: A/B track A8 alongside current production for 2-4 weeks
2. **B**: If A8 holds up, use for position sizing (1x baseline, 2x when prob_top_decile high)
3. **G**: Run separate A8 variants with sector-conditional ranking
4. After 1 month: decide whether to ship A8 as primary or keep as overlay

## Risks to mitigate

1. **Survivorship bias** — current 117-ticker universe excluded delisted names
   Action: when retraining, include delisted tickers from 2020-2024

2. **Calibration on 10% base rate** — different from 50% base rate model
   Action: separate isotonic calibrator for A8

3. **Cross-sectional ranking at prediction time** — must score all 117 tickers concurrently
   Action: Pipeline C needs adjustment

4. **Volatile regime dependence** — A8 may struggle in calm markets
   Action: monitor by VIX regime

## Files
- Model: models/research/A8_top_decile_5d_h5d_20260525.joblib
- Meta: models/research/A8_top_decile_5d_h5d_20260525.meta.json
- Source: docs/audit_findings_may24_evening.md
