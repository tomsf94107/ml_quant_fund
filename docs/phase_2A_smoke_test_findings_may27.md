# Phase 2A Step 4 Smoke Test — Honest Findings May 27 2026 VN

## Test design

5-ticker per-ticker retrain with ML_QUANT_A8_FEATURE=1.
Compare AUC + a8 feature importance.
Goal: validate full retrain BEFORE 32 min commitment.

## Results

| Ticker | Test AUC | a8_imp | n_features |
|---|---|---|---|
| NVDA | 0.5015 | **0.0** | 98 |
| AAPL | 0.4439 | **0.0** | 98 |
| TSLA | 0.5176 | **0.0** | 98 |
| MSFT | 0.5962 | **0.0** | 98 |
| EME  | 0.3922 | **0.0** | 98 |

**Mean AUC: 0.4903**  (similar to default-flag baseline 0.4758)
**a8_prob_top_decile importance: ZERO in every model**

## Root cause — redundancy

A8 prob is highly correlated with existing per-ticker features:

| Existing feature | Correlation with a8_prob |
|---|---|
| volatility_10d | +0.79 |
| volatility_5d | +0.69 |
| vwap | -0.59 |
| bb_width | +0.57 |
| atr | +0.53 |
| rev_growth_yoy | +0.47 |
| macd | -0.40 |

Per-ticker trees prefer the RAW correlated features over A8's linear
combination. Trees are doing the right thing — A8's signal is already
accessible through its components.

## What this means

**Phase 2A approach (A8 as per-ticker feature) FAILS BY DESIGN.**
A8 doesn't add new information; it's a re-encoding of existing inputs.

This is a legitimate scientific finding. The 5-ticker smoke test
saved ~30 min of full retrain time.

## A8 still has value — just not as per-ticker feature

A8's AUC 0.677 is real CROSS-SECTIONAL alpha. It works because it
ranks tickers WITHIN A DATE, not within a single ticker's history.
The cross-sectional ranking captures relative outperformance that
per-ticker models can't see by construction.

## Path forward — use A8 as separate decision layer

| Phase | Approach | How |
|---|---|---|
| 2H | OVERLAY filter | Downgrade BUYs with low a8_prob |
| 3B | POSITION SIZING | Scale weights by a8_prob |
| 4C | STAGE 1 SCREENER | A8 selects top 20, main model picks 5 |
| 4G | SECTOR-CONDITIONAL A8 | Rank within sector |

## Decision

- **Code KEPT in place** (flag-gated, default OFF, doesn't affect production)
- **`.env` flag REMOVED** (no production impact)
- **Pipeline B 03:00 cron tomorrow** runs unchanged (clean retrain)
- **NEXT: Phase 2H** (overlay) likely best near-term alternative

## Files involved
- features/builder.py (A8 join logic, flag-gated)
- models/classifier.py (FEATURE_COLUMNS extension, flag-gated)
- data/a8_oos_panel.parquet (93,668 rows, walk-forward OOS predictions)
- scripts/generate_a8_oos_panel.py (panel generator)
- models/train_top_decile.py (A8 training script)

All A8 artifacts remain valid for downstream phases (2H, 3B, 4C, 4G).
