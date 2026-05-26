# Session Learnings — Tue May 26 2026 VN

## What we found

### Bug 1: Insider features blind to slow signal (FIXED)
insider_7d and insider_21d windows too short to catch quarterly insider
activity. CFO sales spread over weeks don't aggregate to meaningful values
in rolling-7d window.

Fix: Added insider_60d and insider_90d. Now NVDA -3.2M insider sells visible
to model. Was completely invisible before.

### Bug 2: Multiplier compound inflates probabilities (FIXED via env var)
7 multipliers (risk, sent, regime, options, squeeze, intraday, fg) compound
multiplicatively. Net 1.05-1.08x on average. Pushes borderline BUYs
(prob_raw 0.62) into high-confidence bucket (prob_eff 0.68+).

Result: bucket-level calibration broken even though aggregate is good.

Fix: env var ML_QUANT_DISABLE_MULTIPLIERS=1 opts out of multiplier compound.
Default behavior unchanged.

### Architectural finding 1: GLOBAL stale after per-ticker retrain
GLOBAL models 90 features (May 24). Per-ticker now 97 features (May 26 PM).
Path A A/B comparison invalid until GLOBAL retrained.
Action: Run scripts/post_pipeline_B_retrain_global.sh after Pipeline B.

### Architectural finding 2: Dark pool data only, no lit flow
2.1M institutional_trades rows ALL is_dark_pool=1.
inst_dp_signed_flow_5d is identical to inst_signed_flow_5d (filter is no-op).
UW has /api/lit-flow endpoint but we don't ingest it.

### Architectural finding 3: 13F holdings data unused
earnings_monitor.db.institutional_holdings has 10K rows of quarterly 13F
filings. NVDA: 497 institutions filed. Not loaded into model.
Future feature engineering opportunity.

### Architectural finding 4: BYND/penny stocks have NO inst data
0 institutional rows for BYND. Penny stocks excluded.
Combined with structural blind spots (price level, DTC), explains why model
gives false positives on penny stocks.

## What we shipped (5 commits)

1. f13ecf7 — insider_60d/90d features
2. 2b264e0 — Phase 1 calibration env var
3. a46d8f9 — MASTER_TODO Phase 1 status
4. 3fd71c6 — Dashboard Prob Raw column
5. df7b36b — Phase 2 deferred + data gaps documented

Plus:
- scripts/verify_post_retrain_nvda.sql
- scripts/post_pipeline_B_retrain_global.sh
- docs/MASTER_TODO_LIST.md substantially updated

## What's running

Pipeline B retrain in progress with ML_QUANT_INST_FEATURES=1 +
ML_QUANT_DISABLE_MULTIPLIERS=1. ETA ~16:30 VN.

Will produce 97-feature per-ticker models, prob_eff = prob_raw.

## Open items

1. Run GLOBAL retrain post-Pipeline-B (15-30 min)
2. Verify NVDA signal change with new features
3. Compare today's BUY count vs yesterday
4. Document journal entry

## Rule #1 quality

Every change verified before commit:
- Phase 1 env var: 4-ticker A/B test before commit
- Insider 60d: BYND specifically verified -898K matches UW data
- Dashboard column: AST check before commit
- Each commit individually tested, individually pushed

## Calibration baseline (Pre-Fix)

For reference, pre-fix calibration on post-May-8 data:
- h=1 >=0.80 BUYs: predicted 89%, hit 33% (-55pp)
- h=3 0.65-0.70: predicted 65%, hit 39% (-26pp)
- h=5 >=0.80 BUYs: predicted 84%, hit 40% (-44pp)

Friday May 29 should show improvement on new predictions.
