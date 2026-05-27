# Phase 2A Execution Plan — Wed May 28 2026 onwards

## Context (read May 27 PM VN)

A8 = cross-sectional top-decile model:
- v1: OOS AUC 0.677 (92 features, no interactions)
- v2: OOS AUC 0.684 (95 features, with vol_x_short etc.)

Per-ticker baseline AUC: 0.51-0.53 multi-year
Global ranker Q5-Q1 spread: +1.04pp

**Why A8 matters: macro-independent ticker-picking alpha.** Macro features
get 70% importance in per-ticker model but only 2% in A8. Cross-sectional
ranking lets ticker-specific signals shine.

## Decision: USE A8 v1, NOT v2

Why v1 over v2:
- v1 features match our existing 97-feature schema → clean integration
- v2's interaction features (vol_x_short etc.) NOT in our builder
- Adding interactions = scope creep
- +0.7pp AUC difference not worth integration complexity
- We can validate Phase 2A first with v1, add interactions later if needed

## Phase 2A scope (Tue/Wed/Thu May 28-30)

### Day 1 (Wed May 28): Walk-forward A8 OOS predictor

Goal: Generate panel of (date, ticker, a8_prob_top_decile) where each
prediction uses ONLY data BEFORE that date.

Steps:
1. Read models/train_cross_sectional.py — understand current A8 training
2. Write scripts/generate_a8_oos_panel.py:
   - For each prediction date D in our outcomes table:
     - Train A8 on dates < D - buffer (e.g. 5-day buffer for safety)
     - Score A8 on date D for all 125 tickers
     - Append to output panel
3. Cache result to data/a8_oos_panel.parquet
4. Verify: prob ~ 0.10 average (top-decile base rate)

Effort: 4-6 hours (most of Day 1)

### Day 2 (Thu May 29): Builder integration

1. Add column a8_prob_top_decile to FEATURE_COLUMNS
2. In features/builder.py, read from data/a8_oos_panel.parquet
3. Join on (ticker, date)
4. Handle missing values (early dates before A8 had training data)
5. Verify NVDA, AAPL: a8_prob_top_decile populated for recent dates

Effort: 2-3 hours

### Day 3 (Fri May 30): Validation

1. Trigger full Pipeline B retrain (~32 min) with 98 features
2. Compare OOS AUC: with vs without a8_prob_top_decile
3. Per-ticker AUC and per-horizon AUC
4. Q5-Q1 spread on the ranker

Success criteria: +1.0pp AUC sustained on Apr-May 2026 holdout.

If pass: Phase 2A succeeds. Deploy.
If fail: keep A8 trained but don't include as feature. Move to Phase 2H (overlay).

## RISK MITIGATION

### Rule #1 audit notes for tomorrow

(a) Audit Pipeline C: needs adjustment for production scoring
    Pipeline C step: score A8 BEFORE per-ticker generator runs
    
(b) Silent error: if A8 panel missing for some date, prediction = 0.10 default
    Or use last known A8 prob. Document choice.

(c) Flag flip: ML_QUANT_A8_FEATURE=1 env var to gate
    Default OFF until validation passes

(d) Verify script: scripts/a8_consistency_check.py
    For 5 (ticker, date) samples, compare:
      training panel a8_prob_top_decile
      production cache a8_prob_top_decile
    Must match within rounding

(e) Built not known: train_cross_sectional.py — need to read carefully tomorrow

(f) Test patches with real data: backfill A8 OOS for all 89 BUY dates
    in portfolio_returns_ab table. Compare predictions for same dates.

(g) Gap check: walk-forward A8 training must use ONLY data < D
    Standard purged-CV practice. Already solved in walk_forward.py

(h) Verify chain: 
    Pipeline A → Pipeline B (train A8 + per-ticker) → Pipeline C (score A8 → score per-ticker)
    Each stage tested independently before integration

(i) Compiled ≠ verified: end-to-end smoke test on 1 day before full retrain

## CRITICAL RISK: Look-ahead leakage

**The biggest risk.** If A8 panel uses data from date D when predicting date D,
the entire validation is fake.

Mitigation:
- Strict purge: A8 training uses dates < D - 5 (5 day buffer)
- Validation script asserts no future data leaked
- Compare A8 OOS panel prediction for date D with A8 trained ONLY through D-5
  (should match within rounding)

## NEXT IMMEDIATE STEP (Tomorrow morning Thu May 28 VN)

Run: `head -100 models/train_cross_sectional.py`
Understand A8 training architecture before writing OOS panel generator.

## DECISION DATES

| Date | Action |
|---|---|
| Wed May 28 | Build OOS panel generator |
| Thu May 29 | Builder integration |
| Fri May 30 | Validation + decision |

If Fri May 30 fails: Defer to Phase 2H (cheaper overlay) or revisit.
