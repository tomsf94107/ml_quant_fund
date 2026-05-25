# Phase 2 A — A8 prob_top_decile as feature in main model — DESIGN SPEC

Created May 25, 2026 (Mon VN evening). Implementation Tue-Wed.

## Concept

A8 is currently a separate model that predicts cross-sectional rank (top 10%).
Performance: OOS AUC 0.684, much better than ~0.5 random.

Phase 2 A: feed A8's output (prob_top_decile per ticker per date) as an
INPUT FEATURE to the main per-ticker model. Per-ticker model learns when
'A8 says this is top decile' should boost or temper its own signal.

## Goal

Beat the per-ticker model's current OOS AUC by leveraging A8's complementary
cross-sectional signal.

Expected lift: +1-3pp AUC if the signals are truly complementary, 0pp if A8
is just rediscovering what per-ticker already knows.

## Architecture options

### Option X — Score-all-at-inference (recommended)

Pipeline C orchestration change:
1. Build features for all 125 tickers (already happens)
2. Run A8 model on full panel (batch call, returns prob per (ticker, date))
3. Cache results in /tmp/a8_predictions_TODAY.parquet
4. Per-ticker generator reads a8_prob_top_decile from cache as a feature

Pros:
- No DB schema changes
- Pipeline C handles all orchestration
- A8 runs once, all per-ticker models use same value

Cons:
- Cache file is intermediate state (must be created fresh each day)
- Race condition: if cache stale, prediction uses old A8
- Pipeline C is the source of truth

### Option Y — Cross-sectional service (Option G from yesterday)

1-2 days of work. Detailed yesterday in decision_per_ticker_interactions_over_cross_sectional_service.md.

Defer to Q3 2026 sprint unless we find compelling reasons today.

## Training architecture

Two-stage walk-forward:

Stage 1: Train A8 on data up to date D-1 (cross-sectional, top-decile target)
Stage 2: Score A8 on date D for all 125 tickers (OOS predictions)
Stage 3: Use those A8 predictions as features when training per-ticker on date D+1+
Stage 4: Per-ticker model now has a8_prob_top_decile as 96th feature

Critical: A8 predictions must be OOS (trained on data BEFORE the date being predicted),
otherwise look-ahead leakage.

## Feature definition

`a8_prob_top_decile` = LightGBM prediction from A8 model (cross-sectional top-decile target).
Range: 0.0 to 1.0. Baseline ~0.10.

## Rule #1 audit

(a) Audit Pipeline C orchestration — need to verify it can handle the new batch step.
    Pipeline C currently uses daily_runner_batched. Need to add A8 scoring before it.

(b) Silent error: if A8 fails for some tickers, those rows get a8_prob_top_decile=0.
    Per-ticker model trained with 0 = 'no A8 signal'. Tolerable.

(c) Flag flip: New feature added to FEATURE_COLUMNS. All 125 per-ticker models become
    stale until Pipeline B retrains. Same as Phase 1 D.

(d) Verify script: need a8_consistency_check.py — for a sample (ticker, date), compute
    a8_prob_top_decile in training panel vs production cache. Must match.

(e) Built-not-known: I don't know Pipeline C orchestration in detail. Need to read.

(f) Test patches with real data: backtest Phase 2 A vs baseline on Apr-May 2026 OOS.

(g) Gap-check: walk-forward A8 training must not use date D+1 data when predicting date D.
    Standard issue. Already solved in walk_forward.py framework.

(h) Verify chain: Pipeline A (data) -> Pipeline B (train A8 + train per-ticker) ->
    Pipeline C (score A8 -> score per-ticker -> generate signals) -> outcomes match.

(i) Compiled OK != verified: full integration test required.

## Estimated effort

- Walk-forward A8 OOS prediction generator: 4-6 hr
- features/builder.py integration (read from cache): 2 hr
- Pipeline C orchestration change: 4-6 hr
- Training + AUC validation: 2-3 hr
- End-to-end testing: 4 hr

Total: 2-3 days of focused work.

## Validation

Train per-ticker model WITH a8_prob_top_decile vs WITHOUT.
Measure OOS AUC delta on Apr-May 2026 hold-out.

Success criteria: +1.0pp AUC sustained over hold-out window.

Failure: A8 signal redundant with existing features OR overfits to A8 noise.

## Out of scope for Phase 2 A

- Position sizing changes (Phase 3 B)
- Overlay filter (Phase 2 H — separate spec exists)
- Sector-conditional A8 (Phase 4 G)

## Risk if we skip Phase 2 A

A8's signal stays trapped in a research artifact. Production never benefits from
the cross-sectional alpha we proved exists. Phase 2 H (overlay) is a weaker but
cheaper alternative — uses prob_pct7 instead of A8 prob. If 2 A is too hard, 2 H is plan B.
