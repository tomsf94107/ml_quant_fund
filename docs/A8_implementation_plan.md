# Status (updated May 25, 2026 17:00 VN)

| Phase | Step | Status | AUC | Notes |
|---|---|---|---|---|
| ε (epsilon) | Shadow-log PCT7 | **DEPLOYED** | logged | A1_pct7 model shadow-logged as prob_pct7. accuracy/sink.py, signals/generator.py, scripts/daily_runner.py patched. Verified end-to-end. |
| 1 | D — Feature interactions | TESTED | marginal | vol_x_short, rev_x_peer, low52w_x_short tested. Top importance ranks but ~0 AUC lift. NOT shipped to production. |
| 1 | E — A/B tracking | SUPERSEDED by ε | — | ε already does shadow logging. |
| 2 | A — A8 prob as feature | pending | — | Next: re-train PCT7-INCLUSIVE main model. |
| 2 | H — Overlay scoring filter | pending | — | Filter BUYs with low prob_pct7. |
| 3 | B — Position sizing by A8 | pending | — | After Phase 2 validates signal. |
| 3 | F — Multi-horizon A8 | pending | — | Train h=1, h=3, h=10, h=21. |
| 4 | G — Sector-conditional A8 | pending | — | Highest expected lift. |
| 4 | C — Two-stage screener | pending | — | Most invasive. |

# Implementation Plan (original below)

# A8 Implementation Plan: All 8 Alternatives in Phases

Updated: May 25, 2026 (Sunday late evening)
Source: docs/audit_findings_may24_evening.md + A8 finding

## Background

A8 (top-decile cross-sectional target) achieved OOS AUC 0.677 with macro-independent
ticker-picking alpha (rev_growth importance 0.034, short_pct_float 0.165, vol 0.183).
This is a NEW alpha source distinct from current production (any-positive target,
mostly macro-driven).

Decision: extract maximum value via 8 layered alternatives, in 4 phases.

## Phase 1 — Foundation (Week 1, May 26 - Jun 1)

### D. Feature engineering lessons
**Goal:** Add interaction features informed by A8 finding.
**Specific features to add:**
- `vol_x_short`: volatility_10d * short_pct_float (squeeze candidate)
- `rev_x_peer_rank`: rev_growth_yoy * peer percentile rank
- `low52w_x_short`: low_52w_ratio * short_pct_float (mean-reversion + squeeze)
**Effort:** 1-2 hr
**Files:** features/builder.py, models/classifier.py FEATURE_COLUMNS
**Gate:** New features show non-zero importance in test retrain. AUC delta logged.

### E. A/B tracking
**Goal:** Log A8 prob_top_decile alongside production predictions for every BUY.
**What:** Pipeline C scores all 117 tickers, computes prob_top_decile via A8 model,
attaches to prediction record.
**Effort:** 1-2 hr
**Files:** signals/generator.py, schema migration to predictions table
**Gate:** A/B records flow for 1 week before analysis.

## Phase 2 — Score integration (Week 2, Jun 2 - Jun 8)

### A. A8 prob as new feature in main model
**Goal:** Add prob_top_decile (from A8 model) as a feature in the main model.
**What:** During Pipeline B retrain, score historical panel with A8 model, use
prob_top_decile as input feature. Main model gains squeeze-candidate awareness.
**Effort:** 4-6 hr
**Risk:** Bootstrapping problem — first-time feature, needs A8 model already trained
**Files:** models/train_all.py, models/classifier.py
**Gate:** Backtest shows AUC improvement of at least 1pp.

### H. Overlay scoring on current BUYs
**Goal:** Filter or annotate BUYs by A8 score. Skip BUYs with low prob_top_decile.
**What:** Pipeline C adds A8 score to each prediction. If prob_top_decile < 0.20,
downgrade BUY to HOLD.
**Effort:** 2-3 hr
**Files:** signals/generator.py
**Gate:** Live A/B test for 1 week shows hit rate improvement.

## Phase 3 — Capital allocation (Week 3, Jun 9 - Jun 15)

### B. Position sizing by A8
**Goal:** Scale position sizes by prob_top_decile.
**Scheme:** Baseline 1x, scale to 2x at prob > 0.5, 0.5x at prob < 0.2
**Effort:** 4-6 hr
**Files:** signals/generator.py, portfolio construction logic
**Gate:** Backtest shows Sharpe improvement of at least 0.1.

### F. Multi-horizon A8
**Goal:** Train A8 at h=1, h=3, h=10, h=21. Different horizons may capture
different ticker dynamics.
**Effort:** 4-6 hr
**Files:** models/train_cross_sectional.py
**Gate:** At least one new horizon shows AUC > 0.65 and engages different feature mix.

## Phase 4 — Architectural changes (Week 4, Jun 16 - Jun 22)

### G. Sector-conditional top-decile
**Goal:** Rank within sectors instead of universe. Top 10% per sector per date.
**What:** Use sector_etf_map to define peer group, rank within. Trains separate A8
variant.
**Effort:** 4-6 hr
**Files:** models/train_cross_sectional.py, scripts/save_experiment_artifact.py
**Gate:** AUC competitive with universe-wide A8 (within 2pp).

### C. Two-stage screener
**Goal:** A8 selects top 20 daily, main model picks 5 from those.
**Most invasive change.** Replaces stage 1 of inference.
**Effort:** 1-2 days
**Files:** signals/generator.py, Pipeline C orchestration
**Gate:** End-to-end shadow trading shows hit rate at or above current.

## Stop conditions

- If a phase shows AUC delta < 0.5pp on real OOS, pause and reassess.
- If A/B tracking shows A8 underperforms in live data, abort B/H/C.
- If overfitting suspected (train AUC >> test AUC), strip back features.

## Files to track
- models/research/A8_top_decile_5d_h5d_20260525.joblib (the A8 model)
- All implementations land on research-track branch first, merge to main after gate.

## Engineering checklist per phase
- [ ] Rule #1 audit before code
- [ ] Backup file before edit (.bak.YYYYMMDD)
- [ ] AST validate after edit
- [ ] Unit test or smoke test
- [ ] Backtest on holdout
- [ ] A/B for at least 5 trading days
- [ ] Document findings before promoting
