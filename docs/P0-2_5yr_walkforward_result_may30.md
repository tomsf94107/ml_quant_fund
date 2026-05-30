# P0-2 — 5-Year PIT Walk-Forward Result (May 30 2026)

Closes P0-2, and with it C3 + C4. Per-ticker model edge question is ANSWERED.

## Setup
- Backfilled outcomes 2020-01-02 -> 2026-05 (688k rows, canonical target close.shift(-h)/close-1).
- Batch A1: 40 regime-balanced tickers, WEEKLY cadence, 5yr, h=5, --config production.
- PIT panel (training_mode=True, UW-free): 13,360 builds, 0 failures, 13,360 rows, 102 live feats.
- Scored via walk_forward_backtest, 5 folds, 5d embargo. n_oos = 10,688.

## Result — NO EDGE
| | value |
|---|---|
| pooled OOS AUC | **0.4866** |
| mean test AUC | 0.4936 |
| pooled OOS acc | 49.6% |
| pooled Brier | 0.257 (=random) |
| mean train AUC | 0.7406 |
| train->test gap | **0.247 (every fold)** |

Per-fold test AUC: 0.495 / 0.440 / 0.502 / 0.538. No regime above ~0.54; most at/below 0.50.

## Verdict
1. The May 23 per-ticker 0.44 was REAL, not a small-sample/3-month artifact. Reproduces at 5yr, n=10,688.
2. The May 27 "weak-but-real, regime-dependent edge" rescue is REFUTED. Across 5yr / all regimes, OOS = chance.
3. The stable pathology is a 0.247 train->test gap in EVERY fold: model fits noise, generalizes to nothing.
4. Harness auto-diagnosis concurs: "OOS AUC at coin flip. No detectable signal. Pivot: build new alpha sources rather than tune current model."

## 3-AUC reconciliation (C4) — final
- Per-ticker (this test): ~0.49 OOS. The honest number. No edge.
- WF stacks 0.51-0.53: different METHOD (time-series stacks), not the per-ticker classifier's OOS.
- Path A 0.58: different ARCHITECTURE (cross-sectional GLOBAL). NOT yet honestly validated the same way.
- The gap was never a bug. Three methods measuring three things. The per-ticker leg is the weak one and is now disproven.

## Consequences
- P3 ROADMAP_HYBRID_ADVISOR sizing work stays DEAD (not blocked - disproven). Cannot size on a coin-flip per-ticker signal.
- Next real question: does Path A (GLOBAL cross-sectional, 0.58) survive a 5yr PIT WF of its own? That is the pivot, per the harness verdict ("build new alpha sources").
- Caveats on this result: survivorship (universe = tickers with recent outcomes; pre-2026 delistings absent); IPOs in set have phantom pre-IPO outcome rows but they drop at the empty-feature join; ETFs excluded by design (separate run B3).

## A2 confirmation
A2 (other 40 tickers, same config) launched to confirm. Expected: same ~0.49. Does not change verdict.


## A2 CONFIRMATION (May 30, ran ~same night)
Second 40-ticker half, same config. Combined n_oos = 21,376.
| metric | A1 | A2 |
|---|---|---|
| pooled OOS AUC | 0.4866 | 0.4931 |
| mean test AUC | 0.4936 | 0.5002 |
| train->test gap | 0.247 | 0.230 |
Per-fold A2: 0.472 / 0.464 / 0.518 / 0.547 — same shape as A1 (early<0.50, fold4 mild, uniform ~0.23 gap).
VERDICT LOCKED: per-ticker model = coin flip across 80 tickers / 5yr / 21,376 OOS. No edge. Not regime-specific. C3/C4/P0-2 closed.
