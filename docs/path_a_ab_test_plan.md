# Path A A/B Test Plan — Cross-Sectional Global Model

**Date created:** May 23, 2026
**Decision date:** June 23, 2026 (~30 days of production data)

---

## What

Test whether the new cross-sectional GLOBAL model (trained on pooled
data from all 125 tickers) outperforms the per-ticker model
architecture currently in production.

Per-ticker models REMAIN PRIMARY. GLOBAL is logged in parallel for
comparison only. Production behavior (BUY/HOLD signals) is unchanged
during this A/B period.

## Why

Validation findings May 23 2026:
- Per-ticker models systematically ignore cross-sectional features
  (8-K importance=0, rev_growth importance=0, inst importance=0)
- Pooled cross-sectional analysis showed real alpha for those features
- Cross-sectional global model (Path A) shows:
  - OOS AUC: h=1d 0.522, h=3d 0.580, h=5d 0.589 (all CIs > 0.50)
  - Top decile h=5d: 75% hit rate, +5.04% avg return
  - 5-fold TimeSeriesSplit: recent fold Sharpe h=5d=4.36 (10bp cost-adj)

These are the strongest OOS results in the project's history.

## Implementation (Option 3)

**SignalResult dataclass:** new field `today_prob_up_global: Optional[float]`

**signals/generator.py:** computes GLOBAL prediction in parallel with
per-ticker. Wrapped in try/except — GLOBAL failure NEVER blocks pipeline.

**accuracy/sink.py:** `log_prediction()` accepts `prob_up_global` param.

**accuracy.db.predictions:** new column `prob_up_global REAL`.

**scripts/daily_runner.py:** passes `sig.today_prob_up_global` to
`log_prediction_to_db()` (both main and watchlist call sites).

## Observability (3 layers)

### Layer 1: Bootstrap check
`run_daily()` verifies GLOBAL models load before predicting. Logs:
A/B GLOBAL models loaded: h=1d OK (90 features) ...
If any horizon fails, warning logged but pipeline continues.

### Layer 2: Daily summary
At end of `run_daily()`, prints:
A/B summary: 375 predictions, GLOBAL coverage 372/375 (99%)
Means: per-ticker=0.512  GLOBAL=0.524  |Δ|avg=0.04  max=0.18
BUY agreement: both=38  only_per-ticker=5  only_GLOBAL=4

### Layer 3: Standalone health script
`python scripts/check_global_ab_health.py [--since 2026-05-25]`
Reports correlation, BUY agreement, and hit rate by source.

## Decision criteria at June 23

Run `python scripts/check_global_ab_health.py --since 2026-05-25`.

Decision matrix:
| GLOBAL hit rate vs per-ticker | Action |
|---|---|
| GLOBAL ≥ per-ticker + 2pp consistently | **Promote GLOBAL to primary** (refactor) |
| GLOBAL < per-ticker | **Keep per-ticker**, kill A/B |
| Within ±1pp | **Extend A/B 4 more weeks** to Jul 21 |

## Files modified

| File | Change |
|---|---|
| `accuracy/sink.py` | log_prediction accepts prob_up_global |
| `signals/generator.py` | SignalResult adds today_prob_up_global; generate_signals computes it |
| `scripts/daily_runner.py` | passes prob_up_global through; Layer 1 + 2 |
| `scripts/check_global_ab_health.py` | NEW Layer 3 script |
| `models/train_cross_sectional.py` | NEW: trains GLOBAL models |
| `models/saved/GLOBAL_ensemble_{1,3,5}d.joblib` | NEW model files |
| `accuracy.db.predictions.prob_up_global` | NEW column |

## GLOBAL model retraining

Pipeline B currently retrains per-ticker models. **GLOBAL models will
go stale** unless retrained. Add to Pipeline B Stage 2 manually for
now, then formalize if A/B promotes GLOBAL.

Command to retrain GLOBAL:
PYTHONPATH=. ML_QUANT_INST_FEATURES=1 python -m models.train_cross_sectional

Runtime: ~10-15 min (panel build) + 3 × 2s training = ~15 min.

## Calendar events to create

Title: **ML Quant Fund — Path A A/B Decision Review**
Date: Tuesday, June 23, 2026
Action:
1. Run `python scripts/check_global_ab_health.py --since 2026-05-25`
2. Review docs/path_a_ab_test_plan.md
3. Apply decision matrix above
4. If promote: refactor generator.py line 737 to use GLOBAL predict as primary
5. If kill: drop prob_up_global column, remove A/B code

## Open risk items to watch

1. GLOBAL model staleness — retraining cadence not formalized
2. The "extra_columns" warning fires 6x per ticker (per-ticker + GLOBAL ×3 horizons) — log noise increased; tomorrow's Option D refactor addresses
3. Need to add GLOBAL retraining to Pipeline B
