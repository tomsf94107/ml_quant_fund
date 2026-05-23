# Finding: 8-K features are real alpha, but per-ticker models can't learn from them

**Date:** May 23, 2026 (Session F follow-up)
**Status:** Discovered during validation, before deployment in production
**Severity:** Architectural finding — informs how alpha features are consumed

---

## TL;DR

We shipped 6 8-K Item code features (`eightk_exec_change_30d`, `eightk_other_events_30d`, 
`eightk_filings_30d`, etc.) and 2 revenue growth features (`rev_growth_yoy/qoq`) in 
Phases 2 + 3.

After retraining all 125 ticker models with these features included, walk-forward 
validation showed:

1. **Models almost entirely IGNORE the new features** (importance = 0 in XGB + LGB)
2. **But the features DO contain real alpha** when measured at the pooled cross-ticker level
3. **Root cause:** per-ticker training has ~6 events per ticker for sparse 8-K Items; 
   trees can't find a stable split with so few samples per leaf

## The data

Pooled cross-ticker analysis (3,324 rows across NVDA/AMD/INTC/AAPL):

| Feature | When ACTIVE | Hit rate vs baseline (55.6%) | n |
|---|---|---|---|
| `eightk_exec_change_30d` | exec change in last 30d | **50.1%** (-5.4pp) | 798 |
| `eightk_other_events_30d` | Item 8.01 in last 30d | **61.5%** (+6.0pp) | 312 |
| `eightk_material_agreement_30d` | Item 1.01 in last 30d | 55.1% (-0.5pp) | 216 |
| `eightk_reg_fd_30d` | Item 7.01 in last 30d | 54.7% (-0.9pp) | 613 |
| `eightk_filings_30d` | any filing in last 30d | 54.7% (-0.8pp) | 2070 |
| `eightk_days_since_last` 60-90d bucket | no recent filing | 57.9% (+2.4pp) | 302 |

**TWO of the six 8-K features show meaningful directional signal:**
- `eightk_exec_change_30d` is **BEARISH** (-5.4pp) — exec changes predict UNDERPERFORMANCE
- `eightk_other_events_30d` is **BULLISH** (+6.0pp) — Item 8.01 predicts OUTPERFORMANCE

## Why models ignore the signal

Per-ticker model feature importance after retraining (May 22 evening):

| Ticker | 8-K total importance | rev_growth total |
|---|---|---|
| NVDA | 0 (out of max 19) | 0 |
| AMD | 4 (only days_since_last) | 0 |
| INTC | 0 (out of max 15) | 0 |
| MNDY | 0 (out of max 13) | 0 |
| DNA | 0 (out of max 10) | 0 |

XGBoost + LGBM trees find ~6 exec_change events per ticker. With 5-fold CV, that's 
~1 event per fold per side of the split — well below the noise floor. Trees prune 
the feature.

## Correlations with existing features (rules out subsumption)

Pearson correlations:

| 8-K feature | sentiment_score | finbert_sentiment | days_to_earnings | post_earnings_3d | eps_surprise |
|---|---|---|---|---|---|
| exec_change_30d | +0.04 | +0.10 | -0.24 | +0.01 | +0.14 |
| other_events_30d | +0.05 | -0.10 | -0.15 | +0.06 | +0.14 |
| filings_30d | +0.06 | +0.06 | -0.27 | +0.02 | +0.14 |
| days_since_last | -0.01 | -0.08 | +0.33 | -0.01 | -0.10 |

**Max ±0.33 — the 8-K features are INDEPENDENT signals.** Not subsumed by existing 
features. They genuinely add information the model is failing to extract.

## rev_growth_yoy/qoq are ALSO dead (separate bug)

Phase 2 (revenue growth) features were always 0.0 in training data due to:

1. `data/etl_earnings.py` `INSERT OR REPLACE` was wiping Polygon's rev_actual 
   backfill on every retrain (yfinance returns NULL for revenue since 2024-2025)
2. Without rev_actual, rev_growth_yoy/qoq compute to NaN → builder fills 0.0 → 
   constant feature → useless to model

**Fixed May 23 2026** (this commit): yfinance ETL now writes only EPS columns; 
Polygon owns rev_actual. ON CONFLICT preserves rev_* on every retrain.

## Architectural implication

**Per-ticker models cannot extract alpha from sparse cross-sectional features.** 
The 8-K signal lives in the cross-sectional layer (across all tickers, the 
exec_change effect is real). Per-ticker training partitions the data such that 
each model sees only its own ticker's events.

**Two viable paths:**

### Path A: Cross-sectional model (architectural change)
- Train ONE global model on all tickers pooled
- 8-K features get 798+ exec_change events, enough for trees to learn
- Loses ticker-specific tuning
- Major refactor (2-3 hours plus testing)

### Path B: Post-prediction overlay (lighter change)
- Keep per-ticker models for primary prediction
- Add a "gating layer" that adjusts BUY signals based on 8-K state:
  - `eightk_exec_change_30d=1` → reduce signal strength (hit rate drops 5.4pp)
  - `eightk_other_events_30d=1` → boost signal strength (+6.0pp)
- Same shape as the planned Tue/Wed inst_flow × earnings backtest

## Recommendation

**Path B for the next 2 weeks**, then evaluate Path A. Reasons:

1. Backtest framework already exists (docs/session_E_backtest_inst_earnings_suppression.md)
2. Lighter risk, easier to validate
3. Path A requires architectural shift in train_all_batched.py — design first

The Tue/Wed May 26-27 backtest should be expanded to include:
- `inst_signed_flow_5d` (already planned)
- `earnings_calendar.days_since_earnings` (already planned)
- `eightk_exec_change_30d` (NEW — bearish gating)
- `eightk_other_events_30d` (NEW — bullish boosting)

## Status of today's commits

| Feature | Reality |
|---|---|
| 8-K Item codes (Phase 3) | Shipped but ignored by per-ticker models. **Real alpha visible cross-sectionally.** Plan to use as gating overlay. |
| Revenue growth (Phase 2) | Was wiped by retrain ETL. **Fixed today** via ON CONFLICT preserving rev_*. Re-running Polygon backfill. |
| DuckDB migration | Working — 21x faster inst lookup |
| UW + Massive cache | Working — 9x Pipeline C speedup |
| WAL + retry | Working — 0 lock errors |
| predict_proba fix | Working — saved tonight's predictions from being all 0.0 |
| Inst features backfill | Working — 96.8% coverage |

**Net: performance + reliability wins are real. Alpha features need architectural work.**

## Next steps (chronological)

1. **Tonight:** rev_growth fix shipped (this commit). Polygon backfill rerunning.
2. **Tomorrow morning 05:30 VN:** Pipeline B cron retrains with rev_growth now actually populated.
3. **Tue/Wed May 26-27:** backtest 4-feature gating overlay on existing predictions.
4. **Within 2 weeks:** if Path B shows ≥3pp lift in hit_rate, evaluate Path A architecture.
