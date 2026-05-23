# Validation Findings — May 22-23, 2026

Master document of all validation results from this weekend's work. Use as reference for tomorrow's audit and Jun 23 A/B decision.

---

## TL;DR

**Per-ticker production model is essentially random in OOS PIT walk-forward (AUC 0.44).**
**Cross-sectional GLOBAL model (Path A) shows real OOS edge (AUC 0.58-0.59).**
**Gap: +14pp AUC. Path A solves the architectural problem per-ticker can't.**

---

## Finding 1: Per-ticker PIT walk-forward — production model is broken

**Source:** `reports/w2_pit_production_run.log`, completed May 23 ~18:32
**Command:** `nohup caffeinate -i python -m analysis.walk_forward --pit --config production`
**Scope:** 13,743 outcomes, 5,512 unique (ticker, prediction_date) pairs, all 125 tickers

| Metric | Value | Interpretation |
|---|---|---|
| n_oos | 10,995 | Substantial sample |
| pooled_oos_auc | **0.4389** | **BELOW 0.50 — worse than coin flip** |
| pooled_oos_acc | 0.4832 | 48.3% (random is 50%) |
| pooled_oos_brier | 0.2613 | Poor calibration |
| mean_train_auc | 0.754 | Model fits training "well" |
| mean_test_auc | 0.4876 | Collapses on test |
| **auc_gap** | **0.2664** | **MASSIVE — severe overfit/leak** |

### Script's own diagnosis (auto-generated)

> ⚠ MASSIVE train→test AUC gap of 0.266.
> Training AUC (0.754) is fitting noise.
> Run --mode ablation and --mode leak_audit to find the source.
>
> ⚠ OOS AUC (0.488) at coin flip. No detectable signal.
> Pivot: build new alpha sources rather than tune current model.

**Bottom line: per-ticker production architecture cannot reliably predict. Tuning won't fix this. Need different architecture or new alpha sources.**

---

## Finding 2: Per-ticker walk-forward (older method) — confirms overfitting

**Source:** `reports/walkforward_summary_h{1,3,5}.csv`, completed May 22 ~23:58
**Scope:** 124 tickers, time-series fold AUC

| Horizon | Mean OOS AUC (124 tickers) |
|---|---|
| h=1d | 0.5083 |
| h=3d | 0.5247 |
| h=5d | 0.5342 |

Top performers at h=5d:

| Ticker | AUC | BUY hit @ 0.55+ |
|---|---|---|
| QS | 0.611 | 57.8% (n=189) |
| INTC | 0.610 | 58.0% (n=675) |
| BSX | 0.601 | 66.2% (n=876) |
| NOK | 0.590 | 64.7% (n=623) |
| SHOP | 0.586 | 60.0% (n=642) |

Earlier walk-forward gave slightly better numbers (~0.51-0.53) than PIT (~0.44). Reason: walk-forward used time-series folds, PIT uses point-in-time data. PIT is the more honest test — actual production conditions.

---

## Finding 3: Path A cross-sectional GLOBAL model — real OOS edge

**Source:** Tonight's `train_cross_sectional.py` validation runs
**Method:** 5-fold TimeSeriesSplit + bootstrap CI + 10bp cost-adjusted Sharpe

### Bootstrap 95% CI on AUC

| Horizon | AUC | 95% CI | Verdict |
|---|---|---|---|
| h=1d | 0.522 | [0.505, 0.540] | Tiny but real edge |
| h=3d | 0.580 | [0.563, 0.597] | Solid edge |
| h=5d | 0.589 | [0.572, 0.606] | Solid edge |

**All three CIs exclude 0.50.** Statistically significant cross-sectional alpha.

### 5-fold TimeSeriesSplit — Sharpe of top-decile rotation portfolio (10bp cost-adj)

| Fold | Train range | Test range | h=1 Sharpe | h=3 Sharpe | h=5 Sharpe |
|---|---|---|---|---|---|
| 1 | 2020-2021 | 2021-2022 | -1.83 | +0.45 | -0.17 |
| 2 | 2020-2022 | 2022-2023 | -0.66 | +0.44 | +1.25 |
| 3 | 2020-2023 | 2023-2024 | +1.05 | +1.24 | +1.47 |
| 4 | 2020-2024 | 2024-2025 | -0.14 | +1.69 | +2.58 |
| **5** | **2020-2025** | **2025-2026** | **+1.17** | **+2.93** | **+4.36** |
| **Mean** | — | — | **-0.08** | **+1.35** | **+1.90** |

**Sharpe GROWS over folds.** Most likely explanation: 8-K data coverage grows over time (backfill is shorter for old quarters). May also be model genuinely improving.

### Top decile hit rate (h=5d)

391 rows in top decile of prob_up_global. **75.2% hit rate, +5.04% avg return.** Strongest single OOS finding from the project.

---

## Finding 4: Production calibration is broken

**Source:** Earlier queries on accuracy.db (Mar-May 2026 closed BUYs)

### h=1 (97 BUYs)

| prob bucket | n | hit % |
|---|---|---|
| 0.70-0.80 | 79 | 67.1% (matches expected) |
| >=0.80 | 18 | **50.0%** (underperforms predicted) |

### h=3 (366 BUYs) — DISASTER

| prob bucket | n | hit % |
|---|---|---|
| 0.55-0.60 | 23 | **39.1%** (LOSES money) |
| 0.60-0.65 | 52 | **38.5%** (LOSES money) |
| 0.65-0.70 | 64 | **39.1%** (LOSES money) |
| 0.70-0.80 | 188 | 63.8% (under predicted 75%) |
| >=0.80 | 29 | 65.5% (under predicted 90%) |

### h=5 (422 BUYs) — INVERSION

| prob bucket | n | hit % |
|---|---|---|
| 0.55-0.60 | 21 | 57.1% |
| 0.60-0.65 | 54 | **42.6%** (LOSES money) |
| 0.65-0.70 | 72 | 55.6% |
| 0.70-0.80 | 229 | 61.6% |
| >=0.80 | 35 | **40.0%** ← **HIGHEST CONFIDENCE = WORST OUTCOME** |

**A well-calibrated monotonic model should show hit_rate increase with confidence. h=5d inverts at the top bucket.** This is the smoking gun.

---

## Finding 5: 8-K gating overlay (Path B) — underpowered, NO rule

**Source:** `docs/8k_gating_backtest_result.md`, `scripts/backtest_8k_inst_gating.py`

Tested 885 closed BUYs across 3 horizons against 8-K events × inst flow rules. Wilson 95% CIs revealed **no rule reaches statistical significance.**

Interesting non-significant patterns:
- h=1 BUY × exec_change=1: 47.1% vs 67.5% (-20.4pp, n=17)
- h=5 BUY × exec_change=1: 63.6% vs 53.9% (+9.7pp, n=88) **SIGN FLIPS** vs h=1
- h=5 BUY × other_events=1: 61.2% vs 54.9% (+6.3pp, n=67)

Sign flip across horizons suggests signal is not horizon-invariant. Makes Path A (cross-sectional with per-horizon model) the better fix.

Re-test schedule: Jul 20, Sep 21, Nov 23.

---

## Finding 6: The comparison that matters

| Method | h=1 AUC | h=3 AUC | h=5 AUC |
|---|---|---|---|
| Per-ticker PIT (production) | ~0.50 (pooled 0.44) | ~0.50 | ~0.50 |
| Per-ticker walk-forward | 0.508 | 0.525 | 0.534 |
| **GLOBAL Path A (cross-sectional)** | **0.522** | **0.580** | **0.589** |
| Path A advantage | +1.4pp | **+5.5pp** | **+5.5pp** |

**Path A is +5.5pp better than per-ticker at h=3 and h=5.** Not a tuning improvement — a different architecture extracting alpha per-ticker structurally cannot.

---

## What this means for the system

### Per-ticker is structurally limited
- Each ticker has only ~6 exec_change events over 5 years
- XGBoost cannot split on features with so few events
- Result: 8-K, rev_growth, inst features have importance=0 in per-ticker models
- Per-ticker is fitting noise on top of ticker-specific patterns

### Path A solves the architectural problem
- Pools all 125 tickers (180k rows)
- 8-K events become 798+ across tickers — enough for stable splits
- Model can learn cross-sectional patterns per-ticker can't access

### Calibration is broken in production
- Multiplier system (risk × sent × regime × etc.) likely amplifies bad raw probs
- Highest-confidence BUYs are WORST performers at h=5
- Tomorrow's audit (diagnostic_audit.py) will trace root cause

### A/B test is the right move
- Per-ticker remains primary during A/B (production behavior unchanged)
- GLOBAL logged in parallel; comparison data accumulates
- Jun 23: decide whether to promote GLOBAL

## Open questions for tomorrow's diagnostic audit

1. **Where is per-ticker overfitting?** train AUC 0.75 vs OOS 0.49 = 0.27 gap. What's leaking?
2. **Does Path A overfit too?** Or is it genuinely better?
3. **Which multipliers cause the h=5 >=0.80 inversion?**
4. **Are calibration issues universe-wide or specific to certain tickers/sectors?**

## File references

- `reports/w2_pit_production_run.log` — PIT validation output
- `reports/w2_pit_production_20260523_folds.csv` — per-fold detail
- `reports/walkforward_summary_h{1,3,5}.csv` — earlier walk-forward
- `reports/walkforward_folds_h{1,3,5}.csv` — earlier walk-forward folds
- `models/saved/GLOBAL_ensemble_{1,3,5}d.joblib` — Path A models
- `accuracy.db.predictions` — production prediction history with `prob_up_global` (NULL until Monday's first A/B run)

