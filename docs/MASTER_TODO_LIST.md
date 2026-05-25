# Master Unified TODO List

**Last updated:** May 25, 2026 (Mon end-of-session VN)
**Sources consolidated:**
- userMemories (session_reset summaries)
- docs/TODO_calibration_audit_priority.md
- docs/TODO_refactor_feature_dataframe.md
- docs/roadmap_4week_quant_completeness.md
- docs/A8_implementation_plan.md
- docs/8k_alpha_finding_and_gating_plan.md
- docs/feature_improvement_plan.md
- docs/path_a_ab_test_plan.md
- docs/audit_findings_may24_evening.md
- All commits and changes from Sun-Mon May 24-25 sessions

---

## P0 — CRITICAL FOUNDATION (blocker, address FIRST)

### C1 — Calibration broken via multipliers
**Status:** Confirmed May 25 evening. Multiplier system destroys calibration.
**Data:** prob_raw is perfectly calibrated (h=3 64.0/64.0%, h=5 63.8/63.8%).
prob_eff (after multipliers) miscalibrated by -30 to -43pp at >=0.80 bucket.
**Action:** Identify destructive multiplier. Options:
- Test each multiplier individually (turn off one at a time, measure cal_err)
- Cap multiplier compounding (max 1.0x net)
- Switch BUY threshold to operate on prob_raw not prob_eff
- Bucket-override sizing (use observed hit rate per bucket)
**Effort:** 4-8 hours
**Blocks:** All position sizing, exit logic, dashboard items (W1.1, W1.2, W1.3)

### C2 — Calibration verification post-fix
After C1, re-run the calibration query to confirm fix.
**Effort:** 30 min

---

## P1 — IN FLIGHT (this week)

### S2 — Phase 2 H promote to active (Fri May 29)
Currently shadow mode. After 5 BD outcomes, decide activate.
**Trigger:** Fri May 29 EOD.

### S3 — Phase epsilon monitoring (Fri May 29)
First 127 outcomes arrive Fri. Run monitor_pct7_ab.py.
**Trigger:** Fri May 29 EOD.

### N1 — Verify Pipeline B picks up 95 features
Tomorrow's automated retrain should produce 95-feature per-ticker models.
**Trigger:** Tue May 26 after Pipeline B completes.

### N2 — Verify SKIP_TICKERS expansion respected
Pipeline A skips GLD, QQQ, SLV, XLB, XLC, XLRE properly.
**Trigger:** Tue May 26 after Pipeline A.

### R2 — prob_raw column patch ui/1_Dashboard.py:504
Display prob_raw alongside prob_up in dashboard.
**Effort:** 10 min.
**Pending since:** Prior session.

---

## P2 — NEAR-TERM (this/next week)

### S1 — Phase 2 A: A8 prob as feature in main model
Spec exists. ~2-3 days implementation.
Walk-forward A8 OOS prediction generator + Pipeline C orchestration.

### A2/A3/A4 — Path A A/B observability layers
Layer 1 bootstrap check, Layer 2 daily summary, Layer 3 health script.
**Decision date:** June 23, 2026.

### S4 — Inst suppression rule backtest
Original Tue/Wed plan. Recommend SKIP per session memory (inst features dead).
**Decision needed:** formal skip or do it.

### R1 — Refactor build_feature_dataframe (separate features from diagnostics)
Spec exists in TODO doc. 1-2 hours. Cleans up extras_columns warnings.

### F1 — Drop dead features
From feature_improvement_plan, was "PRIORITY SHIP TODAY" — unclear if done.
**Verify:** check which dead features remain in FEATURE_COLUMNS.

---

## P3 — 4-WEEK ROADMAP (deferred until calibration fixed)

### Week 1 — Foundations (BLOCKED by C1)
- W1.1 Position sizing by prob_eff bucket
- W1.2 Stop-loss + exit rules
- W1.3 Live performance dashboard
- W1.4 Hit-rate regression test suite

### Week 2 — Risk Management
- W2.1 Portfolio-level risk limits
- W2.2 Volatility-scaled position sizing
- W2.3 Daily P&L attribution
- W2.4 Drawdown monitor + circuit breaker
- W2.5 GLOBAL retraining in Pipeline B

### Week 3 — Alpha Research
- W3.1 Multi-horizon signal aggregation
- W3.2 Slippage + spread cost model
- W3.3 Earnings drift (PEAD) strategy
- W3.4 Path A A/B decision review (Jun 23 trigger)

### Week 4 — Operational Maturity
- W4.1 Model versioning + rollback
- W4.2 Live paper trading + Sharpe vs benchmark
- W4.3 Feature lineage tracking
- W4.4 Macro overlay
- W4.5 Cost-attribution dashboard

---

## P4 — FUTURE PHASES (4-8 weeks out)

### F2 — Sector ETF features expansion (high ROI)
sector_rel_ret exists, expand to more sector signals.
**Effort:** 2-3 hours, expected +0.005-0.015 AUC.

### F3 — FinBERT historical backfill
6-8h setup + 2h compute. Expected +0.005-0.015 AUC.

### F4 — pc_ratio_change_5d + iv_skew_change features
8-12 hours. Expected +0.003-0.010 AUC.

### K2 — Cross-ticker 8-K alpha exploitation
Per-ticker models ignore 8-K features. Cross-sectional path needed.
Partially addressed by Phase epsilon if PCT7 picks up 8-K signal.

### S6 — Q3 2026: Option G cross-sectional service
Proper architecture for peer-rank features.
**Defer per:** docs/decision_per_ticker_interactions_over_cross_sectional_service.md

### S5 — Sept 15, 2026: Inst features reconsideration
After 6 months of inst data depth.

---

## P5 — CLEANUP / COSMETIC

### N3 — Audit ALTER TABLE migrations for missing columns
Check if other recent schema changes have same gap as prob_up_global did.

### N5 — Cleanup apply_*.py scripts in repo root
Old patch scripts. Either delete or archive.

### N6 — Suppress extras_columns warning
Either implement R1 (refactor) or whitelist 5 known extras.

### A8 v2 model
Sits in research/, not deployed. Will be needed for Phase 2 A.

### Stale rev for 5-7 fiscal-year-end cos
Will auto-update. Monitor only.

---

## P6 — FAR FUTURE (deferred from week-4 roadmap)

| Item | Why deferred |
|---|---|
| Bayesian/probabilistic models | Big architectural change |
| Deep learning experimentation | Lower priority than ops gaps |
| Alternative data (satellite, etc.) | Cost-effective only at larger AUM |
| Multi-asset (futures, FX, crypto) | Equity model not yet validated live |
| Options strategies | Need equity validation first |
| Risk-parity portfolio construction | Need vol-scaled sizing first |
| Statistical arbitrage on pairs | Cross-sectional global model is first step |
| Phase 4 LLM extraction | Existing alpha not yet exploited |
| Sessions B/C/D FinBERT (10-Q, 10-K) | Existing FinBERT not yet driving signals |

---

## SCHEDULED CHECKPOINTS

| Date | What |
|---|---|
| Tue May 26 | Pipeline A/B/C auto-run with 95 features. Verify. |
| Fri May 29 EOD | First Phase epsilon outcomes. Run monitor_pct7_ab.py. |
| Fri May 29 EOD | Phase 2 H downgrade precision check |
| Mon Jun 1 | Second batch of outcomes |
| Fri Jun 5 | One week of A/B data. Decision: promote/revert PCT7. |
| Sun Jun 23 | Path A A/B decision review (per path_a_ab_test_plan.md) |
| Mon Sep 15 | Inst features hit 6-month threshold |
| Q3 2026 | Option G cross-sectional service |

---

## HONEST PRIORITIZATION

**The calibration bug is the foundation crack.** Every signal/feature/overlay layered on top of prob_eff inherits the bug. Phase 1 D and Phase 2 H both interact with prob_eff somewhere.

If calibration is fixed:
- Phase 2 H thresholds may need recalibration
- Bucket boundaries shift
- Some "BUYs" today become "HOLDs" automatically

If calibration is NOT fixed:
- Phase 2 H may be selecting against the wrong noise
- Phase 2 A trains on miscalibrated targets
- 4-week roadmap items 1.1, 1.2 build on quicksand

**Tomorrow's first action: fix calibration (C1).**


---

## UPDATE: May 25 2026 late-night calibration diagnostic + research

### What we found

Reran calibration audit (TODO_calibration_audit_priority.md), filtered to post-May-8
data with full multiplier logging. Confirmed:

**Component 1 — Raw model miscalibration**
  h=3 0.65-0.70 bucket (n=64): avg_prob_raw=68.5%, hit=39.1% (-29pp from raw)
  Model itself over-confident in mid-conviction tickers. NOT a multiplier issue here.
  Possibly distribution shift since training; isotonic over-fit; or both.

**Component 2 — Multiplier amplification**
  Net multiplier 1.05-1.08x on >=0.80 BUYs.
  Compounds over-confidence further, pushes borderline picks into high-conviction bucket.
  Detail: regime_mult averages 1.05 (likely BULL multiplier); squeeze_mult 1.017; others ~1.00.

**Critical observations**
  - HOLD calibration is fine (no inflation, all buckets 47-58% hit)
  - prob_raw at h=3/h=5 BUYs averages 64.0%/63.8% with hit rate 64.0%/63.8%
    Aggregate-level prob_raw is PERFECTLY calibrated — but bucket-level isn't
  - h=5 highest-conviction (>=0.80) hit only 40% (-43.6pp error)

### Research: better calibration methods (May 2026 SOTA)

Standard post-hoc calibration methods for binary classifiers:

| Method | Best for | Effort |
|---|---|---|
| Platt scaling | Small cal sets (<1000), robust | 2-3 hr |
| Isotonic regression (CURRENT) | Large cal sets, overfits on tails | (in production) |
| Beta calibration | Where isotonic overfits | 3-4 hr |
| Venn-Abers | Distribution-free, non-stationary | 6-8 hr |
| Box-constrained recalibration | Hard probability bounds + refit | 4-6 hr |

Key findings from recent literature:
- Monotonic methods preserve ranking but fix systematic over/under-confidence
- Box-constrained calibration directly addresses both overconfidence AND
  underconfidence with explicit probability bounds
- Venn-Abers provides distribution-free validity guarantees, suited for
  non-stationary financial markets

### REVISED P0 plan (replaces Option E from earlier in this doc)

Original Option E (cap → recalibrate → bucket override) had unknown blast radius
and weeks of work. Revised plan is faster + more reversible:

**Phase 1 — Disable multipliers (1-2 hr, Tuesday morning)**
  Add env var ML_QUANT_DISABLE_MULTIPLIERS. Set BUY threshold on prob_raw.
  Continue logging prob_eff for shadow comparison.
  Eliminates Component 2 instantly. Reversible.

**Phase 2 — Platt scaling on PCT7 (2-3 hr, Tuesday afternoon)**
  Replace isotonic calibration with Platt scaling (CalibratedClassifierCV
  method='sigmoid'). Platt is the original post-hoc method, robust on small
  calibration sets where isotonic overfits.
  Test on PCT7 first (95 features), measure new cal_err before retraining
  125 per-ticker models.

**Phase 3 — Beta calibration if Platt insufficient (3-4 hr, Wednesday)**
  If Phase 2 doesn't fully fix Component 1, try beta calibration.
  Specifically designed for binary classifiers where isotonic over-fits tails.

**Phase 4 — Box-constrained recalibration (4-6 hr, Thursday)**
  Combines bound enforcement with re-fit. Most thorough fix.

**Phase 5 — Venn-Abers as long-term architecture (8+ hr, future)**
  Distribution-free validity. Better suited to non-stationary markets.

### Why this revised plan is better

- Phase 1 ships in hours, stops the bleeding immediately
- Phase 2 uses well-studied alternative (Platt) instead of just re-fitting broken approach
- Each phase reversible (keep old artifacts)
- Each phase has clear measurement (cal_err per bucket)
- Phases are independent (Phase 2 success doesn't depend on Phase 1)

### Rule #1 audit on Phase 1 (env var disable)

(a) Need to find ALL places reading prob_eff vs prob_raw — signals/generator.py and
    signals/risk_gate.py for BUY decisions, position_sizer.py (not implemented yet),
    dashboard for display.
(b) Some downstream code may expect prob_eff != prob_raw; check for ratios/divisions.
(c) env var ML_QUANT_DISABLE_MULTIPLIERS = clean opt-in flag.
(d) Verify: compute hypothetical BUY count today with vs without; signal change rate.
(e) Need to read multiplier compounding code first.
(f) Run on historical predictions; measure new cal_err per bucket.
(g) Check earnings gate / risk gate / position sizing for multiplier dependencies.
(h) code → flag check → conditional → DB → measure.
(i) End-to-end test before commit.

### Impact on existing plans

- Phase 2 H overlay (shipped tonight): thresholds may need recalibration after
  Phase 1 disables multipliers. Currently overlay fires when prob_pct7 < 0.10
  AND today_signal == BUY. If multipliers disabled, fewer BUYs, fewer overlay fires.
- 4-week roadmap Week 1 items 1.1 (position sizing) and 1.2 (stops) require
  calibrated prob. They can resume after Phase 1.
- Phase 2 A (A8 prob as feature) unaffected — A8 trains on cross-sectional rank,
  not prob_eff.

### Schedule

Tue morning May 26: Phase 1 (disable multipliers via env var)
Tue afternoon: Phase 2 (Platt scaling on PCT7)
Wed: Phase 3 (Beta calibration if needed)
Thu: Phase 4 (box-constrained if needed)
