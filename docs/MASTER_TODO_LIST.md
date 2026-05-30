# Master Unified TODO List

**Last updated:** May 28, 2026 (Thu session VN)
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

## RECENTLY CLOSED (May 28 2026)

### ❌ 4G — Sector-conditional A8 (TESTED + REJECTED May 28)
Hypothesis: rank A8 WITHIN sector vs universe-wide. Result (496 days):
universe +3.43% vs sector-relative +2.68% → sector-relative WORSE -0.75pp.
A8 is cross-sectional alpha; its edge IS concentrating where signal is
strongest regardless of sector. Sector-neutralizing dilutes it. DO NOT REVISIT.
Script: research/test_4g_sector_a8.py.

### 🔬 4G v2 — Sector as CONFIRMATION filter (TESTED May 28, marginal)
User idea: hold names that are BOTH top-10 universe AND top-2 in sector
(intersection, not replacement). Rigor (1451 days, 0 thin days):
mean return wins all periods (full +0.17pp, H1 +0.08, H2 +0.27) BUT Sharpe
is a WASH (0.217 vs 0.212; H2 slightly worse). More return + more risk,
~same efficiency. Like lightly leveraging universe-wide — not a clean edge.
Scripts: research/test_4g_v2_intersection.py, test_4g_v2_rigor.py.
Lesson: first test +0.53pp, rigor +0.17pp — trust the more rigorous one.

### ⏳ NEXT-SESSION ITEMS (from May 28)
- 3B — A8 position sizing: ❌ TESTED + REJECTED May 28 (Sharpe flat).
  This ALSO closes 4G v2's revisit path: 4G v2 was to be tested as a size
  multiplier, but 3B proved sizing-by-signal among already-good picks
  doesn't help. 4G v2 now CLOSED (marginal as filter, flat as sizer).
  Sector+A8 combos (4G, 4G v2) and A8 sizing (3B) all explored — none beat
  universe-wide A8 selection + existing Kelly sizer.

- ❌ 4C — A8-as-SELECTOR: REJECTED (shadow tested Fri May 29 2026).
  Test: research/test_4c_shadow_selector.py (A8 ranks full cross-section ->
  top-20 -> per-ticker model picks 5). Done OFFLINE via a8_oos_panel +
  accuracy.db outcomes (no live shadow needed after all).
  RESULT (h=5, 40 days Mar24-May22, pool=20 pick=5):
                 mean%   win    sharpe   maxDD
    production   0.025   0.829  4.78    -0.09
    4C_selector  0.040   0.750  3.19    -0.46
  4C had higher RAW return but LOWER win, LOWER Sharpe, 5x WORSE maxDD.
  Steamroller trap again. 27% pick-overlap w/ production = genuinely
  different (worse risk-adjusted) portfolio, so the test was meaningful.
  WHY: learning-to-rank cross-sectional selection (Poh-Roberts-Zohren 2021,
  ~3x Sharpe) is LONG-SHORT, large-universe, pre-cost. Long-only + 125-name
  thin breadth = the documented conditions where the edge vanishes
  (cf. arXiv 2302.10175 saw CS decile go negative). Confirmed with our data.
  ★ FIVE tests now agree A8 is at its ceiling as overlay/selector:
    4G (reject), 4G v2 (wash), 3B (flat), Phase 2H (mirage), 4C (fail).
    A8 works as the SELECTION signal it already is (top-decile model);
    stacking more A8-driven logic adds nothing risk-adjusted.
  Files: research/test_4c_shadow_selector.py.
  → See P6 for the preserved long-short revisit hypothesis.
- DSR — Deflated Sharpe Ratio overfit tool (penalizes Sharpe for # trials
  tested + sample + skew). Formalizes "real edge vs multiple-testing luck."
  ★ ABSORBED INTO P3.2 — see "P3 ALPHA PROGRAM" section below. DSR is the
    centerpiece of the gating pipeline; P3.2 adds purged-CV and PBO/CSCV.
- Pipeline C Stage 0 sentiment TIMEOUT — bump 15min timeout for 157-ticker
  universe (predictions ok but sentiment may be partial).

NOTE: F2 sector MAP still valuable (fixes sector_rel_ret as a FEATURE).
4G/v2 were about RANKING — different use. F2 stands.
### ✅ Universe expansion +25/+5 (DONE May 28, commit d73d7b5)
- tickers.txt 125->150: large/liquid (KLAC TXN MCHP NXPI DELL GLW FLEX JBL
  GFS NEE D CHTR BIO), mid-cap (HOOD RDDT PINS MDB CYBR CRWV NBIS RKLB STLA),
  speculative (CRCL FIG AMC)
- tickers_watchlist.txt 2->7: ALT SANA SENS VXRT RC (microcaps)
- Excluded: futures, crypto, foreign-listed, warrants, ETF dupes
- STLA + NBIS added to revenue SKIP_TICKERS (foreign filers)
- All 30 verified >=200d Polygon history

### ✅ Data backfill for 30 new tickers (DONE May 28)
- Revenue (Polygon, 21/30 — foreign+IPO skips expected)
- Earnings/EPS (UW migration, 745 rows, 29/30 — VXRT no UW data)
- UW snapshot (short int/analyst/FTD/seasonality/dark pool, 30/30)
- 8-K items (26/30, 1158 filings/2324 items)
- Sentiment (FinBERT, 30/30)
- Insider Form 4 (in progress)
- Institutional: SKIPPED (dead features until Q4 2026)

### ✅ Earnings source: yfinance -> UW (DONE May 28, commit b5ff0ad)
- UW /api/stock/{ticker}/earnings: 30yr history + pre-computed surprise
  vs yfinance ~4 quarters. New backfill script + --tickers flags on
  daily_uw_snapshot.py and etl_insider.py (reusable backfill tooling).

### ✅ F2 — Sector ETF features (DONE May 28, commit b5ff0ad)
- Rebuilt SECTOR_ETF_MAP for full 157-ticker coverage (was ~30, rest
  wrongly defaulted to XLK). GOOG/META->XLC, HOOD/CRCL->XLF, D/NEE/OKLO->XLU,
  EQIX/DLR/RC->XLRE, etc.
- Added XLC/XLRE/XLB 5d returns (all 11 SPDR sectors now). FEATURE_COLUMNS
  97->100. Fixed silent-error anti-pattern with logging (Rule #1b).

### ✅ Options skew silent failure (FIXED May 28, commit b5f8711)
- Wrong dict keys (put_iv_25d vs iv_25d_put) → KeyError caught by broad
  except → skew failed silently for ALL tickers universe-wide in every
  snapshot run. options_skew_history was empty. Fixed keys + surfaced errors.
- VERIFIED: NVDA/KLAC/HOOD skew now writes. Live tonight (Pipeline C).

---

## RECENTLY CLOSED (May 26-27 2026)

### ✅ C1 — Calibration broken via multipliers (DONE May 26-27)
- May 26: Deployed `ML_QUANT_DISABLE_MULTIPLIERS=1` env var (commit 2b264e0)
- May 27: Found `.env` regression where cron didn't load flag (commit 48a97ea)
- VERIFIED: prob_eff == prob_raw on tonight's Pipeline C output
- prob_raw column added to dashboard (commit 3fd71c6)

### ✅ R1 — Builder warns about extras (DONE May 26, commit 1a995d6)
- df.attrs['feature_cols'] + ['output_only_cols'] suppress 375 warnings/run

### ✅ Per-ticker AUC 0.44 panic (RESOLVED May 27, commit 70f219b)
- Was: window artifact (3 months, 3 regimes)
- Real per-ticker AUC: 0.51-0.53 multi-year
- Documented in docs/three_auc_reconciliation_may27.md

### ✅ Neutralizer audit (DONE May 27, commit f75f1ab)
- Backtested long-only modes
- Found sector mode adds 3-15pp on small window (regime-inflated)
- NOT wired to production. Available as dashboard column instead.

### ✅ Dashboard REC % column (DONE May 27, commit c361e84)
- Conviction-weight position sizing recommendation
- Tooltip explains interpretation ranges
- INFORMATIONAL ONLY

### ✅ REC % A/B framework (DONE May 27, commits 3537781 + 8fe8a73)
- portfolio_returns_ab table (89 rows backfilled)
- Pipeline B Stage 5 auto-populates nightly
- **DECISION DATE: Wed Jun 24 2026**
- Honest finding so far: conviction ≈ equal-weight (<1pp diff)

### ✅ Dead inst features diagnosis (DONE May 27, commit 11b4367)
- Root cause: temporal coverage (10% of training rows)
- Not a bug — needs more data history
- Revisit Q4 2026+ when coverage reaches 30-50%

### ✅ Native NaN + Missing Indicators flags (DONE May 27, commits dcf3e23 + 04427c0)
- Two flags built and tested per-ticker + ranker
- Neither helps. native_nan HURTS ranker (-0.28pp).
- Flags KEPT in code, default OFF for option value

### ❌ S1 / Phase 2A — A8 prob as per-ticker feature (FAILED May 27, commit 5104580)
- Built complete infrastructure: A8 training script (commit 920d03b),
  OOS panel generator (62278ee), 93K-row panel data (7c5b801),
  builder integration (7e65888)
- Smoke test: a8_prob importance = 0 in all 5 test tickers
- Root cause: redundant with existing features (rho +0.79 vol_10d, +0.69 vol_5d)
- A8's alpha exists at CROSS-SECTIONAL level, not per-ticker level
- See docs/phase_2A_smoke_test_findings_may27.md
- **A8 OOS panel data REMAINS VALID for Phase 2H/3B/4C**

---

## P0 — CRITICAL FOUNDATION (blocker, address FIRST)

### C1 — Calibration broken via multipliers ✅ DONE
**Status:** DEPLOYED May 26 + .env fix May 27. See RECENTLY CLOSED section above.
**Original Status:** Confirmed May 25 evening. Multiplier system destroys calibration.
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

### S2 — Phase 2 H (PCT7 ranker) promote — DATE CORRECTED to Tue Jun 2
Currently shadow mode. Shadow logging VERIFIED working (537 preds carry
prob_pct7 as of May 29). BUT 0 outcomes joined: PCT7 is h=5, first preds
were May 25, so 5-BD-forward returns don't close until ~Jun 1-2. The
"Fri May 29" trigger was optimistic — monitor_pct7_ab.py itself says wait.
**Trigger (corrected):** Tue Jun 2, re-run monitor_pct7_ab.py, then decide.

### S3 — Phase epsilon monitoring (Fri May 29)
First 127 outcomes arrive Fri. Run monitor_pct7_ab.py.
**Trigger:** Fri May 29 EOD.

### N1 — Verify Pipeline picks up features ✅ VERIFIED (May 29)
Live models (RC global + per-ticker AAPL/VXRT, trained May 29 06:05) all
carry 100 feature_cols = 96 base + 4 inst, 0 panel transforms. So: inst
features ARE live in production, the retrain loaded .env correctly (inst
flag took effect), and the PANEL_TRANSFORMS flag-OFF default held (no leak).
The "95" was a stale pre-inst expectation; correct live count is 100.
Env-fragility (FEATURE_COLUMNS built at import from os.environ) is
theoretical-only in practice — live path loads dotenv before classifier import.

### N2 — SKIP_TICKERS for training ❌ CLOSED-INVALID (May 29, premise wrong)
The premise (Pipeline A skips these ETFs from training/prediction) was a
MISCONCEPTION. By DESIGN the system trains + predicts GLD/QQQ/SLV: they're
in tickers.txt, training iterates the full universe, and generator.py:628
_ETFS only skips the EQUITY MULTIPLIERS (options flow / short interest) that
ETFs don't have — the 'raise Exception(ETF skip)' jumps to an except that
leaves multipliers neutral, NOT a prediction skip. generator.py:47-51
documents this as a deliberate choice ('let GLD h=1 through rather than
sacrifice winner models'). The only real SKIP_TICKERS (data/etl_polygon_
revenue.py) correctly skips ETFs from REVENUE backfill (ETFs have no revenue).
XLB/XLC/XLRE have 0 models simply because they're not in tickers.txt.
NO CODE CHANGE. (Open future question, NOT a bug: are ETF base-model
predictions good enough to keep? Would need its own OOS analysis if revisited.)

### R2 — prob_raw column patch ✅ SUPERSEDED (May 28)
Original intent: show calibrated prob_raw next to prob_eff so the
distorted multiplier value wasn't the only thing visible.
NOW MOOT: (1) ML_QUANT_DISABLE_MULTIPLIERS=1 → prob_eff == prob_raw
(verified diff=0.0 on live predictions). (2) Phase 2H BLEND/A8 columns
already surface prob_raw. A literal column would duplicate existing data.

---

## P2 — NEAR-TERM (this/next week)

### S1 — Phase 2 A: A8 prob as feature in main model ❌ FAILED (May 27)
See RECENTLY CLOSED section. Redundant with existing features. A8 OOS panel remains
useful for Phase 2H/3B/4C — these are the next priorities.

### S1 — original entry
Spec exists. ~2-3 days implementation.
Walk-forward A8 OOS prediction generator + Pipeline C orchestration.

### A2/A3/A4 — Path A A/B observability layers
Layer 1 bootstrap check, Layer 2 daily summary, Layer 3 health script.
**Decision date:** June 23, 2026.

### S4 — Inst suppression rule backtest ❌ CLOSED-SKIP (decided)
DECISION MADE: permanent skip. Full rationale in the "CONTEXT/REASONS FOR
SKIP" block below (inst_signed_flow_5d has ZERO LGB importance in 100% of
30 tickers; per-ticker models can't extract sparse CS features; thesis was
untested speculation). Reconsider only as a NEW task with an explicit
"audit if signal exists" precondition. No backtest will be run.

### R1 — Refactor build_feature_dataframe (separate features from diagnostics)
⏸ DEFER — CONDITION-TRIGGERED, re-check at Sep 15 review. Revisit R1 ONLY
when a trigger fires: (a) a real train/serve feature mismatch occurs, (b) a
new feature-fetch path doesn't fit the training_mode seam, or (c) the function
grows past ~800 lines. No forced date — a calendar deadline would manufacture
risk for a no-payoff refactor. Re-evaluate these conditions at the Sep 15
inst-features review; if none fired, confirm continued defer.
build_feature_dataframe is ~650 lines and the
single most depended-on function (every model, generator, alpha gate,
dashboard). The key separation (PIT-honest training vs live serving) ALREADY
exists via the training_mode seam (gates live calls in ~6 places, verified
working). The motivating concern (FEATURE_COLUMNS env-fragility) is
theoretical-only: N1 confirmed live models train with the correct 100 features
and .env loads before classifier import. So this refactor = high regression
risk (a silent feature-value change corrupts every prediction) for marginal
benefit + no forcing reason. If revisited: STRANGLER pattern — extract ONE
concern at a time with before/after bit-equality tests on real ticker data,
never a big-bang rewrite. Rule #1(a): don't churn working critical code.

### F1 — Drop dead features
⏸ DEFER → REVISIT Mon Sep 15, 2026 (fold into inst-features 6-mo
reconsideration; PRECONDITION: proper per-feature audit — XGB AND LGB
importance, train-vs-live behavior, A/B status — before removing anything).
feature_importance_history shows avg_imp=0.0 for
~15 features, but this is NOT a reliable kill-list:
  - pc_ratio_snap reads 0.0 yet showed gate IC +0.048 (t=8.9) this session
    -> the importance logging is suspect, not the feature.
  - inst_* are in an ACTIVE A/B until Sept 15 (dropping = kill experiment).
  - Live-API features (pc_ratio, iv_skew, analyst_*) are PIT-gated to 0 in
    training_mode (constant -> 0 splits) but useful live -> dropping breaks
    live serving.
  - Counts vary (n=1998..8277) = logged over inconsistent model sets; avg
    not comparable; exact 0.0 suggests LGB-specific 0-splits, not unused.
No safe, forcing-reason removal available. Revisit only with a proper
per-feature audit (XGB AND LGB importance, train vs live behavior, A/B status)
before removing anything from the canonical FEATURE_COLUMNS set.

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
★ ABSORBED INTO P3.3 — see "P3 ALPHA PROGRAM" section. F3 is the sentiment
slice of P3.3's textual-signals bucket (Lazy Prices, MD&A delta, call tone).

### F4 — pc_ratio_change_5d + iv_skew_change features
8-12 hours. Expected +0.003-0.010 AUC.
★ ABSORBED INTO P3.3 — see "P3 ALPHA PROGRAM" section. F4 is 2 features
out of ~15 in P3.3's options-surface bucket (IV term slope, butterfly,
vol-of-vol, GEX single-name, VRP, etc.).

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

### ★ Long-Short revisit of A8-as-selector (CONDITIONAL — gated on short strategy greenlit)
4C failed long-only (see active list). BUT the learning-to-rank research is
clear that cross-sectional selection's natural habitat is LONG-SHORT — the
loser leg carries ~half the documented alpha that long-only discards.
HYPOTHESIS: A8 ranking-as-selector may add value in a future long-short book.
- Infrastructure already partly there: A8 produces a full-cross-section
  ranking (a8_prob for ~125 names); today only the TOP is used. The BOTTOM
  is what a short leg would screen.
- BLOCKER 1: A8 predicts TOP-decile membership. Low a8_prob = "not a likely
  winner", NOT "likely to fall". A short leg needs a SEPARATE bottom-decile /
  down-move model. 4C selector CODE is reusable; the A8 MODEL is not.
- BLOCKER 2: shorting = major strategy change (borrow cost/availability,
  unlimited downside, margin, squeeze risk [cf. RZLV 25% SI], wash-sale/tax,
  Reg SHO). "Does A8 ranking help" is a tiny piece of "should I short at all".
- GATE TO START: a short strategy must be explicitly greenlit AND a
  bottom-decile model built. Until then, parked.
- Reusable asset: research/test_4c_shadow_selector.py (structured to extend
  to long-short — add a short sleeve from the bottom of the a8 ranking).

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
| Mon Sep 15 | Inst features 6-mo threshold + F1 dead-feature pruning (gated on per-feature audit) + R1 refactor trigger-condition re-check |
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


---

## UPDATE: May 26 2026 — Phase 1 calibration fix + insider_60d shipped

### Phase 1 calibration fix — SHIPPED (commit 2b264e0)

env var `ML_QUANT_DISABLE_MULTIPLIERS=1` opts in to skip the multiplier
compound. Default behavior unchanged.

A/B test results (4 tickers):
  BYND: prob_eff 0.824 -> 0.681  (BUY stays BUY, no longer >=0.80)
  MSFT: 0.606 -> 0.561  (HOLD -> HOLD, no change)
  QQQ:  0.663 -> 0.631  (BUY -> HOLD, correctly downgraded)
  BA:   0.666 -> 0.648  (BUY -> HOLD, correctly downgraded)

When env var active: prob_eff = prob_raw, no inflation.

### Insider feature bug — FIXED (commit f13ecf7)

Discovered via BYND audit: model was BLIND to bearish insider selling because
insider_7d/insider_21d windows were too short to capture quarterly activity.

Real activity invisible:
  BYND: CFO sold 419K shares ($253K) April 13 — was 42 days ago, INVISIBLE
  NVDA: -3.2M net insider sells over 60 days — INVISIBLE
  AAPL: -694K net sells over 60d — 7d window saw 0

Fix:
  features/builder.py: added insider_60d, insider_90d rolling sums
  models/classifier.py: added to FEATURE_COLUMNS (95 -> 97 with inst flag)
  scripts/pipeline_A_ingest.sh: ETL window 7d -> 60d
  60-day backfill: 349 rows persisted (5.6x more than 7d)

### Pipeline B retrain IN PROGRESS

Launched 14:11 VN time with both env vars set. ~3 hours.
Will produce 97-feature per-ticker models that USE insider_60d/90d.
Daily_runner will generate signals with multipliers OFF.

### Status of P0 (calibration foundation)

  Phase 1 (env var to disable multipliers):     ✓ SHIPPED
  Phase 2 (Platt scaling on PCT7):              PENDING — next after retrain
  Phase 3 (Beta calibration if needed):         PENDING
  Phase 4 (Box-constrained recalibration):      PENDING
  Phase 5 (Venn-Abers for non-stationary):      DEFERRED


---

## UPDATE: May 26 2026 afternoon — Phase 2 calibration plan revised

### Phase 2 deferred until Friday May 29 (post Phase 1 outcomes)

KEY ARCHITECTURAL FINDING:
  PCT7 model (LGBOnlyResult) has NO post-hoc isotonic calibration.
  Per-ticker EnsembleResult (XGB+LGB) DOES have isotonic via
  CalibratedClassifierCV cv=5.

  So 'replace isotonic with Platt' only applies to per-ticker models,
  which is huge blast radius (125 models).

PHASE 2 DECISION: Wait for Phase 1 outcomes before deciding.
  Phase 1 (env var disable multipliers) is now generating predictions.
  In 5 business days (Friday May 29), we'll have outcomes for h=5
  predictions and can run calibration audit on new data.

  If calibration GOOD after Phase 1: Phase 2 not needed.
  If calibration STILL broken: proceed to:
    - Option B: Add Platt scaling to PCT7 (2-3 hr, low risk)
    - Option D: Bucket-recalibration based on observed hit rates (4-6 hr)

  Right discipline: VERIFY Phase 1 worked before adding Phase 2.

POST-RETRAIN TASKS (queued):
  1. Retrain GLOBAL models (3 horizons) with 97 features
  2. Run verify_post_retrain_nvda.sql
  3. Compare May 26 BUYs vs May 25 (impact of multipliers OFF + insider features)

NEW DATA QUALITY FINDINGS (audit during retrain wait):
  - BYND/penny stocks have ZERO institutional rows (penny stocks excluded)
  - All 2.1M institutional rows are dark pool (no lit data)
  - inst_block_count_7d shows corr(actual_return)=0.156 (weak but non-zero)
  - inst_block_notional_7d shows corr(actual_return)=-0.123 (counterintuitive)
  - inst_sweep_count_7d ALWAYS 0 (ingestion broken or sweep filter wrong)
  - earnings_monitor.db (29MB) NEVER REFERENCED by builder
  - institutional_holdings (10K rows) NEVER USED in model

ARCHITECTURE NOTE:
  GLOBAL models still 90 features (trained May 24). Per-ticker now 97
  features (retrained May 26 PM). Path A A/B comparison invalidated
  until GLOBAL retrained. Plan: retrain GLOBAL after Pipeline B done.


---

## UPDATE: May 26 PM — DEAD FEATURES AUDIT

Audited LGB feature importance across 30 per-ticker models (h=5).

CRITICAL FINDING: Only 1 of 97 features (lqd_hyg_spread) is alive in >80%
of tickers. 57 features are zero in >75% of models.

DEAD-EVERYWHERE FEATURES (100% zero across 30 tickers):
  - sentiment_score (FinBERT investment was wasted??)
  - All 4 inst features (inst_signed_flow_5d/30d, inst_block_buy_sell_7d, inst_auction_imbal_5d)
  - All FinBERT features (finbert_sentiment, finbert_sentiment_earnings)
  - All earnings-surprise features (eps_surprise, rev_surprise)
  - All post-earnings drift features (post_earnings_1d/3d/5d)
  - is_squeeze_setup (the interaction we shipped yesterday)
  - All rev_growth features (rev_growth_yoy, qoq)
  - All risk multiplier inputs (risk_today, risk_next_*)
  - All short features (short_ratio, short_pct_float)
  - Many calendar/regime features (monday_sentiment, day_of_week, is_month_end, vix_5d_above_25)

INSIGHT: AAPL's insider_60d was top-5 because AAPL had -694K activity.
Other tickers with 0 insider activity got 0 importance. Per-ticker models
can't learn from a sparse signal. 77% of tickers had zero insider_60d in
last 60 days.

IMPLICATIONS:
  1. Per-ticker model effectively uses ~20 features per ticker, not 97
  2. GLOBAL model (cross-sectional pooled) should benefit more from these
  3. PCT7 trains pooled — similar advantage
  4. Adding features to per-ticker is wasted effort if they're sparse globally

PROPOSED ACTIONS (separately):
  A. Drop confirmed-dead features from FEATURE_COLUMNS to reduce noise
     - Estimated 30-50 features could go without loss
     - Keep: bb_lower, igv_vs_sp500_ret_30d, xlv_ret_5d, lqd_hyg_spread,
       insider_60d, rsi_14, atr, return_5d, vol_surge_eod, eightk_other_events_30d,
       ma_5/20/50, vwap, days_to_earnings, vix_close, yield_10y, macd, bb_width,
       obv, xlk_ret_5d, premarket_gap
  B. Train GLOBAL model with all 97 features (cross-sectional advantage)
  C. Use PCT7 as cross-sectional anchor for the new sparse features

CAVEAT: This was only 30 of 125 tickers. Need full audit before action.
Some "dead" features may be alive for specific tickers (penny stocks,
small caps with different dynamics).


---

## DECISION: S4 — Inst suppression rule backtest = SKIPPED

Status: FORMAL SKIP (May 26 PM)

CONTEXT: Pending TODO since May 22 was to backtest a post-earnings + negative-inst-flow
suppression rule using inst_signed_flow_5d + earnings_calendar.

REASONS FOR SKIP:
  1. Dead-features audit (just completed) shows inst_signed_flow_5d has
     ZERO LGB importance in 100% of 30 tickers checked.
  2. Per-ticker models can't extract signal from sparse cross-sectional features.
  3. Many tickers (BYND, penny stocks, small caps) have NO inst data at all.
  4. The thesis was untested speculation. Without already-validated signal,
     no reason to engineer a suppression rule on top.

WHEN TO RECONSIDER:
  - When GLOBAL model is retrained with full 97 features (this week, post-Pipeline-B)
  - When inst data depth reaches 6 months (Sept 2026)
  - If Phase 2 A (A8 prob as feature) shows cross-sectional signal works

This SKIP closes the May 22 TODO item permanently. If reconsidered, re-open as
NEW task with explicit "audit if signal exists" precondition.


---

## UPDATE: May 26 PM — Path A Layer 3 reveals GLOBAL is broken

scripts/check_global_ab_health.py output for 11-day window:

  | Metric | h=1 | h=3 | h=5 |
  |--------|-----|-----|-----|
  | per-ticker std | 0.100 | 0.132 | 0.144 |
  | GLOBAL std | 0.023 | 0.017 | 0.025 |
  | Correlation | 0.005 | -0.098 | -0.071 |
  | Both BUY | 0 | 0 | 0 |

THREE PROBLEMS:

  1. GLOBAL has 5-7x LESS variance than per-ticker
     std 0.017-0.025 means GLOBAL barely deviates from 0.5 prior.
     It's essentially predicting "no information."

  2. GLOBAL has NEGATIVE correlation with per-ticker at h=3/h=5
     Should be ~0 (random) or positive (same signals).
     Slight negative suggests learned patterns conflict.

  3. GLOBAL hits BUY threshold 0% of the time
     Per-ticker hits BUY 57-103 per horizon. GLOBAL: 0.
     A/B comparison invalid until GLOBAL competitive.

INTERPRETATION:
  GLOBAL model is fundamentally weak. Not just feature-stale (90 vs 97).
  GLOBAL probably has data scarcity / collinearity / calibration issues.

ACTIONS:
  1. Retrain GLOBAL after Pipeline B with 97 features (queued)
  2. After retrain: re-run check_global_ab_health.py
  3. If GLOBAL std still <0.05: investigate data scarcity / training script
  4. June 23 Path A decision date may need extension if GLOBAL is broken

KEY DECISION CRITERIA at June 23 STILL VALID:
  - If GLOBAL hit rate >= per-ticker + 2pp: promote GLOBAL
  - If GLOBAL never reaches BUY: keep per-ticker, kill A/B
  - Current trajectory: kill A/B unless retrain dramatically improves


---

## UPDATE: May 26 PM — GLOBAL root cause + research-based revised plan (E then C)

DIAGNOSTIC:
  RAW LightGBM predictions BEFORE isotonic calibration:
    NVDA, AAPL, MSFT, TSLA all give EXACTLY 0.513 (same to 4 decimals!)
    BYND gives 0.470 (slightly different — penny stock features differ)

  GLOBAL has learned ONE prediction for most large-cap stocks.
  Pooled classification with shared features makes most stocks look
  identical to the trees.

ROOT CAUSE:
  Training pooled on heterogeneous tickers WITHOUT a discrimination signal.
  Trees can't split on features that don't vary cross-sectionally between
  large-cap tech (NVDA/AAPL/MSFT/TSLA are too similar in our feature space).

RESEARCH (industry standard for cross-sectional quant):
  - Hedge funds train RANKING models, not classifiers
    (predict relative position not absolute up/down)
  - LightGBM has LambdaRank/LambdaMART for this
  - Cliff Asness Global Alpha: locate underpriced equities via factor ranking
  - Yanfu: index-enhanced products via cross-sectional factor models

REVISED PLAN (Option E then Option C):

  STEP E (tonight after Pipeline B, ~30 min):
    Retrain GLOBAL with 97 features (insider_60d/90d + interactions)
    Hypothesis: insider features differentiate tickers (BYND -898K vs
    NVDA -3.2M). If model picks them up, variance increases. If not,
    confirms the diagnosis.

  STEP C (this week, ~4-6 hours):
    Refactor GLOBAL as RANKING model via LightGBMRanker(objective='lambdarank')
    - Group rows by prediction_date (cross-section per day)
    - Target: rank stocks within each day by future return
    - Output: percentile 0-1 within day (relative outperformance prob)
    - Calibrate output to interpretable scale
    - A/B test against per-ticker as before

  BACKUP (Option F if C is too much work):
    Use GLOBAL as Phase 2 H-style meta-label filter
    - GLOBAL serves as confidence check, not primary signal
    - Shadow mode for 2 weeks before promoting

  KILL (if everything fails):
    Accept per-ticker is right architecture. Remove Path A A/B logging.

WHY OPTIONS NOT CHOSEN:
  Option A (ticker_id): defeats cross-sectional purpose
  Option B (deeper trees): doesn't address feature non-discrimination
  Option D (kill premature): worth trying ranking first


---

## UPDATE: May 26 PM — GLOBAL root cause + research-based revised plan (E then C)

DIAGNOSTIC:
  RAW LightGBM predictions BEFORE isotonic calibration:
    NVDA, AAPL, MSFT, TSLA all give EXACTLY 0.513 (same to 4 decimals!)
    BYND gives 0.470 (slightly different — penny stock features differ)

  GLOBAL has learned ONE prediction for most large-cap stocks.
  Pooled classification with shared features makes most stocks look
  identical to the trees.

ROOT CAUSE:
  Training pooled on heterogeneous tickers WITHOUT a discrimination signal.
  Trees can't split on features that don't vary cross-sectionally between
  large-cap tech (NVDA/AAPL/MSFT/TSLA are too similar in our feature space).

RESEARCH (industry standard for cross-sectional quant):
  - Hedge funds train RANKING models, not classifiers
    (predict relative position not absolute up/down)
  - LightGBM has LambdaRank/LambdaMART for this
  - Cliff Asness Global Alpha: locate underpriced equities via factor ranking
  - Yanfu: index-enhanced products via cross-sectional factor models

REVISED PLAN (Option E then Option C):

  STEP E (tonight after Pipeline B, ~30 min):
    Retrain GLOBAL with 97 features (insider_60d/90d + interactions)
    Hypothesis: insider features differentiate tickers (BYND -898K vs
    NVDA -3.2M). If model picks them up, variance increases. If not,
    confirms the diagnosis.

  STEP C (this week, ~4-6 hours):
    Refactor GLOBAL as RANKING model via LightGBMRanker(objective='lambdarank')
    - Group rows by prediction_date (cross-section per day)
    - Target: rank stocks within each day by future return
    - Output: percentile 0-1 within day (relative outperformance prob)
    - Calibrate output to interpretable scale
    - A/B test against per-ticker as before

  BACKUP (Option F if C is too much work):
    Use GLOBAL as Phase 2 H-style meta-label filter
    - GLOBAL serves as confidence check, not primary signal
    - Shadow mode for 2 weeks before promoting

  KILL (if everything fails):
    Accept per-ticker is right architecture. Remove Path A A/B logging.

WHY OPTIONS NOT CHOSEN:
  Option A (ticker_id): defeats cross-sectional purpose
  Option B (deeper trees): doesn't address feature non-discrimination
  Option D (kill premature): worth trying ranking first


---

## UPDATE: May 26 evening — comprehensive doc audit + missing critical items

Reviewed all 27 doc files. Found CRITICAL items not in MASTER_TODO.

### NEW P0 — Per-ticker AUC 0.44 finding (validation_findings_may23.md)

  From May 23 PIT walk-forward (production model on 10,995 outcomes):
    pooled_oos_auc = 0.4389  (BELOW 0.50 — worse than coin flip)
    pooled_oos_acc = 0.4832  (random = 50%)
  
  If true, per-ticker BUY signals are BELOW CHANCE. The model has negative
  edge. Today's BUYs aren't reliable.
  
  Tonight's bucket-level calibration audit showed prob_raw aggregate is
  PERFECTLY calibrated (64.0/64.0% h=3). But that's CONDITIONAL on signal
  being BUY. Aggregate calibration ≠ predictive edge.
  
  CRITICAL: investigate this contradiction. Either:
    (a) The May 23 PIT walk-forward had bug we haven't fixed
    (b) Per-ticker model is genuinely broken OOS
    (c) Different test windows show different results (regime dependency)

### NEW P0 — 3-AUC reconciliation (diagnostic_audit_design.md May 23 PRIORITY #1)

  Three OOS AUCs disagree by up to 14pp:
    Per-ticker PIT production WF: 0.44
    Per-ticker walk-forward stacks: 0.51-0.53
    Path A 5-fold validation: 0.58-0.59
  
  Cannot make any production decisions until we know which is honest.
  Was identified as PRIORITY #1 on May 23. Still not done.
  
  Tests required (3-5 hours):
    Test 1: Run PIT WF on GLOBAL (Path A) — does 0.58 hold or collapse?
    Test 2: Per-ticker WF with same test window as PIT
    Test 3: Identify leakage in per-ticker setup
    Test 4: Compare Path A 5-fold split vs PIT logic

### NEW P1 — A8 model best alpha not yet integrated (A8_interpretation_and_action_plan.md)

  A8 = "is ticker in top 10% by 5d forward return" target.
  OOS AUC = 0.677 (vs per-ticker 0.44, GLOBAL classifier 0.58, ranker top-quintile +1.56pp).
  
  A8 is by construction cross-sectional (top-decile membership can't be
  predicted by macro). Highest verified alpha in the entire codebase.
  
  Status:
    - Spec exists: phase_2A_a8_as_feature_spec.md
    - Model trained: A8_v2 in models/research/ (not deployed)
    - Path to production: A8 prob as feature in main model
    - Effort: 2-3 days
  
  Today's Option C ranker is RELATED but different (continuous rank vs
  top-decile classification). Both could coexist as signals.

### NEW P2 — ROADMAP_HYBRID_ADVISOR.md (May 8 4-week vision)

  Sprint-level breakdown of dashboard UX, continuous sizing, tooltips,
  manual trade log. Status: design doc, not implementation. Target was
  May 8 - June 5, 2026 (4 weeks). 
  
  Components:
    - Manual trade log (closed loop on Atom's actual trades)
    - Continuous position sizing (replace HIGH/MED/LOW tiers)
    - Tooltips explaining each metric (dyslexia-friendly UX)
    - Bloomberg/WorldQuant-style hybrid dashboard
  
  Most BLOCKED by P0 (need to know if per-ticker has edge before sizing).

### NEW P2 — 8K gating backtest result (8k_gating_backtest_result.md)

  Status: result exists but action unclear. Need to review.

### NEW P3 — Sprint_W1 cost model + survivorship audit

  - Survivorship: CLEAN (verified May 17, no fix needed)
  - Cost model: SHIPPED to analysis/fitness_scorer.py (may be uncommitted)
  - h=3/h=5 confidence cap retracted (cap is 0.95, dormant)
  
  Verify cost_model is committed (search git log for "cost_model").

### NEW P3 — Session E phase 4 LLM extraction (FUTURE)

  Earnings call transcript LLM extraction. Far future, deferred behind:
    - Validating current FinBERT signal works
    - Per-ticker AUC issue resolved

### Doc redundancy / cleanup

Some docs are likely stale or superseded:
  - audit_findings_may24_morning.md (superseded by evening)
  - feature_improvement_plan.md (older, partially executed)
  - decision_per_ticker_interactions_over_cross_sectional_service.md
    (Option C ranker IS the cross-sectional service we deferred to Q3 — 
     decision may be reversible if ranker proves itself)

### REVISED PRIORITIES

**P0 (foundation, blockers):**
  - C3 (NEW): 3-AUC reconciliation
  - C4 (NEW): Per-ticker AUC 0.44 investigation

**P1 (this week, requires P0):**
  - S1 evolved: A8 + Ranker integration via Phase 2A
  - Phase 2H promotion to active (Fri May 29)
  - Phase ε monitoring (Fri May 29 → Tue Jun 2)

**P2 (deferred until P0 resolved):**
  - ROADMAP_HYBRID_ADVISOR 4-week dashboard
  - Drop dead features (F1)
  - Most 4-week roadmap items


---

# ═══════════════════════════════════════════════════════════════════════════
# SESSION END — May 26 2026 VN (22:30, after 15+ hours)
# ═══════════════════════════════════════════════════════════════════════════

## ✅ COMPLETED TODAY (15 commits)

### Bug fixes shipped
- ✅ **C1 — Multiplier inflation** (commit 2b264e0) — env var ML_QUANT_DISABLE_MULTIPLIERS=1
  - Verified: BYND went from BUY 0.824 (inflated) → HOLD 0.553 (honest)
  - prob_eff = prob_raw across all predictions
- ✅ **Insider blindness 60d/90d** (commit f13ecf7) — NVDA -3.2M sells now visible
- ✅ **GLOBAL classifier broken** (commit 17232e9 → 11254f0) — replaced with LightGBMRanker
  - OOS validated: Q5 vs Q1 spread +1.56pp / 5d (commit e3790ad)
  - Integrated into daily_runner shadow logging (commit ce4f0db)
- ✅ **375 extras_columns warnings → 0** (commit 1a995d6) — R1 refactor via df.attrs
- ✅ **Dashboard prob_raw column** (commit 3fd71c6) — R2 done

### Documentation shipped
- ✅ **MASTER_TODO comprehensive audit** (commit 01b8e53) — found P0 missing items
- ✅ **Calibration ECE tracker** (commit 4e4649d) — scripts/calibration_metric_tracker.py
- ✅ **Dead features audit** (commit 83359ba) — 57/97 features dead in >75% of tickers
- ✅ **S4 inst suppression formally skipped** (commit 1557f5c)
- ✅ **Phase 2 calibration deferred** (commit df7b36b) — wait for Fri May 29 outcomes
- ✅ **GLOBAL revised plan** (commit 572deec) — Option E (failed) → Option C (success)
- ✅ **MASTER_TODO Phase 1 status** (commit a46d8f9)

### Verification completed
- ✅ **N1 — Pipeline B picks up 97 features** (verified, was 95, +insider_60d/90d)
- ✅ **N2 — SKIP_TICKERS respected** (verified)

---

## 🔴 LEFT ON TODO LIST — by priority

### P0 — CRITICAL (still unresolved)

#### C3 — Per-ticker AUC 0.44 investigation  ✅ DONE (P0-2, May 30)
- **Problem:** Per-ticker model OOS AUC = 0.4389 (May 23 PIT WF, 10,995 outcomes)
- **Below 0.50** — worse than coin flip
- **Contradiction:** aggregate prob_raw is calibrated 64.0/64.0% h=3
- **Effort:** 3-5 hours
- **Trigger:** Wednesday May 27 or sooner if conviction allows

#### C4 — 3-AUC reconciliation  ✅ DONE (P0-2 + P0-3, May 30)
- Per-ticker PIT: 0.44
- Per-ticker WF stacks: 0.51-0.53
- Path A 5-fold: 0.58-0.59
- **Tests required:** PIT WF on GLOBAL, same-window per-ticker WF, leakage hunt
- **Effort:** 3-5 hours
- **Blocks:** all production decisions (P1 below)

### P1 — IN FLIGHT (this week, scheduled)

#### S2 — Phase 2 H promote to active (Fri May 29 EOD)
- Status: shadow mode, 5 BD outcomes due Friday
- Action: run monitor_pct7_ab.py, decide if PCT7 filter ready

#### S3 — Phase ε monitoring (Fri May 29 EOD)
- First 127 outcomes from Tue May 25 BUYs arrive Friday
- Action: validate prob_pct7 vs actual hit rate

#### C2 — Calibration verification post-fix (Tue Jun 2)
- First h=5 outcomes for today's predictions
- Re-run calibration audit with multipliers OFF + insider features
- Compare ECE before/after

#### Phase 2 A — A8 prob as feature (NEW from doc audit)
- A8 has OOS AUC 0.677 (best signal in codebase)
- Model trained (`models/research/A8_v2_*`), spec exists
- Effort: 2-3 days
- Target: next week

#### Ranker A/B decision (Fri Jun 5, one week of data)
- Compare per-ticker BUYs alone vs ranker-confirmed BUYs
- Decision: filter (Phase 2 H style), promote, or kill

### P2 — NEAR-TERM (this/next week)

#### F1 — Drop dead features (skipped tonight, high-risk)
- 57/97 features dead in >75% of tickers (audit complete)
- Risk: features dead in 30 sampled tickers may be alive in others
- Effort: 1-2 hours with proper cross-validation
- Defer until: P0 resolved (then we'll trust the AUC math)

#### A2/A3/A4 — Path A A/B observability layers
- Layer 1 bootstrap check (done — verified models load)
- Layer 2 daily summary (in code, needs verification)
- Layer 3 health script (in code)
- Decision: June 23

### P3 — DEFERRED (BLOCKED by P0)

#### ROADMAP_HYBRID_ADVISOR.md 4-week vision (May 8)
- Sprint W1: position sizing by prob_eff bucket
- Sprint W2: portfolio-level risk, volatility-scaled sizing
- Sprint W3: alpha research, A8 deployment
- Sprint W4: operational maturity, manual trade log
- **All blocked** until C3/C4 resolved — can't size positions on negative-edge signals

### P4 — FUTURE PHASES (4-8 weeks)

- Sector ETF features expansion (high ROI per doc)
- FinBERT historical backfill
- pc_ratio_change_5d + iv_skew_change features
- Cross-ticker 8-K alpha exploitation
- Option G cross-sectional service (Q3 2026)
- Inst features reconsideration (Sept 15)

### P5 — CLEANUP / COSMETIC

- Audit ALTER TABLE migrations for missing columns
- Cleanup apply_*.py scripts in repo root
- ✅ Suppress extras_columns warning (DONE today via R1)

### P6 — FAR FUTURE

- Session E phase 4: LLM extraction from earnings transcripts
- Q3+ alpha research

---

## 📅 SCHEDULED CHECKPOINTS

| Date | Event |
|---|---|
| **Wed May 27** | Verify last night's automated Pipeline B/C succeeded with ranker logging |
| **Fri May 29 EOD** | Phase 2 H promote decision + Phase ε first outcomes |
| **Tue Jun 2** | First h=5 outcomes for May 26 predictions |
| **Fri Jun 5** | One-week ranker A/B decision |
| **Mon Jun 8** | First A8 deployment (if not blocked by P0) |
| **Mon Jun 23** | Path A A/B final decision |

---

## 🎯 IMMEDIATE NEXT STEPS (Wed May 27 morning)

1. Verify Pipeline C completed last night
2. Confirm `prob_up_global_ranker` populated in today's DB rows
3. Decide priority: P0 investigation OR continue building (Phase 2A A8)
4. (Optional, low risk) Run feature_validator.py to check 97-feature consistency

---

## 🧠 ENGINEERING DEBT acquired today

- We added `ML_QUANT_DISABLE_MULTIPLIERS` env var (Rule #1(a): flags=debt)
  - Plan to audit/remove: after Friday's outcomes prove Phase 1 is correct
- We added `df.attrs['feature_cols']` metadata — fragile in pandas
  - Tested: copy/iloc/fillna/reset_index/column-add all preserve
  - Not tested: groupby/merge (not used in our path)
  - Monitor: if any pipeline starts emitting noise again, attrs were stripped

---

## 📊 SESSION METRICS

- **Commits:** 15
- **Files changed:** ~12 (builder, ensemble, generator, sink, daily_runner, classifier, dashboard, wrappers, multiple docs)
- **New files created:** 6 (train_global_ranker.py, calibration_metric_tracker.py, post_pipeline_B_retrain_global.sh, verify_post_retrain_nvda.sql, option_c_oos_validation.md, session_may26_learnings.md)
- **Models trained:** 130 (126 per-ticker + 1 GLOBAL classifier × 3 horizons + 3 GLOBAL rankers)
- **Tests run:** Pipeline B full retrain, Pipeline C in flight, 3-ticker integration test
- **Bugs uncovered:** GLOBAL classifier non-discriminative, 57 dead features, per-ticker negative OOS AUC contradiction
- **Lines added to MASTER_TODO:** ~250

End of May 26 session checkpoint.


---

# ═══════════════════════════════════════════════════════════════════════════
# May 26 — Signal_System_Addendum.docx AUDIT (parallel Claude session)
# ═══════════════════════════════════════════════════════════════════════════

## Source
External doc from parallel "monitor session" (Claude + Atom), proposing
architectural changes to signal system. Audited against codebase + research
literature (Cohen/Malloy/Pomorski 2012 on insider trade informativeness).

## Verdict
~40% genuine insight, ~40% post-hoc narrative, ~20% factually wrong.
Cherry-pick the genuine items; reject the multiplier framework (contradicts
today's Phase 1 calibration fix).

## VERIFIED facts from doc
- portfolio/neutralizer.py EXISTS as orphan (7049 bytes, May 14) — not wired
- features/alpha_transformations.py EXISTS as orphan (10959 bytes, May 13) — not wired
- earnings_monitor.db has rich tables NEVER consumed by builder:
    darkpool_prints, form4_parsed, form4_transactions, institutional_holdings,
    insider_trades, edgar_filings, short_interest_snapshots
- No cohort feature in builder/classifier
- pc_ratio, put_call, iv_skew features DO exist in builder (doc's claim was wrong here)

## WRONG/RISKY in doc (reject)
- 0.967 training / 0.510 live AUC number — NOT from our codebase
  (our honest WF: h=1 0.511, h=3 0.528, h=5 0.540)
- 9-row "Confidence Multiplier Framework" — would re-break calibration we just fixed
- positioning_extremity_dampener — n=1 fit on NVDA May 20, not validated
- "prediction_features" terminology — not our schema

## NEW ITEMS TO ADD

### P1 NEW — Wire earnings_monitor.db into builder (HIGH ROI)

Currently the monitor DB has 8 tables of rich data we don't consume.
Most valuable: form4_parsed (insider trades classified), darkpool_prints,
institutional_holdings.

Tasks:
  - F1a: Wire form4_parsed → opportunistic vs routine flag (Cohen 2012)
  - F1b: Wire darkpool_prints → dp_7d_skew, dp_block_count features
  - F1c: Wire institutional_holdings → quarterly position deltas (handle stale data!)

Effort: 4-6 hours per source after schema discovery
Validation: walk-forward AUC delta with/without feature, ECE check
Blocker: Verify schema before each wire (Rule #1g — gap-check subsystem)

### P1 NEW — Opportunistic vs routine insider classification

Research basis: Cohen, Malloy, Pomorski (2012) — "Decoding Inside Information"
  - Routine trades: predictable monthly pattern, NOT informative
  - Opportunistic trades: 82bp/month abnormal returns (long-short portfolio)
  - Both opportunistic BUYS and SELLS predict future returns (asymmetric weights)
  - 10b5-1 sales = routine (preset plan); CEO discretionary buys = opportunistic

Implementation:
  - Flag insiders with detectable monthly trading pattern as "routine"
  - Keep insider_60d/90d net (current) AS WELL AS opportunistic_only variants
  - Separate features: insider_buy_opp_30d, insider_sell_opp_30d (do NOT net)
  - CEO/CFO/EVP discretionary flag (P-code, not 10b5-1) as binary

Effort: 2-3 days
Test cases: NVDA -3.2M (CFO/CEO?), BYND CFO sale, STX CFO+EVP $45.86M
Validation: separate AUC for opp vs routine; routine should approach 0.50

### P2 NEW — Cohort percentile rank feature

Current state: ranker (Option C, shipped today) already does cross-sectional
ranking. May be REDUNDANT with this proposal.

Decision needed BEFORE building:
  - Test if existing ranker captures cohort-rank signal
  - If yes: skip this feature (redundant)
  - If no: build as separate feature with sector-ETF grouping

Effort: 1 day (mostly validation, not new code)

### P2 NEW — Ticker collision filter for news

Doc's observation matches our reality. Common collisions:
  OPEN → "A-SHARE OPEN LOWER" (Chinese market)
  AI   → word "AI" 
  S    → generic letter
  CNC  → "CNC machining"
  ALK  → "ALK-positive" (cancer drug)
  UREN → confused with IREN

Fix: per-ticker disambiguation dict
  - Canonical company name
  - Allowed context terms (sector, products)
  - Blacklist phrases

Effort: 1-2 days build + manual curation
Validation: false-positive rate audit on news catalysts last 30 days

### P2 NEW — Triage orphan modules

portfolio/neutralizer.py and features/alpha_transformations.py exist
but are not imported anywhere in production path.

Decision per file:
  (a) Wire into builder/signals — validate works first
  (b) Formally archive to deprecated/ folder with git note
  (c) Delete if duplicates existing functionality

Effort: 2-4 hours per file
Rule #1: audit before flip — gap-check what they do vs existing code

### P3 NEW — 8-K body parsing (already in K2 backlog)

Doc proposes corporate_event_signature feature requiring 8-K body parsing.
Aligns with existing K2 (cross-ticker 8-K alpha exploitation).
Effort: 2-3 days for parser; longer for feature validation.

### REJECT explicitly

1. Confidence multiplier framework (Doc §3.2)
   Reason: contradicts Phase 1 fix (commit 2b264e0). Multipliers destroyed
   calibration. ECE was 12-21pp pre-fix. Adding 5 more multipliers (monitor
   confidence, positioning_extremity, insider_cap, news_credibility,
   cohort_strength) rebuilds the exact problem we removed.
   Alternative: any "signal" the doc proposes must be a model FEATURE, not
   a post-model multiplier. Let the model learn the weight via training.
   ECE must stay <5pp post-deployment.

2. positioning_extremity_dampener (Doc §2.4)
   Reason: n=1 fit on NVDA May 20. The "stock fell because marginal buyers
   exhausted" is one hypothesis. Could equally be sell-the-news, macro,
   profit-taking. Doc proposes "backtest 200 earnings prints" but that's
   not done — feature is hypothesis, not finding.
   Reconsider after: someone actually runs the 200-print backtest.

3. News credibility tiering (Doc §3.4)
   DEFER — requires source-extraction infrastructure we don't have.
   UW news endpoint returns 404s (per doc). Need data acquisition first.

## CROSS-REFERENCES

Items above relate to existing TODOs:
  - F1 (drop dead features) ⇄ "orphan modules" decision
  - K2 (cross-ticker 8-K) ⇄ doc's corporate_event_signature
  - Phase 2A (A8 as feature) ⇄ separate from this but builds same pattern

End of Signal_System_Addendum audit.


---

## P3 — ALPHA PROGRAM (6-phase combinatorial alpha pipeline, May 28 deep research)

**Source:** Two May 28 deep-research reports — alpha signals catalog + hedge-fund
combinatorial methodology. Saved as `docs/alpha_signals_catalog_research_20260528.md`.

**Core principle (Report 2):** infrastructure BEFORE signals. Gating defenses
(DSR/PBO/purged-CV) must exist before signal-pool expansion or we ship overfit
garbage. This is why P3.1 and P3.2 precede P3.3+.

**Yield expectation (honest):** Expect <10% of expanded candidates to survive
gating. End state: ~10-30 truly new validated features, not hundreds.

**Total timeline:** ~10-12 weeks. Phases gated; do not skip.

### P3.0 — Prerequisite: (4C RESOLVED — rejected May 29, see active list)
- 4C verdict made: REJECTED. P3 no longer gated on 4C.
- P3 may now begin whenever prioritized (next genuinely-open work item).
- Dates: Jun 5-8 (existing schedule)
- Status: gated by 4C shadow trade (build May 29-30, accumulate Jun 1-5)

### P3.1 — Operator library + PIT audit (~1 week)
- Goal: Build the ~20 operator vocabulary from Report 2 §II.A
- Deliverable: `analysis/operators.py` with rank, ts_rank, ts_mean, ts_std,
  ts_delta, ts_decay_linear, ts_corr, ts_min/max, ts_argmin/max, scale,
  neutralize, group_rank, vector_neut, signedpower, etc.
- Also: point-in-time audit of feature store (no look-ahead)
- Gate to P3.2: all 20 operators pass unit tests; PIT audit clean
- Dates: Jun 9-16

### P3.2 — Gating pipeline (~2 weeks) ★ absorbs existing "DSR overfit tool"

★★ P3.2 STATUS: BUILT + RUN (Fri May 29 2026). VERDICT BELOW. ★★
Deliverables shipped + committed:
- analysis/alpha_gate_stats.py (DSR/PSR/expected-max math + unit tests)
- analysis/alpha_gate.py (rank-IC over full 3390-feature panel, 579 dates,
  vs Massive fwd returns; extreme-value t-threshold sqrt(2 ln N) + mean_IC
  magnitude floor 0.02 + macro/sector exclusion + dedup-by-base)
- analysis/alpha_gate_incremental.py (transform-vs-raw uplift)
- analysis/alpha_gate_results_h5.csv (full scored output)

HONEST VERDICT (the important part):
- Gate WORKS: from 3390 features it independently rediscovered the catalog's
  high-evidence CS signals (short_ratio, pc_ratio, eps_surprise, post-earnings
  drift, squeeze, reversal). Strong validation it is calibrated right.
- BUT the 3390-feature panel is ~95% REDUNDANT with the model's existing 96
  FEATURE_COLUMNS. Of 30 gated CS survivors, ~29 have a base already in the
  model. Only 2 bases NOT in model (pc_ratio_snap, macd_signal); their
  transforms barely beat raw (+0.001 / worse).
- The 'unused 3390-panel = big alpha' premise is DISPROVEN. Real opportunity
  is ~4-6 smoothing transforms (rev_growth_qoq__ts_std, ma_20__ts_std,
  rsi_14__ts_mean, bb_upper__ts_delta) adding ~+0.012 to +0.020 ABSOLUTE IC
  over raw bases. Modest; XGBoost can partly learn them from raw anyway.
- LESSON: uplift_pct is a TRAP (tiny raw_IC denominators give absurd % like
  31000%). Judge by ABSOLUTE IC uplift, not percentage.

REVISED DOWNSTREAM PLAN:
- P3.4 (combinatorial expansion): DOWNGRADED. Expansion already exists (panel)
  and is mostly redundant. Not worth a big effort.
- P3.5 (feed survivors to model): NARROWED to A/B testing ONLY the ~4-6
  defensible transforms above; expect marginal gain; gate on Sharpe not raw.
- Net P3 outcome: proved the model is already well-built; avoided the overfit
  trap of dumping 3390 correlated features into XGBoost. Rigor WAS the value.

★★ P3.5 STATUS: CLOSED — NEGATIVE RESULT (Fri May 29 2026). ★★
Wired the 7 gate-surviving transforms behind ML_QUANT_PANEL_TRANSFORMS=1
(builder.py + classifier.py, mirrors ML_QUANT_INST_FEATURES; OFF=true no-op,
verified OUTPUT_COLUMNS 117->124 / FEATURE_COLUMNS +7 only when ON).
Walk-forward A/B (h=5, NVDA/TSLA/JPM, baseline vs transforms-ON):
  NVDA AUC 0.514 -> 0.501 (-0.013)
  TSLA AUC 0.547 -> 0.527 (-0.020)
  JPM  AUC 0.571 -> 0.543 (-0.029)
Transforms make per-ticker AUC WORSE on all 3 (accuracy moves = noise).
WHY: gate measured CROSS-SECTIONAL IC but classifier is PER-TICKER -> CS
signal doesn't transfer; is_squeeze_setup__ts_argmax is a within-ticker
constant (mean 19.0); ts_std/ts_mean transforms are reconstructable by XGBoost
from raw ma/rsi/bb -> redundant correlated columns that hurt generalization.
DECISION: flag stays OFF (default). Patch retained as documented rejected
experiment (reversible, harmless no-op). P3 arc complete: 3390-panel offers
NO usable per-ticker alpha beyond the existing 96 features. Tested, not assumed.
ALSO FIXED (separate standing bug found via Rule#1 gap-check): builder.py
training_mode insider unpack expected 3 but _load_insider returns 5 -> the
training_mode feature build was crashing. Committed separately.
FRAGILITY FLAGGED (not fixed): FEATURE_COLUMNS built at import-time from
os.environ; no load_dotenv() in entrypoints -> count depends on import order.
A/B used explicit inline env to avoid confounding. Future refactor candidate.

ORIGINAL P3.2 SPEC (for reference):
- Goal: Build the multiple-testing + overfitting defense pipeline
- Deliverable: `analysis/alpha_gate.py` with:
  - Deflated Sharpe Ratio (Bailey-López de Prado 2014)
  - Probability of Backtest Overfitting via CSCV (Bailey-Borwein-LdP-Zhu)
  - Purged & embargoed k-fold CV (López de Prado AFML ch.7)
  - Incremental-IR vs current book (correlation cap 0.7)
- Threshold framework: |t|>3.0 AND DSR>0.95 AND PBO<0.3 AND net Sharpe>0.5
- Gate to P3.3: pipeline correctly reproduces today's 4G/3B verdicts as rejects
- Dates: Jun 16-30

### P3.3 — Bounded seed pool (~2 weeks) ★ absorbs F3 + F4
- Goal: Hand-curate 30-50 base signals from the catalog (NOT 100+)
- Deliverable: `research/seed_signals.yml` + feature builders for each
- Includes from catalogs:
  - Textual: Lazy Prices 10-K similarity delta, MD&A sentiment delta,
    earnings-call FinBERT tone delta vs prior call (subsumes F3)
  - Options surface: IV term-slope, butterfly, vol-of-vol, GEX single-name,
    VRP single-name, pc_ratio_change_5d, iv_skew_change (subsumes F4)
  - Network: GDELT co-mention centrality, customer-supplier lead-lag
  - Microstructure: Amihud illiquidity, overnight/intraday return split
  - Event: 8-K item-specific drift, opportunistic-insider cluster buys
  - Cross-asset: Treasury/DXY/oil lead-lag, BTC contagion proxy
  - Higher moments: realized skew, kurtosis, coskewness
- Gate to P3.4: each seed produces non-NaN features, passes sanity check
- Dates: Jul 1-14

### P3.4 — Combinatorial expansion + gating (~3 weeks)
- Goal: Parameter sweep on seeds → ~1-3k variants → run through P3.2 gate
- Deliverable: `research/alpha_expansion/` results table; survivors documented
- Variant axes: lookback windows {5,10,20,60,120}, decay {0,3,5,10},
  neutralization {sector, size-bucket, none}, transforms {rank, zscore, ts_rank}
- Track N religiously (feeds DSR's selection-bias correction)
- Gate to P3.5: survivor list with DSR/PBO/incremental-IC scores; <10% expected
- Dates: Jul 15-Aug 7

### P3.5 — HRP clustering + ensemble integration (~2 weeks)
- Goal: Cluster survivors by correlation (HRP — López de Prado 2016), combine
  within family by inverse-vol, feed to XGBoost/LightGBM as new features
- Deliverable: extended FEATURE_COLUMNS; retrained models; A/B backtest
- Also: bundle F1 (drop dead features) into this retrain
- Gate to P3.6: combined ensemble Sharpe > current baseline (purged-CV measured)
- Dates: Aug 8-21

### P3.6 — Production monitoring + decay (ongoing)
- Goal: Per-alpha live IC log, decay alerts, shadow A/B lane in Pipeline C
- Deliverable: monitoring dashboard; daily per-alpha IC tracking; auto-retire
  rules (e.g. trailing-60d IC IR < half backtest → flag)
- Crowding detection: correlation with public factors
- Dates: Aug 22+

---

## P3 ALPHA SIGNAL BACKLOG (catalog from May 28 deep research)

A reference inventory of candidate signals. **Test threshold:** standalone IC
>= 0.02 at any horizon, OR SHAP top-quartile when added, OR +50bps top-decile
precision at 5d. **Drop threshold:** corr > 0.8 with existing, sign-flip OOS,
OR live IC < 0 for 2 quarters.

Full catalog with citations: `docs/alpha_signals_catalog_research_20260528.md`

### Tier 1 — Cheap to build, high-evidence (target for P3.3 seed pool)
- Overnight vs intraday return decomposition (Lou-Polk-Skouras 2019 JFE)
- Amihud illiquidity, rolling 21d (Amihud 2002)
- Insider cluster buys, 30d window (Cohen-Malloy-Pomorski 2012, 82bps/mo)
- Short-interest CHANGE (we have level, add Δ)
- Days-to-cover (SI / 20d avg volume)
- 52-week-high proximity (George-Hwang 2004)
- Residual / idiosyncratic momentum (Blitz-Huij-Martens 2011)
- Heston-Sadka same-calendar-month seasonality (2008 JFE)
- days_to_FOMC / NFP / CPI indicators (Savor-Wilson 2014)
- MAX (max daily return prior 21d, long-only avoid filter)
- IVOL filter (Ang-Hodrick-Xing-Zhang 2006)

### Tier 2 — Options-surface signals (from UW, high leverage at 1-5d)
- IV term-slope (Vasquez) — slope of ATM IV across tenors
- IV term-curvature / butterfly
- 25Δ risk reversal dynamics (Δ and z-score)
- Skew-butterfly / smile convexity
- IV-RV spread per stock (Bollerslev-Tauchen-Zhou VRP)
- Cremers-Weinbaum call-put IV spread
- GEX / dealer-gamma proxy at ticker level
- Vol-of-vol (single name)

### Tier 3 — Fundamentals (Polygon-buildable, slower but useful)
- Gross profitability (Novy-Marx 2013)
- Cash-based operating profitability (Ball et al 2016)
- Piotroski F-Score
- Beneish M-Score
- Asset growth, YoY (Cooper-Gulen-Schill 2008)
- Net share issuance, 12mo (Pontiff-Woodgate 2008)
- Analyst forecast revision momentum (Chan-Jegadeesh-Lakonishok)
- Analyst forecast dispersion (Diether-Malloy-Scherbina)

### Tier 4 — Network / textual / events (some need new data)
- Lazy Prices: 10-K/10-Q YoY similarity (Cohen-Malloy-Nguyen 2020 JF)
- MD&A / risk-section FinBERT sentiment delta
- Earnings-call tone delta vs prior call
- GDELT news co-mention network centrality
- Customer-supplier momentum (Cohen-Frazzini 2008, needs link data)
- ETF mispricing / unexpected flow (Ben-David-Franzoni-Moussawi)
- 8-K item-specific drift (item codes 1.01, 5.02, 2.02)
- Form 144 + officer-specific Form 4 (opportunistic vs routine)
- Patent-value shock (Kogan-Papanikolaou-Seru-Stoffman 2017 QJE)
- Cross-asset lead-lag (Treasury, DXY, oil → equity)
- TIPS breakeven inflation term structure (FRED)
- Corporate credit spread stress (HY OAS, BAA-AAA)

### Higher moments
- Realized skewness (20-60d)
- Realized kurtosis (tail thickness)
- Co-skewness with market

### Decayed / arbitraged — do NOT prioritize:
- Sloan accruals (Mohanram 2014)
- PEAD on large/liquid stocks (Martineau 2022 — alive only in microcaps)
- Pre-FOMC drift standalone (Kurov-Wolfe-Gilbert 2021)
- BAB headline construction (Novy-Marx-Velikov 2022)

### Interaction ideas worth testing once tier 1-2 land:
- Quality × Value (QMJ — Asness-Frazzini-Pedersen 2019)
- SUE × IVOL (PEAD concentrated in high-IVOL — Mendenhall 2004)
- Short interest × news sentiment (Engelberg-Reed-Ringgenberg 2012)
- IV skew × earnings proximity
- VIX regime × momentum (momentum crashes, Daniel-Moskowitz 2016)


---

## P3 — ALPHA FACTORY (replaces obsolete "4-Week Roadmap"; supersedes May-8 P3)

CONTEXT: Old P3 was BLOCKED by C1 and assumed the per-ticker model as base.
C1 is DONE; P0-2 (May 30, n=21,376) proved per-ticker = coin flip (OOS AUC 0.487/0.493).
So the factory's PURPOSE is to find edge where the current per-ticker architecture has none:
new data axes + cross-sectional/GLOBAL objective + rigorous gating. NOT more transforms of
existing OHLCV features (P3.2 proved that panel 95% redundant; P3.5 proved transforms HURT
per-ticker). ~70% of this infra already exists — this roadmap is FINISH + FORMALIZE.

TWO LOAD-BEARING LESSONS (violate these and the factory manufactures false positives):
  L1 (from P3.5): GATE ON THE PRODUCTION OBJECTIVE, not cross-sectional IC. A signal that
     passes CS rank-IC can still lower per-ticker/live AUC. alpha_gate.py currently gates CS-IC.
  L2 (from P3.2): EXPANSION APPLIES TO NEW DATA AXES ONLY. Transforming existing features
     adds nothing (95% redundant). alpha_transformations flag stays OFF for current features.

### Stage 0 — Operator library + PIT audit  ✅ MOSTLY DONE
- Have: features/alpha_transformations.py (17 ops), analysis/build_alpha_panel.py (panels,
  group/neutralize, per-date parquet, resumable).
- Remaining: PIT-correctness test suite over the panel (no look-ahead); document operator list.
- KILL: n/a (infra). TRIGGER: as-needed before Stage 1 on new data.

### Stage 1 — Candidate generation from NEW DATA  🔨 PARTIAL



### Stage 1 DATA AUDIT (May 30) — every new axis is forward-only or throttle-limited

Grep + DB-verified. The deep 5yr purged-gate process (used for per-ticker P0-2 and
GLOBAL eval) CANNOT run on any new axis — the historical data does not exist.

| Axis | Blocker | Gateable now? |
|---|---|---|
| Options VRP | options_skew_history stores skew_25d only; put_iv/call_iv columns BLANK (source='massive' dropped IV legs). No IV level -> no VRP. 10wk span (2026-03-21+). | No |
| Options IV-skew change | skew_25d IS stored, 10wk -> ~50 dates. Too few to gate. | Forward-only |
| Options IV term-structure / O-S / GEX | never logged; UW greeks current-only, no backfill | No |
| 10-K Lazy Prices | 10-Ks NOT stored anywhere. finbert_filings is 8-K ONLY (1yr, data/sentiment.db) | No |
| 8-K sentiment | 1yr (2025-05+, 108 tk, 2846 rows) but event-sparse | Marginal |
| GDELT co-mention | historical archive WORKS (_fetch_gdelt_headlines takes as_of date) BUT ~10s/call + frequent 429s. Co-mention needs all-tickers-per-date -> ~156 calls/date -> days-to-weeks backfill. | Yes but operationally painful |

CONCLUSION: Stage 1 is NOT build-and-gate-tonight. It is build-extractors + log-forward
+ gate-in-months (same regime as SELL/PCT7 forward validation). No new axis has deep
gateable history.

HIGHEST-EV ACTION (done May 30): fix options IV-level logging so put_iv/call_iv are
stored -> VRP + IV-skew-change start accumulating forward from today -> gateable ~Aug-Sep.
20-min fix beats a week-long GDELT crawl for a moderate-value signal.

DEFERRED: GDELT co-mention backfill (throttled multi-day crawler) — moderate EV, high
effort. Revisit only if forward-logged options signals fail to gate.



### Stage 1 DATA AUDIT (May 30) — every new axis is forward-only or throttle-limited

Grep + DB-verified. The deep 5yr purged-gate process (used for per-ticker P0-2 and
GLOBAL eval) CANNOT run on any new axis — the historical data does not exist.

| Axis | Blocker | Gateable now? |
|---|---|---|
| Options VRP | options_skew_history stored skew_25d only; put_iv/call_iv were BLANK. FIXED May 30 (daily_massive_skew.py now logs IV legs) -> VRP accumulates forward. 10wk span so far. | Forward, ~Aug-Sep |
| Options IV-skew change | skew_25d stored, 10wk -> ~50 dates. Too few to gate. | Forward-only |
| Options IV term-structure / O-S / GEX | never logged; UW greeks current-only, no backfill | No |
| 10-K Lazy Prices | 10-Ks NOT stored. finbert_filings is 8-K ONLY (1yr, data/sentiment.db) | No |
| 8-K sentiment | 1yr (2025-05+, 108 tk, 2846 rows) but event-sparse | Marginal |
| GDELT co-mention | historical archive WORKS but ~10s/call + frequent 429s; needs all-tickers-per-date -> days-to-weeks backfill | Yes but high-effort/moderate-EV. DEFERRED. |

CONCLUSION: Stage 1 is NOT build-and-gate-tonight. It is build-extractors + log-forward
+ gate-in-months (same regime as SELL/PCT7 forward validation).

DONE May 30: fixed options IV-level logging (daily_massive_skew.py) -> VRP + IV-skew-change
now accumulate forward. Gateable ~Aug-Sep when >=250 dates exist.

FOLLOW-UP: other skew writers (daily_uw_snapshot.py, generator.py, intraday_builder.py,
write_skew_from_log.py) still drop IV legs — patch for consistency when convenient. Trigger: before VRP gate (~Aug).

DEFERRED: GDELT co-mention backfill — moderate EV, multi-day throttled crawl. Revisit only
if forward-logged options signals fail to gate.

### Stage 1 — VERIFIED build list (May 30, grep-audited against codebase)

GLOBAL cross-sectional eval: 5yr purged WF OOS AUC = 0.496 (smoke, 3tk, n=3807).
The plan-doc 0.58 was a single 2mo April holdout (validate_oos), no embargo — a leak.
BOTH architectures (per-ticker P0-2 0.49, cross-sectional 0.50) coin-flip on current
features. Architecture search OVER. Only live branch = NEW DATA AXES.

UW options endpoints ALREADY FETCHED (data in hand, features not built):
  /api/stock/{t}/stock-state (spot), /greeks?expiry (per-strike greeks/IV),
  /options-volume. So VRP/O/S/GEX/IV-slope are FEATURE-ENGINEERING, not new data.

Build order (ROI = info-per-hour, all inputs already available):
  1. VRP (IV30d - realized vol) — inputs in hand, ~2h. HIGHEST ROI. NOT built (grep-confirmed).
  2. IV-skew CHANGE (1d/5d delta+z of iv_skew_25d) — have level only, ~1h. NOT built.
  3. IV term-structure slope (single-name 30d vs 60d IV) — 2nd expiry call, ~half day.
     NOT built (vix_term_structure is VIX, not single-name IV — grep-confirmed).
  4. O/S ratio (options-volume / stock volume) — endpoint fetched, ~2h. NOT built.
  5. GEX (Sigma OI*gamma*spot) — greeks endpoint has it, ~full day. Lowest pri (noisy/assumption-heavy).

ALREADY BUILT (do NOT rebuild — grep-confirmed in classifier.py):
  - 8-K item-type features: eightk_exec_change_30d, material_agreement, reg_fd, other_events — LIVE.

Partial / different-than-catalog:
  - GDELT: sentiment headlines only (etl_gdelt.py), NO co-mention/centrality network. Network = new build.
  - Lazy-Prices 10-K text delta: NOT built. etl_finbert_filings scores per-section sentiment, not YoY similarity.

Each new feature -> through the 6-gate dual pipeline (alpha_gate.py). Keep only dual_survivors.

- Have: build_alpha_panel.py generation engine; data/alpha_sources.py (analyst revisions/yf).
- Build: high-feasibility LOW-CROWDING sources the catalog flags, in priority of
  (feasibility-on-our-data x expected incremental-IC x low crowding):
    1. Options-surface geometry from UW: IV-skew change/level, IV term-structure slope,
       VRP (IV-RV), O/S volume ratio, single-name GEX proxy.
    2. EDGAR text deltas (free): Lazy-Prices 10-K/10-Q YoY similarity; item-specific 8-K drift.
    3. GDELT co-mention network centrality (in stack, uncrowded).
    4. UW Form-4 opportunistic-vs-routine insider classification.
    5. Overnight/intraday return decomposition (cheap, unarbitraged).
- Track N (trial count) religiously — feeds DSR. Seed ~30-50 base expr -> ~1-3k variants MAX.
- KILL per source: if best variant fails Stage-2 gate, log as rejected experiment, move on.
- TRIGGER: after Stage 2 objective-fix (below). First source: UW options-surface (we pay for UW).

### Stage 2 — Gating pipeline  ✅ STRONG, 2 GAPS
- Have (real): alpha_gate.py (CS rank-IC + Harvey-Liu-Zhu |t|>3.0), alpha_gate_stats.py
  (deflated_sharpe, expected_max_sharpe, probabilistic_sharpe), alpha_gate_incremental.py
  (absolute uplift vs raw base), fitness_scorer.py, walk_forward.py (purged k-fold + embargo).
- GAP A ✅ DONE (May 30, commit 77547df): per-ticker TS-IC gate added beside CS-IC in alpha_gate.py.
  Result h5: 32 CS survivors -> 14 pass per-ticker t>3, but only 6 SAME-SIGN (dual_survivor).
  6 same-sign = earnings/fundamentals-event signals (post_earnings_3d/5d, eps_surprise, rev_growth_qoq, obv, bb_width).
  8 opposite-sign = objective-conditional traps (vwap/rsi/macd/52w/intraday: trend CS, mean-revert per-ticker).
  27/32 CS survivors flagged divergent — quantifies the P3.5 footgun the old gate was blind to.
- GAP A (was): add a PER-TICKER / production-objective gate beside the CS-IC gate.
  A candidate must improve the metric the live model is scored on, not just CS-IC.
- GAP B ✅ DONE (May 30): cscv_pbo added to alpha_gate_stats.py + wired as 6th gate. Result h5: PBO=0.151 (PASS <0.3) on 70 base signals -> SELECTION IS NOT OVERFIT. The surviving signals are real, not best-of-noise. Combined with P0-2: signals have real CS information, per-ticker architecture can't extract it. Stage 2 COMPLETE (6 gates: HLZ, EVT, MAG, DSR, TS-IC+sign, PBO).
- SIX-GATE BAR (clear ALL): purged-CV rank-IC IR>0.3; |t|>3.0; DSR>0.95; PBO<0.3;
  net long-only Sharpe>0.5; incremental (corr-to-book<0.7 AND incremental-IR>0).
- KILL: <single-digit % survival is EXPECTED, not failure (per both research docs).
- TRIGGER: GAP A before any new Stage-1 source is trusted. *Also gate GLOBAL/Path A 0.58 here.*

### Stage 3 — Combination (HRP)  ❌ MISSING — BUILD
- Build portfolio/hrp_combine.py: cluster gated survivors by correlation (dendrogram),
  inverse-variance allocate within family; feed family composites + raw survivors to XGBoost
  (Gu-Kelly-Xiu: trees capture interactions; ensemble IS the combiner). Keep isotonic calibration.
- KILL: if combined survivors' OOS Sharpe < best single survivor, combination adds nothing.
- TRIGGER: once >=5 survivors exist from Stage 2.

### Stage 4 — Production integration  ⚠ PARTIAL
- Wire survivors into features/builder.py behind config flag (pattern exists for transforms).
- Gate: OOS validation on held-out window BEFORE live; A/B shadow lane vs incumbent.
- Rule #1: a survivor is not "done" until committed + tests + imported in builder +
  in feature panel + on cron + nonzero importance + documented.
- KILL: live IC<0 for 2 consecutive quarters -> retire (Stage 5).
- TRIGGER: per-survivor after Stage 3.

### Stage 5 — Monitoring / decay  ❌ MISSING — BUILD
- Build scripts/alpha_decay_monitor.py: daily rank-IC per alpha; alert if trailing-60d IC IR
  < half backtest or negative k weeks; monthly DSR/PBO recompute with UPDATED N; retire on collapse.
- Per-alpha daily log: date, signal hash, rank-IC(1/3/5d), turnover, gross/net, corr-to-book.
- KILL: n/a (this IS the kill mechanism). TRIGGER: once first survivor goes live (Stage 4).

### Sequence / dependency
S0(done) -> Stage2 GAP A (objective fix) -> Stage1 first source (UW options) -> gate ->
Stage3 HRP (>=5 survivors) -> Stage4 wire one -> Stage5 monitor. GLOBAL/Path A validation
runs through Stage2 in parallel (it's the one untested architecture that might already have edge).

### Immediate next action (highest leverage)
Stage 2 GAP A — make the gate score the production objective, not CS-IC. This single fix makes
every one of the ~250 catalog signals testable HONESTLY and would have caught the P3.5 false positive.
Build it BEFORE generating new candidates and BEFORE trusting the GLOBAL 0.58.


## P3+ — WEAK-SIGNAL EXTRACTION PLAN (May 30, from OOS-collapse research)

### Framing — READ FIRST (so future-you/AI doesn't panic)
"At the ceiling" is NOT "dying." The realistic OOS ceiling for 1-5d equity direction is
AUC ~0.55 / IC ~0.05 (Kamalov 2019; Gu-Kelly-Xiu landmark SUCCESS = 0.4% monthly R2 -> Sharpe 1.35).
Being near it = you measured HONESTLY, not failed. Every real quant lives here. The funds that
think they're above it have undiscovered leaks (that was us at 0.967, and the 0.58 GLOBAL leak).

Verified state (this session):
- Signal is REAL (dual-gate PBO=0.15, not overfit selection) but WEAK (CS mean_IC ~0.02-0.05).
- Both architectures coin-flip OOS (per-ticker P0-2 0.49 n=21376; GLOBAL purged 0.496) -- because
  IC~0.03 MATHEMATICALLY maps to AUC~0.515 (AUC ~= 0.5 + IC/2). Not a model defect. The ceiling.
- So the job is NOT rescue-the-models. It is: extract PORTFOLIO-level edge from weak signals
  (combine -> size -> manage decay), OR prove honestly 150-long-only-daily can't clear costs.

Decay is MANAGED, not terminal. Publication-decay (McLean-Pontiff 26%/58%) hits PUBLIC factor-zoo
signals -> defense is many + uncrowded + rotate. A portfolio of many weak decorrelated alphas with
active rotation does NOT go to zero; you retire faded ones, others carry the book (Stage E).

The REAL open risk = BREADTH. IR = IC * sqrt(breadth). 150 names, long-only (transfer coef <1),
daily = structurally low breadth. Whether enough weak alphas combine to clear costs is UNKNOWN and
is exactly what Stage C tests. If Stage C combined-IC ~= 0 across regimes -> the fix is STRUCTURAL
(more names / longer horizon / relax long-only), NOT "give up" and NOT "tune the model more."

### Stage A — Re-baseline on rank-IC, demote AUC (CHEAP, FIRST)
Research Stage 1. AUC hides weak signal (AUC~=0.5+IC/2). The gate already uses rank-IC; model evals
(walk_forward.py, eval_global_pit.py) still headline AUC. 
ACTION: add per-date rank-IC (Spearman pred vs fwd ret), IC distribution (median + quartiles),
and top-decile-minus-universe spread to both harnesses. Report IC as headline, AUC as footnote.
COST: low. 
KILL GATE: median per-date rank-IC > 0.02 with positive lower quartile -> real signal, proceed to B.
  If centered at 0 with tight dispersion -> signal absent -> STOP, go to structural levers.
  If IC swings sign by regime -> non-stationary -> note for regime work (but see caution below).

### Stage B — Re-evaluate learning-to-rank HONESTLY (top remedy, model already exists)
Research's #1 remedy: ranking optimizes ORDERING (what long-only needs), ~3x Sharpe vs classification
(Poh et al. 2021). models/train_global_ranker.py (LightGBMRanker lambdarank) EXISTS but had ZERO OOS
eval -- only in-sample predict(X[:1000]). The 0.58 was a leaky single 2mo split.
ACTION: run the ranker through purged-WF + rank-IC (ranker outputs ranks -> rank-IC is the native
metric, convenient). Compare ranker rank-IC vs classifier rank-IC on the SAME purged folds.
COST: medium (adapt eval harness to score a ranker).
KILL GATE: ranker rank-IC > classifier rank-IC by a real margin -> ranking is the architecture.
  If equal/worse -> ranking didn't help here, proceed to C on classifier scores anyway.

### Stage C — Build alpha-combination layer / HRP (WHERE EDGE IS MADE)
Alpha Factory Stage 3. The core remedy: many weak decorrelated alphas -> portfolio Sharpe via
IR=IC*sqrt(breadth) (Kakushadze: 101 alphas, ~16% pairwise corr, none strong alone). Have ~6
dual-survivors + panel builder; NO combiner.
ACTION: build portfolio/hrp_combine.py -- cluster gated survivors by correlation (dendrogram),
inverse-variance weight, recursive bisection (Lopez de Prado 2016). Backtest combined book's
rank-IC/Sharpe vs best single survivor. Feed composites + raw survivors to the ranker/ensemble.
COST: medium-high (new module).
KILL GATE (THE BIG ONE): combined OOS net-of-cost Sharpe > best single survivor AND > 0.5
  -> real portfolio edge, proceed to D. If combined IC ~= 0 across regimes -> 150-long-only-daily
  lacks breadth -> STRUCTURAL decision (expand universe / lengthen horizon / relax long-only), NOT
  build D, NOT tune more.

### Stage D — Bet sizing: fractional Kelly + explicit cost model
Research: a 51% edge IS tradeable if sized right + costs controlled. Full Kelly catastrophically
overbets a mis-estimated edge -> use QUARTER-to-HALF Kelly (half = ~75% growth at ~half variance).
Cost model exists in fitness_scorer.py but not in the live/sizing path.
ACTION: size positions by combined-signal strength via fractional Kelly off calibrated probs;
deduct realistic per-turnover cost (10bps/turn already modeled). Calibration (Platt/isotonic) folds
in HERE -- it can't make edge, only makes probs honest for Kelly.
COST: medium.
KILL GATE: net-of-cost Sharpe stays > 0.5 AFTER realistic turnover (research: costs can eat the
  entire thin edge at 1-5d holding). If costs kill it -> lengthen horizon (less turnover).

### Stage E — Decay monitoring (Alpha Factory Stage 5)
Research: expect >=26% OOS haircut (McLean-Pontiff); rotate alphas as they fade. Same pattern as
SELL/PCT7 forward validation.
ACTION: scripts/alpha_decay_monitor.py -- daily live rank-IC per alpha; alert when trailing-60d IC
< half backtest or negative k weeks; retire on collapse, surface replacement candidates.
COST: medium. TRIGGER: once first combined signal goes live (after D).

### DEFERRED / CAUTIONED (research says low-priority at our scale)
- Triple-barrier + meta-labeling: benefit "strategy-dependent" (Hudson&Thames). Bigger lift,
  uncertain payoff. Revisit after A-C prove signal is combinable.
- Regime-conditioning: research EXPLICITLY warns it fragments scarce data + increases overfitting
  on a 150-name book. Against it for now. Only if Stage A shows clean regime-split IC.
- Calibration as standalone: "cannot manufacture edge." Folds into Stage D, not its own stage.

### STRUCTURAL LEVERS (may matter more than any code -- the real ceiling is breadth)
If Stage C kills on breadth, these are the actual remedies, in order of feasibility:
1. Lengthen horizon (daily -> weekly): less cost drag, more signal-per-trade. Cheapest structural fix.
2. Expand universe (>150 names): directly raises breadth (IR=IC*sqrt(breadth)).
3. Relax long-only: recovers the short leg (~half the breadth). Hardest (borrow, infra, risk).
OPEN QUESTION for Atom: which of these is actually movable? If breadth is the ceiling, the
highest-value "remedy" is a constraint change, not more model work.

### Sequence
A (rank-IC rebaseline, cheap, reinterprets tonight) -> B (test ranking, top remedy, model exists)
-> C (HRP combine, where edge is made -- THE decision gate) -> D (size) -> E (monitor).
Each stage has a kill gate. A or C at IC~=0 -> STOP, pull a structural lever, don't grind noise.


### STAGE A VERDICT (May 30) — per-name direction fails; cross-sectional ranking weakly positive; BREADTH is the proven lever

Ran rank-IC on BOTH panels at full scale. Read against Stage A kill gate:

| Metric | GLOBAL (149tk, n=169348, 978 dates) | Per-ticker (40tk, 202 dates) | Gate | Pass? |
|---|---|---|---|---|
| rank_ic_median | 0.0112 | 0.0111 | >0.02 | no |
| rank_ic_q25 | -0.113 | -0.205 | >0 | no |
| rank_ic_pos_frac | 0.531 | 0.525 | >0.55 | no |
| rank_ic_t | 1.664 | 0.969 | >3.0 | no |
| pooled OOS AUC | 0.479 | 0.487/0.493 | (footnote) | coin-flip |

CRITICAL READ — do NOT misread this as "everything fails / worse than coin flip":
- AUC/accuracy (~0.48) measures PER-NAME DIRECTION ("will NVDA go up?"). That fails. Expected.
  The research says this is near-impossible at the ceiling AND it is the WRONG target. Stop chasing it.
- rank-IC (+0.011 median, pos on 53% of days) measures CROSS-SECTIONAL RANKING ("is NVDA a better
  bet than INTC today?"). That is WEAKLY POSITIVE — not zero, not negative. Pointed the right way,
  just not yet significant. This is the metric that maps to portfolio Sharpe.
- These are DIFFERENT QUESTIONS. A system can fail per-name direction while being faintly-right at
  cross-sectional ranking. The ranking is what makes money in a portfolio.

THE HOPEFUL NUMBER (easy to miss): IC t-stat DOUBLED 0.97 -> 1.66 going from 40 -> 149 names, with
the SAME IC (~0.011). That is IR = IC*sqrt(breadth) confirmed EMPIRICALLY. The weak signal does not
die with scale — it gets MORE significant. The signal is not dead, it is UNDER-LEVERAGED.

GATE DECISION: Stage A fails on current features for BOTH architectures. AUC was NOT masking real IC
(we checked with the right metric on the right cross-sectional panel at full scale). So:
- Do NOT build Stage B (ranker eval) or Stage C (HRP combine) on current features — would be
  polishing a 0.011-median/q25-negative signal = grinding noise. The plan's kill gate says STOP here.
- ROUTE TO STRUCTURAL LEVERS. The t-stat doubling proves the binding constraint is BREADTH, not the
  model. "How do I improve accuracy" is the wrong question; the right one is "how do I add breadth so
  the ranking t-stat clears 3."

BREADTH MATH: to get t from 1.66 -> 3.0 at constant IC needs ~(3/1.66)^2 ~= 3.3x more breadth.
Levers (the OPEN QUESTION for Atom — highest-value decision in the project now):
  1. Use ALL 3 horizons (1d/3d/5d) as partly-independent bets — ~3x breadth, ZERO new data. Cheapest.
     Currently h=5 tested alone. This ALONE could approach the needed multiple.
  2. Expand universe >149 names — directly raises breadth.
  3. Relax long-only — recovers short leg, ~2x effective breadth (rank bottom too, not just top). Hardest.

HONEST RISK: IC 0.011 is genuinely weak; breadth might not scale far enough to beat costs. That is the
real unknown — but "might not scale far enough" is NOT "no path exists." The path exists (rank + breadth);
the question is whether the signal rides it to profitability. Only way to know: add breadth, re-measure.

NEXT (next session): pick a breadth lever. Cheapest first test = combine 1d/3d/5d rank-IC (lever 1,
no new data) and see if pooled t-stat rises toward 3. If yes -> breadth is the answer, build toward it.
If even 3-horizon + full universe stays t<2 -> signal too weak for long-only daily, consider horizon
lengthening (daily->weekly) or accept the system as non-viable at this scale.


### STAGE A — HORIZON CURVE + POOLING TEST (May 30) — both cheap breadth levers DEAD; signal lives at h=5

Full rank-IC horizon sweep, GLOBAL cross-sectional, 5yr purged WF, 149 tickers:
| Horizon | IC t-stat | pos_frac | read |
|---|---|---|---|
| h=1  | -0.42 | 0.500 | dead (coin-flip ranking) |
| h=3  | +1.47 | 0.502 | rising |
| h=5  | +1.66 | 0.531 | PEAK |
| h=10 | +1.32 | 0.505 | falling |
| h=20 | -1.50 | 0.473 | NEGATIVE (ranking actively wrong) |

Signal peaks at h=5 and decays to negative by h=20. Lives specifically in the 3-5 day band.

LEVER 1 (pool horizons) — DEAD: corr(h3_IC, h5_IC)=0.571 -> only partial breadth (~1.13x, not sqrt2),
AND h1/h20 are negative so can't pool the full set. analysis/horizon_ic_corr.py.
LEVER 2 (lengthen horizon daily->weekly) — DEAD: signal peaks h=5, negative by h=20. Lengthening
HURTS. The 5-day mark is the operating point; do not extend.

So both FREE/cheap levers eliminated with data. Remaining breadth levers are the harder structural ones:
- LEVER 3a MORE NAMES (untested): pure breadth at constant IC. The t-stat doubling (per-ticker 0.97
  -> 149-name 1.66) is direct evidence more names raises significance. Cleanest remaining test:
  expand universe well beyond 149, re-measure h=5 rank-IC t-stat. To reach t=3 from 1.66 at constant
  IC needs ~3.3x breadth -> ballpark ~500 names if breadth scales as sqrt(N) cleanly.
- LEVER 3b SHORT LEG / relax long-only (untested): recovers bottom of ranking, ~2x effective breadth.
  Hardest operationally (borrow, infra, risk). But long-only currently discards half the ranking signal
  (you can act on top names but not short the bottom).

NEXT SESSION: the real decision is 3a vs 3b. Cheapest test = LEVER 3a, expand universe (more tickers
in tickers.txt) and re-run eval_global_pit --horizon 5. If t-stat climbs toward 3 with more names ->
breadth via universe is the path. If it stalls -> long-only daily at this IC may be non-viable; short
leg (3b) or accept the system can't clear costs.

CONFIRMED OPERATING POINT: h=5, cross-sectional ranking (NOT per-name direction, NOT h1, NOT h20).
