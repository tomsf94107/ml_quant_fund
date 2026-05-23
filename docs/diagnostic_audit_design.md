# Diagnostic Audit Design — scripts/diagnostic_audit.py

**Date created:** May 23, 2026 (Saturday evening)
**Build date:** Sunday May 24, 2026
**Effort:** 2-4 hours
**Goal:** Systematic, reproducible audit of the ML pipeline. Replace ad-hoc queries with permanent tooling.

---

## Why this exists

Production accuracy audit revealed broken calibration (h=5d >=0.80 conf hit 40%, h=3d 0.55-0.65 conf hit 39%). One-off queries miss things. A real quant fund has diagnostic tooling that runs after every model retrain.

## PRIORITY #1 (BEFORE the 5-layer audit): Reconcile the three conflicting AUCs

Tonight's validation produced THREE different OOS AUCs that disagree by up to 14pp. This is the most fundamental issue. If we cannot trust our validation numbers, we cannot make the Jun 23 A/B decision.

### The three numbers

| Test | OOS AUC | Method |
|---|---|---|
| Per-ticker PIT production WF | **0.44** | analysis.walk_forward --pit --config production |
| Per-ticker walk-forward stacks | **0.51-0.53** | models.walk_forward (older time-series folds) |
| Path A 5-fold validation | **0.58-0.59** | train_cross_sectional.py validate_oos |

### Why they shouldn't disagree this much

A well-designed validation should give similar numbers regardless of method. 14pp gap means one of:

1. PIT is correct, others have leakage
2. Architecture difference (per-ticker structurally bad, GLOBAL truly better) → +5pp lift real
3. Test windows differ (Mar-May 2026 is hard period for per-ticker only)
4. Implicit leakage in Path A validation through feature engineering

### Specific tests to reconcile

**Test 1: Run PIT walk-forward on Path A GLOBAL model**
- Same `analysis.walk_forward --pit` framework
- Train cross-sectional model PIT-correctly (refit at each prediction date)
- Compare OOS AUC to tonight's 0.58-0.59
- If 0.58 holds → Path A is real
- If collapses to 0.50 → Path A is also leaky

**Test 2: Re-run per-ticker walk-forward with same test window as PIT**
- Use Mar-May 2026 as test
- See if walk-forward still gives 0.51 or also drops to 0.44
- Tells us if it's a window issue or methodology issue

**Test 3: Identify the leakage in current per-ticker setup**
- Train/test gap 0.27 means model is fitting noise heavily
- Investigate: feature engineering steps that use FULL panel before fold split
- Feature normalization, calibration fit, target encoding

**Test 4: Compare Path A 5-fold split logic vs PIT**
- 5-fold TimeSeriesSplit DOES use only past data per fold
- But each row's features may have been computed using FULL history
- Refit feature engineering per-fold to check

### Output

A reconciliation report:
- Which validation methodology is honest?
- What is the TRUE OOS AUC for per-ticker and GLOBAL?
- Should we trust Path A's 0.58-0.59 or expect it to be lower in production?
- Are the conditions for Jun 23 A/B decision based on reliable numbers?

### Time estimate

3-5 hours. Critical to do BEFORE building the 5-layer audit tool, because the audit tool needs to test what we believe is true. Right now, we don't know what's true.

---

## Five layers of audit

### Layer 1: Data integrity
Verify the foundation isn't corrupt.

Checks:
- [ ] Training data row counts per ticker (no truncated tickers)
- [ ] Targets (target_1d/3d/5d) have no NaN where they shouldn't
- [ ] Outcomes table reconciled with predictions (no orphans)
- [ ] Universe stability (ticker list unchanged or changes documented)
- [ ] Feature build dates vs prediction dates (no train data after prediction)
- [ ] Polygon revenue coverage (after recent fix)
- [ ] 8-K coverage by date range
- [ ] inst feature coverage

### Layer 2: Feature quality
Are individual features useful?

Checks:
- [ ] Feature distribution: mean, std, NaN rate per feature
- [ ] Mutual information of each feature with target_5d
- [ ] Spearman correlation per feature with actual_up
- [ ] Distribution drift: KS test of train vs prod feature values
- [ ] Feature importance from current models (XGB built-in + permutation)
- [ ] Identify features that LOOK informative (high MI) but are IGNORED by model (importance=0)
- [ ] Per-feature stationarity (does its signal change over time?)

### Layer 3: Model fitting quality
Is the model itself good?

Checks:
- [ ] Calibration plot: predicted prob vs actual rate by bucket
- [ ] Expected Calibration Error (ECE)
- [ ] Brier score on held-out data
- [ ] Log loss on held-out
- [ ] AUC on held-out (per horizon)
- [ ] Are predictions monotone in confidence? (regression line on prob vs hit)
- [ ] Is reliability diagram U-shaped (overconfident) or S-shaped (underconfident)?

### Layer 4: Production transformations
The pipeline from raw model output → final signal.

Checks:
- [ ] prob_raw distribution
- [ ] Each multiplier (risk/sent/regime/options/squeeze/intraday/fg) distribution
- [ ] Compound multiplier effect: prob_raw × all_mults → prob_eff
- [ ] Calibration of prob_raw alone vs prob_eff
- [ ] Per-multiplier hit rate contribution (BUYs where mult>1 vs mult<1)
- [ ] Confidence threshold gating (BUYs near the boundary 0.55-0.60)
- [ ] Multiplier explanation (which mult drove each BUY)

### Layer 5: Reality checks
Sanity validations.

Checks:
- [ ] BUYs that LOST big (>-5%) — common pattern?
- [ ] BUYs that WON big (>+5%) — common pattern?
- [ ] Per-ticker performance (are some tickers persistently wrong?)
- [ ] Per-sector performance (sector concentration of losses?)
- [ ] Time pattern (Monday BUYs vs Friday, post-earnings vs not)
- [ ] Regime conditional performance (BULL vs BEAR vs VOLATILE)
- [ ] Compare BUYs to HOLDs: are BUYs differentiated or random?

## Output format
==============================================================
ML QUANT FUND DIAGNOSTIC AUDIT
Date: YYYY-MM-DD
Model versions: per-ticker={hash}, GLOBAL={hash}
LAYER 1: DATA INTEGRITY ─────────────────────────────────
[✓] PASS  Training rows by ticker: median 1487, min 218 (CAVA), max 1587
[✓] PASS  Targets: 0 NaN in 180,871 rows
[✗] FAIL  3 outcomes orphaned (predictions older than 7 days with no actual_up)
Tickers: AAPL 2026-05-15, MSFT 2026-05-16, TSM 2026-05-15
Action: investigate why reconciliation missed these
[!] WARN  Polygon revenue: 10 tickers in SKIP_TICKERS (foreign filers/ETFs)
LAYER 2: FEATURE QUALITY ─────────────────────────────────
Per-feature mutual info with target_5d:
Top informative:
eightk_exec_change_30d:  MI=0.024 (BUT importance=0 in models)
inst_signed_flow_5d:     MI=0.019 (importance=0)
rev_growth_yoy:          MI=0.015 (importance=0)
Distribution drift (train 2020-2025 vs prod Mar-May 2026):
finbert_sentiment: KS p=0.001 ← DRIFT
short_pct_float:   KS p=0.034 ← DRIFT
[✗] FAIL  Cross-sectional features have signal but model ignores them.
Confirms Path A architecture is correct fix.
LAYER 3: MODEL FITTING ───────────────────────────────────
Calibration table (h=5d):
prob bucket      n      pred%   actual%   abs_err
<0.55         1234     0.45     0.51      0.06
0.55-0.60      450     0.58     0.51      0.07
0.60-0.65      280     0.62     0.49      0.13  ← MISCAL
0.65-0.70      180     0.67     0.56      0.11  ← MISCAL
0.70-0.80      150     0.75     0.62      0.13  ← MISCAL

=0.80          35     0.85     0.40      0.45  ← SEVERE MISCAL

[✗] CRITICAL  ECE = 0.18 (acceptable < 0.05)
[✗] CRITICAL  Inversion at high confidence bucket >=0.80
Predicted 85%, actual 40%
LAYER 4: PRODUCTION TRANSFORMS ───────────────────────────
prob_raw distribution: μ=0.52, σ=0.08
risk_mult distribution: μ=0.96, σ=0.07
sent_mult distribution: μ=1.00, σ=0.04
regime_mult distribution: μ=1.02, σ=0.13 (BULL=1.05, BEAR=0.92, VOL=0.65)
options_mult distribution: μ=1.00, σ=0.05
prob_raw → prob_eff calibration:
prob_raw 0.60-0.65 bucket actual hit: 52% (close to predicted)
prob_eff 0.60-0.65 bucket actual hit: 39% ← inflated by mults
[✗] FINDING  Multipliers DESTROY calibration in the 0.60-0.65 bucket
[!] Action   Re-test prob_raw-only signal generation
LAYER 5: REALITY CHECKS ──────────────────────────────────
Biggest BUY losses (last 30 days):
RZLV 2026-05-08 h=3 prob=0.78 actual=-12%
NIO  2026-05-12 h=5 prob=0.81 actual=-9%
...
[!] Common pattern: high-confidence BUYs on volatile small-caps
[!] Action       Consider universe filter or per-ticker conviction caps
==============================================================
SUMMARY
CRITICAL ISSUES: 3
WARNINGS: 2
PASSING: 47/52 checks
TOP PRIORITY FIXES:

Severe miscalibration at high-conf bucket (>=0.80 hits 40%)
Multipliers inflate prob into miscalibrated range
Feature drift in finbert_sentiment

RECOMMENDED NEXT ACTIONS:

Re-fit isotonic calibration on recent data
Consider capping multiplier compound effect
Investigate finbert_sentiment drift
Build position sizing on bucket actual hit rates, not predicted probs


## Implementation modules
scripts/diagnostic_audit.py     # entry point + report formatting
audit/layer1_data_integrity.py
audit/layer2_feature_quality.py
audit/layer3_model_fitting.py
audit/layer4_production_transforms.py
audit/layer5_reality_checks.py
audit/report_formatter.py

## Testing approach

For each layer, write the check, then deliberately break the input to verify the check fires. Examples:
- Layer 1: insert NaN into training target, verify it's caught
- Layer 3: feed perfectly-calibrated predictions, verify ECE near 0
- Layer 4: artificially boost multipliers, verify miscal is detected

## Integration

Once built, hook into:
- Pipeline B post-retrain: run audit, alert if anything regresses
- Pipeline C pre-prediction: run Layer 1+4 subset to catch data issues
- Manual: `python scripts/diagnostic_audit.py [--full | --quick | --layer 3]`

## Findings persist

Each run saves to `audit_reports/audit_YYYY-MM-DD.md` for history.
Track over time: is ECE improving or regressing?

## Rule #1 checklist applied to this audit

- (a) Audit BEFORE changes: this is the audit
- (b) No silent errors: every check fails loud
- (c) Flags=debt: no flags introduced; audit reports findings
- (d) Verify scripts: each layer has a verification test
- (e) Build-not-known cold: this DOC defines what we'll know
- (f) Real data: audit runs on accuracy.db (production data)
- (g) Gap-check after bug: audit IS the gap-check tool
- (h) Verify chain: layers cover data → features → model → transforms → reality
- (i) Compiled OK ≠ verified: layer tests confirm checks fire


---

## Audit timeline (added May 23 evening)

### Phase B (start Sunday May 24, ~5-7 days)
Build 5-layer diagnostic tool, run it, fix critical findings. Sequential:
- Sun May 24 (start of phase B): Build diagnostic_audit.py + ALL 5 layers
- Sun May 24 (later): Run on current data, read findings
- Sun May 24 (continued): Begin manual audit of signals/generator multiplier chain
- Mon May 25: Continue manual audit + Pipeline A→B→GLOBAL→C run
- Mon May 25 (later): Diagnostic on fresh Pipeline data; ETL path audit begins
- Tue May 26: Continue ETL audit + fixes
- Wed May 27 onward: Synthesize, apply fixes, re-run diagnostic to verify

Bottom line: real work starts SUNDAY. Not Tuesday.

### Continuous (week 2 onward)
- 5-layer diagnostic runs after every Pipeline B retrain
- Alerts on calibration regression, feature drift, outcome gaps
- New features/modules get audit checks added before they ship

### Phase A (full systematic audit)
**Date: Saturday June 27 - Sunday July 5, 2026** (~8-9 days)

Triggered by:
- Path A A/B decision (Jun 23) settles model architecture
- 4-week roadmap items mostly complete
- Calibration issue resolved
- ~28 days of production A/B data available

Scope: every code file, every data source, every connection
Output: docs/full_system_audit_2026Q2.md
Followup: Q3 development guided by findings

### Why this sequencing

1. Fix the immediate calibration problem first (Phase B)
2. Run continuous diagnostics to catch regressions (no slip)
3. Audit the FULL system after architecture is stable (Phase A)
4. Avoid auditing code that may be deprecated by A/B decision
