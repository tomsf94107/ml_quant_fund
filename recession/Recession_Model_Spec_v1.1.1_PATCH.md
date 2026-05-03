# Recession Model Spec v1.1.1 — Patch Document

*Patch revision: May 3, 2026 (evening). Modifies v1.1.0 based on Step (b) exploration findings.*

This is a **patch document**, not a full re-spec. Changes here apply on top of `Recession_Model_Spec_v1.1.md`. After review, these changes get folded into the main spec as v1.1.1 and the patch is archived.

---

## 0. Why this patch

The Step (b) data exploration (run May 3, 2026 against the actual `recession.db`) produced findings that meaningfully refine four v1.1.0 decisions. None of the changes alter the core architecture (5 targets, 5 model families, 4 horizons, lead-time metric, alert taxonomy). They tighten input definitions and rescue T2 from being unusable.

Specifically the exploration revealed:

1. **Sahm rule (AUC 0.545) and CFNAI (0.530) are weak at h=12** — they're coincident, not 12-month leading. CFNAI jumps to 0.647 at h=6 vs T2, confirming it's a coincident/short-horizon signal. The v1.1.0 plan to add 4 more labor features primarily for h=12 forecasting needs reassignment.

2. **BAA10Y at h=12 vs T1 is nearly noise (0.518) but at h=6 vs T2 is strong (0.739).** The v1.1.0 T5 threshold of "BAA10Y > rolling 90th percentile" almost never fires because the BAA10Y range is too narrow. Threshold needs absolute calibration, not percentile.

3. **SP500 raw level at h=12 hits ✗** (1/8 episodes show recession-zone movement) because raw price is a monotonically trending series. Hamilton detrending in Step 4 was already planned, but the spec needs to commit *which* SP500 transformations are produced and used downstream.

4. **T1/T2 disagreement is large and exactly as the research brief predicted.** BAA10Y, EBP, CFNAI predict T2 drawdowns much better than T1 recessions (Δ < −0.10). T10Y3M, REAL_FFR_GAP, DTWEXBGS, DCOILWTICO predict T1 much better than T2 (Δ > +0.10). This validates having T1 and T5 as separate targets — and tells us *which* features anchor each target.

---

## 1. Changes from v1.1.0

### Change A: Reassign labor expansion to M5 primary inputs

**Section 4.2 (Tier 2: Labor market)** — modify the v1.1.0 additions:

| Feature | v1.1.0 status | v1.1.1 status |
|---|---|---|
| `SAHMREALTIME` | All horizons primary | M5 primary; available to h=1, h=3 in M4 only; **excluded** from h=6, h=12 model fits |
| `JTSQUR` | All horizons primary | Same as above |
| `ICSA` | All horizons primary | Same as above |
| `CCSA` (NEW v1.1) | All horizons primary | M5 primary; same restriction |
| `AWHMAN` (NEW v1.1) | All horizons primary | M5 primary; same restriction |
| `TEMPHELPS` (NEW v1.1) | All horizons primary | M5 primary; same restriction |
| `JTSLDR` (NEW v1.1) | All horizons primary | M5 primary; same restriction |

**Rationale:** All seven labor features are coincident or short-horizon. Forcing them into h=12 fits adds ~7 features × 4 horizons = 28 parameter slots that can't generalize. Restricting to h=1/h=3 in M4 (the only model with regularization strong enough to handle them well) and M5 (where they belong by design) is correct.

**Implementation note:** This is a config-time decision in Step 4's feature pipeline. A new column in the feature registry: `eligible_horizons` (default: all). Labor features get `eligible_horizons = ['h=0', 'h=1', 'h=3']` (h=0 = M5 coincident).

### Change B: Recalibrate T5 thresholds to absolute, not rolling-percentile

**Section 2.5 (Market Stress Layer T5)** — replace the threshold definitions:

| Condition | v1.1.0 threshold | v1.1.1 threshold |
|---|---|---|
| Yield curve | T10Y3M < 0 | T10Y3M < 0 (unchanged) |
| Credit spread | BAA10Y > 90th-percentile of trailing 60 months | **BAA10Y > 3.5%** (absolute) |
| Equity drawdown | SP500 ≤ 90% of 12-month rolling max | SP500 ≤ 90% of 12-month rolling max (unchanged) |
| Financial conditions | NFCI > 0.5 | **NFCI > 0.25** (one quarter standard deviation above the Z-scored mean of zero — captures meaningfully-tighter conditions without firing on every above-average month) |

T5 fires when **2 or more** conditions are true. Same firing rule as v1.1.0.

**Rationale (data-driven):**
- BAA10Y range across 1986-2026 is ~1.4% to 6.2%. The 90th percentile rolls between ~2.5% and ~3.5% depending on the trailing 60-month window. With p90 logic, T5 only fires during the very top of credit cycles. With absolute > 3.5%, T5 fires during 2008-09, 2020 COVID, 2015-16 energy bust, 2002 dot-com — all genuine market stress events, all visible in the time series plot.
- NFCI is a Z-scored series with mean ≈ 0 by construction. NFCI > 0.5 only fires in deep crises (~7% of months). NFCI > 0 fires too often (~35-40% of months) — would put T5 in a near-permanent firing state. NFCI > 0.25 fires in ~20-25% of months — captures the "meaningfully tighter than usual" regime without dominating the composite.

**Calibration sanity check** (to run after Step 3.6): T5 firing months should *qualitatively* line up with periods of genuine market stress — i.e., T5 should fire during 2008-09, 2020 COVID, 2002 dot-com, 2015-16 energy bust, and 2022 Fed tightening. T5 should *not* fire during clearly calm periods (2004-06, 2013-14, 2017-19). T5 and T2 are conceptually different (T5 = current-month stress, T2 = forward 6-month drawdown) so a fixed overlap percentage isn't meaningful; check the firing pattern visually instead. If T5 fires in <5% or >25% of all months, recalibrate the BAA10Y threshold up or down accordingly.

### Change C: Commit SP500 transformations as features

**Section 4.4 (Tier 4: Financial conditions)** — add two derived features alongside raw SP500:

| Feature | Derivation | Purpose |
|---|---|---|
| `SP500` | Raw monthly EOP close (existing) | Source data; T5 drawdown condition |
| **`SP500_RET_12M`** (new) | log(SP500_t / SP500_{t-12}) | Stationary equity-momentum signal for h=12 forecasts |
| **`SP500_DRAWDOWN_12M`** (new) | SP500_t / max(SP500_{t-11..t}) − 1 | Distance below 12-month rolling max; ranges in [−1, 0] |

**Implementation timing — IMPORTANT:** Both derived features are computed at **ingestion time** (new Step 3.7, before Step 4), not in Step 4's feature pipeline. They need to land in `features_monthly` so PCA, at-risk transforms, and Hamilton detrending (§4.13) can apply naturally as if they were primary features.

**Refactor opportunity:** `SP500_DRAWDOWN_12M` is mathematically identical to T2's drawdown computation (1 if drawdown ≤ −0.15 else 0 — T2 just thresholds it). After Step 3.7 ships, T2 should read `SP500_DRAWDOWN_12M` from `v_features_latest` instead of recomputing it inline in `ingest_targets`. This eliminates duplicated logic.

**Rationale:** The exploration showed raw SP500 fails the pre-recession behavior test (1/8 episodes ✓) because the trend dominates the variation. The two derived forms are stationary, more interpretable, and strictly more useful than raw price for any modeling. Committing them in the spec means they survive the Hamilton-detrending step in §4.13 as a known input, not an afterthought.

### Change D: Document feature horizon-target eligibility

**Section 4.16 (NEW)** — add a feature-routing table that codifies findings from the exploration:

| Feature | Best horizon (vs T1) | Best horizon (vs T2) | Routing |
|---|---|---|---|
| T10Y3M | h=12 (0.80) | h=6 (0.60) | T1-leading |
| DTWEXBGS | h=12 (0.80) | h=6 (0.69) | T1-leading + T5 secondary |
| NFCI | h=12 (0.80) | h=6 (0.74) | Both targets, all horizons |
| DRTSCILM | h=12 (0.73) | h=6 (0.83) | Both targets |
| INDPRO | h=12 (0.72) | h=6 (0.62) | T1-leading |
| REAL_FFR_GAP | h=12 (0.71) | h=6 (0.51) | T1-leading |
| EBP | h=12 (0.65) | h=6 (0.78) | **T5 primary**, T1 secondary |
| DCOILWTICO | h=12 (0.61) | h=6 (0.51) | T1-leading |
| NAPMPI | h=12 (0.59) | h=6 (0.66) | Both targets |
| ICSA | h=12 (0.59) | h=6 (0.51) | M5 (coincident) |
| JTSQUR | h=12 (0.58) | h=6 (0.59) | M5 (coincident) |
| BAA10Y | h=12 (0.52) | h=6 (0.74) | **T5 primary**, exclude h=12 vs T1 |
| SAHMREALTIME | h=12 (0.55) | h=6 (0.54) | M5 (coincident only) |
| CFNAI | h=12 (0.53) | h=6 (0.65) | M5 (coincident only) |

This table is illustrative based on existing 17 features. The 15 net-new v1.1.1 additions (4 labor: CCSA, AWHMAN, TEMPHELPS, JTSLDR; 2 housing: NAHB, EXHOSLUSM495S; 3 inflation: CPILFESL, PCEPILFE, CES0500000003; 2 term-structure: T10Y2Y, near-term forward; 1 diffusion breadth, 1 missingness flag, 2 SP500 derivations) get added with placeholder routing and the AUC values are populated as part of Step 4.

**Caveats on the routing table — read before using:**

- **DTWEXBGS small-sample risk.** DTWEXBGS data starts in 2006, covering only 2 NBER recessions (2008, 2020). Its AUC of 0.802 vs T1 should be treated as "high-but-thin" — the regularization in M4 should be aggressive on this feature, and M1/M2 (unregularized probit) should not weight it heavily. In a pre-2006 backtest fold the feature is unavailable; in post-2006 folds it's effectively fitting 2 events.

- **PERMIT and ISRATIO routing nuance.** Both have AUC < 0.55 vs T1 and < 0.55 vs T2. They go into M4 (where regularization handles weak features gracefully) but should be **excluded from M1 and M2** (unregularized linear models, where weak features add coefficient-noise without payoff). Tag in the registry: `model_eligible = ['M3', 'M4', 'M5']`.

- **Coincident vs. leading routing for labor features.** Per Change A, all labor features (Sahm, JTSQUR, ICSA, CCSA, AWHMAN, TEMPHELPS, JTSLDR) are routed to M5 (coincident regime model) and to M4 at h=0/h=1/h=3 only. M5 itself stays narrow per its original 4-input Stock-Watson/Chauvet-Piger spec (W875RX1, PAYEMS, INDPRO, CMRMTSPL); the labor features feed M4-short, not M5's factor inputs. This preserves M5's direct comparability with the published RECPROUSM156N benchmark.

**Rationale:** Without this routing, the model fitting in Steps 5–8 would treat all features symmetrically across all (target, horizon) cells. That's 17 × 5 × 4 = 340 cells — most of them noise. The routing reduces the search space dramatically and forces feature use to match where the signal actually lives.

---

## 2. Changes NOT made (and why)

To be explicit about what we considered but rejected:

| Considered | Decision | Reason |
|---|---|---|
| Drop BAA10Y entirely | **No** | Strong T5 predictor (0.739) — losing it costs the credit channel |
| Drop SAHMREALTIME / CFNAI / ICSA | **No** | Weak at h=12 but useful for M5; just don't use them at h=12 |
| Increase the T5 threshold count from 2-of-4 to 3-of-4 | **No** | 2-of-4 already gives sane firing pattern in time series; 3-of-4 too restrictive |
| Add T6 "credit-only stress" target | **No** | Redundant with T5; would split data too thin |
| Defer T2 entirely until Yahoo is more reliable | **No** | We have the CSV committed to git; reliability is now a non-issue |
| Re-enable T3 because Page 11 is rich | **No** | Page 11 confirmed to be a static playbook display (Section 5 of last gap-check) |
| Drop COPPER_GOLD definitively | **Yes** in v1.1.1 | Both gold series gone from FRED; v2 backlog item |

---

## 3. Implementation sequence

These changes fold into existing Step 3.5 / 3.6 / 4 work, plus one **new Step 3.7** for SP500 derived features.

| Step | Action |
|---|---|
| 3.5 (next) | Add the 13 originally-planned v1.1 features to ingestion (labor, housing, inflation, term-structure, engineered). Tag labor features with `eligible_horizons = ['h=0', 'h=1', 'h=3']` per Change A |
| 3.6 (next) | Implement T5 with v1.1.1 absolute thresholds (BAA10Y > 3.5%, NFCI > 0.25). Run sanity check: T5 should fire in 5-25% of all months and visually match known stress periods. |
| **3.7 (new)** | Compute `SP500_RET_12M` and `SP500_DRAWDOWN_12M` at ingestion time, store in `features_monthly`. Refactor T2 to read `SP500_DRAWDOWN_12M` from `v_features_latest` instead of recomputing inline. |
| 4 (later) | Hamilton detrending (§4.13) over the now-15 v1.1 features. PCA reduction. At-risk threshold transforms. Implement the §4.16 routing table as data in the feature registry. |
| 5–8 (later) | Models read `eligible_horizons` and skip cells that don't apply. Without the routing: 15 v1.1 features × 5 targets × 4 horizons + existing features = ~340 cells. With routing: ~150 effective fits. |
| Done check | Section 13's smoke-test list gets two items added: T5 fires in 5-25% of all months and visually matches known stress periods (2008-09, 2020, 2022, etc.); M5 ≈ Chauvet-Piger published RECPROUSM156N within 5pp |

---

## 4. Updated v1.1.1 changelog entry

To be appended to the main spec's §14:

> ### v1.1.1 (May 3, 2026)
>
> Patch revision after Step (b) exploration. Four data-justified refinements:
>
> - **Change A:** Labor features (Sahm, JTSQUR, ICSA + 4 v1.1 additions) reassigned from "all horizons primary" to M5/h=1/h=3 only. Coincident-by-design; weak at h=12.
> - **Change B:** T5 (Market Stress) thresholds switched from rolling-percentile to absolute (BAA10Y > 3.5%, NFCI > 0.25). Percentile logic almost never fired given the narrow BAA10Y range; absolute thresholds calibrated to fire in 5-25% of months matching known stress periods.
> - **Change C:** SP500 transformations (`SP500_RET_12M`, `SP500_DRAWDOWN_12M`) committed as features. Raw price is non-stationary; the model needs the derived forms.
> - **Change D:** Feature horizon-target routing documented as §4.16. Reduces cells to fit by ~50% and enforces "use each feature where it actually has signal."
>
> **No architecture changes.** All other v1.1.0 decisions stand: 5 targets, 5 model families (M1-M5), 4 horizons, 4 metrics, 5-tier alert taxonomy, confidence score formulation.

---

*End of v1.1.1 patch.*
