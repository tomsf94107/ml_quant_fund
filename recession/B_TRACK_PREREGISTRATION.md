# B-Track — Short-Horizon Recession Model Series

## Pre-Registration Document (v1.0)

**Status: PRE-REGISTERED. Written before any B-track model is run.**

This document locks the B-track design *before* results are seen. Nothing
below may be changed on the basis of a B-track result — that is the rule
that makes B-track a legitimate test rather than a fishing expedition.
Any change after the first run must be recorded as a dated amendment with
a reason, and the original kept.

---

## 1. Why B-Track Exists

A-track (the M1-M5 ladder at T1/h=12) concluded: the Treasury yield-curve
spread dominates at the 12-month horizon; no other model class robustly
beats it out-of-sample. Two independent diagnostics then pointed past
that result:

- The **horizon scan** found macro features — NFCI especially — come
  alive at *shorter* horizons (h=3, h=6), where the yield curve is weaker.
- The **nested likelihood-ratio test** confirmed NFCI and INDPRO carry
  statistically real signal at h=12 (p=0.0001) that the AUC delta was too
  underpowered to show — though that signal did not translate to OOS
  skill *at h=12*.

B-track tests the open question both raised: **at short horizons, does
the macro signal become real out-of-sample predictive skill?** A-track
answered "no" at h=12. B-track asks the same question at h=3 and h=6,
where the evidence says the answer might be "yes".

B-track is a SEPARATE, pre-registered experiment. The horizon scan was
the hypothesis generator; B-track is the test. Per the D+ anti-leakage
discipline, the scan's results may NOT be used to pick B-track's design
beyond what is fixed in this document.

---

## 2. Design Decisions (F1-F4) — LOCKED

### F1 — Horizons

The institutional **3-6-12 recession-horizon ladder** (NY Fed publishes
at 12 months; the Conference Board and most institutional recession
dashboards run 3-6-12; h=9 is skipped as decision-redundant between 6 and
12). h=12 is already complete from A-track. B-track adds **h=3 and h=6**.

Rationale: a horizon ladder is a layered alarm system, not a single
forecast — h=12 is the early signal (more lead, noisier), h=3 is the
confirmation signal (less lead, more reliable). Both are wanted.

### F2 — Models

| Horizon | Models run | Notes |
|---|---|---|
| h=3  | M1, M2, M3, M2-binary | M5 optional (regime model at near-coincident horizon) |
| h=6  | M1, M2, M3, M2-binary | |
| h=12 | **M2-binary only** | M1-M5 NOT re-run — A-track verdict stands; M2-binary added as a single targeted test |

Model definitions:
- **M1** — single-feature yield-curve probit (`[T10Y3M]`). The baseline.
  Non-negotiable: it is what B-track models must beat.
- **M2** — L2-regularized logit on the 4-feature set
  `[T10Y3M, NFCI, INDPRO, REAL_FFR_GAP]`. The model the horizon scan
  flagged as coming alive at short horizons.
- **M3** — random forest on the same feature set. One tree model, to test
  whether nonlinearity adds anything a linear model cannot.
- **M2-binary** — **the headline new idea.** M2's linear logit, but with
  the macro features passed through an **at-risk (binarized)
  transformation**: each predictor is converted to an indicator of an
  "unusually weak" state via a threshold estimated *from training data
  only*. Motivated by Billakanti & Shin (FRB Philadelphia, Dec 2025),
  which found binarized predictors consistently improve OOS recession
  forecasting and often make linear models competitive with ML methods,
  with gains concentrated around recession onset.

**M4 is deliberately excluded.** A-track found M4 (XGBoost) redundant
with M3 at h=12 — both tree models, M4 added nothing M3 did not. Re-running
it would be a duplicate angle and an extra multiple-comparison, not new
information.

**Why M2-binary instead of more model classes:** the FRB Philadelphia
result is explicit that once binarization is applied, additional
tree-based methods provide little incremental benefit, because the
transformation itself captures the relevant nonlinearity. The
highest-value new "angle" at short horizons is therefore a feature
transformation on the linear model — not a fifth model class.

### F3 — Success Criterion (pre-registered, primary + confirmatory)

**Primary criterion — OOS AUC, seed-stable.** A B-track model "beats the
baseline" at a horizon only if its walk-forward mean fold AUC exceeds
M1's at that horizon by more than the **pre-registered seed-noise band of
0.03** (the A-track standard; for stochastic models the seed strip must
confirm the edge survives reseeding). A numeric edge inside the band is
NOT a win — same rule that correctly rejected M3's and M4's apparent
h=12 wins.

**Confirmatory criterion — forecast encompassing test.** Following the
FRB Philadelphia (2025) encompassing approach: a probit regression of the
realized recession indicator on the log-odds of the two competing OOS
forecasts (M1 and the challenger). If the challenger's coefficient is
significant and M1's is not, the challenger contains information M1 does
not. This is reported alongside the AUC verdict; a B-track model is
considered a genuine improvement only if it passes the primary criterion
AND the encompassing test agrees.

Both criteria are fixed here, before any run. Neither threshold may move
on the basis of a result.

### F4 — Infrastructure

**Reuse, do not rebuild.** The walk-forward harness, the feature builder,
and the seed-stability strips are horizon-agnostic and already validated
(269/269 tests). B-track reuses them unchanged; the embargo simply
changes from 12 to the B-track horizon (the harness already takes the
horizon as a parameter). New code is minimal and limited to:
  - `features/at_risk.py` — the binarization transformation.
  - `models/m2_binary.py` — M2 with the at-risk transform.
  - `validation/b_track.py` — the driver + encompassing test + report.

No change to any A-track module.

---

## 3. Pre-Registered Hypotheses

Stated before any run, so the result is a test and not a story:

- **H1.** At h=3 and h=6, at least one of {M2, M3, M2-binary} beats the
  M1 yield-curve baseline on the primary OOS-AUC criterion. (The horizon
  scan and nested test predict this; B-track tests it OOS.)
- **H2.** M2-binary is the strongest challenger at short horizons —
  binarization converts the in-sample macro signal into OOS skill where
  plain continuous M2 could not.
- **H3 (h=12 control).** M2-binary does NOT beat M1 at h=12. If it does,
  the A-track conclusion is refined, not overturned (OOS skill is still
  the bar); if it does not, the A-track verdict is reconfirmed and
  strengthened.

A result that contradicts a hypothesis is reported as-is. The honest
outcomes — including "B-track also finds nothing, the yield curve wins at
every horizon" — are all acceptable findings.

---

## 4. What Would Falsify / Confirm

- **B-track succeeds** if H1 holds: a model robustly beats M1 OOS at a
  short horizon and passes the encompassing test. The deliverable then
  extends to a short-horizon model alongside M1.
- **B-track is null** if no model clears the pre-registered band at any
  horizon. That is still a publishable finding: it would mean the macro
  signal, though statistically real (nested test), never becomes
  generalizable OOS skill — at any horizon. The yield curve wins
  end-to-end.

Both are real results. B-track is designed so the null outcome is as
informative as the positive one.

---

## 5. Build Order

1. `features/at_risk.py` + tests — the binarization transform.
2. `models/m2_binary.py` + tests — M2 with the transform.
3. `validation/b_track.py` + tests — driver, encompassing test, report.
4. Run B-track on real data at h=3, h=6, h=12.
5. Report against the pre-registered criteria. Commit.

---

*Pre-registered. Do not edit on the basis of a result. Amendments must be
dated, reasoned, and appended — never overwrite.*
