# Experiment A — Candidate Feature Evaluation (Tight Pre-Registered Test)

## Pre-Registration Document (v1.0)

**Status: PRE-REGISTERED. Written before any Experiment-A model is run.**

This document locks Experiment A's design *before* results are seen.
Nothing below may be changed on the basis of an A result. Any change
after the first run must be recorded as a dated amendment, with the
original kept. This is the same discipline used for the B-track
pre-registration.

---

## 1. Why Experiment A Exists

The feature audit (Step C) established the project's central unused
asset: **25 features are populated with real data — decades deep — and
have never been tested.** Every model (M1-M5, M2-binary) uses the same
6-feature core. The audit's 29 "registry-only genuine gap" features,
minus the 4 empty `skip_v1` ones, are ~25 populated, untested series.

Experiment A does NOT test all 25. Testing 25 features and keeping
whatever scores best would be a multiple-comparison fishing expedition —
precisely the overfitting trap the project's whole discipline (seed
bands, the D+ rule, the B-track pre-registration) exists to prevent.

Experiment A is the **tight** test: a small, pre-registered set of the
strongest candidates, chosen by economic reasoning *before* any run,
each tested at the horizon where theory says it should matter. The
broader sweep of the remaining features is a SEPARATE, later experiment
(Experiment B) with its own pre-registration and an explicit
multiple-comparison correction. A is not "B with fewer features"; A is
the sharp confirmatory test, B is the broad exploratory one.

---

## 2. The Candidates — LOCKED (5 features)

Chosen by economic reasoning, not by peeking at results. Each is a
populated registry-only feature from the Step-C audit.

| Candidate | Tier | What it is | Why it is a strong candidate |
|---|---|---|---|
| `EBP` | yield_credit | Excess Bond Premium (Gilchrist-Zakrajsek 2012) — the component of corporate credit spreads not explained by default risk | The single most-cited modern recession predictor after the yield curve. Captures credit-market stress / risk appetite the curve does not. Used in Federal Reserve research. |
| `NEAR_TERM_FORWARD` | yield_credit | Near-term forward spread (the 6-quarter-ahead forward 3-month rate minus the current 3-month rate) | Engstrom-Sharpe (Federal Reserve, 2018) argue this spread *outperforms* the conventional 10y-3m term spread as a recession predictor. A direct, theory-grounded challenger to the project's M1 baseline. |
| `BAA10Y` | yield_credit | Moody's Seasoned Baa Corporate Bond Yield minus the 10-year Treasury yield | The project's *chosen* credit feature — it was selected in v1.0.2 to replace the deprecated `BAMLH0A0HYM2`. Yet it was never wired into any model. Testing it answers directly whether the v1.0.2 credit-feature decision delivered usable signal. |
| `ICSA` | labor | Initial unemployment claims (weekly, aggregated monthly) | The canonical short-horizon recession signal — claims turn up fast at cycle turns. B-track established that non-curve features dominate at SHORT horizons; claims are the textbook example. |
| `T10Y2Y` | yield_credit | 10-year minus 2-year Treasury spread | The other canonical yield-curve measure. Tests whether the 2-year point carries information beyond M1's 3-month point — a clean, well-defined question. |

**Why these five.** EBP and NEAR_TERM_FORWARD are the two features the
recession-prediction literature most directly pits against the term
spread. BAA10Y is the project's own chosen-but-unwired credit feature.
ICSA is the strongest short-horizon candidate and connects to the
B-track finding. T10Y2Y is the most direct "is M1's curve definition the
right one" check. A focused, theory-driven set — not a sweep.

**Data-depth caveat (pre-registered).** Candidate history lengths differ:
`EBP` from 1973, `ICSA` from 1967, `T10Y2Y` from 1976, `BAA10Y` from
1986, `NEAR_TERM_FORWARD` from 1983 (~518 monthly rows). NEAR_TERM_FORWARD
and BAA10Y have the shortest histories — their walk-forward AUC (Gate 2)
rests on fewer recessions and therefore carries wider uncertainty. A
borderline Gate-2 result for those two must be read with that in mind;
it is not a reason to discount them, but it is a pre-registered caveat.

**Explicitly NOT in Experiment A** (deferred to Experiment B): `CCSA`,
`SAHMREALTIME`, `CFNAI`, `NAPMPI`, the SP500 family, `UMCSENT`, the
inflation tier, `DCOILWTICO`, `DTWEXBGS`, `DRTSCILM`, `AWHMAN`,
`TEMPHELPS`, `JTSLDR`, `JTSQUR`, `COVID_DUMMY`. These are real candidates
but testing them belongs in the separately-corrected broad sweep.
`SAHMREALTIME` is specifically deferred because the Sahm rule is a
*coincident* recession identifier, not a *leading* predictor — testing
it at a forecast horizon tests it where it is not designed to work; it
belongs in B with that caveat.

---

## 3. Horizons — LOCKED

Each candidate is tested at the horizon(s) where theory says it should
matter. This is pre-registered to prevent horizon-shopping.

| Candidate | Horizon(s) tested | Rationale |
|---|---|---|
| `EBP` | h=12, h=6, h=3 | Credit stress propagates over months — it is neither a pure 12-month lead nor purely coincident. h=12 is the literature's standard recession horizon; h=6 is the B-track "handoff zone" where a credit-channel signal should be strongest; h=3 is the near-coincident check. |
| `NEAR_TERM_FORWARD` | h=12 | A direct rival to the term spread at the conventional 12-month forecast horizon. |
| `BAA10Y` | h=12 | A credit-stress signal evaluated at the standard 12-month recession horizon, directly comparable to the M1 baseline. |
| `ICSA` | h=3 | Claims are a SHORT-horizon signal. B-track established h=3 is where non-curve features win; testing claims at h=12 would test it where theory does not expect it to work. |
| `T10Y2Y` | h=12 | A yield-curve measure; same horizon as the M1 baseline it is compared to. |

Total: **7 (candidate, horizon) tests** — EBP×3, NEAR_TERM_FORWARD×1,
BAA10Y×1, ICSA×1, T10Y2Y×1. Seven is a small, pre-registered number; the
multiple-comparison exposure is low and is addressed in §4.

---

## 4. Success Criterion — LOCKED (two gates, both required)

A candidate is judged to carry real signal at a horizon only if it
passes BOTH:

**Gate 1 — statistical significance (nested likelihood-ratio test).**
A nested LR test: restricted model = M1 (T10Y3M only); full model =
M1 + the one candidate feature. Both unpenalised logits, fit on the
common rows where the candidate, T10Y3M, and the label are all present.
The candidate passes Gate 1 if the LR-test p-value is below the
**pre-registered, multiple-comparison-corrected threshold**:

    alpha = 0.05 / 7 = 0.00714   (Bonferroni, 7 pre-registered tests)

The Bonferroni correction is applied because A runs 7 tests; using a
plain 0.05 across 7 tests would inflate the family-wise false-positive
rate. 0.00714 is fixed here, before any run.

**Gate 2 — out-of-sample skill (walk-forward AUC).**
The candidate must also improve out-of-sample discrimination. Walk-forward
AUC of (M1 + candidate) must exceed walk-forward AUC of M1-alone by more
than the **pre-registered seed-noise band of 0.03** — the same band used
in A-track and B-track. In-sample significance without OOS improvement is
explicitly NOT a pass (this is exactly the nested-test-vs-OOS gap the
project already found for the macro features at h=12).

A candidate "carries real signal" only if it clears Gate 1 (corrected
significance) AND Gate 2 (OOS band). Either alone is not a pass.

---

## 5. Pre-Registered Hypotheses

Stated before any run:

- **HA1.** At h=12, at least one of `EBP`, `NEAR_TERM_FORWARD`, `T10Y2Y`
  passes both gates against the M1 baseline. (The literature predicts
  EBP and the near-term forward spread should.)
- **HA2.** `NEAR_TERM_FORWARD` is the strongest h=12 challenger —
  consistent with Engstrom-Sharpe's claim that it outperforms the term
  spread.
- **HA3.** `ICSA` passes both gates at h=3 — claims are a genuine
  short-horizon signal, consistent with the B-track finding that
  non-curve features dominate at short horizons.

A result contradicting a hypothesis is reported as-is. The honest null
outcome — "no candidate beats the M1 baseline on both gates" — is a valid
and publishable finding: it would mean the yield curve, as M1 uses it,
already captures the available signal.

---

## 6. What Would Falsify / Confirm

- **A succeeds** for a candidate if it clears both gates. That candidate
  then becomes a justified addition to a model's feature set — wired in,
  re-tested, documented (the project's "built but not wired" standard).
- **A is null** if no candidate clears both gates. That is still a real
  result: it would mean the project's 6-feature core is not obviously
  improvable by the four strongest named rivals, and it would refocus
  effort on the B sweep or on model structure rather than features.

---

## 7. Method Notes — to prevent leakage

- The nested LR test and the walk-forward harness are reused unchanged
  except that the candidate feature set is now a parameter (the existing
  `nested_lr_test` hard-codes the 4-feature set; Experiment A adds a
  generalised version that takes `[T10Y3M, <candidate>]`).
- All point-in-time discipline is inherited: features are loaded through
  `build_feature_dataframe` with `as_of`/`train_cutoff`, vintage-aware;
  the walk-forward embargo is unchanged.
- Each (candidate, horizon) test runs on its own common-rows axis (the
  months where that candidate, T10Y3M, and the label are all present) —
  candidates have different history lengths (`NEAR_TERM_FORWARD` starts
  1983, `EBP` 1973, etc.), so axes are not shared across candidates. The
  M1 baseline is re-fit on each candidate's axis for an exact pairwise
  comparison.

---

## 8. Build Order

1. `validation/candidate_eval.py` — a generalised nested LR test
   (baseline vs baseline+candidate) + walk-forward AUC comparison + the
   two-gate verdict. Reuses `nested_test` and `walk_forward` internals.
2. Tests for `candidate_eval.py` (mock DB, known-answer).
3. Run Experiment A on the real DB for the 7 pre-registered tests.
4. Report against the two pre-registered gates. Commit.

---

*Pre-registered. Do not edit on the basis of a result. Amendments must be
dated, reasoned, and appended — never overwrite.*
