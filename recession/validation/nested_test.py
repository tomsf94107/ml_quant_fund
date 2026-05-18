"""
recession/validation/nested_test.py

v1.1.3 — the nested likelihood-ratio test.

WHY THIS MODULE EXISTS
----------------------
The M-ladder concluded that M2 (4 features) does not beat M1 (yield curve
alone) at T1/h=12, on the evidence of an AUC delta (M2 0.777 vs M1 0.796).
A research review of the nested-model literature flagged a real weakness
in that reasoning:

  When comparing two NESTED models (one's features are a subset of the
  other's), an AUC-difference test is known to be EXCESSIVELY CONSERVATIVE
  and LOW-POWERED. The standard references (Demler/Pencina/D'Agostino;
  Vickers et al.) show an added predictor can be genuinely associated with
  the outcome yet produce no significant AUC gain — and that the DeLong
  AUC test is the wrong tool for nested models, because under the null its
  distribution is not even normal. The correct tool is a likelihood-ratio
  (or Wald) test on the added coefficients in the underlying regression.

So "the macro features add nothing at h=12" deserves a PROPER test, not an
eyeballed AUC delta. This module provides it.

WHY IT DOES NOT REUSE M1 / M2
-----------------------------
A valid likelihood-ratio test requires:
  1. UNPENALISED maximum-likelihood fits. The LR statistic's chi-square
     null distribution assumes plain ML estimates; a penalised
     (regularised) fit breaks that assumption. M2 is L2-regularised
     (sklearn LogisticRegression) — so M2's fit cannot be used.
  2. The restricted model genuinely NESTED in the full model — same link
     function, restricted = full with the extra coefficients set to zero.
     M1 is a PROBIT, M2 is a LOGIT — different families, not nested.

Therefore this module fits its OWN nested pair, purpose-built for the test:
  - RESTRICTED: unpenalised logit on [T10Y3M] only.
  - FULL:       unpenalised logit on [T10Y3M, NFCI, INDPRO, REAL_FFR_GAP].
Both unpenalised, both logit => genuinely nested => the LR statistic is
chi-square with df = 3 (the three added features) under the null
"the three macro features carry no signal beyond the yield curve".

This is a DIAGNOSTIC, separate from the M-ladder. It does not change M1-M5
or any production choice. It tells you, with a valid test, whether the
project's "features add nothing at h=12" conclusion holds up.

INTERPRETATION
--------------
  p >= 0.05 : fail to reject the null. The three macro features do not
              carry detectable signal beyond the yield curve at h=12 —
              this CONFIRMS the M-ladder conclusion with a proper test.
  p <  0.05 : reject the null. The features DO carry joint signal that a
              linear model can use — even though the AUC delta did not
              show it. This would mean the AUC comparison was masking a
              real (if small) effect, exactly the nested-model pitfall.

The test is run on the SAME walk-forward training discipline is NOT used
here — this is an in-sample full-history nested test, the standard setting
for a likelihood-ratio test of coefficient significance. (Out-of-sample
skill is the M-ladder's job; this module answers a different question:
"is the association statistically real".)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm

from recession.features.builder import build_feature_dataframe


# the nested pair: restricted is a strict subset of full
RESTRICTED_FEATURES = ["T10Y3M"]
FULL_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]
ADDED_FEATURES = ["NFCI", "INDPRO", "REAL_FFR_GAP"]   # full minus restricted


def _fit_logit(X: pd.DataFrame, y: pd.Series):
    """Fit an unpenalised logit. Returns the statsmodels result, or None
    if it fails to converge / the target is degenerate."""
    y_arr = np.asarray(y, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return None
    Xd = sm.add_constant(X, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.Logit(y_arr, Xd).fit(disp=0, maxiter=200)
        if not np.all(np.isfinite(result.params)):
            return None
        return result
    except Exception:
        return None


def nested_lr_test(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict:
    """Likelihood-ratio test: do the 3 macro features carry joint signal
    beyond the yield curve at the given target/horizon?

    Fits the nested pair (restricted: T10Y3M only; full: 4 features) on
    the common rows where all 4 features and the label are present, both
    unpenalised logits, and computes:

        LR statistic = 2 * (llf_full - llf_restricted)
        df           = 3   (the three added features)
        p-value      = chi-square survival function

    Returns a dict with the statistic, df, p-value, both log-likelihoods,
    the per-feature Wald p-values from the full model, and a verdict.
    """
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year

    fr = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=FULL_FEATURES, **build_kwargs,
    )
    X_all = fr.X[FULL_FEATURES]
    y_all = fr.y
    # common rows: every feature present AND label present
    mask = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[mask]
    y = y_all.loc[mask].astype(int)

    if len(X) < 30:
        return {"error": f"only {len(X)} usable rows — too few for the test"}

    # standardise features (numerical stability; does not affect the LR
    # statistic, which is invariant to linear reparameterisation)
    Xz = (X - X.mean()) / X.std(ddof=0)

    restricted = _fit_logit(Xz[RESTRICTED_FEATURES], y)
    full = _fit_logit(Xz[FULL_FEATURES], y)

    if restricted is None or full is None:
        return {"error": "a nested model failed to converge — test "
                          "cannot be computed"}

    # both fits MUST be on the same rows for the LR test to be valid
    if int(restricted.nobs) != int(full.nobs):
        return {"error": f"row mismatch (restricted {int(restricted.nobs)} "
                          f"vs full {int(full.nobs)}) — test invalid"}

    llf_r = float(restricted.llf)
    llf_f = float(full.llf)
    lr_stat = 2.0 * (llf_f - llf_r)
    df = len(ADDED_FEATURES)
    # the full model must fit at least as well as the restricted one;
    # tiny negative values can occur from convergence noise — clamp.
    if lr_stat < 0:
        lr_stat = 0.0
    p_value = float(stats.chi2.sf(lr_stat, df))

    # per-feature Wald p-values from the full model (which of the added
    # features, if any, is individually significant)
    wald_p = {}
    for feat in ADDED_FEATURES:
        if feat in full.pvalues.index:
            wald_p[feat] = float(full.pvalues[feat])

    reject = p_value < 0.05
    if reject:
        verdict = (
            f"REJECT the null (p={p_value:.4f}). The three macro features "
            f"carry joint signal beyond the yield curve that a linear "
            f"model can use — even though the M-ladder's AUC delta did "
            f"not show it. This is the known nested-model pitfall: an AUC "
            f"comparison is underpowered for nested models. The features "
            f"are statistically associated; whether that translates to "
            f"out-of-sample skill is a separate question the M-ladder "
            f"already addressed (it did not).")
    else:
        verdict = (
            f"FAIL TO REJECT the null (p={p_value:.4f}). The three macro "
            f"features do not carry detectable joint signal beyond the "
            f"yield curve at {target}/{horizon}. This CONFIRMS the "
            f"M-ladder conclusion — 'the extra features add nothing at "
            f"h=12' — with a proper likelihood-ratio test, not just an "
            f"AUC delta. The conclusion now rests on the correct tool.")

    return {
        "n_obs": int(full.nobs),
        "llf_restricted": llf_r,
        "llf_full": llf_f,
        "lr_statistic": lr_stat,
        "df": df,
        "p_value": p_value,
        "wald_p_values": wald_p,
        "reject_null": reject,
        "verdict": verdict,
        "target": target,
        "horizon": horizon,
    }


def print_nested_test_report(result: dict) -> None:
    """Print the nested likelihood-ratio test report."""
    print("=" * 70)
    print("NESTED LIKELIHOOD-RATIO TEST — do the macro features carry")
    print("signal beyond the yield curve?")
    print("=" * 70)
    if result.get("error"):
        print(f"  {result['error']}")
        print("=" * 70)
        return

    print(f"  target / horizon : {result['target']} {result['horizon']}")
    print(f"  observations     : {result['n_obs']}")
    print(f"  restricted model : logit on [T10Y3M]")
    print(f"  full model       : logit on [T10Y3M, NFCI, INDPRO, "
          f"REAL_FFR_GAP]")
    print()
    print(f"  log-likelihood (restricted): {result['llf_restricted']:.3f}")
    print(f"  log-likelihood (full)      : {result['llf_full']:.3f}")
    print(f"  LR statistic = 2*(llf_full - llf_restricted) = "
          f"{result['lr_statistic']:.4f}")
    print(f"  reference distribution: chi-square, df = {result['df']}")
    print(f"  p-value: {result['p_value']:.4f}")
    print()
    if result["wald_p_values"]:
        print("  per-feature Wald p-values (full model) — which added")
        print("  feature, if any, is individually significant:")
        for feat, p in result["wald_p_values"].items():
            mark = " *" if p < 0.05 else ""
            print(f"    {feat:>14}: p={p:.4f}{mark}")
    print()
    print("-" * 70)
    # wrap the verdict
    words = result["verdict"].split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 68:
            print(line)
            line = "  " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        print(line)
    print("=" * 70)
