"""
recession/validation/candidate_eval.py

Experiment A — candidate feature evaluation.

Implements the design locked in A_PREREGISTRATION.md. Tests whether a
candidate feature carries real recession signal BEYOND the M1 yield-curve
baseline, via two pre-registered gates that BOTH must pass:

  GATE 1 — statistical significance (nested likelihood-ratio test).
    Restricted model = logit on [T10Y3M]; full model = logit on
    [T10Y3M, candidate]. LR statistic = 2*(llf_full - llf_restricted),
    df = 1, p-value from the chi-square survival function. The candidate
    passes Gate 1 if p < the pre-registered Bonferroni-corrected
    threshold ALPHA_CORRECTED (0.05 / 7 tests = 0.00714).

  GATE 2 — out-of-sample skill (walk-forward AUC).
    Walk-forward AUC of (M1 + candidate) must exceed walk-forward AUC of
    M1-alone by more than the pre-registered seed-noise band (0.03).
    In-sample significance without OOS improvement is NOT a pass — that
    is exactly the nested-test-vs-OOS gap the project already found.

A candidate "carries real signal" at a horizon only if it clears BOTH
gates. Either alone is not a pass.

Reuse: the nested test logic mirrors recession.validation.nested_test
(generalised here to an arbitrary single candidate); the OOS comparison
uses the standard walk_forward harness with M1Probit. No new modeling.
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
from recession.validation.walk_forward import walk_forward
from recession.models.m1_probit import M1Probit


BASELINE_FEATURE = "T10Y3M"

# pre-registered constants (A_PREREGISTRATION.md section 4)
ALPHA_CORRECTED = 0.05 / 7      # Bonferroni, 7 pre-registered tests
SEED_NOISE_BAND = 0.03         # the A-track / B-track OOS band

# the 7 pre-registered (candidate, horizon) tests — LOCKED
PREREGISTERED_TESTS = [
    ("EBP", "h=12"),
    ("EBP", "h=6"),
    ("EBP", "h=3"),
    ("NEAR_TERM_FORWARD", "h=12"),
    ("BAA10Y", "h=12"),
    ("ICSA", "h=3"),
    ("T10Y2Y", "h=12"),
]


# =============================================================================
# Gate 1 — nested likelihood-ratio test
# =============================================================================

def _fit_logit(X: pd.DataFrame, y: pd.Series):
    """Unpenalised logit. Returns the statsmodels result, or None on
    failure / degenerate target. (Mirrors nested_test._fit_logit.)"""
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


def gate1_nested_lr(
    candidate: str,
    target: str,
    horizon: str,
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict:
    """Gate 1: nested LR test, M1 (T10Y3M) vs M1 + candidate.

    Both logits are fit on the common rows where T10Y3M, the candidate,
    and the label are all present. Returns the LR statistic, df=1, the
    p-value, and whether it clears ALPHA_CORRECTED.
    """
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year

    feats = [BASELINE_FEATURE, candidate]
    try:
        fr = build_feature_dataframe(
            target=target, horizon=horizon,
            as_of="today", train_cutoff="today",
            feature_subset=feats, **build_kwargs,
        )
    except Exception as e:
        return {"gate": 1, "error":
                f"could not build features for {feats}: {e}"}
    cols = [c for c in feats if c in fr.X.columns]
    if BASELINE_FEATURE not in cols or candidate not in cols:
        return {"gate": 1, "error":
                f"missing column(s): need {feats}, got {cols}"}

    # common rows: both features and the label present
    mask = fr.X[cols].notna().all(axis=1) & fr.y.notna()
    n = int(mask.sum())
    if n < 60:
        return {"gate": 1, "error":
                f"only {n} common rows — too few for a nested test"}

    X = fr.X.loc[mask, cols]
    y = fr.y.loc[mask]

    restricted = _fit_logit(X[[BASELINE_FEATURE]], y)
    full = _fit_logit(X[[BASELINE_FEATURE, candidate]], y)
    if restricted is None or full is None:
        return {"gate": 1, "error":
                "a nested logit failed to fit (degenerate target?)"}

    lr_stat = 2.0 * (full.llf - restricted.llf)
    df = 1
    p_value = float(stats.chi2.sf(lr_stat, df))
    # the candidate's own Wald p in the full model
    wald_p = (float(full.pvalues.get(candidate, np.nan))
              if candidate in full.pvalues.index else np.nan)

    passes = p_value < ALPHA_CORRECTED
    return {
        "gate": 1, "n": n,
        "lr_stat": float(lr_stat), "df": df, "p_value": p_value,
        "wald_p": wald_p,
        "alpha": ALPHA_CORRECTED, "passes": passes,
    }


# =============================================================================
# Gate 2 — out-of-sample walk-forward AUC
# =============================================================================

def gate2_oos_auc(
    candidate: str,
    target: str,
    horizon: str,
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **walk_forward_kwargs,
) -> dict:
    """Gate 2: walk-forward AUC of (M1 + candidate) vs M1-alone, on a
    shared fold axis. Passes if the AUC edge exceeds SEED_NOISE_BAND.
    """
    feats = [BASELINE_FEATURE, candidate]

    # shared axis: months where T10Y3M AND the candidate are both present
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    probe = None
    try:
        probe = build_feature_dataframe(
            target=target, horizon=horizon,
            as_of="today", train_cutoff="today",
            feature_subset=feats, **build_kwargs,
        )
    except Exception as e:
        return {"gate": 2, "error":
                f"could not build features for {feats}: {e}"}
    cols = [c for c in feats if c in probe.X.columns]
    if BASELINE_FEATURE not in cols or candidate not in cols:
        return {"gate": 2, "error":
                f"missing column(s): need {feats}, got {cols}"}
    axis = probe.X.index[probe.X[cols].notna().all(axis=1)]

    common = dict(target=target, horizon=horizon,
                  min_history_year=min_history_year, db_path=db_path,
                  restrict_to_months=axis, **walk_forward_kwargs)

    # M1 baseline on its own
    base = walk_forward(
        model_factory=M1Probit, feature_subset=[BASELINE_FEATURE],
        model_columns=[BASELINE_FEATURE],
        model_name=f"M1 ({horizon})", **common,
    )
    # M1 + candidate (M1Probit is a generic logit — it uses the columns given)
    augmented = walk_forward(
        model_factory=M1Probit, feature_subset=feats,
        model_columns=feats,
        model_name=f"M1+{candidate} ({horizon})", **common,
    )

    a_base = base.mean_fold_auc
    a_aug = augmented.mean_fold_auc
    if a_base is None or a_aug is None:
        return {"gate": 2, "error": "insufficient scoreable folds",
                "baseline_auc": a_base, "augmented_auc": a_aug}

    edge = a_aug - a_base
    passes = edge > SEED_NOISE_BAND
    return {
        "gate": 2,
        "baseline_auc": a_base, "augmented_auc": a_aug,
        "edge": edge, "band": SEED_NOISE_BAND, "passes": passes,
        "n_folds": base.n_folds,
    }


# =============================================================================
# the driver — both gates, one candidate/horizon
# =============================================================================

def evaluate_candidate(
    candidate: str,
    horizon: str,
    *,
    target: str = "T1",
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **walk_forward_kwargs,
) -> dict:
    """Run BOTH gates for one (candidate, horizon). A candidate carries
    real signal only if both gates pass."""
    g1 = gate1_nested_lr(candidate, target, horizon,
                         db_path=db_path, min_history_year=min_history_year)
    g2 = gate2_oos_auc(candidate, target, horizon,
                       db_path=db_path, min_history_year=min_history_year,
                       **walk_forward_kwargs) if "error" not in g1 \
        else {"gate": 2, "error": "skipped — Gate 1 errored"}

    g1_pass = bool(g1.get("passes"))
    g2_pass = bool(g2.get("passes"))
    both = g1_pass and g2_pass

    if "error" in g1 or "error" in g2:
        verdict = ("INCONCLUSIVE — a gate could not be evaluated: "
                   + "; ".join(x["error"] for x in (g1, g2)
                                if "error" in x))
    elif both:
        verdict = (f"PASS — {candidate} at {horizon} clears BOTH gates "
                   f"(LR p={g1['p_value']:.5f} < {ALPHA_CORRECTED:.5f}; "
                   f"OOS edge {g2['edge']:+.4f} > {SEED_NOISE_BAND}). It "
                   f"carries recession signal beyond the M1 yield curve.")
    elif g1_pass and not g2_pass:
        verdict = (f"NO — {candidate} at {horizon} is statistically "
                   f"significant (LR p={g1['p_value']:.5f}) but does NOT "
                   f"improve OOS skill (edge {g2['edge']:+.4f}, within the "
                   f"{SEED_NOISE_BAND} band). In-sample signal that does "
                   f"not generalise — not a pass.")
    elif g2_pass and not g1_pass:
        verdict = (f"NO — {candidate} at {horizon} shows an OOS edge "
                   f"({g2['edge']:+.4f}) but fails the corrected "
                   f"significance bar (LR p={g1['p_value']:.5f} "
                   f">= {ALPHA_CORRECTED:.5f}). Not a pass.")
    else:
        verdict = (f"NO — {candidate} at {horizon} fails both gates "
                   f"(LR p={g1.get('p_value', float('nan')):.5f}; "
                   f"OOS edge {g2.get('edge', float('nan')):+.4f}). No "
                   f"signal beyond the M1 yield curve.")

    return {
        "candidate": candidate, "horizon": horizon,
        "gate1": g1, "gate2": g2,
        "gate1_pass": g1_pass, "gate2_pass": g2_pass,
        "passes_both": both, "verdict": verdict,
    }


def run_experiment_a(
    *,
    target: str = "T1",
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **walk_forward_kwargs,
) -> dict:
    """Run all 7 pre-registered (candidate, horizon) tests."""
    results = []
    for candidate, horizon in PREREGISTERED_TESTS:
        results.append(evaluate_candidate(
            candidate, horizon, target=target, db_path=db_path,
            min_history_year=min_history_year, **walk_forward_kwargs))
    return {"results": results,
            "alpha": ALPHA_CORRECTED, "band": SEED_NOISE_BAND}


def print_experiment_a_report(report: dict) -> None:
    """Print the Experiment-A report against the two pre-registered gates."""
    print("=" * 74)
    print("EXPERIMENT A — CANDIDATE FEATURE EVALUATION")
    print("=" * 74)
    print(f"  Gate 1 (significance): nested LR test, p < "
          f"{report['alpha']:.5f}  (Bonferroni 0.05/7)")
    print(f"  Gate 2 (OOS skill):    walk-forward AUC edge > "
          f"{report['band']}")
    print(f"  a candidate passes only if it clears BOTH gates")
    print()
    print(f"  {'candidate':>18} {'horizon':>8} {'LR p':>10} "
          f"{'OOS edge':>10} {'G1':>4} {'G2':>4} {'verdict':>9}")
    print("  " + "-" * 70)
    for r in report["results"]:
        g1, g2 = r["gate1"], r["gate2"]
        p = g1.get("p_value")
        p_s = f"{p:.5f}" if p is not None else "err"
        edge = g2.get("edge")
        edge_s = f"{edge:+.4f}" if edge is not None else "err"
        g1m = "PASS" if r["gate1_pass"] else "no"
        g2m = "PASS" if r["gate2_pass"] else "no"
        v = ("PASS" if r["passes_both"]
             else "INCONCL." if "INCONCLUSIVE" in r["verdict"]
             else "no")
        print(f"  {r['candidate']:>18} {r['horizon']:>8} {p_s:>10} "
              f"{edge_s:>10} {g1m:>4} {g2m:>4} {v:>9}")

    # full verdicts
    print()
    print("-" * 74)
    for r in report["results"]:
        print(f"  {r['candidate']} @ {r['horizon']}:")
        words = r["verdict"].split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line); line = "    " + w
            else:
                line += (" " if line.strip() else "") + w
        if line.strip():
            print(line)
        print()

    # overall
    passed = [r for r in report["results"] if r["passes_both"]]
    print("=" * 74)
    print("  OVERALL — EXPERIMENT A")
    if passed:
        for r in passed:
            print(f"  PASS: {r['candidate']} @ {r['horizon']} — carries "
                  f"signal beyond the M1 yield curve. Justified to wire "
                  f"into a model (then re-test + document).")
    else:
        print("  No candidate clears both pre-registered gates. The M1")
        print("  yield curve, as used, already captures the available")
        print("  signal among the five strongest named rivals. This is")
        print("  the pre-registered null outcome — a valid finding.")
    print("=" * 74)
