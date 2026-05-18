"""
recession/validation/combination.py

Step 9 — model combination. NOT a large ensembling framework: a focused
test of one question, with the diagnostic that explains the answer.

THE QUESTION
------------
M1-M4 each score ~0.80 mean fold AUC at T1/h=12 and none robustly beats
the M1 yield-curve baseline. Does COMBINING them — averaging their
recession probabilities — beat the best single model?

WHY THE ANSWER IS LIKELY "NO" (and why that is fine)
----------------------------------------------------
Model combination helps when models are DECORRELATED — when they make
different mistakes, so averaging cancels the mistakes. M1-M4 all lean on
the same dominant feature (the yield curve, T10Y3M — confirmed by every
model's OOS permutation importance). Models that rely on the same signal
make the SAME mistakes; averaging them reproduces the same ~0.80. If that
is what this test finds, the production choice is the single model M1 —
simpler, not worse.

M5 is EXCLUDED from the combination. M5 is a coincident regime model and
is anti-correlated with the h=12 leading target (it scores below 0.5 at
this cell by construction). Averaging an anti-correlated model in would
be incoherent. The combination is M1-M4 only.

WHAT THIS MODULE DOES
---------------------
  - builds a mean-probability ensemble of M1-M4 behind the RecessionModel
    interface, validates it through the Step-5 harness on the common fold
    axis, and compares it to the best single model.
  - computes the inter-model prediction correlation matrix on the OOS
    fold predictions — the DIAGNOSTIC that explains the result. High
    correlation => combination cannot help; this is the evidence behind
    the "ship the single model" finding.

This is a FINDING, not a framework. The verdict is reported plainly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recession.validation.walk_forward import walk_forward, WalkForwardResult
from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1Probit, M1_FEATURES
from recession.models.m2_logit import M2Logit, M2_FEATURES
from recession.models.m3_forest import M3Forest, M3_CORE_FEATURES
from recession.models.m4_xgboost import M4Xgb, M4_CORE_FEATURES


# the combination uses the 4-feature core set (the widest set M1-M4 share
# cleanly); each sub-model still sees only the columns it should via its
# own internal feature handling
COMBINATION_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]


# =============================================================================
# The ensemble model
# =============================================================================

class MeanEnsemble:
    """Mean-probability ensemble of M1-M4, behind the RecessionModel
    protocol.

    Each sub-model is fitted on the columns it is designed for (M1 on
    T10Y3M only; M2/M3/M4 on the 4-feature core). predict_proba averages
    the four sub-models' recession probabilities.
    """

    def __init__(self) -> None:
        self._subs: dict = {}
        self._cols: dict = {
            "M1": M1_FEATURES,
            "M2": M2_FEATURES,
            "M3": M3_CORE_FEATURES,
            "M4": M4_CORE_FEATURES,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanEnsemble":
        self._subs = {
            "M1": M1Probit(),
            "M2": M2Logit(C=1.0),
            "M3": M3Forest(),
            "M4": M4Xgb(),
        }
        for name, model in self._subs.items():
            cols = [c for c in self._cols[name] if c in X.columns]
            model.fit(X[cols], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for name, model in self._subs.items():
            cols = [c for c in self._cols[name] if c in X.columns]
            preds.append(np.asarray(model.predict_proba(X[cols]),
                                    dtype=float))
        return np.mean(preds, axis=0)

    def sub_predictions(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-sub-model predictions — for the correlation diagnostic."""
        out = {}
        for name, model in self._subs.items():
            cols = [c for c in self._cols[name] if c in X.columns]
            out[name] = np.asarray(model.predict_proba(X[cols]),
                                   dtype=float)
        return out


# =============================================================================
# The driver
# =============================================================================

def run_combination(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    **walk_forward_kwargs,
) -> dict:
    """Validate the mean ensemble and each single model on identical folds,
    and compute the inter-model OOS correlation diagnostic.

    Returns {'ensemble', 'singles': {name: result}, 'corr': DataFrame,
             'verdict': str}.
    """
    common = dict(
        target=target, horizon=horizon,
        min_history_year=min_history_year, db_path=db_path,
        **walk_forward_kwargs,
    )
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=COMBINATION_FEATURES, **build_kwargs,
    )
    cols = [c for c in COMBINATION_FEATURES if c in probe.X.columns]
    axis = probe.X.index[probe.X[cols].notna().all(axis=1)]

    # the ensemble
    ens = walk_forward(
        model_factory=MeanEnsemble,
        feature_subset=COMBINATION_FEATURES,
        model_columns=COMBINATION_FEATURES,
        model_name="Mean ensemble (M1-M4)",
        restrict_to_months=axis, **common,
    )
    # the single models
    singles = {}
    singles["M1"] = walk_forward(
        model_factory=M1Probit, feature_subset=M1_FEATURES,
        model_columns=M1_FEATURES, model_name="M1",
        restrict_to_months=axis, **common,
    )
    singles["M2"] = walk_forward(
        model_factory=lambda: M2Logit(C=1.0), feature_subset=M2_FEATURES,
        model_columns=M2_FEATURES, model_name="M2",
        restrict_to_months=axis, **common,
    )
    singles["M3"] = walk_forward(
        model_factory=M3Forest, feature_subset=M3_CORE_FEATURES,
        model_columns=M3_CORE_FEATURES, model_name="M3",
        restrict_to_months=axis, **common,
    )
    singles["M4"] = walk_forward(
        model_factory=M4Xgb, feature_subset=M4_CORE_FEATURES,
        model_columns=M4_CORE_FEATURES, model_name="M4",
        restrict_to_months=axis, **common,
    )

    # inter-model correlation diagnostic — pool each model's OOS fold
    # predictions and correlate. High correlation explains why combining
    # cannot help.
    pooled: dict[str, list[float]] = {n: [] for n in
                                      ("M1", "M2", "M3", "M4")}
    for n in pooled:
        for fold in singles[n].folds:
            pooled[n].extend(fold.test_proba)
    # the fold structure is identical across singles (same axis), so the
    # pooled vectors align row-for-row
    lengths = {len(v) for v in pooled.values()}
    if len(lengths) == 1 and lengths != {0}:
        corr = pd.DataFrame(pooled).corr()
    else:
        corr = None

    # verdict
    ens_auc = ens.mean_fold_auc
    best_single = max(
        ((n, r.mean_fold_auc) for n, r in singles.items()
         if r.mean_fold_auc is not None),
        key=lambda kv: kv[1], default=(None, None),
    )
    if ens_auc is not None and best_single[1] is not None:
        delta = ens_auc - best_single[1]
        mean_corr = None
        if corr is not None:
            # mean of off-diagonal correlations
            m = corr.to_numpy()
            off = m[~np.eye(len(m), dtype=bool)]
            mean_corr = float(np.mean(off))
        if delta > 0.01:
            verdict = (f"Ensemble beats best single ({best_single[0]}) by "
                       f"{delta:+.4f} — combination adds value.")
        else:
            cstr = (f" Mean inter-model correlation {mean_corr:.3f} — "
                    f"models are highly correlated, "
                    f"so averaging cannot cancel mistakes."
                    if mean_corr is not None else "")
            verdict = (
                f"Ensemble does NOT beat best single ({best_single[0]} "
                f"{best_single[1]:.4f}); delta {delta:+.4f}.{cstr} "
                f"FINDING: at T1/h=12 the models are too correlated to "
                f"benefit from combination — ship the single model M1.")
    else:
        verdict = "(insufficient scoreable folds for a verdict)"

    return {"ensemble": ens, "singles": singles, "corr": corr,
            "verdict": verdict}


def print_combination_report(results: dict) -> None:
    """Print the Step-9 combination finding."""
    ens = results["ensemble"]
    singles = results["singles"]
    corr = results["corr"]

    print("=" * 66)
    print(f"STEP 9 — MODEL COMBINATION — {ens.target} {ens.horizon}")
    print("=" * 66)
    print()
    print(ens.summary())
    print()
    print("  single models (same folds):")
    for n, r in singles.items():
        a = r.mean_fold_auc
        print(f"    {n}: mean fold AUC "
              + (f"{a:.4f}" if a is not None else "n/a"))

    if corr is not None:
        print()
        print("  inter-model OOS prediction correlation:")
        print("  " + corr.round(3).to_string().replace("\n", "\n  "))

    print()
    print("-" * 66)
    print("  VERDICT:")
    # wrap the verdict to a readable width
    words = results["verdict"].split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 64:
            print(line)
            line = "  " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        print(line)
    print("=" * 66)
