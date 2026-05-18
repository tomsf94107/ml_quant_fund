"""
recession/models/m2_logit.py

M2 — the regularized multi-feature logit. The second rung of the model
ladder. M1 established that an UNREGULARIZED probit cannot use more than
the single yield-curve feature (the Estrella result: extra correlated
features fit in-sample noise and degrade OOS — M1 mean fold AUC 0.796
single-feature vs 0.771 four-feature).

M2 ASKS THE QUESTION M1 LEFT OPEN
---------------------------------
"Do the extra features pay off IF the model is regularized?" M2 is a
logistic regression with an L2 penalty on the four §4.16 T1-leading
features. The L2 penalty shrinks coefficients toward zero, which is
exactly the medicine for the overfitting that sank M1-extended. If M2
beats M1's 0.796 baseline, regularization rescued the features. If it
does not, even regularized linear models cannot beat the yield curve here
and the next gain must come from nonlinearity (M3/M4).

M2's FEATURE SET (§4.16-compliant)
----------------------------------
    T10Y3M         yield curve          (T1-leading, h=12 AUC 0.80)
    NFCI           financial conditions (T1-leading, h=12 AUC 0.80)
    INDPRO         real activity         (T1-leading, h=12 AUC 0.72)
    REAL_FFR_GAP   monetary stance       (T1-leading, h=12 AUC 0.71)

Same four features as the M1-extended variant — deliberately. M1-extended
failed them unregularized; M2 tests whether L2 rescues the same set.
Excluded per §4.16 (unchanged from M1): DTWEXBGS (thin), EBP / BAA10Y
(T5-primary), labor features (coincident), PERMIT / ISRATIO (M3/M4/M5).

REGULARIZATION STRENGTH — DECISION (i+), May 2026
-------------------------------------------------
C (sklearn's inverse-regularization-strength) is fixed a priori, NOT
searched. C is selected on the test set by nobody — that would be
leakage. Nested cross-validation (tuning C in an inner loop per fold) was
rejected: the usable recession sample is too small. An early fold's
training window holds only 2-3 recession episodes; an inner validation
split of that contains zero or one recession, so the inner loop would
"choose" C from noise — worse than a fixed value because it looks
principled while being random.

Instead:
  - HEADLINE C = 1.0, pre-committed. With z-scored features and L2 this
    is a weakly-informative prior: coefficients of order ~1 are expected,
    which is C≈1. This is the value a Bayesian writes down before seeing
    data, not an arbitrary default.
  - A C-SENSITIVITY STRIP is reported across C in {0.1, 0.3, 1, 3, 10}.
    This is for DISPLAY ONLY — the headline is always the pre-committed
    C=1.0. The strip shows whether the M2-vs-M1 conclusion is robust to C
    or hangs on one cherry-pickable value. No selection on it => no leak.

Nested-CV tuning is logged for v2: revisit when the usable recession
sample is large enough for a stable inner loop.

INTERFACE
---------
M2Logit implements the RecessionModel protocol (fit / predict_proba), so
the Step-5 walk-forward harness validates it with zero new harness code.
sklearn.LogisticRegression lives strictly behind that interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from recession.validation.walk_forward import walk_forward, WalkForwardResult
from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1Probit, M1_FEATURES, M1_EXTENDED_FEATURES


# M2's feature set — the four §4.16 T1-leading features.
M2_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]

# Regularization — decision (i+).
M2_DEFAULT_C = 1.0                       # pre-committed headline value
M2_C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]   # sensitivity strip — display only


# =============================================================================
# The model
# =============================================================================

class M2Logit:
    """Regularized (L2) logistic-regression recession model.

    Implements the RecessionModel protocol. fit() z-scores features on
    TRAINING-fold statistics only (L2 penalizes coefficient magnitude, so
    features must be on a common scale or the penalty is arbitrary across
    units), then fits an L2-penalized logit. predict_proba() applies the
    same train-derived standardization and returns P(recession).

    Parameters
    ----------
    C : float
        Inverse L2 regularization strength (sklearn convention — smaller C
        = stronger shrinkage). Fixed per instance; never tuned on test data.
    """

    def __init__(self, C: float = M2_DEFAULT_C) -> None:
        self.C = C
        self._model: Optional[LogisticRegression] = None
        self._feature_names: list[str] = []
        self._mean: Optional[pd.Series] = None
        self._std: Optional[pd.Series] = None
        self._fallback_rate: Optional[float] = None

    # -- RecessionModel interface ------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M2Logit":
        """Fit the L2 logit on training features X and binary target y."""
        self._feature_names = list(X.columns)

        # Standardize on TRAINING stats only — no leakage.
        self._mean = X.mean()
        self._std = X.std().replace(0.0, 1.0)   # guard constant columns
        Xz = ((X - self._mean) / self._std).to_numpy()
        y_arr = np.asarray(y, dtype=int)

        # Degenerate target (all 0 or all 1) — logit is undefined. Fall
        # back to predicting the base rate, consistent with M1Probit.
        if len(np.unique(y_arr)) < 2:
            self._fallback_rate = float(y_arr.mean())
            self._model = None
            return self

        # L2-penalized logistic regression. lbfgs handles L2 + is stable
        # on small samples. class_weight balanced is deliberately NOT used
        # — M1 was unweighted, and weighting would change the comparison
        # from "regularization" to "regularization + reweighting".
        self._model = LogisticRegression(
            penalty="l2", C=self.C, solver="lbfgs", max_iter=1000,
        )
        try:
            self._model.fit(Xz, y_arr)
            self._fallback_rate = None
        except Exception:
            self._fallback_rate = float(y_arr.mean())
            self._model = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(recession) for each row of X."""
        if self._model is None:
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(X), rate, dtype=float)

        Xz = ((X[self._feature_names] - self._mean) / self._std).to_numpy()
        # column 1 = P(class==1) = P(recession)
        return np.asarray(self._model.predict_proba(Xz)[:, 1], dtype=float)

    # -- inspection --------------------------------------------------------

    def coefficients(self) -> Optional[dict]:
        """The fitted (standardized-feature) coefficients, or None if the
        model fell back to base-rate. Useful for checking that L2 actually
        shrank the coefficients vs an unpenalized fit."""
        if self._model is None:
            return None
        return {
            "intercept": float(self._model.intercept_[0]),
            **{name: float(c) for name, c in
               zip(self._feature_names, self._model.coef_[0])},
        }


def _m2_factory(C: float):
    """Return a zero-arg model factory for the harness at a given C."""
    return lambda: M2Logit(C=C)


# =============================================================================
# The driver — M2 vs the M1 baseline, plus the C-sensitivity strip
# =============================================================================

def run_m2(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    c_grid: Optional[list[float]] = None,
    **walk_forward_kwargs,
) -> dict:
    """Run M2 validation and the head-to-head against the M1 baseline.

    M2's headline is the pre-committed C=1.0 logit. Also runs:
      - M1 (single-feature probit baseline) on the SAME folds, so the
        M2-vs-M1 comparison is exact.
      - a C-sensitivity strip: M2 at each C in c_grid. DISPLAY ONLY — the
        headline is always C=1.0; the strip just shows robustness.

    All runs share one common fold axis (months where all M2 features
    exist) so every number is directly comparable.

    Returns
    -------
    dict with keys:
        'm2'        : WalkForwardResult — M2 at the headline C (1.0)
        'm1'        : WalkForwardResult — the M1 baseline, same folds
        'c_strip'   : dict[float, WalkForwardResult] — M2 at each grid C
    """
    if c_grid is None:
        c_grid = M2_C_GRID

    common = dict(
        target=target, horizon=horizon,
        min_history_year=min_history_year,
        db_path=db_path,
        **walk_forward_kwargs,
    )

    # One common fold axis for everything: months where all M2 features
    # exist (the binding constraint; M1's single feature is a subset).
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=M2_FEATURES,
        **build_kwargs,
    )
    feat_cols = [c for c in M2_FEATURES if c in probe.X.columns]
    all_present = probe.X[feat_cols].notna().all(axis=1)
    common_axis = probe.X.index[all_present]

    # M2 at the headline C.
    m2 = walk_forward(
        model_factory=_m2_factory(M2_DEFAULT_C),
        feature_subset=M2_FEATURES,
        model_columns=M2_FEATURES,
        model_name=f"M2 logit L2 (C={M2_DEFAULT_C}, 4 features)",
        restrict_to_months=common_axis,
        **common,
    )

    # M1 baseline — same folds, for the head-to-head.
    m1 = walk_forward(
        model_factory=M1Probit,
        feature_subset=M1_FEATURES,
        model_columns=M1_FEATURES,
        model_name="M1 baseline (T10Y3M probit)",
        restrict_to_months=common_axis,
        **common,
    )

    # C-sensitivity strip — display only, no selection.
    c_strip = {}
    for C in c_grid:
        c_strip[C] = walk_forward(
            model_factory=_m2_factory(C),
            feature_subset=M2_FEATURES,
            model_columns=M2_FEATURES,
            model_name=f"M2 logit L2 (C={C})",
            restrict_to_months=common_axis,
            **common,
        )

    return {"m2": m2, "m1": m1, "c_strip": c_strip}


def print_m2_report(results: dict) -> None:
    """Pretty-print the M2 vs M1 head-to-head and the C-sensitivity strip."""
    m2 = results["m2"]
    m1 = results["m1"]
    strip = results["c_strip"]

    print("=" * 66)
    print(f"M2 VALIDATION — {m2.target} {m2.horizon}")
    print("=" * 66)
    for r in (m2, m1):
        print()
        print(r.summary())

    print()
    print("-" * 66)
    if m2.mean_fold_auc is not None and m1.mean_fold_auc is not None:
        delta = m2.mean_fold_auc - m1.mean_fold_auc
        print(f"  M2 logit L2 (C={M2_DEFAULT_C})  mean fold AUC: "
              f"{m2.mean_fold_auc:.4f}")
        print(f"  M1 baseline (yield curve)  mean fold AUC: "
              f"{m1.mean_fold_auc:.4f}")
        print(f"  delta (M2 - M1): {delta:+.4f}")
        if delta > 0:
            print("  M2 BEATS the M1 baseline — L2 regularization rescued "
                  "the extra features.")
        else:
            print("  M2 does NOT beat the M1 baseline — even regularized, "
                  "linear models cannot")
            print("  out-predict the yield curve here. Next gain must come "
                  "from nonlinearity (M3/M4).")

    # C-sensitivity strip — robustness display.
    print()
    print("-" * 66)
    print("  C-sensitivity strip (DISPLAY ONLY — headline is the "
          f"pre-committed C={M2_DEFAULT_C}):")
    print(f"  {'C':>8} {'mean fold AUC':>15} {'vs M1':>10}")
    m1_auc = m1.mean_fold_auc
    for C, r in strip.items():
        a = r.mean_fold_auc
        if a is None:
            print(f"  {C:>8} {'n/a':>15}")
            continue
        vs = (f"{a - m1_auc:+.4f}" if m1_auc is not None else "n/a")
        flag = " *headline" if C == M2_DEFAULT_C else ""
        print(f"  {C:>8} {a:>15.4f} {vs:>10}{flag}")
    # robustness verdict
    if m1_auc is not None:
        all_aucs = [r.mean_fold_auc for r in strip.values()
                    if r.mean_fold_auc is not None]
        if all_aucs:
            n_beat = sum(1 for a in all_aucs if a > m1_auc)
            print(f"  M2 beats M1 at {n_beat}/{len(all_aucs)} grid values "
                  f"-> {'robust' if n_beat in (0, len(all_aucs)) else 'C-DEPENDENT (fragile)'}.")
    print("=" * 66)
