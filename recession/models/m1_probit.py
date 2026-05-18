"""
recession/models/m1_probit.py

M1 — the static probit model. The econometric baseline of the recession
model project: every later model (M2-M8) must beat M1's out-of-sample AUC.

WHAT "STATIC PROBIT" MEANS
-------------------------
"Static" = fixed coefficients, no time-series dynamics (no lags of the
target, no regime switching). It is the classic Estrella-Mishkin style
recession probit, generalised from one regressor to a small fixed set.

M1's FEATURE SET (Option B, locked in Step 5 planning against spec §4.16)
-------------------------------------------------------------------------
M1 at the primary cell (T1, h=12) uses four features, each of which §4.16
identifies as T1-leading or all-horizon with real h=12 signal, and none of
which §4.16 excludes from an unregularised probit:

    T10Y3M         yield curve         (h=12 vs T1 AUC 0.80)
    NFCI           financial conditions (h=12 vs T1 AUC 0.80)
    INDPRO         real activity        (h=12 vs T1 AUC 0.72)
    REAL_FFR_GAP   monetary stance      (h=12 vs T1 AUC 0.71)

Excluded per §4.16: DTWEXBGS (thin, 2006-start — "do not weight heavily in
M1/M2"), EBP / BAA10Y (T5-primary; BAA10Y explicitly excluded h=12 vs T1),
labor features (coincident — M5/M4-short only), PERMIT / ISRATIO
(model_eligible = M3/M4/M5 only).

THE B+ SUB-BASELINE
-------------------
run_m1() also fits and reports a single-feature T10Y3M-only probit — the
canonical Estrella-Mishkin recession probit — on the IDENTICAL walk-forward
folds. A one-feature probit cannot overfit, so its OOS AUC is an
unoverfittable floor: if the four-feature M1 cannot beat it, that is a
loud diagnostic signal. This is exactly two fits (full M1 + the floor) —
not a feature search.

INTERFACE
---------
M1Probit implements the RecessionModel protocol (fit / predict_proba) from
walk_forward.py, so the harness validates it identically to every other
model. statsmodels.Probit lives strictly behind that interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# statsmodels emits noisy convergence warnings on small/quasi-separable
# folds; we handle non-convergence explicitly below, so silence the chatter.
import warnings

import statsmodels.api as sm

from recession.validation.walk_forward import walk_forward, WalkForwardResult
from recession.features.builder import build_feature_dataframe


# -----------------------------------------------------------------------------
# M1's feature set — DECISION (b), May 2026
# -----------------------------------------------------------------------------
# M1 is the canonical single-feature yield-curve recession probit
# (Estrella-Mishkin 1998; the NY Fed's published recession-probability
# series uses exactly this — the 10yr-3mo Treasury spread alone).
#
# This was decided AFTER the B+ walk-forward result. We initially locked a
# 4-feature M1 (Option B: T10Y3M+NFCI+INDPRO+REAL_FFR_GAP). The B+ run on
# real data showed the single-feature probit BEATS the 4-feature one out of
# sample — mean fold AUC 0.796 vs 0.771, and far better calibrated (Brier
# 0.14 vs 0.21). That is the textbook Estrella result: adding macro features
# to an UNREGULARIZED yield-curve probit fits in-sample noise and degrades
# OOS. Spec §4.16's own logic agrees — it routes correlated/weak features to
# the regularized models M3/M4, not to an unregularized probit.
#
# So M1 = the single-feature probit. The 4-feature variant is retained and
# still reported (M1_EXTENDED_FEATURES) as recorded evidence, but it is not
# "M1". The baseline every later model (M2-M8) must beat is M1's OOS AUC.
M1_FEATURES = ["T10Y3M"]

# The 4-feature variant — retained for the record. Does NOT beat M1.
M1_EXTENDED_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]


# =============================================================================
# The model
# =============================================================================

class M1Probit:
    """Static probit recession model implementing the RecessionModel protocol.

    Wraps statsmodels.Probit. fit() standardises features (z-score, stats
    learned on the training fold only) for numerical stability, then fits a
    probit MLE. predict_proba() applies the same standardisation and returns
    P(recession).
    """

    def __init__(self) -> None:
        self._result = None              # statsmodels fitted result
        self._feature_names: list[str] = []
        self._mean: Optional[pd.Series] = None
        self._std: Optional[pd.Series] = None
        self._fallback_rate: Optional[float] = None   # used if MLE fails

    # -- RecessionModel interface ------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M1Probit":
        """Fit the probit on training features X and binary target y."""
        self._feature_names = list(X.columns)

        # Standardise — z-score using TRAINING stats only (no leakage).
        self._mean = X.mean()
        self._std = X.std().replace(0.0, 1.0)   # guard constant columns
        Xz = (X - self._mean) / self._std

        # Design matrix with intercept.
        Xd = sm.add_constant(Xz, has_constant="add")
        y_arr = np.asarray(y, dtype=float)

        # Degenerate target (all 0 or all 1) — probit MLE is undefined.
        # Fall back to predicting the base rate.
        if len(np.unique(y_arr)) < 2:
            self._fallback_rate = float(y_arr.mean())
            self._result = None
            return self

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Probit(y_arr, Xd)
                self._result = model.fit(disp=0, maxiter=100)
            # Quasi-separation can yield non-finite params — treat as failure.
            if not np.all(np.isfinite(self._result.params)):
                raise ValueError("non-finite probit params")
            self._fallback_rate = None
        except Exception:
            # MLE failed to converge / separated — fall back to base rate.
            self._fallback_rate = float(y_arr.mean())
            self._result = None

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(recession) for each row of X."""
        if self._result is None:
            # fallback model — constant base-rate prediction
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(X), rate, dtype=float)

        Xz = (X[self._feature_names] - self._mean) / self._std
        Xd = sm.add_constant(Xz, has_constant="add")
        return np.asarray(self._result.predict(Xd), dtype=float)

    # -- inspection --------------------------------------------------------

    def summary(self) -> str:
        """statsmodels summary — the econometric sanity check (coefficients,
        standard errors, significance). Available because M1 is a real
        statsmodels probit underneath, not a black box."""
        if self._result is None:
            return "(M1Probit fell back to base-rate; no probit fit available)"
        return str(self._result.summary())


# =============================================================================
# The B+ driver — full M1 + the sub-baseline floor, identical folds
# =============================================================================

def run_m1(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    **walk_forward_kwargs,
) -> dict[str, WalkForwardResult]:
    """Run M1 validation: the single-feature M1 AND the 4-feature variant.

    Both are validated through walk_forward() with identical settings on one
    common fold axis, so their OOS metrics are directly comparable.

    Per decision (b): 'm1' is the canonical single-feature yield-curve
    probit — the project baseline. 'extended' is the 4-feature variant,
    retained as recorded evidence (it does not beat M1 OOS).

    Args:
        target: target id (default 'T1' — the primary recession cell).
        horizon: forecast horizon (default 'h=12').
        min_history_year: passed to the builder; default 1986 trims the
            sparse pre-1986 era so all features are present.
        db_path: optional DB path.
        **walk_forward_kwargs: forwarded to walk_forward (min_train_months,
            test_window_months, step_months, threshold).

    Returns:
        {'m1': WalkForwardResult,        # the single-feature baseline
         'extended': WalkForwardResult}  # the 4-feature variant (evidence)
    """
    common = dict(
        target=target, horizon=horizon,
        min_history_year=min_history_year,
        db_path=db_path,
        **walk_forward_kwargs,
    )

    # Both runs must use IDENTICAL folds to be comparable. M1 (1 feature)
    # and the extended variant (4 features) have different feature-
    # availability windows, so we compute one common axis — months where
    # ALL of the extended feature set is present (the binding constraint,
    # since M1's single feature is a subset) — and force both runs onto it
    # via restrict_to_months.
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=M1_EXTENDED_FEATURES,
        **build_kwargs,
    )
    feat_cols = [c for c in M1_EXTENDED_FEATURES if c in probe.X.columns]
    all_present = probe.X[feat_cols].notna().all(axis=1)
    common_axis = probe.X.index[all_present]

    m1 = walk_forward(
        model_factory=M1Probit,
        feature_subset=M1_FEATURES,
        model_columns=M1_FEATURES,
        model_name="M1 probit (T10Y3M — Estrella-Mishkin yield-curve baseline)",
        restrict_to_months=common_axis,
        **common,
    )

    extended = walk_forward(
        model_factory=M1Probit,
        feature_subset=M1_EXTENDED_FEATURES,
        model_columns=M1_EXTENDED_FEATURES,
        model_name="M1-extended (4-feature variant — does not beat baseline)",
        restrict_to_months=common_axis,
        **common,
    )

    return {"m1": m1, "extended": extended}


def print_m1_report(results: dict[str, WalkForwardResult]) -> None:
    """Pretty-print M1 (single-feature baseline) vs the 4-feature variant.

    The verdict compares MEAN FOLD AUC — the valid headline metric for an
    overlapping-fold walk-forward. Pooled AUC is not used for the verdict
    (overlapping test windows + per-fold refit make pooled probabilities
    non-comparable).
    """
    m1 = results["m1"]
    ext = results["extended"]
    print("=" * 66)
    print(f"M1 VALIDATION — {m1.target} {m1.horizon}")
    print("=" * 66)
    for r in (m1, ext):
        print()
        print(r.summary())
    print()
    print("-" * 66)
    if m1.mean_fold_auc is not None and ext.mean_fold_auc is not None:
        # M1 (single feature) is the baseline. The extended 4-feature
        # variant is reported for the record; per decision (b) it does not
        # beat M1 OOS, which is the expected Estrella result.
        delta = ext.mean_fold_auc - m1.mean_fold_auc
        print(f"  M1 baseline (single-feature) mean fold AUC: "
              f"{m1.mean_fold_auc:.4f}")
        print(f"  M1-extended (4-feature)      mean fold AUC: "
              f"{ext.mean_fold_auc:.4f}   (delta {delta:+.4f})")
        print(f"  [scoreable folds: M1 {m1.n_scoreable_folds}, "
              f"extended {ext.n_scoreable_folds}]")
        if delta > 0:
            print("  NOTE: the 4-feature variant beat the baseline — "
                  "worth revisiting decision (b).")
        else:
            print("  As expected: the 4-feature variant does NOT beat the "
                  "single-feature baseline.")
            print("  (Estrella result — extra features need regularization; "
                  "that is M3/M4's role.)")
        print()
        print(f"  >>> PROJECT BASELINE = M1 mean fold AUC "
              f"{m1.mean_fold_auc:.4f} — every later model must beat this.")
    else:
        print("  (insufficient scoreable folds for a verdict)")
    print("=" * 66)
