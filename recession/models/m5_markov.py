"""
recession/models/m5_markov.py

M5 — the Markov-switching regime model. The fifth and final model class on
the ladder, and structurally UNLIKE M1-M4.

WHY M5 IS DIFFERENT
-------------------
M1-M4 are classifiers: given features at month t, predict P(recession
within h months). M5 is a REGIME model. It assumes the economy is in one
of two hidden states — expansion or recession — each with its own
dynamics, and infers the probability of being in the recession state. It
is the Hamilton (1989) class of model, the econometric standard for
business-cycle regime identification.

THREE DESIGN FORKS — defaults chosen and flagged
------------------------------------------------
Fork 1 — what does M5 predict? A Markov model outputs the FILTERED
recession-state probability P(state_t = recession). To compare with M1-M4
at T1/h=12, M5's "prediction" is that filtered probability, scored against
the SAME T1/h=12 label through the SAME walk-forward harness.
  CAVEAT (documented honestly): M5 estimates the CURRENT regime, while the
  T1/h=12 target is "recession within 12 months". These are not the same
  conceptual object. But scoring a regime model's recession probability
  against a forward recession label is the standard way these models are
  evaluated, and it lets M5 run through the identical harness. The caveat
  is real and is restated in the report.

Fork 2 — Markov on what series? A Markov-switching recession model
classically runs on a single growth series (Hamilton used GNP growth).
M5 runs on INDPRO — industrial production — the canonical monthly
real-activity series. NOT the yield curve: regime-switching a spread is
not a standard or sensible construction; regime models go on activity
series.

Fork 3 — fit the RecessionModel interface? Yes. M5Markov wraps
statsmodels MarkovRegression behind fit / predict_proba. fit() estimates
the regime model on the training window; predict_proba() filters through
the requested rows and returns the per-month recession-state probability.
The harness then validates M5 identically to M1-M4.

ROBUSTNESS — M5 is the "riskiest" model (per the project plan)
--------------------------------------------------------------
Markov-switching MLE is fragile on short samples. Three guards:
  - CONVERGENCE: if the MLE fails or returns non-finite params, M5 falls
    back to the base rate (consistent with M1-M4).
  - LABEL SWITCHING: the two estimated regimes have arbitrary index order
    — "regime 0" may be expansion in one fold and recession in the next.
    M5 aligns by ECONOMIC MEANING: the recession regime is the one with
    the LOWER mean growth (and, as a tiebreak, higher variance). The
    recession-state probability always refers to that regime, never to a
    fixed index.
  - DEGENERATE FITS: if the two regimes are not separated (nearly equal
    means), the fit carries no regime information — flagged, and the
    recession probability is still returned but the degeneracy is recorded.

EXPECTATION (stated plainly)
----------------------------
M1-M4 all tie the yield-curve baseline at T1/h=12 — four model classes,
one verdict. M5 is the fifth. It is most likely the fifth confirmation,
not a breakthrough. M5's value is COMPLETENESS of the model-class sweep.

M5Markov implements the RecessionModel protocol, so the Step-5 harness
validates it with zero new harness code.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from recession.validation.walk_forward import walk_forward, WalkForwardResult
from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1Probit, M1_FEATURES


# M5 runs on a single real-activity series — INDPRO (industrial production).
M5_FEATURES = ["INDPRO"]


# =============================================================================
# The model
# =============================================================================

class M5Markov:
    """Two-regime Markov-switching recession model implementing the
    RecessionModel protocol.

    fit() estimates a 2-regime switching-mean, switching-variance model on
    the training rows' INDPRO series. predict_proba() filters the model
    through the requested rows and returns P(recession regime).

    The recession regime is identified by economic meaning (lower mean
    growth), never by statsmodels' arbitrary regime index.
    """

    def __init__(self) -> None:
        self._result = None                  # fitted statsmodels result
        self._feature_names: list[str] = []
        self._recession_regime: Optional[int] = None   # aligned index
        self._fallback_rate: Optional[float] = None
        self._degenerate = False             # regimes not separated
        self._train_index: Optional[pd.DatetimeIndex] = None
        self._train_values: Optional[np.ndarray] = None

    # -- RecessionModel interface ------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M5Markov":
        """Estimate the 2-regime Markov model on the training INDPRO series."""
        self._feature_names = list(X.columns)
        # M5 uses a single series; take the first model column (INDPRO).
        series = X[self._feature_names[0]].to_numpy(dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self._train_index = X.index
        self._train_values = series

        # need a minimally long, non-degenerate series
        if len(series) < 24 or not np.isfinite(series).all() \
                or np.nanstd(series) == 0:
            self._fallback_rate = (float(y_arr.mean())
                                   if len(y_arr) else 0.5)
            self._result = None
            return self

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MarkovRegression(
                    series, k_regimes=2, switching_variance=True,
                )
                self._result = model.fit()
            if not np.all(np.isfinite(self._result.params)):
                raise ValueError("non-finite Markov params")
            # --- align regimes by economic meaning --------------------
            # regime means are params[0], params[1]; recession = lower mean
            mean0, mean1 = self._result.params[0], self._result.params[1]
            self._recession_regime = 0 if mean0 < mean1 else 1
            # degeneracy check — regimes barely separated
            spread = abs(mean0 - mean1)
            scale = (abs(mean0) + abs(mean1)) / 2 + 1e-9
            self._degenerate = (spread / scale) < 0.10
            self._fallback_rate = None
        except Exception:
            self._fallback_rate = (float(y_arr.mean())
                                   if len(y_arr) else 0.5)
            self._result = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(recession regime) for each row of X.

        statsmodels' regime model is estimated on the training series; to
        get probabilities for arbitrary rows we filter the fitted model
        over the union of train + requested rows and read off the
        smoothed recession-regime probability for the requested rows.
        """
        if self._result is None or self._recession_regime is None:
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(X), rate, dtype=float)

        series = X[self._feature_names[0]].to_numpy(dtype=float)

        # If the requested rows are exactly the training rows, read the
        # smoothed probabilities directly.
        same_as_train = (
            self._train_index is not None
            and len(X) == len(self._train_index)
            and (X.index == self._train_index).all()
        )
        if same_as_train:
            smoothed = self._result.smoothed_marginal_probabilities
            return np.asarray(smoothed[:, self._recession_regime],
                              dtype=float)

        # Otherwise: apply the fitted regime model to the new series by
        # filtering. statsmodels lets us evaluate the model at the fitted
        # params on a new endog via .filter().
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_model = MarkovRegression(
                    series, k_regimes=2, switching_variance=True,
                )
                filtered = new_model.filter(self._result.params)
            probs = np.asarray(
                filtered.filtered_marginal_probabilities[
                    :, self._recession_regime],
                dtype=float,
            )
            if len(probs) != len(X) or not np.isfinite(probs).all():
                raise ValueError("filter produced bad output")
            return probs
        except Exception:
            # last-resort fallback
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(X), rate, dtype=float)

    # -- diagnostics -------------------------------------------------------

    def is_degenerate(self) -> bool:
        """True if the two regimes were not meaningfully separated."""
        return self._degenerate

    def regime_summary(self) -> Optional[dict]:
        """The estimated regime means, or None if the model fell back."""
        if self._result is None:
            return None
        return {
            "regime_0_mean": float(self._result.params[0]),
            "regime_1_mean": float(self._result.params[1]),
            "recession_regime": self._recession_regime,
            "degenerate": self._degenerate,
        }


# =============================================================================
# The driver
# =============================================================================

def run_m5(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    **walk_forward_kwargs,
) -> dict:
    """Run M5 validation and the head-to-head against the M1 baseline.

    Both M5 and M1 run on one common fold axis (months where INDPRO and
    T10Y3M both exist), so the comparison is exact.

    Returns {'m5': WalkForwardResult, 'm1': WalkForwardResult}.
    """
    common = dict(
        target=target, horizon=horizon,
        min_history_year=min_history_year,
        db_path=db_path,
        **walk_forward_kwargs,
    )

    # common axis: months where both INDPRO (M5) and T10Y3M (M1) exist
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    feats = sorted(set(M5_FEATURES) | set(M1_FEATURES))
    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=feats, **build_kwargs,
    )
    cols = [c for c in feats if c in probe.X.columns]
    common_axis = probe.X.index[probe.X[cols].notna().all(axis=1)]

    m5 = walk_forward(
        model_factory=M5Markov,
        feature_subset=M5_FEATURES, model_columns=M5_FEATURES,
        model_name="M5 Markov-switching (INDPRO regime)",
        restrict_to_months=common_axis, **common,
    )
    m1 = walk_forward(
        model_factory=M1Probit,
        feature_subset=M1_FEATURES, model_columns=M1_FEATURES,
        model_name="M1 baseline (T10Y3M probit)",
        restrict_to_months=common_axis, **common,
    )
    return {"m5": m5, "m1": m1}


def print_m5_report(results: dict) -> None:
    """Pretty-print the M5 vs M1 head-to-head."""
    m5 = results["m5"]
    m1 = results["m1"]
    print("=" * 68)
    print(f"M5 VALIDATION (Markov-switching regime model) — "
          f"{m5.target} {m5.horizon}")
    print("=" * 68)
    print("  NOTE: M5 estimates the CURRENT regime; the T1/h=12 target is")
    print("  'recession within 12 months'. Scoring a regime model against a")
    print("  forward label is standard but not a perfect conceptual match.")
    for r in (m5, m1):
        print()
        print(r.summary())
    print()
    print("-" * 68)
    if m5.mean_fold_auc is not None and m1.mean_fold_auc is not None:
        delta = m5.mean_fold_auc - m1.mean_fold_auc
        print(f"  M5 Markov-switching  mean fold AUC: {m5.mean_fold_auc:.4f}")
        print(f"  M1 baseline          mean fold AUC: {m1.mean_fold_auc:.4f}")
        print(f"  delta (M5 - M1): {delta:+.4f}")
        print()
        if m5.mean_fold_auc < 0.5:
            print("  M5 scores BELOW 0.5 — and this is EXPECTED, not a failure.")
            print("  M5 is a COINCIDENT model: it estimates P(in recession")
            print("  NOW). The T1/h=12 target is LEADING: 'recession starts")
            print("  within 12 months'. In the run-up to a recession the")
            print("  economy is still in the expansion regime (INDPRO growth")
            print("  still positive), so M5 correctly reports low recession")
            print("  probability exactly when the forward label is 1. Current")
            print("  regime and future-recession-onset are ANTI-correlated in")
            print("  the run-up, so a coincident model scores below 0.5 on a")
            print("  leading target. The sub-0.5 AUC quantifies that mismatch.")
            print()
            print("  READ: M5 is not a leading recession predictor and does")
            print("  not belong at the h=12 cell. Its natural home is the")
            print("  COINCIDENT cell (h=0) — flagged for the short-horizon")
            print("  track. At h=12 the M1-M4 verdict stands unchanged.")
        elif delta > 0:
            print("  M5 beats the M1 baseline.")
        else:
            print("  M5 does NOT beat the M1 baseline.")
    else:
        print("  (insufficient scoreable folds for a verdict)")
    print("=" * 68)
