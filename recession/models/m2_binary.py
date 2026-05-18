"""
recession/models/m2_binary.py

B-track — M2-binary, the headline new model.

WHAT IT IS
----------
M2-binary is M2's L2-regularized logit, but with the macro features first
passed through the at-risk BINARIZATION transform (recession/features/
at_risk.py): each predictor becomes a 0/1 indicator of an "unusually weak"
state instead of a continuous standardized value.

WHY
---
Billakanti & Shin (FRB Philadelphia, Dec 2025) found that binarized
predictors consistently improve out-of-sample recession forecasting, and
often make a linear model competitive with flexible ML methods — because
the signal for a rare event like a recession lies in whether an indicator
has crossed into adverse territory, not in its exact value. The B-track
pre-registration names M2-binary as the primary new challenger to the
yield-curve baseline at short horizons.

THE LEAKAGE DISCIPLINE
----------------------
The at-risk transform's thresholds are quantiles that MUST come from
training data only. M2-binary therefore:
  - fit(): fits the AtRiskTransform on the TRAINING rows, then fits the
    logit on the binarized training features.
  - predict_proba(): applies the already-fitted transform to the rows it
    is given (train or test). The thresholds never see test data.
The walk-forward harness fits on the training fold and scores the test
fold — so the binarization is point-in-time correct.

DESIGN NOTES
------------
- The binarized features are already 0/1, so they are fed to the logit
  DIRECTLY — no z-scoring. (Standardizing a 0/1 indicator is pointless
  and slightly distorts the regularization.) This is the one structural
  difference from M2.
- Same L2 penalty and same C as M2. The comparison M2 vs M2-binary must
  isolate the BINARIZATION, not confound it with a regularization change.
- Same degenerate-target base-rate fallback as M1/M2.

M2Binary implements the RecessionModel protocol (fit / predict_proba), so
the Step-5 walk-forward harness validates it with no new harness code.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from recession.features.at_risk import AtRiskTransform, DEFAULT_AT_RISK_QUANTILE


# M2-binary uses the same 4-feature set as M2.
M2_BINARY_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]


class M2Binary:
    """L2-regularized logit on at-risk-binarized features. Implements the
    RecessionModel protocol."""

    def __init__(self, C: float = 1.0,
                 quantile: float = DEFAULT_AT_RISK_QUANTILE) -> None:
        self.C = C
        self.quantile = quantile
        self._model: Optional[LogisticRegression] = None
        self._transform: Optional[AtRiskTransform] = None
        self._feature_names: list[str] = []
        self._fallback_rate: Optional[float] = None

    # -- RecessionModel interface ------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M2Binary":
        """Fit the at-risk transform on training rows, then the logit on
        the binarized training features."""
        self._feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=int)

        # fit the binarization on TRAINING rows only — no leakage
        self._transform = AtRiskTransform(quantile=self.quantile)
        Xb = self._transform.fit(X).transform(X)
        # binarized features may carry NaN where the input was NaN;
        # the harness masks on notna upstream, but guard here too
        Xb = Xb.fillna(0.0)

        # degenerate target — base-rate fallback, consistent with M1/M2
        if len(np.unique(y_arr)) < 2:
            self._fallback_rate = float(y_arr.mean())
            self._model = None
            return self

        # binarized features are already 0/1 — fed to the logit directly,
        # no z-scoring. L2 regularization (sklearn's default penalty) with
        # the same C as M2, so the M2-vs-M2-binary comparison isolates the
        # binarization. (penalty="l2" is not passed explicitly: it is the
        # default, and the explicit kwarg is deprecated in sklearn 1.8.)
        self._model = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=1000,
        )
        try:
            self._model.fit(Xb.to_numpy(), y_arr)
            self._fallback_rate = None
        except Exception:
            self._fallback_rate = float(y_arr.mean())
            self._model = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(recession) for each row of X."""
        if self._model is None or self._transform is None:
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(X), rate, dtype=float)

        Xb = self._transform.transform(X[self._feature_names]).fillna(0.0)
        return np.asarray(
            self._model.predict_proba(Xb.to_numpy())[:, 1], dtype=float)

    # -- inspection --------------------------------------------------------

    def coefficients(self) -> Optional[dict]:
        """The fitted logit coefficients on the binarized features, or
        None if the model fell back."""
        if self._model is None:
            return None
        return {
            "intercept": float(self._model.intercept_[0]),
            **{name: float(c) for name, c in
               zip(self._feature_names, self._model.coef_[0])},
        }

    def at_risk_thresholds(self) -> Optional[dict]:
        """The fitted at-risk thresholds (for reporting), or None."""
        if self._transform is None:
            return None
        return self._transform.thresholds


def m2_binary_factory(C: float = 1.0,
                      quantile: float = DEFAULT_AT_RISK_QUANTILE):
    """Zero-arg model factory for the walk-forward harness."""
    return lambda: M2Binary(C=C, quantile=quantile)
