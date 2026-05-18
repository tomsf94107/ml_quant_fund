"""
recession/features/at_risk.py

B-track — the "at-risk" binarization transform.

WHAT IT IS
----------
Each continuous predictor is converted into a 0/1 indicator of an
"unusually weak" state: 1 when the predictor is in adverse territory,
0 otherwise. Motivated by Billakanti & Shin (FRB Philadelphia, Dec 2025),
which found binarized predictors consistently improve out-of-sample
recession forecasting — the relevant signal for a rare event like a
recession often lies in whether an indicator has crossed into unusually
adverse territory, not in its exact value.

THE LEAKAGE RULE — the single thing this module gets right
----------------------------------------------------------
The threshold that defines "unusually weak" is a QUANTILE of the
predictor. That quantile MUST be estimated from TRAINING DATA ONLY. A
threshold fit on the whole sample would peek at the future — the test
period's values would have helped set the cut. That is look-ahead
leakage, and it is exactly the kind of bug the whole recession project
has been built to avoid.

So the transform is a fit/apply pair, mirroring sklearn:
  - fit(X_train): estimate one threshold per column from the training
    rows, and remember each column's adverse DIRECTION.
  - transform(X): apply the stored thresholds to any rows (train OR test).
The walk-forward harness fits on the training fold and applies to the
test fold — the thresholds never see test data.

ADVERSE DIRECTION
-----------------
"Unusually weak" is not always "low". For a yield spread or industrial
production, adverse = unusually LOW. For a financial-conditions index
like NFCI, adverse = unusually HIGH (tight conditions). The direction is
declared per feature in ADVERSE_DIRECTION; the transform fires the
indicator on the correct tail.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# adverse direction per feature: "low" => indicator fires when the value
# is unusually LOW; "high" => fires when unusually HIGH.
ADVERSE_DIRECTION = {
    "T10Y3M": "low",        # an inverted / low yield spread is adverse
    "INDPRO": "low",        # weak industrial production is adverse
    "REAL_FFR_GAP": "high", # a high real funds-rate gap (tight policy) adverse
    "NFCI": "high",         # high NFCI = tight financial conditions = adverse
}

# default quantile defining "unusually weak": the worst 25% of training
# history. Pre-registered as a fixed constant — NOT tuned on results.
DEFAULT_AT_RISK_QUANTILE = 0.25


class AtRiskTransform:
    """Fit/apply binarizer. Thresholds are estimated from training data
    only, then applied to any rows."""

    def __init__(self, quantile: float = DEFAULT_AT_RISK_QUANTILE) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0,1), got {quantile}")
        self.quantile = quantile
        self._thresholds: dict[str, float] = {}
        self._directions: dict[str, str] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "AtRiskTransform":
        """Estimate one threshold per column from the TRAINING rows X.

        For a 'low'-adverse feature the threshold is the `quantile`
        quantile (the bottom tail); for a 'high'-adverse feature it is the
        `1 - quantile` quantile (the top tail).
        """
        self._thresholds = {}
        self._directions = {}
        for col in X.columns:
            series = X[col].dropna()
            if len(series) == 0:
                # no training data for this column — skip it (transform
                # will pass it through as all-zero)
                continue
            direction = ADVERSE_DIRECTION.get(col, "low")
            self._directions[col] = direction
            if direction == "low":
                thr = float(series.quantile(self.quantile))
            else:  # "high"
                thr = float(series.quantile(1.0 - self.quantile))
            self._thresholds[col] = thr
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored thresholds. Returns a DataFrame of 0/1
        indicators, same index and columns as X.

        A NaN input stays NaN in the output (downstream code masks on
        notna, exactly as for continuous features).
        """
        if not self._fitted:
            raise RuntimeError("AtRiskTransform.transform called before fit")
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in self._thresholds:
                # column unseen at fit time (or empty) — pass through as 0
                out[col] = 0.0
                continue
            thr = self._thresholds[col]
            direction = self._directions[col]
            vals = X[col]
            if direction == "low":
                ind = (vals <= thr).astype(float)
            else:  # "high"
                ind = (vals >= thr).astype(float)
            # preserve NaN: where the input was NaN, output is NaN
            ind = ind.where(vals.notna(), other=np.nan)
            out[col] = ind
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on X then transform X. Use ONLY when X is the
        training set — never on a train+test combined frame."""
        return self.fit(X).transform(X)

    @property
    def thresholds(self) -> dict[str, float]:
        """The fitted per-column thresholds (for inspection / reporting)."""
        return dict(self._thresholds)
