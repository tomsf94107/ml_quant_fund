"""
models/wrappers.py — Lightweight model result wrappers for non-ensemble experiments.

Required for joblib pickling: classes must live in an importable module so
pickle.load can find them by qualified name (not __main__).

Each class must implement:
    feature_cols: list[str]
    predict_proba(df_or_array) -> np.ndarray  # P(class=1) probabilities

This duck-typed interface matches what predict_proba_ensemble() calls.

USAGE:
    from models.wrappers import LGBOnlyResult
    wrapped = LGBOnlyResult(
        feature_cols=FEATURE_COLUMNS,
        model=trained_lgb_booster,
        metrics={"oos_auc": 0.72},
    )
    joblib.dump(wrapped, path)   # picklable
    loaded = joblib.load(path)   # unpicklable
    prob = loaded.predict_proba(df)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LGBOnlyResult:
    """LightGBM-only model wrapper (no XGB, no isotonic).

    Used for experiments where the C1-architecture LGB-only model outperforms
    the XGB+LGB+isotonic ensemble (Phase 1 D finding, May 25 2026).
    """
    feature_cols: list
    model: Any  # lightgbm.Booster
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, X):
        """Return P(class=1) for each row in X.

        Accepts DataFrame (subsets to feature_cols) or ndarray.
        """
        if isinstance(X, pd.DataFrame):
            # Add any missing columns the model expects with 0.0 (same convention
            # as EnsembleResult.predict_proba).
            working = X.copy()
            for c in self.feature_cols:
                if c not in working.columns:
                    working[c] = 0.0
            working[self.feature_cols] = working[self.feature_cols].fillna(
                working[self.feature_cols].median()
            )
            X = working[self.feature_cols].values.astype(np.float32)
        return self.model.predict(X)

@dataclass
class GlobalRankerResult:
    """Wrapper for a trained LightGBMRanker that's predict_proba compatible.
    
    Provides predict_proba(X)[:, 1] interface so downstream code can swap
    classifier -> ranker without changes. Uses sigmoid of raw rank scores
    normalized by std for output 0-1 range.
    """
    ranker: object        # LGBMRanker
    feature_cols: list
    horizon: int
    ticker: str           # always "GLOBAL"
    metrics: dict

    def predict_proba(self, X):
        import numpy as np
        raw = self.ranker.predict(X)
        s = np.std(raw)
        if s == 0:
            s = 1.0
        proba_up = 1.0 / (1.0 + np.exp(-raw / s))
        return np.column_stack([1 - proba_up, proba_up])

