# models/ensemble.py
# ─────────────────────────────────────────────────────────────────────────────
# XGBoost + LightGBM soft-voting ensemble.
#
# Why ensemble works:
#   - XGBoost overfits some patterns that LightGBM handles better (and vice versa)
#   - Averaging probabilities reduces variance without increasing bias
#   - Expected accuracy gain: +1-3% vs single model
#   - Brier score improves because miscalibration errors partially cancel out
#
# The ensemble model is saved alongside XGB models as:
#   models/saved/{TICKER}_ensemble_{horizon}d.joblib
#
# train_all.py trains both — generator.py uses ensemble if available,
# falls back to XGB-only otherwise.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import joblib

from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=None)
def _cached_joblib_load(path_str):
    """Cache joblib.load results across calls. Models are immutable post-train.
    
    Cache key is the file path string. First call loads from disk; subsequent
    calls return the cached object. Reduces disk I/O during pipeline runs that
    re-load the same models repeatedly.
    """
    return joblib.load(path_str)
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/saved"))


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleResult:
    ticker:       str
    horizon:      int
    xgb_model:    object    # CalibratedClassifierCV wrapping XGBClassifier
    lgb_model:    object    # CalibratedClassifierCV wrapping LGBMClassifier
    weights:      tuple     # (xgb_weight, lgb_weight) — learned from val set
    feature_cols: list[str]
    metrics:      dict = field(default_factory=dict)

    def model_path(self) -> Path:
        return MODEL_DIR / f"{self.ticker}_ensemble_{self.horizon}d.joblib"

    def save(self) -> Path:
        path = self.model_path()
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, ticker: str, horizon: int) -> "EnsembleResult":
        path = MODEL_DIR / f"{ticker}_ensemble_{horizon}d.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No ensemble model for {ticker} {horizon}d")
        return _cached_joblib_load(str(path))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of XGB and LGB probabilities."""
        from models.classifier import FEATURE_COLUMNS
        working = X.copy()
        for c in FEATURE_COLUMNS:
            if c not in working.columns:
                working[c] = 0.0
        working[FEATURE_COLUMNS] = working[FEATURE_COLUMNS].fillna(
            working[FEATURE_COLUMNS].median()
        )
        Xf = working[FEATURE_COLUMNS]

        p_xgb = self.xgb_model.predict_proba(Xf)[:, 1]
        p_lgb = self.lgb_model.predict_proba(Xf)[:, 1]

        w_xgb, w_lgb = self.weights
        p_avg = w_xgb * p_xgb + w_lgb * p_lgb
        return p_avg


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHTGBM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def _train_lgbm(X_train, y_train, X_val, y_val,
                params: Optional[dict] = None,
                sample_weights=None) -> object:
    """Train a calibrated LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    lgb_params = params or {
        "n_estimators":     300,
        "learning_rate":    0.05,
        "max_depth":        5,
        "num_leaves":       31,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    clf = LGBMClassifier(**lgb_params)
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "callbacks": [],   # suppress output
    }
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    clf.fit(X_train, y_train, **fit_kwargs)

    cal = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    cal.fit(X_val, y_val)
    return cal


def _optimal_weights(p_xgb: np.ndarray, p_lgb: np.ndarray,
                     y_true: np.ndarray) -> tuple[float, float]:
    """
    Find the optimal blend weight that maximizes ROC-AUC on validation set.
    Tries w_xgb in [0.3, 0.7] in steps of 0.1.
    """
    best_auc  = 0.0
    best_w    = (0.5, 0.5)

    for w in np.arange(0.3, 0.8, 0.1):
        p_blend = w * p_xgb + (1 - w) * p_lgb
        try:
            auc = roc_auc_score(y_true, p_blend)
            if auc > best_auc:
                best_auc = auc
                best_w   = (round(w, 1), round(1 - w, 1))
        except Exception:
            continue

    return best_w


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(
    ticker:       str,
    df:           pd.DataFrame,
    horizon:      int = 1,
    test_size:    float = 0.2,
    xgb_params:   Optional[dict] = None,
    lgb_params:   Optional[dict] = None,
    save:         bool = True,
    verbose:      bool = True,
) -> EnsembleResult:
    """
    Train XGBoost + LightGBM ensemble for one ticker/horizon.

    Parameters
    ----------
    ticker     : e.g. "AAPL"
    df         : output of build_feature_dataframe() + add_forecast_targets()
    horizon    : 1, 3, or 5
    xgb_params : override default XGB params (uses tuned params if available)
    lgb_params : override default LGB params
    save       : save to disk
    verbose    : print metrics
    """
    from models.classifier import (
        FEATURE_COLUMNS, _prepare_xy, _risk_sample_weights,
        XGB_PARAMS, TrainResult
    )
    from xgboost import XGBClassifier

    target_col = f"target_{horizon}d"
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in df")

    # Fill missing features
    working = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in working.columns:
            working[c] = 0.0

    X, y = _prepare_xy(working, target_col)

    if len(X) < 100:
        raise ValueError(f"{ticker}: only {len(X)} rows — need at least 100")

    split    = int(len(X) * (1 - test_size))
    X_train  = X.iloc[:split];  X_val  = X.iloc[split:]
    y_train  = y.iloc[:split];  y_val  = y.iloc[split:]

    sw = _risk_sample_weights(working.loc[X_train.index])

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    # Use tuned params if available
    from models.tuner import get_params_for
    xgb_p = xgb_params or get_params_for(ticker, horizon) or XGB_PARAMS.copy()

    base_xgb = XGBClassifier(**xgb_p)
    fit_kw = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sw is not None:
        fit_kw["sample_weight"] = sw
    base_xgb.fit(X_train, y_train, **fit_kw)

    cal_xgb = CalibratedClassifierCV(base_xgb, method="isotonic", cv=5)
    cal_xgb.fit(X_train, y_train)

    # ── Train LightGBM — use tuned params if available ────────────────────────
    tuned_lgb = lgb_params or get_params_for(ticker, horizon, "lgb") if hasattr(get_params_for, '__call__') else lgb_params
    try:
        from models.tuner import get_params_for as _gp
        tuned_lgb = lgb_params or _gp(ticker, horizon, model="lgb")
    except Exception:
        tuned_lgb = lgb_params
    cal_lgb = _train_lgbm(X_train, y_train, X_val, y_val,
                           params=tuned_lgb, sample_weights=sw)

    # ── Optimal blend weights ─────────────────────────────────────────────────
    p_xgb   = cal_xgb.predict_proba(X_val)[:, 1]
    p_lgb   = cal_lgb.predict_proba(X_val)[:, 1]
    weights = _optimal_weights(p_xgb, p_lgb, y_val.values)

    # ── Evaluate ensemble ─────────────────────────────────────────────────────
    w_xgb, w_lgb = weights
    p_ens   = w_xgb * p_xgb + w_lgb * p_lgb
    y_pred  = (p_ens >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_val, p_ens)
    except Exception:
        auc = float("nan")

    metrics = {
        "accuracy":    round(accuracy_score(y_val, y_pred),   4),
        "roc_auc":     round(auc,                              4),
        "brier_score": round(brier_score_loss(y_val, p_ens),  4),
        "xgb_weight":  w_xgb,
        "lgb_weight":  w_lgb,
        "n_train":     len(X_train),
        "n_test":      len(X_val),
    }

    if verbose:
        print(
            f"  {ticker:<6} | ensemble_{horizon}d | "
            f"acc={metrics['accuracy']:.3f}  "
            f"auc={metrics['roc_auc']:.3f}  "
            f"brier={metrics['brier_score']:.3f}  "
            f"w=({w_xgb:.1f}xgb+{w_lgb:.1f}lgb)"
        )

    result = EnsembleResult(
        ticker=ticker, horizon=horizon,
        xgb_model=cal_xgb, lgb_model=cal_lgb,
        weights=weights, feature_cols=FEATURE_COLUMNS,
        metrics=metrics,
    )

    if save:
        result.save()

    return result


def predict_proba_ensemble(
    ticker:  str,
    df:      pd.DataFrame,
    horizon: int = 1,
    result:  Optional[EnsembleResult] = None,
) -> pd.Series:
    """
    Return ensemble probability for each row. Falls back to XGB if no ensemble.
    """
    if result is None:
        try:
            result = EnsembleResult.load(ticker, horizon)
        except FileNotFoundError:
            # Fall back to XGB-only
            from models.classifier import predict_proba
            return predict_proba(ticker, df, horizon)

    from models.classifier import FEATURE_COLUMNS
    working = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in working.columns:
            working[c] = 0.0
    working[FEATURE_COLUMNS] = working[FEATURE_COLUMNS].fillna(
        working[FEATURE_COLUMNS].median()
    )

    proba = result.predict_proba(working[FEATURE_COLUMNS])
    return pd.Series(proba, index=df.index, name=f"prob_up_{horizon}d")
