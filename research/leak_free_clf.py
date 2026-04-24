"""
R1 — leak-free classifier. Isolates effect of fixing classifier.py:382.

Prod: calibrates on (X_test, y_test)          → LEAK
R1:   calibrates on held-out slice of X_train → NO LEAK

Same single-fold split as prod, so only variable = leak fix.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

from models.classifier import (
    FEATURE_COLUMNS,
    _prepare_xy, _get_xgb_params, _risk_sample_weights,
    RISK_ALPHA, RISK_WEIGHT_FLOOR,
)

R1_MODEL_DIR = Path("research/models_r1")
R1_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_leak_free(df, target_col, ticker, horizon,
                    test_size=0.2, calib_frac=0.2):
    """Single-fold train with isotonic calibration on held-out train slice."""
    X, y = _prepare_xy(df, target_col)
    if len(X) < 60:
        raise ValueError(f"{ticker} {target_col}: {len(X)} rows < 60")

    # Prod-style single split
    split_idx = int(len(X) * (1 - test_size))
    X_tr_all, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr_all, y_te = y.iloc[:split_idx], y.iloc[split_idx:]

    # Inner split — last calib_frac of train for calibration
    split_in = int(len(X_tr_all) * (1 - calib_frac))
    X_in,  y_in  = X_tr_all.iloc[:split_in], y_tr_all.iloc[:split_in]
    X_cal, y_cal = X_tr_all.iloc[split_in:], y_tr_all.iloc[split_in:]

    # Risk weights on inner train only
    w_all = _risk_sample_weights(df.loc[X_tr_all.index], RISK_ALPHA, RISK_WEIGHT_FLOOR)
    w_in = w_all[:split_in] if w_all is not None else None

    # Fit XGB on inner train
    clf = XGBClassifier(**_get_xgb_params(ticker, horizon))
    fit_kwargs = {"eval_set": [(X_cal, y_cal)], "verbose": False}
    if w_in is not None:
        fit_kwargs["sample_weight"] = w_in
    clf.fit(X_in, y_in, **fit_kwargs)

    # Raw + calibrated probs on test
    raw_cal = clf.predict_proba(X_cal)[:, 1]
    raw_te  = clf.predict_proba(X_te)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)
    cal_te = iso.transform(raw_te)

    try:
        auc_raw = roc_auc_score(y_te, raw_te)
        auc_cal = roc_auc_score(y_te, cal_te)
    except ValueError:
        auc_raw = auc_cal = float("nan")

    return {
        "ticker": ticker, "horizon": horizon,
        "n_train_inner": len(X_in), "n_calib": len(X_cal), "n_test": len(X_te),
        "auc_raw": round(auc_raw, 4),
        "auc_calibrated": round(auc_cal, 4),
        "accuracy": round(accuracy_score(y_te, (cal_te >= 0.5).astype(int)), 4),
        "brier":    round(brier_score_loss(y_te, cal_te), 4),
        "log_loss": round(log_loss(y_te, np.clip(cal_te, 1e-6, 1-1e-6)), 4),
        "y_test": y_te.values, "p_test": cal_te,
    }
