# core/helpers_xgb.py
"""
Centralised ML helpers.
Import with:
    from core.helpers_xgb import train_xgb_predict, RISK_ALPHA
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Exposed knob so the app can show/tune it
RISK_ALPHA: float = 0.30  # 0..1 (higher = stronger down-weighting)


def _ensure_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add/ensure a 'Return' pct-change column."""
    if "Close" in df.columns:
        df["Return"] = df["Close"].pct_change()
    elif "Return_1D" in df.columns:
        df["Return"] = df["Return_1D"]
    else:
        raise ValueError("Missing both 'Close' and 'Return_1D'")
    return df


def train_xgb_predict(
    df: pd.DataFrame,
    horizon: int | None = None,
    horizon_days: int | None = None,
):
    """
    Fit XGBoost on TA features (plus risk features if present) and predict
    `horizon` days ahead. Returns: model, X_test, y_test, y_pred, fallback_note
    """
    # allow either parameter name
    if horizon is None:
        horizon = horizon_days or 1

    # -------- data prep
    df = _ensure_return(df).copy()
    df["Target"] = df["Close"].shift(-horizon)

    base_feats = ["Close", "MA5", "MA10", "MA20", "Return"]
    risk_feats = [
        c
        for c in ["risk_today", "risk_next_1d", "risk_next_3d", "risk_prev_1d"]
        if c in df.columns
    ]
    feats = base_feats + risk_feats

    # basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=base_feats + ["Target"]).copy()
    for c in risk_feats:
        df[c] = df[c].fillna(0.0)

    if len(df) < 10:
        last_price = df["Close"].iloc[-1]
        fallback_pred = np.array([last_price] * min(len(df), 3))
        note = "⚠️ Fallback prediction – too few rows for training"
        return None, df[["Close"]].tail(len(fallback_pred)), None, fallback_pred, note

    X = df[feats].fillna(0.0).astype("float32")
    y = df["Target"].astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.07,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=4,
        tree_method="hist",
        random_state=42,
    )

    # ---- risk-aware weighting (down-weight high-risk days)
    fit_kwargs = {"eval_set": [(X_test, y_test)], "verbose": False}
    if "risk_today" in X_train.columns:
        r = X_train["risk_today"].astype(float)
        rnorm = r / max(1.0, float(r.max()))
        sample_w = (1.0 - RISK_ALPHA * rnorm).clip(0.5, 1.0).values  # never < 0.5x
        fit_kwargs["sample_weight"] = sample_w

    model.fit(X_train, y_train, **fit_kwargs)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, None
