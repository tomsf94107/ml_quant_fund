# core/helpers_xgb.py
"""
Centralised ML helpers.
Import everywhere with:

    from core.helpers_xgb import train_xgb_predict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# ───────────────────────── internal helper ──────────────────────────
def _ensure_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add/ensure a 'Return' pct-change column."""
    if "Close" in df.columns:
        df["Return"] = df["Close"].pct_change()
    elif "Return_1D" in df.columns:
        df["Return"] = df["Return_1D"]
    else:
        raise ValueError("Missing both 'Close' and 'Return_1D'")
    return df


# ─────────────────────── XGB trainer & predictor ────────────────────
def train_xgb_predict(
        df: pd.DataFrame,
        horizon: int | None = None,
        horizon_days: int | None = None
):
    """
    Fit XGBoost on TA features and predict `horizon` days ahead.
    Accepts either `horizon` or legacy `horizon_days` keyword.
    Returns: model, X_test, y_test, y_pred, fallback_note
    """
    # allow either parameter name
    if horizon is None:
        horizon = horizon_days or 1

    # --------------- data prep
    df = _ensure_return(df)
    df["Target"] = df["Close"].shift(-horizon)

    REQUIRED = ["Close", "MA5", "MA10", "MA20", "Return"]
    df = df.dropna(subset=REQUIRED + ["Target"])

    if len(df) < 10:
        last_price    = df["Close"].iloc[-1]
        fallback_pred = np.array([last_price] * min(len(df), 3))
        note = "⚠️ Fallback prediction – too few rows for training"
        return None, df[["Close"]].tail(len(fallback_pred)), None, fallback_pred, note

    # --------------- train
    X, y = df[REQUIRED], df["Target"]
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
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, None
