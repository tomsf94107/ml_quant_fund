# core/helpers_xgb.py
"""
Centralised ML helpers.
Import with:
    from core.helpers_xgb import train_xgb_predict, RISK_ALPHA
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Exposed knob so the app can show/tune it
RISK_ALPHA: float = 0.30  # 0..1 (higher = stronger down-weighting)

# Insider DB rollup features (from SQLite insider_flows merged into the feature frame)
INSIDER_DB_FEATURES: List[str] = ["ins_net_shares_7d_db", "ins_net_shares_21d_db"]

# Columns we never feed directly to the model (ids / targets / obvious leakage)
_EXCLUDE_EXACT = {
    "ticker", "date", "ds", "actual", "y", "yhat", "yhat_lower", "yhat_upper",
    "Signal", "Prob", "Prob_eff", "GateBlock", "Target",
}
# Prefixes to exclude (target/leakage conventions)
_EXCLUDE_PREFIXES = ("ret_", "return_", "Gate", "Signal")


# ----------------------------- utilities ------------------------------------
def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Close' column exists. If not, try common alternatives and alias to 'Close'.
    """
    if "Close" in df.columns:
        return df
    for alt in ("close", "Adj Close", "adj_close", "ClosePrice", "close_price"):
        if alt in df.columns:
            d = df.copy()
            d["Close"] = d[alt]
            return d
    raise ValueError("No 'Close' (or alternative) column available for training.")


def _ensure_ma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure MA5/MA10/MA20 exist; if missing, compute simple moving averages on Close.
    """
    d = _ensure_close(df).copy()
    for w in (5, 10, 20):
        col = f"MA{w}"
        if col not in d.columns:
            d[col] = d["Close"].rolling(w, min_periods=1).mean()
    return d


def _ensure_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add/ensure a 'Return' pct-change column using 'Close'."""
    d = _ensure_close(df).copy()
    if "Return" not in d.columns:
        if "Return_1D" in d.columns:
            d["Return"] = pd.to_numeric(d["Return_1D"], errors="coerce")
        else:
            d["Return"] = d["Close"].pct_change()
    return d


def _pick_feature_columns(df: pd.DataFrame, extra_allow: Optional[List[str]] = None) -> List[str]:
    """
    Robust numeric feature selector:
      - keeps numeric columns
      - excludes identifiers/targets/leakage
      - ensures insider DB rollups are included when present
    """
    if df is None or len(df) == 0:
        return []

    num = df.select_dtypes(include=[np.number])
    cols: List[str] = []
    for c in num.columns:
        if c in _EXCLUDE_EXACT:
            continue
        if any(c.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        cols.append(c)

    # Ensure insider DB rollups included if present
    for c in INSIDER_DB_FEATURES:
        if c in num.columns and c not in cols:
            cols.append(c)

    # Force-include any requested extras that are numeric
    if extra_allow:
        for c in extra_allow:
            if c in num.columns and c not in cols:
                cols.append(c)

    return cols


# ----------------------------- main API -------------------------------------
def train_xgb_predict(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    horizon_days: Optional[int] = None,
    feature_cols: Optional[List[str]] = None,
):
    """
    Fit XGBoost on technical + available risk/insider features and
    predict `horizon` days ahead.

    Args:
        df: Feature frame (must include a price column; 'Close' or aliasable)
        horizon / horizon_days: future shift (days) to predict
        feature_cols: optional explicit feature list; if None, auto-select with _pick_feature_columns

    Returns:
        model, X_test, y_test, y_pred, fallback_note
    """
    # allow either parameter name
    if horizon is None:
        horizon = horizon_days or 1

    # -------- data prep
    d = df.copy()
    d = _ensure_close(d)
    d = _ensure_ma(d)
    d = _ensure_return(d)

    # Target is future Close
    d["Target"] = d["Close"].shift(-horizon)

    # Clean infinities
    d = d.replace([np.inf, -np.inf], np.nan)

    # Must have at least Close + Target to learn; keep rows where those exist
    d = d.dropna(subset=["Close", "Target"]).copy()

    # Not enough rows → fallback (naive hold-last)
    if len(d) < 10:
        last_price = d["Close"].iloc[-1] if len(d) else np.nan
        fallback_pred = np.array([last_price] * min(len(d), 3)) if len(d) else np.array([])
        note = "⚠️ Fallback prediction – too few rows for training"
        return None, d[["Close"]].tail(len(fallback_pred)), None, fallback_pred, note

    # Feature matrix / target
    if feature_cols is None:
        # Auto-pick from all numeric columns (includes MA*, Return, risk_*, insider_*, etc.)
        feature_cols = _pick_feature_columns(d, extra_allow=INSIDER_DB_FEATURES)

        # Safety: ensure the basic TA columns exist in selection
        for base in ("Close", "MA5", "MA10", "MA20", "Return"):
            if base in d.columns and base not in feature_cols:
                feature_cols.append(base)

    X = d[feature_cols].astype("float32").fillna(0.0)
    y = d["Target"].astype("float32")

    # Train/test split (time-ordered)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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
        rmax = float(max(1.0, r.max()))
        rnorm = r / rmax
        sample_w = (1.0 - RISK_ALPHA * rnorm).clip(0.5, 1.0).values  # never < 0.5x
        fit_kwargs["sample_weight"] = sample_w

    model.fit(X_train, y_train, **fit_kwargs)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, None


__all__ = [
    "train_xgb_predict",
    "RISK_ALPHA",
    "_pick_feature_columns",
    "INSIDER_DB_FEATURES",
]
