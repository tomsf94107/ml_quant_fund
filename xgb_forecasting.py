# xgb_forecasting.py

import os
import datetime
import joblib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from forecast_feature_engineering import build_feature_dataframe
import streamlit as st  # optional (used elsewhere; safe to keep)

MODEL_DIR = "models"
LOG_DIR = "forecast_logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Insider DB rollup features (from SQLite insider_flows) ---
INSIDER_DB_FEATURES = ["ins_net_shares_7d_db", "ins_net_shares_21d_db"]

# Columns we should not feed into the model directly
_EXCLUDE_EXACT = {
    "ticker", "date", "ds", "actual",
    "y", "yhat", "yhat_lower", "yhat_upper",
    "Signal", "Prob", "Prob_eff", "GateBlock",
}
# Avoid obvious leakage-style prefixes
_EXCLUDE_PREFIXES = ("ret_", "return_", "Gate", "Signal")


def _safe_model_path(ticker: str, template: str) -> str:
    path = template.format(ticker=ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _pick_feature_columns(df: pd.DataFrame,
                          target_col: str = "close",
                          extra_allow: Optional[List[str]] = None) -> List[str]:
    """
    Select numeric, non-target, non-leaky feature columns.
    Always include insider DB rollups if present.
    """
    if df is None or df.empty:
        return []
    num = df.select_dtypes(include=[np.number]).copy()

    cols: List[str] = []
    for c in num.columns:
        if c == target_col:
            continue
        if c in _EXCLUDE_EXACT:
            continue
        if any(c.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        cols.append(c)

    # Ensure insider DB rollups are included if present
    for c in INSIDER_DB_FEATURES:
        if c in num.columns and c not in cols:
            cols.append(c)

    # Allowlist any extras explicitly
    if extra_allow:
        for c in extra_allow:
            if c in num.columns and c not in cols and c != target_col:
                cols.append(c)

    return cols


def _prepare_xy(df_features: pd.DataFrame,
                target_col: str = "close",
                feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare X, y with robust feature selection and NA handling.
    - Keep rows where y is finite.
    - Fill X NaNs with 0 after clipping infs.
    """
    df = df_features.copy()

    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")

    # Ensure numeric and drop rows with NaN target
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = np.isfinite(y)
    df = df.loc[mask].copy()
    y = y.loc[mask]

    # Choose feature columns
    if feature_cols is None:
        feature_cols = _pick_feature_columns(df, target_col=target_col)

    if not feature_cols:
        raise ValueError("No valid feature columns selected for training.")

    # Build X
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y, feature_cols


def train_xgb_predict(
    ticker: str,
    df_features: pd.DataFrame = None,
    target_col: str = "close",
    model_save_path: str = f"{MODEL_DIR}/xgb_model_{{ticker}}.pkl",
    retrain: bool = False,
    return_model: bool = False,
    log_results: bool = True,
    feature_cols: Optional[List[str]] = None,  # NEW: allow override
):
    """
    Train (or load) an XGBRegressor to fit 'target_col' using features from build_feature_dataframe().
    Returns y_pred (np.ndarray) by default, or (model, X, y_pred) if return_model=True.
    """
    # 1) Build features if not provided
    if df_features is None:
        df_features = build_feature_dataframe(ticker)

    # 2) Prepare X, y
    try:
        X, y, used_cols = _prepare_xy(df_features, target_col=target_col, feature_cols=feature_cols)
    except Exception as e:
        return None, f"❌ Feature prep failed: {e}"

    # 3) Load or train model
    model_path = _safe_model_path(ticker, model_save_path)
    if retrain or not os.path.exists(model_path):
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        joblib.dump({"model": model, "feature_cols": used_cols, "target_col": target_col}, model_path)
    else:
        payload = joblib.load(model_path)
        model = payload["model"]
        # If stored feature list differs (schema drift), align columns gracefully
        stored_cols = payload.get("feature_cols", used_cols)
        # Add any missing stored columns with zeros
        for c in stored_cols:
            if c not in X.columns:
                X[c] = 0.0
        # Keep same column order as training
        X = X[stored_cols]
        used_cols = stored_cols

    # 4) In-sample predictions (consistent with your current workflow)
    y_pred = model.predict(X)

    # 5) Metrics
    try:
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2  = r2_score(y, y_pred)
    except Exception:
        mae = mse = r2 = np.nan

    if log_results and np.isfinite(mae):
        log_xgb_metrics(ticker, mae, mse, r2)

    if return_model:
        return model, X, y_pred

    return y_pred


def forecast_today(ticker: str):
    try:
        df = build_feature_dataframe(ticker)
        y_pred = train_xgb_predict(ticker, df, retrain=False, log_results=True)
        # train_xgb_predict may return (None, err) if feature prep failed
        if isinstance(y_pred, tuple) and len(y_pred) == 2 and isinstance(y_pred[1], str):
            return y_pred[1]
        return y_pred[-1]
    except Exception as e:
        return f"⚠️ Forecast failed: {e}"


def log_xgb_metrics(ticker, mae, mse, r2):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    row = pd.DataFrame({
        "date": [today],
        "ticker": [ticker],
        "mae": [mae],
        "mse": [mse],
        "r2": [r2],
    })

    log_path = os.path.join(LOG_DIR, f"{ticker}_xgb_log.csv")
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        row = pd.concat([existing, row], ignore_index=True)

    row.to_csv(log_path, index=False)


def load_latest_feature_row(ticker: str):
    df = build_feature_dataframe(ticker)
    latest = df.dropna(subset=["close"]).tail(1).drop(columns=["close"], errors="ignore")
    return latest
