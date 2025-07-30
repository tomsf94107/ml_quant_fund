# xgb_forecasting.py

import os
import pandas as pd
import numpy as np
import joblib
import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from forecast_feature_engineering import build_feature_dataframe
import streamlit as st

MODEL_DIR = "models"
LOG_DIR = "forecast_logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def train_xgb_predict(
    ticker: str,
    df_features: pd.DataFrame = None,
    target_col: str = "close",
    model_save_path: str = f"{MODEL_DIR}/xgb_model_{{ticker}}.pkl",
    retrain: bool = False,
    return_model: bool = False,
    log_results: bool = True
):
    if df_features is None:
        df_features = build_feature_dataframe(ticker)

    df = df_features.dropna().copy()
    if target_col not in df.columns:
        return None, f"Missing target column '{target_col}'"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model_path = model_save_path.format(ticker=ticker)

    if retrain or not os.path.exists(model_path):
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    if log_results:
        log_xgb_metrics(ticker, mae, mse, r2)

    if return_model:
        return model, X, y_pred

    return y_pred


def forecast_today(ticker: str):
    try:
        df = build_feature_dataframe(ticker)
        y_pred = train_xgb_predict(ticker, df, retrain=False, log_results=True)
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
        "r2": [r2]
    })

    log_path = os.path.join(LOG_DIR, f"{ticker}_xgb_log.csv")
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        row = pd.concat([existing, row], ignore_index=True)

    row.to_csv(log_path, index=False)


def load_latest_feature_row(ticker: str):
    df = build_feature_dataframe(ticker)
    latest = df.dropna().tail(1).drop(columns=["close"], errors="ignore")
    return latest
