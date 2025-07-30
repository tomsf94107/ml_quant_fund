# train_forecast_model_v1.1

import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from forecast_feature_engineering import build_feature_dataframe, add_forecast_targets

TICKERS = [
    "NVO", "AAPL", "PFE", "NFLX", "AMD", "NVDA", "PYPL", "GOOG", "SMCI",
    "CNC", "CRWD", "META", "MRNA", "DDOG", "UNH", "SHOP", "PLTR", "TSM"
]

TARGET_COLUMNS = ["target_1d", "target_3d", "target_5d"]
FEATURE_COLUMNS = ["close", "volume", "rsi_14", "macd", "ema_10", "sentiment"]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model_for_ticker(ticker):
    df = build_feature_dataframe(ticker)
    df = add_forecast_targets(df)

    for target_col in TARGET_COLUMNS:
        if target_col not in df.columns:
            print(f"⚠️ {target_col} missing in {ticker} — skipping.")
            continue

        df_clean = df.dropna(subset=[target_col])
        if df_clean.empty:
            print(f"⚠️ No data available for {ticker} - {target_col} (after dropping NaNs). Skipping.")
            continue

        X = df_clean[FEATURE_COLUMNS]
        y = df_clean[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_{target_col}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ Trained {ticker} - {target_col} | Accuracy: {acc:.2f} | Saved to {model_path}")

def train_all_models():
    for ticker in TICKERS:
        train_model_for_ticker(ticker)

if __name__ == "__main__":
    train_all_models()

