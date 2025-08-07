# train_forecast_model.py

import os
import pickle

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from forecast_feature_engineering import build_feature_dataframe, add_forecast_targets

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = [
    "NVO", "AAPL", "PFE", "NFLX", "AMD", "NVDA", "PYPL", "GOOG", "SMCI",
    "CNC", "CRWD", "META", "MRNA", "DDOG", "UNH", "SHOP", "PLTR", "TSM",
    "FIG", "SNOW", "MP", "OPEN", "DUOL", "BSX", "JNJ", "AXP", "TSLA",
    "ZM"
]

TARGET_COLUMNS   = ["target_1d", "target_3d", "target_5d"]

BASE_FEATURES    = [
    "close", "volume",
    "return_1d", "return_3d", "return_5d",
    "ma_5", "ma_10", "ma_20",
    "volatility_5d", "volatility_10d",
    "rsi_14", "macd", "macd_signal",
    "bollinger_upper", "bollinger_lower", "bollinger_width",
    "volume_zscore", "volume_spike",
    "sentiment_score",
    "is_pandemic",
]

INSIDER_FEATURES = ["insider_net_shares", "insider_7d", "insider_21d"]
FEATURE_COLUMNS  = BASE_FEATURES + INSIDER_FEATURES

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_with_insider_and_targets(ticker: str):
    # build all features (price, pandemic dummy, insider flows…)
    df = build_feature_dataframe(ticker)
    # add binary up/down targets
    df = add_forecast_targets(df, horizon_days=(1, 3, 5))
    return df


def train_model_for_ticker(ticker: str):
    print(f"\n=== {ticker} ===")
    try:
        df = prepare_with_insider_and_targets(ticker)
    except ValueError as e:
        # e.g. yf.download found no data
        print(f"⚠️  Skipping {ticker}: {e}")
        return

    # skip if no rows remain
    if df.shape[0] == 0:
        print(f"⚠️  Skipping {ticker}: no data after feature engineering")
        return

    for target_col in TARGET_COLUMNS:
        if target_col not in df.columns:
            print(f"⚠️  Missing {target_col} for {ticker}, skipping.")
            continue

        X = df[FEATURE_COLUMNS]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        model_path = os.path.join(MODEL_DIR, f"{ticker}_{target_col}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ {ticker} | {target_col:<10} — Acc: {acc:.3f} → {model_path}")


def train_all_models():
    for ticker in TICKERS:
        train_model_for_ticker(ticker)


if __name__ == "__main__":
    train_all_models()
