# train_forecast_model_v1.2

import os
import pickle

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from forecast_feature_engineering import build_feature_dataframe, add_forecast_targets
from data.etl_insider import fetch_insider_trades

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = [
    "NVO", "AAPL", "PFE", "NFLX", "AMD", "NVDA", "PYPL", "GOOG", "SMCI",
    "CNC", "CRWD", "META", "MRNA", "DDOG", "UNH", "SHOP", "PLTR", "TSM",
    "FIG","SNOW","TSM","MP","OPEN","DUOL","BSX","SMCI","JNJ","AXP","TSLA","SHOP"
    ,"ZM","META"
]

TARGET_COLUMNS  = ["target_1d", "target_3d", "target_5d"]
BASE_FEATURES   = [
    "close", "volume",
    "return_1d", "return_3d", "return_5d",
    "ma_5", "ma_10", "ma_20",
    "volatility_5d", "volatility_10d",
    "rsi_14", "macd", "macd_signal",
    "bollinger_upper", "bollinger_lower", "bollinger_width",
    "volume_zscore", "volume_spike",
    "sentiment_score",
]
INSIDER_FEATURES = ["insider_net_shares", "insider_7d", "insider_21d"]

FEATURE_COLUMNS = BASE_FEATURES + INSIDER_FEATURES

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_with_insider(ticker: str):
    # 1) build price & tech + targets
    df = build_feature_dataframe(ticker)
    df = add_forecast_targets(df, horizon_days=(1, 3, 5))

    # ensure we have a date column
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})

    df["date"] = pd.to_datetime(df["date"])

    # 2) pull insider trades
    ins = fetch_insider_trades(ticker, mode="sheet-first")
    ins = ins.rename(columns={"ds": "date", "net_shares": "insider_net_shares"})
    ins["date"] = pd.to_datetime(ins["date"])

    # 3) merge
    df = df.merge(ins[["date", "insider_net_shares"]], on="date", how="left")
    df["insider_net_shares"] = df["insider_net_shares"].fillna(0.0)

    # 4) rolling summaries
    df["insider_7d"]  = df["insider_net_shares"].rolling(window=7).sum().fillna(0)
    df["insider_21d"] = df["insider_net_shares"].rolling(window=21).sum().fillna(0)

    # drop any rows missing our base features or targets
    df = df.dropna(subset=BASE_FEATURES + TARGET_COLUMNS)
    return df


def train_model_for_ticker(ticker):
    print(f"\n=== {ticker} ===")
    df = prepare_with_insider(ticker)

    for target_col in TARGET_COLUMNS:
        if target_col not in df.columns:
            print(f"⚠️  Missing {target_col} for {ticker}, skipping.")
            continue

        # select X/y
        X = df[FEATURE_COLUMNS]
        y = df[target_col]

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # train
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)

        # eval
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        # persist
        model_path = os.path.join(MODEL_DIR, f"{ticker}_{target_col}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ {ticker} | {target_col:<10} — Acc: {acc:.3f} → {model_path}")


def train_all_models():
    for ticker in TICKERS:
        train_model_for_ticker(ticker)


if __name__ == "__main__":
    train_all_models()
