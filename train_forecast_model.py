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

def prepare_with_insider(ticker):
    df = build_feature_dataframe(ticker)
    try:
        ins = fetch_insider_trades(ticker, mode="sheet-first")
        # detect date column
        if "ds" in ins.columns:
            ins_date_col = "ds"
        elif "date" in ins.columns:
            ins_date_col = "date"
        else:
            # nothing to merge
            raise KeyError(f"No date column in insider trades: {ins.columns.tolist()!r}")

        # parse and normalize
        ins[ins_date_col] = pd.to_datetime(ins[ins_date_col]).dt.normalize()
        ins_ts = ins.set_index(ins_date_col)["net_shares"].to_dict()

        # map into df
        df["insider_net_shares"] = (
            df["date"].dt.normalize().map(ins_ts).fillna(0).astype(float)
        )
    except Exception as e:
        print(f"⚠️ prepare_with_insider: insider merge failed for {ticker}: {e}")
        df["insider_net_shares"] = 0.0

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
