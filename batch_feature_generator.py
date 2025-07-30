# batch_feature_generator.py

import os
import pandas as pd
from forecast_feature_engineering import build_feature_dataframe, add_forecast_targets

# ✅ Default list (you can customize)
TICKER_LIST = [
    "AAPL", "NVO", "UNH", "MRNA", "PFE",
    "CRCL", "CNC", "PLTR", "TSM", "NVDA",
    "SMCI", "CRWD", "DDOG", "AMD", "GOOG",
    "META", "TSLA", "NFLX", "PYPL", "SHOP"
]

# ✅ Output directory
os.makedirs("data", exist_ok=True)

def generate_all_features(tickers=TICKER_LIST):
    for ticker in tickers:
        print(f"⏳ Processing {ticker}...")
        try:
            df = build_feature_dataframe(ticker)
            df = add_forecast_targets(df)
            path = f"data/{ticker}_features.csv"
            df.to_csv(path, index=False)
            print(f"✅ Saved {ticker} to {path}")
        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")

if __name__ == "__main__":
    generate_all_features()
