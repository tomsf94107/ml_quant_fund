# retrain_forecasts.py
import pandas as pd
import os
from forecast_utils import run_auto_retrain_all

def load_tickers(path="tickers.csv"):
    if not os.path.exists(path):
        print("❌ tickers.csv not found.")
        return []
    return [line.strip().upper() for line in open(path, "r") if line.strip()]

def main():
    print("🚀 Starting scheduled retraining...")
    tickers = load_tickers()
    if not tickers:
        print("⚠️ No tickers found.")
        return

    run_auto_retrain_all(tickers)
    print("✅ Retraining complete.")

if __name__ == "__main__":
    main()
