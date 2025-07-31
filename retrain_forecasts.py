import os
os.environ["DISABLE_SHAP"] = "1"  # ✅ Disable SHAP for stability during batch retrain

import pandas as pd
from forecast_utils import run_auto_retrain_all, _latest_log, forecast_today_movement

TICKER_FILE = "tickers.csv"
OUTPUT_PATH = "forecast_metrics.csv"

def load_tickers(path=TICKER_FILE):
    if not os.path.exists(path):
        print("❌ tickers.csv not found.")
        return []
    return [line.strip().upper() for line in open(path, "r") if line.strip()]

def main():
    print("🚀 Starting scheduled retraining...")
    print("📂 Current working directory:", os.getcwd())

    raw_tickers = load_tickers()
    print(f"🔍 Loaded tickers: {raw_tickers}")
    if not raw_tickers:
        print("⚠️ No tickers found.")
        return

    tickers_to_retrain = []
    for ticker in raw_tickers:
        if not _latest_log(ticker):
            print(f"📉 No forecast log for {ticker} — generating one now...")
            forecast_today_movement(ticker)
        if _latest_log(ticker):
            tickers_to_retrain.append(ticker)
        else:
            print(f"❌ Still no log found for {ticker}, skipping.")

    if not tickers_to_retrain:
        print("⚠️ No tickers with valid logs, exiting.")
        return

    print(f"🔁 Final tickers for retraining: {tickers_to_retrain}")
    eval_df = run_auto_retrain_all(tickers_to_retrain)

    if isinstance(eval_df, pd.DataFrame):
        print(f"📊 eval_df shape: {eval_df.shape}")
    else:
        print("❌ eval_df is not a DataFrame — aborting.")
        return

    if not eval_df.empty:
        eval_df.to_csv(OUTPUT_PATH, index=False)
        print("📈 Saved forecast_metrics.csv at:", OUTPUT_PATH)
    else:
        print("⚠️ No evaluation metrics to save — writing fallback CSV.")
        pd.DataFrame(columns=["ticker", "mae", "mse", "r2"]).to_csv(OUTPUT_PATH, index=False)
        print("📝 Wrote empty forecast_metrics.csv with headers.")

    if os.path.exists(OUTPUT_PATH):
        print("✅ File successfully created at:", OUTPUT_PATH)
    else:
        print("❌ File was NOT found after saving!")

    print("✅ Retraining complete.")

if __name__ == "__main__":
    main()
