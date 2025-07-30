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

    eval_df = run_auto_retrain_all(tickers)

    # ✅ Save evaluation results for GitHub Actions to upload
    if isinstance(eval_df, pd.DataFrame) and not eval_df.empty:
        eval_df.to_csv("forecast_metrics.csv", index=False)
        print("📈 Saved forecast_metrics.csv for upload.")
    else:
        print("⚠️ No evaluation metrics to save.")

    print("✅ Retraining complete.")


if __name__ == "__main__":
    main()
