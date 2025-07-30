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
    print("📂 Current working directory:", os.getcwd())

    tickers = load_tickers()
    print(f"🔍 Loaded tickers: {tickers}")
    if not tickers:
        print("⚠️ No tickers found.")
        return

    eval_df = run_auto_retrain_all(tickers)

    if isinstance(eval_df, pd.DataFrame):
        print(f"📊 eval_df shape: {eval_df.shape}")
    else:
        print("❌ eval_df is not a DataFrame")
        return

    if not eval_df.empty:
        output_path = "forecast_metrics.csv"  # 🔄 Save directly in current dir
        eval_df.to_csv(output_path, index=False)
        print("📈 Saved forecast_metrics.csv at:", output_path)

        if os.path.exists(output_path):
            print("✅ File successfully created at:", output_path)
        else:
            print("❌ File was NOT found after saving!")
    else:
        print("⚠️ No evaluation metrics to save.")

    print("✅ Retraining complete.")

if __name__ == "__main__":
    main()
