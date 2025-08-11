# retrain_forecasts.py

import os
import sys
import json
import traceback
import pathlib
import logging
from typing import List

# ✅ Make logs stream immediately in Actions
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ✅ Disable SHAP for stability during batch retrain
os.environ["DISABLE_SHAP"] = "1"

# ------------------------------------------------------------------------------
# Setup: stable working directory, logging, required folders
# ------------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

for p in ["forecast_logs", "models"]:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

TICKER_FILE = "tickers.csv"
OUTPUT_PATH = "forecast_metrics.csv"

# If running in Actions and you pass a JSON secret, surface it for your code.
GCP_SVC = os.getenv("GCP_SERVICE_ACCOUNT")
if GCP_SVC:
    try:
        # Some helper libs expect a file; others can read from env
        os.environ["GCP_SERVICE_ACCOUNT_JSON"] = GCP_SVC
        logging.info("GCP service account detected in environment.")
    except Exception:
        logging.warning("Could not expose GCP service account to env.")

# ------------------------------------------------------------------------------
# Imports that depend on repo path come after we pin CWD
# ------------------------------------------------------------------------------
try:
    import pandas as pd
    from forecast_utils import run_auto_retrain_all, _latest_log, forecast_today_movement
except Exception as e:
    logging.error("Failed to import modules. CWD=%s", os.getcwd())
    logging.error("sys.path=%s", sys.path)
    logging.error("Import error: %s", e)
    traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def load_tickers(path: str = TICKER_FILE) -> List[str]:
    if not os.path.exists(path):
        logging.error("❌ tickers.csv not found at %s", path)
        return []
    with open(path, "r") as f:
        out = [line.strip().upper() for line in f if line.strip()]
    return out


def latest_log_exists(ticker: str) -> bool:
    try:
        return bool(_latest_log(ticker))
    except Exception as e:
        logging.warning("latest_log check failed for %s: %s", ticker, e)
        return False

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> int:
    logging.info("🚀 Starting scheduled retraining…")
    logging.info("📂 Working directory: %s", os.getcwd())

    raw_tickers = load_tickers()
    logging.info("🔍 Loaded %d tickers: %s", len(raw_tickers), raw_tickers)

    if not raw_tickers:
        logging.warning("⚠️ No tickers found in %s. Exiting.", TICKER_FILE)
        # Graceful exit (0) so the cron doesn’t alert if list is intentionally empty
        return 0

    tickers_to_retrain: List[str] = []

    # Ensure each ticker has at least one fresh log
    for ticker in raw_tickers:
        try:
            if not latest_log_exists(ticker):
                logging.info("📉 No forecast log for %s — generating one…", ticker)
                try:
                    forecast_today_movement(ticker)
                except Exception as e:
                    logging.warning("Failed initial forecast for %s: %s", ticker, e)

            if latest_log_exists(ticker):
                tickers_to_retrain.append(ticker)
            else:
                logging.warning("❌ Still no log for %s — skipping.", ticker)
        except Exception as e:
            logging.warning("Ticker precheck failed for %s: %s", ticker, e)

    if not tickers_to_retrain:
        logging.warning("⚠️ No tickers with valid logs. Nothing to retrain.")
        return 0

    logging.info("🔁 Final tickers for retraining: %s", tickers_to_retrain)

    try:
        eval_df = run_auto_retrain_all(tickers_to_retrain)
    except Exception as e:
        logging.error("run_auto_retrain_all crashed: %s", e)
        traceback.print_exc()
        return 1

    if not isinstance(eval_df, pd.DataFrame):
        logging.error("❌ eval_df is not a DataFrame — aborting.")
        return 1

    logging.info("📊 eval_df shape: %s", getattr(eval_df, "shape", None))

    try:
        if not eval_df.empty:
            eval_df.to_csv(OUTPUT_PATH, index=False)
            logging.info("📈 Saved forecast_metrics.csv → %s", OUTPUT_PATH)
        else:
            logging.warning("⚠️ No evaluation metrics — writing fallback CSV.")
            pd.DataFrame(columns=["ticker", "mae", "mse", "r2"]).to_csv(OUTPUT_PATH, index=False)
            logging.info("📝 Wrote empty forecast_metrics.csv with headers.")
    except Exception as e:
        logging.error("Failed writing %s: %s", OUTPUT_PATH, e)
        traceback.print_exc()
        return 1

    if os.path.exists(OUTPUT_PATH):
        logging.info("✅ File successfully created at: %s", OUTPUT_PATH)
    else:
        logging.error("❌ File was NOT found after saving!")
        return 1

    logging.info("✅ Retraining complete.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception as e:
        logging.error("Retrain FAILED: %s", e)
        traceback.print_exc()
        sys.exit(1)
