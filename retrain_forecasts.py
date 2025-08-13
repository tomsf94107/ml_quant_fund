#!/usr/bin/env python3
"""
Nightly/Manual retrain entrypoint.

Guarantees:
- Writes a run log to forecast_logs/retrain_<UTC>.log
- Writes a JSON summary to forecast_logs/summary_<UTC>.json
- Always creates forecast_metrics.csv (headers if no data)
- Exits 0 for "nothing to do" cases; 1 only for real failures

Env knobs (optional):
- FORCE_RETRAIN=1        -> retrain all loaded tickers even if no prior logs
- TICKER_FILE=<path>     -> override tickers.csv location
- OUTPUT_PATH=<path>     -> override forecast_metrics.csv path
- LOG_LEVEL=INFO|DEBUG   -> override logging level
- GCP_SERVICE_ACCOUNT    -> passed through for any downstream code
"""

import os, sys, json, traceback, pathlib, logging, datetime as dt, tempfile, shutil
from typing import List

# Fast log streaming in Actions
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Disable SHAP during batch retrain for stability
os.environ["DISABLE_SHAP"] = "1"

ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)

# ---------- configuration ----------
TICKER_FILE = os.getenv("TICKER_FILE", "tickers.csv")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "forecast_metrics.csv")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# GCP service account passthrough if used elsewhere
if os.getenv("GCP_SERVICE_ACCOUNT"):
    os.environ["GCP_SERVICE_ACCOUNT_JSON"] = os.getenv("GCP_SERVICE_ACCOUNT")

# ---------- logging (console + file) ----------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

log_dir = pathlib.Path("forecast_logs")
log_dir.mkdir(parents=True, exist_ok=True)

ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
file_log_path = log_dir / f"retrain_{ts}.log"
fh = logging.FileHandler(file_log_path, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(fh)
logging.info("File logging enabled → %s", file_log_path)

# Run summary (always written)
stats = {
    "timestamp_utc": ts,
    "cwd": str(ROOT),
    "loaded": 0,
    "with_log": 0,
    "retrained": 0,
    "no_log": [],
    "errors": {},
    "force_retrain": os.getenv("FORCE_RETRAIN") == "1",
    "ticker_file": TICKER_FILE,
    "output_path": OUTPUT_PATH,
    "github_sha": os.getenv("GITHUB_SHA"),
    "github_ref": os.getenv("GITHUB_REF"),
}

# ---------- safe helpers ----------
def _write_csv_atomic(df, path: str):
    """Write CSV atomically to avoid partial files on abrupt termination."""
    tmp = pathlib.Path(tempfile.mkstemp(prefix="metrics_", suffix=".csv")[1])
    try:
        df.to_csv(tmp, index=False)
        shutil.move(str(tmp), path)
    finally:
        try:
            tmp.unlink(missing_ok=True)  # py3.8+: ignore type checker
        except Exception:
            pass


# ---------- imports that rely on repo path ----------
try:
    import pandas as pd
    from forecast_utils import run_auto_retrain_all, _latest_log, forecast_today_movement
except Exception as e:
    logging.error("Import failure: %s", e)
    traceback.print_exc()
    # still write summary
    with open(log_dir / f"summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({**stats, "errors": {"__import__": str(e)}}, f, indent=2)
    sys.exit(1)


# ---------- IO ----------
def load_tickers(path=TICKER_FILE) -> List[str]:
    p = pathlib.Path(path)
    if not p.exists():
        logging.error("tickers.csv not found at %s", p)
        return []
    with p.open("r", encoding="utf-8") as fh_:
        return [ln.strip().upper() for ln in fh_ if ln.strip()]


def latest_log_exists(ticker: str) -> bool:
    try:
        return bool(_latest_log(ticker))
    except Exception as e:
        logging.warning("latest_log check failed for %s: %s", ticker, e)
        return False


# ---------- main ----------
def main() -> int:
    logging.info("🚀 Starting scheduled retraining… CWD=%s", os.getcwd())

    raw = load_tickers()
    stats["loaded"] = len(raw)
    logging.info("🔍 Loaded %d tickers: %s", len(raw), raw)

    if not raw:
        logging.warning("No tickers found — writing header CSV and exiting 0.")
        _write_csv_atomic(pd.DataFrame(columns=["ticker", "mae", "mse", "r2"]), OUTPUT_PATH)
        return 0

    tickers_to_retrain: List[str] = []

    for t in raw:
        try:
            if not latest_log_exists(t):
                logging.info("No forecast log for %s — generating one…", t)
                try:
                    forecast_today_movement(t)
                except Exception as e:
                    logging.warning("Initial forecast failed for %s: %s", t, e)
            if latest_log_exists(t):
                tickers_to_retrain.append(t)
                stats["with_log"] += 1
            else:
                stats["no_log"].append(t)
        except Exception as e:
            stats["errors"][t] = str(e)

    if not tickers_to_retrain and os.getenv("FORCE_RETRAIN") == "1":
        logging.info("FORCE_RETRAIN=1 — retraining all loaded tickers.")
        tickers_to_retrain = raw[:]

    if not tickers_to_retrain:
        logging.warning("No tickers with valid logs — writing header CSV and exiting 0.")
        _write_csv_atomic(pd.DataFrame(columns=["ticker", "mae", "mse", "r2"]), OUTPUT_PATH)
        return 0

    logging.info("🔁 Final tickers for retraining: %s", tickers_to_retrain)

    try:
        eval_df = run_auto_retrain_all(tickers_to_retrain)
    except Exception as e:
        logging.error("run_auto_retrain_all crashed: %s", e)
        traceback.print_exc()
        return 1

    if not isinstance(eval_df, pd.DataFrame):
        logging.error("eval_df is not a DataFrame")
        return 1

    logging.info("📊 eval_df shape: %s", getattr(eval_df, "shape", None))

    try:
        if not eval_df.empty:
            _write_csv_atomic(eval_df, OUTPUT_PATH)
            stats["retrained"] = (
                int(eval_df["ticker"].nunique()) if "ticker" in eval_df.columns else len(tickers_to_retrain)
            )
            logging.info("Saved metrics → %s", OUTPUT_PATH)
        else:
            _write_csv_atomic(pd.DataFrame(columns=["ticker", "mae", "mse", "r2"]), OUTPUT_PATH)
            logging.warning("Empty eval_df — wrote header CSV.")
    except Exception as e:
        logging.error("Failed to write %s: %s", OUTPUT_PATH, e)
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        # always write summary JSON
        try:
            summary_path = pathlib.Path("forecast_logs") / f"summary_{ts}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logging.info("Wrote run summary JSON → %s", summary_path)
        except Exception as e:
            logging.error("Failed to write summary JSON: %s", e)
    sys.exit(code)
