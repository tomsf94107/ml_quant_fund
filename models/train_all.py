# models/train_all.py
# ─────────────────────────────────────────────────────────────────────────────
# Offline batch trainer. Run this script to retrain all models.
# DO NOT import this from Streamlit pages — training happens offline.
#
# Usage:
#   python -m ml_quant_fund.models.train_all
#   python -m ml_quant_fund.models.train_all --tickers AAPL NVDA MSFT
#   python -m ml_quant_fund.models.train_all --horizon 1
#
# Outputs:  models/saved/{ticker}_target_{h}d.joblib  for each ticker × horizon
#           models/saved/training_report.csv           summary of all runs
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from features.builder import build_feature_dataframe, add_forecast_targets
from models.classifier import (
    train_model, TARGET_HORIZONS, MODEL_DIR
)

# ── Ticker universe (your original 28 + room to grow) ─────────────────────────
DEFAULT_TICKERS: list[str] = [
    "NVO",  "AAPL", "PFE",  "NFLX", "AMD",  "NVDA", "PYPL", "GOOG",
    "SMCI", "CNC",  "CRWD", "META", "MRNA", "DDOG", "UNH",  "SHOP",
    "PLTR", "TSM",  "FIG",  "SNOW", "MP",   "OPEN", "DUOL", "BSX",
    "JNJ",  "AXP",  "TSLA", "ZM",
]

TRAIN_START = "2018-01-01"   # 6+ years gives good regime diversity


# ══════════════════════════════════════════════════════════════════════════════

def train_one_ticker(
    ticker: str,
    horizons: tuple[int, ...] = TARGET_HORIZONS,
    verbose: bool = True,
) -> list[dict]:
    """
    Build features + train all horizon models for one ticker.
    Returns a list of result dicts (one per horizon).
    """
    rows = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {ticker}")
        print(f"{'─'*60}")

    # Build features once, reuse for all horizons
    try:
        df = build_feature_dataframe(ticker, start_date=TRAIN_START)
        df = add_forecast_targets(df, horizons=horizons)
    except Exception as e:
        print(f"  ✗ Feature build failed: {e}")
        for h in horizons:
            rows.append({
                "ticker": ticker, "horizon": h,
                "status": "FAILED", "error": str(e),
                "accuracy": None, "roc_auc": None,
                "brier_score": None, "n_train": None,
            })
        return rows

    for h in horizons:
        t0 = time.time()
        try:
            result = train_model(ticker, df, horizon=h, verbose=verbose)
            rows.append({
                "ticker":      ticker,
                "horizon":     h,
                "status":      "OK",
                "error":       None,
                "accuracy":    result.metrics["accuracy"],
                "roc_auc":     result.metrics["roc_auc"],
                "brier_score": result.metrics["brier_score"],
                "n_train":     result.metrics["n_train"],
                "elapsed_s":   round(time.time() - t0, 1),
            })
        except Exception as e:
            print(f"  ✗ Training failed for {ticker} horizon={h}d: {e}")
            rows.append({
                "ticker": ticker, "horizon": h,
                "status": "FAILED", "error": str(e),
                "accuracy": None, "roc_auc": None,
                "brier_score": None, "n_train": None,
            })

    return rows


def train_all(
    tickers: list[str] | None = None,
    horizons: tuple[int, ...] = TARGET_HORIZONS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train all tickers × all horizons. Returns a summary DataFrame.
    Also saves models/saved/training_report.csv.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    print(f"\n{'═'*60}")
    print(f"  ML Quant Fund — Batch Training")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Horizons: {horizons}")
    print(f"  Start   : {TRAIN_START}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*60}")

    all_rows = []
    for ticker in tickers:
        rows = train_one_ticker(ticker, horizons=horizons, verbose=verbose)
        all_rows.extend(rows)

    report = pd.DataFrame(all_rows)

    # Save report
    report_path = MODEL_DIR / "training_report.csv"
    report.to_csv(report_path, index=False)

    # Print summary
    ok    = report[report["status"] == "OK"]
    fail  = report[report["status"] == "FAILED"]

    print(f"\n{'═'*60}")
    print(f"  DONE — {len(ok)} models trained, {len(fail)} failed")
    if len(ok) > 0:
        print(f"  Mean accuracy : {ok['accuracy'].mean():.3f}")
        print(f"  Mean ROC-AUC  : {ok['roc_auc'].mean():.3f}")
        print(f"  Mean Brier    : {ok['brier_score'].mean():.3f}")
    if len(fail) > 0:
        print(f"\n  Failed tickers:")
        for _, row in fail.iterrows():
            print(f"    {row['ticker']} h={row['horizon']}d — {row['error']}")
    print(f"  Report saved → {report_path}")
    print(f"{'═'*60}\n")

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Quant Fund models")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated tickers to train (default: all 28)",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Single horizon to train (1, 3, or 5). Default: all three.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-model output",
    )
    args = parser.parse_args()

    horizons = (args.horizon,) if args.horizon else TARGET_HORIZONS
    train_all(
        tickers=args.tickers,
        horizons=horizons,
        verbose=not args.quiet,
    )
