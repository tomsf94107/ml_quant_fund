#!/usr/bin/env python3
"""
scripts/sector_accuracy.py

Per-sector accuracy report. Joins predictions + outcomes from accuracy.db
with sector/bucket info from tickers_metadata.csv.

By default shows only ACTIVE tickers (those in tickers.txt).
Use --include-inactive to also show historical/dropped tickers.

Usage:
    python scripts/sector_accuracy.py                       # active only, all-time
    python scripts/sector_accuracy.py --days 30             # last 30 days
    python scripts/sector_accuracy.py --horizon 1           # only h=1d
    python scripts/sector_accuracy.py --buy-only            # only BUY signals
    python scripts/sector_accuracy.py --include-inactive    # include old tickers
    python scripts/sector_accuracy.py --csv                 # CSV output
"""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

ROOT = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
DB_PATH = ROOT / "accuracy.db"
META_PATH = ROOT / "tickers_metadata.csv"
TICKERS_PATH = ROOT / "tickers.txt"


def load_metadata() -> pd.DataFrame:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}")
    return pd.read_csv(META_PATH)


def load_active_tickers() -> set:
    if not TICKERS_PATH.exists():
        raise FileNotFoundError(f"Missing {TICKERS_PATH}")
    return set(
        line.strip().upper()
        for line in TICKERS_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def load_predictions_outcomes(days: int | None = None,
                               horizon: int | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    where_clauses = []
    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        where_clauses.append(f"p.prediction_date >= '{cutoff}'")
    if horizon is not None:
        where_clauses.append(f"p.horizon = {horizon}")
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
        SELECT p.ticker, p.prediction_date, p.horizon,
               p.prob_up, p.signal, p.confidence,
               o.actual_up, o.actual_return
        FROM predictions p
        JOIN outcomes    o ON p.ticker          = o.ticker
                          AND p.prediction_date = o.prediction_date
                          AND p.horizon         = o.horizon
        WHERE {where_sql}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def compute_sector_accuracy(joined: pd.DataFrame, buy_only: bool = False) -> pd.DataFrame:
    if buy_only:
        joined = joined[joined["signal"] == "BUY"].copy()

    if joined.empty:
        return pd.DataFrame()

    if buy_only:
        joined["pred_up"] = 1
    else:
        joined["pred_up"] = (joined["prob_up"] >= 0.5).astype(int)

    joined["correct"] = (joined["pred_up"] == joined["actual_up"]).astype(int)

    grouped = joined.groupby("bucket", dropna=False).agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
        avg_prob_up=("prob_up", "mean"),
        avg_return=("actual_return", "mean"),
        n_tickers=("ticker", "nunique"),
    ).reset_index()

    grouped = grouped.round({
        "accuracy": 3,
        "avg_prob_up": 3,
        "avg_return": 4,
    })
    return grouped.sort_values("n", ascending=False)


def print_section(title: str, df: pd.DataFrame, total_rows: int):
    if df.empty:
        print(f"\n{title}: No data")
        return
    print(f"\n{title}")
    print("=" * 80)
    print(df.to_string(index=False))
    overall_n = df["n"].sum()
    weighted_acc = (df["accuracy"] * df["n"]).sum() / overall_n if overall_n else 0
    print(f"\n  Subtotal: {overall_n} predictions across {len(df)} sectors, weighted accuracy {weighted_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Per-sector accuracy report")
    parser.add_argument("--days", type=int, default=None,
                        help="Lookback days (default: all-time)")
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5], default=None,
                        help="Single horizon to analyze (default: all)")
    parser.add_argument("--buy-only", action="store_true",
                        help="Only count BUY signals")
    parser.add_argument("--include-inactive", action="store_true",
                        help="Also show inactive/historical tickers (default: active only)")
    parser.add_argument("--csv", action="store_true",
                        help="Output as CSV instead of formatted table")
    args = parser.parse_args()

    meta = load_metadata()
    active = load_active_tickers()
    joined = load_predictions_outcomes(days=args.days, horizon=args.horizon)

    if joined.empty:
        print("No prediction outcomes found for the given filters.")
        return

    # Merge metadata
    joined = joined.merge(meta[["ticker", "bucket", "tier"]],
                          on="ticker", how="left")
    joined["bucket"] = joined["bucket"].fillna("UNKNOWN")
    joined["is_active"] = joined["ticker"].isin(active)

    # Title
    title_filters = []
    if args.days:
        title_filters.append(f"last {args.days} days")
    if args.horizon:
        title_filters.append(f"h={args.horizon}d")
    if args.buy_only:
        title_filters.append("BUY signals only")
    filters_str = " (" + ", ".join(title_filters) + ")" if title_filters else ""

    if args.csv:
        active_joined = joined[joined["is_active"]]
        active_acc = compute_sector_accuracy(active_joined, buy_only=args.buy_only)
        print(active_acc.to_csv(index=False))
        return

    print(f"\n{'='*80}")
    print(f"Per-Sector Accuracy{filters_str}")
    print(f"{'='*80}")
    print(f"Active tickers: {len(active)} (from tickers.txt)")
    print(f"Total predictions in window: {len(joined)}")

    # Active sector accuracy
    active_joined = joined[joined["is_active"]]
    active_acc = compute_sector_accuracy(active_joined, buy_only=args.buy_only)
    print_section("ACTIVE TICKERS — Per-Sector Accuracy", active_acc, len(active_joined))

    # Inactive sector accuracy
    if args.include_inactive:
        inactive_joined = joined[~joined["is_active"]]
        if not inactive_joined.empty:
            inactive_acc = compute_sector_accuracy(inactive_joined, buy_only=args.buy_only)
            print_section("INACTIVE/HISTORICAL TICKERS", inactive_acc, len(inactive_joined))
            inactive_tickers = sorted(inactive_joined["ticker"].unique())
            print(f"\n  Inactive tickers: {', '.join(inactive_tickers)}")

    # Overall
    if args.buy_only:
        all_buys = joined[joined["signal"] == "BUY"].copy()
        all_buys["correct"] = (all_buys["actual_up"] == 1).astype(int)
        if not all_buys.empty:
            print(f"\nOverall (all data, BUY only): {all_buys['correct'].mean():.3f} on {len(all_buys)} BUYs")
    else:
        joined["pred_up"] = (joined["prob_up"] >= 0.5).astype(int)
        joined["correct"] = (joined["pred_up"] == joined["actual_up"]).astype(int)
        print(f"\nOverall (all data): {joined['correct'].mean():.3f} on {len(joined)} predictions")

    print()


if __name__ == "__main__":
    main()
