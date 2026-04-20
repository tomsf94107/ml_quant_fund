#!/usr/bin/env python3
"""
scripts/daily_uw_snapshot.py
─────────────────────────────────────────────────────────────────────────────
Fetches dark pool + options skew for all tickers from Unusual Whales API.
Saves to accuracy.db tables:
  - dark_pool_history
  - options_skew_history

Run daily after market close (cron: 0 5 * * 2-6 Vietnam = 4 PM ET)
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

ROOT    = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
DB_PATH = ROOT / "accuracy.db"

sys.path.insert(0, str(ROOT))


def init_tables():
    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS dark_pool_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT NOT NULL,
                ticker       TEXT NOT NULL,
                dp_ratio     REAL NOT NULL,
                dp_volume    INTEGER NOT NULL,
                total_volume INTEGER NOT NULL,
                dp_signal    TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                UNIQUE(date, ticker)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_skew_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                skew_25d    REAL,
                put_iv_25d  REAL,
                call_iv_25d REAL,
                iv_rank     REAL,
                skew_signal TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE(date, ticker)
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_date ON dark_pool_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_ticker ON dark_pool_history(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skew_date ON options_skew_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skew_ticker ON options_skew_history(ticker)")
        conn.commit()

    print("Tables ready")


def load_tickers() -> list[str]:
    tickers_file = ROOT / "tickers.txt"
    tickers = [
        t.strip() for t in tickers_file.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]
    # Skip ETFs for options skew (no options data)
    return tickers


ETF_LIST = {"SPY","QQQ","GLD","SLV","XLF","XLE","XLV","XLI","XLU"}


def run_snapshot(snapshot_date: str = None):
    if snapshot_date is None:
        snapshot_date = str(date.today() - timedelta(days=1))

    print(f"\n{'='*60}")
    print(f"  UW Daily Snapshot — {snapshot_date}")
    print(f"{'='*60}")

    init_tables()
    tickers = load_tickers()

    from features.dark_pool import get_dark_pool_ratio
    from features.options_flow import get_25delta_skew

    from datetime import datetime
    now = datetime.now().isoformat()

    dp_ok = dp_fail = skew_ok = skew_fail = 0

    with sqlite3.connect(DB_PATH) as conn:
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i:3d}/{len(tickers)}] {ticker:<6}", end=" ", flush=True)

            # Dark pool
            try:
                dp = get_dark_pool_ratio(ticker, snapshot_date)
                if dp.get("error") is None and dp["dp_ratio"] > 0:
                    conn.execute("""
                        INSERT OR REPLACE INTO dark_pool_history
                            (date, ticker, dp_ratio, dp_volume, total_volume, dp_signal, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (snapshot_date, ticker, dp["dp_ratio"],
                          dp["dp_volume"], dp["total_volume"],
                          dp["dp_signal"], now))
                    dp_ok += 1
                    print(f"dp={dp['dp_ratio']:.1%}", end=" ")
                else:
                    dp_fail += 1
                    print(f"dp=err", end=" ")
            except Exception as e:
                dp_fail += 1
                print(f"dp=err", end=" ")

            # Options skew (skip ETFs)
            if ticker not in ETF_LIST:
                try:
                    skew = get_25delta_skew(ticker)
                    if skew.get("error") is None and skew.get("skew_25d") is not None:
                        conn.execute("""
                            INSERT OR REPLACE INTO options_skew_history
                                (date, ticker, skew_25d, put_iv_25d, call_iv_25d,
                                 iv_rank, skew_signal, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (snapshot_date, ticker, skew["skew_25d"],
                              skew["put_iv_25d"], skew["call_iv_25d"],
                              skew["iv_rank"], skew["skew_signal"], now))
                        skew_ok += 1
                        print(f"skew={skew['skew_25d']:+.3f} {skew['skew_signal']}")
                    else:
                        skew_fail += 1
                        print(f"skew=err")
                except Exception:
                    skew_fail += 1
                    print(f"skew=err")
            else:
                print(f"skew=ETF")

            # Rate limit — 120 req/min = 0.5s per ticker
            time.sleep(0.5)

        conn.commit()

    print(f"\n{'='*60}")
    print(f"  Dark pool:  {dp_ok} ok  {dp_fail} failed")
    print(f"  Skew:       {skew_ok} ok  {skew_fail} failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()
    run_snapshot(args.date)
