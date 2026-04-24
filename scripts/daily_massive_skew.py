#!/usr/bin/env python3
"""
scripts/daily_massive_skew.py
─────────────────────────────────────────────────────────────────────────────
Fetches 25-delta options skew from Massive API for all tickers.
Saves to options_skew_history table.

Useful when UW API is rate limited — Massive Options Starter ($29/mo) has
unlimited API calls.

Usage:
    python scripts/daily_massive_skew.py                   # today
    python scripts/daily_massive_skew.py --date 2026-04-24 # specific date
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import sqlite3
import time
import argparse
from datetime import date, datetime
from pathlib import Path

ROOT = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DB_PATH = ROOT / "accuracy.db"


def load_tickers() -> list[str]:
    tickers = []
    for fname in ["tickers.txt", "tickers_watchlist.txt"]:
        fp = ROOT / fname
        if fp.exists():
            tickers += [t.strip().upper() for t in fp.read_text().splitlines()
                        if t.strip() and not t.startswith("#")]
    return list(dict.fromkeys(tickers))


def run_snapshot(snapshot_date: str = None):
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    print(f"\n{'='*60}")
    print(f"  Massive Skew Snapshot — {snapshot_date}")
    print(f"{'='*60}")

    tickers = load_tickers()
    print(f"Tickers: {len(tickers)}")

    # Init table if missing
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_skew_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                skew_25d    REAL,
                iv_rank     REAL,
                skew_signal TEXT NOT NULL,
                source      TEXT,
                created_at  TEXT NOT NULL,
                UNIQUE(date, ticker)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skew_date ON options_skew_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skew_ticker ON options_skew_history(ticker)")
        conn.commit()

    # Add source column if missing (migration)
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute("ALTER TABLE options_skew_history ADD COLUMN source TEXT DEFAULT 'unknown'")
            conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    from features.massive_options import get_25delta_skew_massive

    ok = fail = 0
    now = datetime.now().isoformat()

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i:>3}/{len(tickers)}] {ticker:<6} ", end="", flush=True)

            try:
                result = get_25delta_skew_massive(ticker)
                if result.get("skew_25d") is not None:
                    conn.execute("""
                        INSERT OR REPLACE INTO options_skew_history
                            (date, ticker, skew_25d, iv_rank, skew_signal, source, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (snapshot_date, ticker, result["skew_25d"],
                          result.get("iv_rank"), result["skew_signal"],
                          "massive", now))
                    print(f"skew={result['skew_25d']:+.4f} {result['skew_signal']}")
                    ok += 1
                else:
                    print(f"FAIL: {result.get('error', 'no data')}")
                    fail += 1
            except Exception as e:
                print(f"ERROR: {e}")
                fail += 1

            # Commit every 20 tickers to avoid losing progress
            if i % 20 == 0:
                conn.commit()

            # Rate limit — be gentle (Massive is unlimited but still)
            time.sleep(0.1)

        conn.commit()

    print(f"\n{'='*60}")
    print(f"  Massive skew: {ok} ok  {fail} failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_snapshot(args.date)
