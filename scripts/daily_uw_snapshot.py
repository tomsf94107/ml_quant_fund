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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS institutional_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT NOT NULL,
                ticker       TEXT NOT NULL,
                inst_score   REAL,
                units_changed INTEGER,
                inst_signal  TEXT NOT NULL,
                filing_date  TEXT,
                created_at   TEXT NOT NULL,
                UNIQUE(date, ticker)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inst_date ON institutional_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inst_ticker ON institutional_history(ticker)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                report_date     TEXT NOT NULL,
                report_time     TEXT,
                expected_move   REAL,
                pre_drift_3d    REAL,
                post_drift_3d   REAL,
                actual_eps      REAL,
                est_eps         REAL,
                updated_at      TEXT NOT NULL,
                UNIQUE(ticker, report_date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_earn_ticker ON earnings_cache(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_earn_date ON earnings_cache(report_date)")
        conn.commit()

    print("Tables ready")


def load_tickers() -> list[str]:
    tickers_file = ROOT / "tickers.txt"
    tickers = [
        t.strip() for t in tickers_file.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]
    # Add watchlist tickers
    wl_file = ROOT / "tickers_watchlist.txt"
    if wl_file.exists():
        wl = [t.strip().upper() for t in wl_file.read_text().splitlines()
              if t.strip() and not t.startswith("#")]
        tickers = list(dict.fromkeys(tickers + wl))
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

            # Institutional ownership (quarterly — cache weekly)
            try:
                from features.uw_signals import get_institutional_score
                inst = get_institutional_score(ticker)
                if inst.get("error") is None and inst.get("inst_score") is not None:
                    conn.execute("""
                        INSERT OR REPLACE INTO institutional_history
                            (date, ticker, inst_score, units_changed, inst_signal, filing_date, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (snapshot_date, ticker, inst["inst_score"],
                          inst["units_changed"], inst["inst_signal"],
                          inst["filing_date"], now))
                    print(f"inst={inst['inst_signal']}", end=" ")
                else:
                    print(f"inst=err", end=" ")
            except Exception:
                print(f"inst=err", end=" ")
            print()

            # Rate limit — 120 req/min = 0.5s per ticker
            time.sleep(0.5)

        conn.commit()

    print(f"\n{'='*60}")
    print(f"  Dark pool:  {dp_ok} ok  {dp_fail} failed")
    print(f"  Skew:       {skew_ok} ok  {skew_fail} failed")
    print(f"  Institutional: saved to DB")

    # ── Earnings cache ────────────────────────────────────────────────────
    print("  Fetching earnings calendar from UW...")
    uw_key  = os.getenv("UW_API_KEY", "")
    uw_hdrs = {"Authorization": f"Bearer {uw_key}"}
    earn_ok = earn_fail = 0

    with sqlite3.connect(DB_PATH) as conn:
        for ticker in tickers:
            try:
                r = requests.get(
                    f"https://api.unusualwhales.com/api/earnings/{ticker}",
                    headers=uw_hdrs, timeout=8
                )
                if r.status_code != 200:
                    earn_fail += 1
                    time.sleep(0.3)
                    continue
                data = r.json().get("data", [])
                for e in data:
                    rd = e.get("report_date", "")
                    if not rd:
                        continue
                    exp_move  = float(e.get("expected_move_perc", 0) or 0)
                    pre_drift = float(e.get("pre_earnings_move_3d", 0) or 0) if e.get("pre_earnings_move_3d") else None
                    post_drift= float(e.get("post_earnings_move_3d", 0) or 0) if e.get("post_earnings_move_3d") else None
                    try:
                        actual_eps = float(e.get("actual_eps") or 0) if e.get("actual_eps") else None
                        est_eps    = float(e.get("street_mean_est") or 0) if e.get("street_mean_est") else None
                    except Exception:
                        actual_eps = est_eps = None
                    conn.execute("""
                        INSERT OR REPLACE INTO earnings_cache
                            (ticker, report_date, report_time, expected_move,
                             pre_drift_3d, post_drift_3d, actual_eps, est_eps, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ticker, rd, e.get("report_time",""), exp_move,
                          pre_drift, post_drift, actual_eps, est_eps, now))
                earn_ok += 1
                time.sleep(0.3)
            except Exception:
                earn_fail += 1
                time.sleep(0.3)
        conn.commit()

    print(f"  Earnings:   {earn_ok} ok  {earn_fail} failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()
    run_snapshot(args.date)
