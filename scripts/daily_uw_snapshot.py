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
import requests
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS short_interest_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                market_date TEXT NOT NULL,
                si_float    REAL,
                days_to_cover REAL,
                si_signal   TEXT,
                updated_at  TEXT NOT NULL,
                UNIQUE(ticker, market_date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seasonality_cache (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker               TEXT NOT NULL,
                month                INTEGER NOT NULL,
                avg_change           REAL,
                positive_months_perc REAL,
                seasonal_signal      TEXT,
                updated_at           TEXT NOT NULL,
                UNIQUE(ticker, month)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyst_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                date            TEXT NOT NULL,
                analyst_score   REAL,
                upgrades_30d    INTEGER,
                downgrades_30d  INTEGER,
                avg_target      REAL,
                analyst_signal  TEXT,
                updated_at      TEXT NOT NULL,
                UNIQUE(ticker, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ftd_cache (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker     TEXT NOT NULL,
                date       TEXT NOT NULL,
                ftd_shares INTEGER,
                ftd_signal TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(ticker, date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_si_ticker ON short_interest_cache(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seas_ticker ON seasonality_cache(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyst_ticker ON analyst_cache(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ftd_ticker ON ftd_cache(ticker)")
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


def run_snapshot(snapshot_date: str = None, mode: str = "full"):
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

    # ── Skip slow signals in post_market mode ────────────────────────────
    if mode == "post_market":
        print(f"  Mode: post_market — dark pool + skew only")
        print(f"{'='*60}\n")
        return

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
            except Exception as earn_err:
                print(f"  {ticker} earnings error: {earn_err}")
                earn_fail += 1
                time.sleep(0.3)
        conn.commit()

    print(f"  Earnings:   {earn_ok} ok  {earn_fail} failed")

    # ── Short interest cache ──────────────────────────────────────────────
    print("  Fetching short interest...")
    si_ok = si_fail = 0
    with sqlite3.connect(DB_PATH) as conn:
        for ticker in tickers:
            try:
                r = requests.get(
                    f"https://api.unusualwhales.com/api/shorts/{ticker}/interest-float/v2",
                    headers=uw_hdrs, timeout=8
                )
                if r.status_code != 200:
                    si_fail += 1
                    time.sleep(0.2)
                    continue
                data = r.json().get("data", [])
                if data:
                    d = data[0]
                    si_float = float(d.get("si_float", 0) or 0)
                    dtc      = float(d.get("days_to_cover", 0) or 0)
                    sig      = "SQUEEZE" if si_float > 0.20 and dtc > 5 else                                "HIGH_SHORT" if si_float > 0.10 else                                "LOW_SHORT" if si_float < 0.02 else "NEUTRAL"
                    conn.execute("""
                        INSERT OR REPLACE INTO short_interest_cache
                            (ticker, market_date, si_float, days_to_cover, si_signal, updated_at)
                        VALUES (?,?,?,?,?,?)
                    """, (ticker, d.get("market_date", snapshot_date),
                          si_float, dtc, sig, now))
                    si_ok += 1
                else:
                    si_fail += 1
                time.sleep(0.2)
            except Exception as earn_err:
                si_fail += 1
                time.sleep(0.2)
        conn.commit()
    print(f"  Short interest: {si_ok} ok  {si_fail} failed")

    # ── Seasonality cache ─────────────────────────────────────────────────
    print("  Fetching seasonality...")
    from datetime import date as _date
    current_month = _date.today().month
    seas_ok = seas_fail = 0
    with sqlite3.connect(DB_PATH) as conn:
        for ticker in tickers:
            try:
                r = requests.get(
                    f"https://api.unusualwhales.com/api/seasonality/{ticker}/monthly",
                    headers=uw_hdrs, timeout=8
                )
                if r.status_code != 200:
                    seas_fail += 1
                    time.sleep(0.2)
                    continue
                data = r.json().get("data", [])
                for m in data:
                    month    = int(m.get("month", 0))
                    avg_ret  = float(m.get("avg_change", 0) or 0)
                    pos_pct  = float(m.get("positive_months_perc", 0.5) or 0.5)
                    sig      = "BULLISH" if avg_ret > 0.02 and pos_pct > 0.6 else                                "BEARISH" if avg_ret < -0.02 and pos_pct < 0.4 else "NEUTRAL"
                    conn.execute("""
                        INSERT OR REPLACE INTO seasonality_cache
                            (ticker, month, avg_change, positive_months_perc,
                             seasonal_signal, updated_at)
                        VALUES (?,?,?,?,?,?)
                    """, (ticker, month, avg_ret, pos_pct, sig, now))
                seas_ok += 1
                time.sleep(0.2)
            except Exception:
                seas_fail += 1
                time.sleep(0.2)
        conn.commit()
    print(f"  Seasonality:    {seas_ok} ok  {seas_fail} failed")

    # ── Analyst cache ─────────────────────────────────────────────────────
    print("  Fetching analyst ratings...")
    from datetime import datetime as _dt, timedelta as _td
    cutoff_dt = (_dt.now() - _td(days=30)).isoformat()
    anal_ok = anal_fail = 0
    with sqlite3.connect(DB_PATH) as conn:
        for ticker in tickers:
            try:
                r = requests.get(
                    f"https://api.unusualwhales.com/api/screener/analysts",
                    headers=uw_hdrs, params={"ticker": ticker}, timeout=8
                )
                if r.status_code != 200:
                    anal_fail += 1
                    time.sleep(0.2)
                    continue
                data    = r.json().get("data", [])
                recent  = [d for d in data if d.get("timestamp","") >= cutoff_dt]
                upgrades   = sum(1 for d in recent if d.get("action","").lower() in
                                ("upgrade","initiated","reiterated") and
                                d.get("recommendation","").lower() in ("buy","strong buy","outperform"))
                downgrades = sum(1 for d in recent if d.get("action","").lower() == "downgrade")
                targets    = [float(d["target"]) for d in recent if d.get("target")]
                avg_tgt    = sum(targets)/len(targets) if targets else 0.0
                score      = (upgrades - downgrades) / max(len(recent), 1)
                sig        = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
                conn.execute("""
                    INSERT OR REPLACE INTO analyst_cache
                        (ticker, date, analyst_score, upgrades_30d,
                         downgrades_30d, avg_target, analyst_signal, updated_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (ticker, snapshot_date, round(score,4),
                      upgrades, downgrades, round(avg_tgt,2), sig, now))
                anal_ok += 1
                time.sleep(0.2)
            except Exception:
                anal_fail += 1
                time.sleep(0.2)
        conn.commit()
    print(f"  Analyst:        {anal_ok} ok  {anal_fail} failed")

    # ── FTD cache ─────────────────────────────────────────────────────────
    print("  Fetching FTDs...")
    ftd_ok = ftd_fail = 0
    with sqlite3.connect(DB_PATH) as conn:
        for ticker in tickers:
            try:
                r = requests.get(
                    f"https://api.unusualwhales.com/api/shorts/{ticker}/ftds",
                    headers=uw_hdrs, timeout=8
                )
                if r.status_code != 200:
                    ftd_fail += 1
                    time.sleep(0.2)
                    continue
                data        = r.json().get("data", [])
                recent_ftds = sum(int(d.get("quantity", 0) or 0) for d in data[:5])
                sig         = "HIGH" if recent_ftds > 500_000 else                               "ELEVATED" if recent_ftds > 100_000 else "NORMAL"
                conn.execute("""
                    INSERT OR REPLACE INTO ftd_cache
                        (ticker, date, ftd_shares, ftd_signal, updated_at)
                    VALUES (?,?,?,?,?)
                """, (ticker, snapshot_date, recent_ftds, sig, now))
                ftd_ok += 1
                time.sleep(0.2)
            except Exception:
                ftd_fail += 1
                time.sleep(0.2)
        conn.commit()
    print(f"  FTDs:           {ftd_ok} ok  {ftd_fail} failed")

    # ── Wikipedia pageviews (FREE) ────────────────────────────────────────
    print("  Fetching Wikipedia pageviews...")
    from features.alt_data import init_wiki_table, fetch_wiki_pageviews, save_wiki_to_db, WIKI_MAPPING
    init_wiki_table()
    wiki_ok = wiki_skip = 0
    for ticker in tickers:
        if ticker.upper() not in WIKI_MAPPING:
            wiki_skip += 1
            continue
        try:
            pv = fetch_wiki_pageviews(ticker, days_back=30)
            if pv:
                save_wiki_to_db(ticker, pv)
                wiki_ok += 1
            time.sleep(0.1)  # Wikipedia: 100 req/sec limit, be nice
        except Exception:
            pass
    print(f"  Wikipedia:      {wiki_ok} ok  {wiki_skip} no mapping")

    # ── SEC 8-K filings (FREE) ────────────────────────────────────────────
    print("  Fetching SEC 8-K filings...")
    from features.alt_data import init_sec_table, fetch_recent_8k, save_8k_to_db
    init_sec_table()
    sec_ok = sec_fail = 0
    for ticker in tickers:
        try:
            filings = fetch_recent_8k(ticker, days_back=30)
            if filings:
                save_8k_to_db(ticker, filings)
                sec_ok += 1
            time.sleep(0.2)  # SEC: 10 req/sec limit
        except Exception:
            sec_fail += 1
    print(f"  SEC 8-K:        {sec_ok} ok  {sec_fail} failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--mode", default="full", choices=["full", "post_market"],
                        help="full = all signals, post_market = dark pool + skew only")
    args = parser.parse_args()
    run_snapshot(args.date, mode=args.mode)
