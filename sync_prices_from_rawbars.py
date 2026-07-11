#!/usr/bin/env python3
"""
sync_prices_from_rawbars.py -- Keep daily_prices current from Massive's raw_bars.

WHY: daily_prices (read by backtests/validation) was filled by yfinance via
fetch_and_pead.py, which isn't scheduled -> it went stale (Jun 26). Meanwhile
massive_client ALREADY caches current adjusted daily bars to raw_bars in the SAME
prices.db (verified current to today, adjusted: raw_bars.close == daily_prices.adj_close
to 0.00% on overlapping dates). So instead of the fragile yfinance fetch, this copies
the fresh Massive bars into daily_prices.

ADDITIVE / SAFE:
  - INSERT OR IGNORE: only dates missing from daily_prices are added; existing rows
    (incl. the deep 2008-2021 history raw_bars lacks) are never touched or overwritten.
  - raw_bars.close is already split/div-adjusted (auto_adjust=true in massive_client),
    so it maps directly to daily_prices.adj_close.
  - Read-mostly: one INSERT OR IGNORE, no deletes, no schema changes.

Idempotent -- safe to run daily (or anytime). No network (reads raw_bars already in db).

RUN
  python sync_prices_from_rawbars.py
  python sync_prices_from_rawbars.py --dry-run     # show what would be added, write nothing
"""
import argparse, os, sqlite3, sys, datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--prices-db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    db = a.prices_db or os.path.join(root, "prices.db")
    if not os.path.isfile(db):
        print(f"[STOP] prices.db not found at {db}"); sys.exit(1)

    conn = sqlite3.connect(db, timeout=60)
    cur = conn.cursor()

    # sanity: both tables exist
    tabs = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("raw_bars", "daily_prices"):
        if t not in tabs:
            print(f"[STOP] table {t} not in prices.db"); sys.exit(1)

    # current state
    dp_max = cur.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
    rb_max = cur.execute("SELECT MAX(d) FROM raw_bars").fetchone()[0]
    dp_rows_before = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    print(f"  daily_prices max date: {dp_max}  ({dp_rows_before:,} rows)")
    print(f"  raw_bars    max date: {rb_max}")

    if rb_max and dp_max and rb_max <= dp_max:
        print("  daily_prices already >= raw_bars; nothing to add.")
        conn.close(); return

    # how many candidate rows would be added (raw_bars dates strictly after dp_max,
    # or ALL raw_bars if daily_prices is somehow empty)
    if dp_max:
        cand = cur.execute(
            "SELECT COUNT(*) FROM raw_bars WHERE d > ? AND close IS NOT NULL", (dp_max,)
        ).fetchone()[0]
        preview = cur.execute(
            "SELECT d, COUNT(*) FROM raw_bars WHERE d > ? AND close IS NOT NULL "
            "GROUP BY d ORDER BY d", (dp_max,)
        ).fetchall()
    else:
        cand = cur.execute("SELECT COUNT(*) FROM raw_bars WHERE close IS NOT NULL").fetchone()[0]
        preview = []

    print(f"  candidate new rows (raw_bars date > {dp_max}): {cand:,}")
    if preview:
        print(f"  new dates to add: {preview[0][0]} .. {preview[-1][0]} ({len(preview)} trading days)")

    if a.dry_run:
        print("  [DRY RUN] no rows written.")
        conn.close(); return

    # additive upsert: only missing (ticker,date) rows; map d->date, close->adj_close.
    # INSERT OR IGNORE respects the PK (ticker,date) so existing rows are untouched.
    cur.execute("""
        INSERT OR IGNORE INTO daily_prices (ticker, date, adj_close)
        SELECT ticker, d, close FROM raw_bars
        WHERE close IS NOT NULL AND d > ?
    """, (dp_max if dp_max else "0000-00-00",))
    conn.commit()

    dp_rows_after = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    dp_max_after = cur.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
    added = dp_rows_after - dp_rows_before
    print(f"  ADDED {added:,} rows | daily_prices now: {dp_rows_after:,} rows, max date {dp_max_after}")
    conn.close()
    print("  done.")

if __name__ == "__main__":
    main()
