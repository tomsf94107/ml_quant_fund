#!/usr/bin/env python3
"""
repair_stale_feeds.py -- find and repair per-ticker TRAILING-EDGE gaps in raw_bars.

WHY THIS EXISTS
  Neither existing tool repairs a feed that STOPPED:
    - price_cache.cached_daily() extends forward from last bar, but its gap write
      is `if gap is not None and not gap.empty:` -- an empty vendor response is a
      SILENT no-op. Nothing logs, nothing retries, nothing alerts.
    - backfill_raw_bars.py gates on EARLIEST-date depth ("already deep"), so a
      ticker with years of history passes even when its trailing edge is months
      stale.
  Result (found 2026-08-14): 10 tickers stale, CYBR dead ~6 months undetected.

WHAT IT DOES
  1. Scans raw_bars for every ticker whose MAX(d) is older than --max-age days.
  2. Refetches MAX(d)+1 .. today via the SAME vendor path production uses
     (massive_client.download auto_adjust=False -> raw bars).
  3. Writes via price_cache._write_raw (same upsert production uses).
  4. Verifies MAX(d) actually advanced, and REPORTS the outcome per ticker:
       REPAIRED      -- rows written, MAX(d) advanced
       NO-VENDOR-DATA-- vendor returned 0 rows => likely DELISTED / symbol dead
       NO-ADVANCE    -- rows written but MAX(d) did not move (investigate)
       FAILED        -- exception
  The NO-VENDOR-DATA case is the one the old code swallowed silently. It is
  reported loudly here: those tickers are candidates for universe RETIREMENT,
  not repair.

USAGE
  python scripts/repair_stale_feeds.py --dry-run
  python scripts/repair_stale_feeds.py
  python scripts/repair_stale_feeds.py --tickers CYBR,SATS,EA
  python scripts/repair_stale_feeds.py --max-age 3 --sleep 0.3

EXIT CODE
  0 = nothing stale, or all stale tickers repaired
  1 = one or more tickers could not be repaired (dead symbols / failures)
      -> suitable for cron alerting
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import date, timedelta

# ET-aware "today" -- VN local date is ahead of ET and causes phantom
# future-date requests to the vendor.
try:
    from zoneinfo import ZoneInfo
    def _today():
        return date.today() if os.environ.get("ML_QUANT_NAIVE_DATE") else \
            __import__("datetime").datetime.now(ZoneInfo("America/New_York")).date()
except Exception:
    def _today():
        return date.today()

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
PRICES_DB = os.path.join(ROOT, "prices.db")


def _read_lines(fn):
    p = os.path.join(ROOT, fn)
    if not os.path.isfile(p):
        return set()
    return {l.strip().upper() for l in open(p) if l.strip()}


def retired_set():
    """Tickers deliberately retired -- delisted names NEVER refresh, so alerting
    on them forever is an alarm that can never clear (and therefore gets
    ignored). Read from tickers_retired.csv written by ticker_lifecycle.py."""
    p = os.path.join(ROOT, "tickers_retired.csv")
    if not os.path.isfile(p):
        return set()
    out = set()
    import csv as _csv
    for i, r in enumerate(_csv.reader(open(p))):
        if i == 0 and r and r[0].strip().lower() in ("ticker", "symbol"):
            continue
        if r and r[0].strip():
            out.add(r[0].strip().upper())
    return out


def active_set():
    """The universe that SHOULD be fresh: runner + watchlist."""
    return _read_lines("tickers.txt") | _read_lines("tickers_watchlist.txt")


def stale_tickers(max_age_days, only=None):
    """Return [(ticker, last_bar, nrows)] whose MAX(d) is older than cutoff."""
    cutoff = (_today() - timedelta(days=max_age_days)).isoformat()
    con = sqlite3.connect(PRICES_DB, timeout=30)
    try:
        rows = con.execute(
            "SELECT ticker, MAX(d) AS last_bar, COUNT(*) AS n "
            "FROM raw_bars GROUP BY ticker HAVING last_bar < ? ORDER BY last_bar",
            (cutoff,)).fetchall()
    finally:
        con.close()
    if only:
        want = {t.strip().upper() for t in only}
        rows = [r for r in rows if r[0].upper() in want]
    return rows, cutoff


def max_d(ticker):
    con = sqlite3.connect(PRICES_DB, timeout=30)
    try:
        r = con.execute("SELECT MAX(d), COUNT(*) FROM raw_bars WHERE ticker=?",
                        (ticker,)).fetchone()
    finally:
        con.close()
    return r if r else (None, 0)


def repair(ticker, last_bar, args):
    """Refetch trailing gap for one ticker. Returns (status, detail)."""
    from datetime import datetime as _dt
    start = (_dt.strptime(last_bar, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
    end = _today().isoformat()
    if start > end:
        return "UP-TO-DATE", f"{start} > {end}"

    if args.dry_run:
        return "DRY", f"would fetch {start}..{end}"

    from features import massive_client as mc
    from features import price_cache as pc

    try:
        raw = mc.download(ticker, start=start, end=end, auto_adjust=False)
    except Exception as e:
        return "FAILED", f"fetch: {e.__class__.__name__}: {e}"[:120]

    if raw is None or len(raw) == 0:
        # THE CASE THE OLD CODE SWALLOWED SILENTLY
        return "NO-VENDOR-DATA", f"vendor returned 0 rows for {start}..{end}"

    try:
        con = pc._conn()
        pc._write_raw(con, ticker, raw)
        con.commit()
        con.close()
    except Exception as e:
        return "FAILED", f"write: {e.__class__.__name__}: {e}"[:120]

    new_max, n = max_d(ticker)
    if new_max and new_max > last_bar:
        return "REPAIRED", f"{last_bar} -> {new_max} (+{len(raw)} rows, n={n})"
    return "NO-ADVANCE", f"wrote {len(raw)} rows but MAX(d) still {new_max}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=int, default=3,
                    help="a feed is stale if MAX(d) older than this many days (default 3)")
    ap.add_argument("--tickers", help="comma list; default = all stale in raw_bars")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.3, help="pause between tickers")
    ap.add_argument("--all", action="store_true",
                    help="scan every ticker in raw_bars, including retired/inactive")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT, PRICES_DB
    if args.root:
        ROOT = os.path.expanduser(args.root)
        PRICES_DB = os.path.join(ROOT, "prices.db")
    if not os.path.isfile(PRICES_DB):
        sys.exit(f"FATAL: {PRICES_DB} not found")
    sys.path.insert(0, ROOT)

    only = args.tickers.split(",") if args.tickers else None
    rows, cutoff = stale_tickers(args.max_age, only)

    retired = set() if args.all else retired_set()
    active = set() if args.all else active_set()
    skipped_retired, skipped_inactive = [], []
    if not args.all:
        kept = []
        for r in rows:
            t = r[0].upper()
            if t in retired:
                skipped_retired.append(t)
            elif active and t not in active:
                skipped_inactive.append(t)
            else:
                kept.append(r)
        rows = kept

    print(f"# repair_stale_feeds  db={PRICES_DB}")
    print(f"# today(ET)={_today()}  cutoff={cutoff}  stale={len(rows)}")
    if skipped_retired:
        print(f"# RETIRED (skipped, will never refresh): {', '.join(sorted(skipped_retired))}")
    if skipped_inactive:
        print(f"# not in active universe (skipped): {', '.join(sorted(skipped_inactive)[:12])}"
              + (" ..." if len(skipped_inactive) > 12 else ""))
    if args.dry_run:
        print("# DRY-RUN: no fetches, no writes")
    if not rows:
        print("# nothing stale.")
        return 0
    print()

    results = []
    for ticker, last_bar, n in rows:
        status, detail = repair(ticker, last_bar, args)
        results.append((ticker, last_bar, n, status, detail))
        print(f"{ticker:8s} last={last_bar}  n={n:<6d} {status:<15s} {detail}")
        if not args.dry_run:
            time.sleep(args.sleep)

    # summary
    print()
    by = {}
    for r in results:
        by.setdefault(r[3], []).append(r[0])
    for status in sorted(by):
        print(f"# {status:<15s} {len(by[status]):3d}  {', '.join(by[status])}")

    dead = by.get("NO-VENDOR-DATA", [])
    if dead:
        print(f"\n# ACTION REQUIRED: vendor has no data for {', '.join(dead)}.")
        print("# Likely delisted/renamed. Verify the corporate action, then RETIRE")
        print("# from the universe -- do not leave them emitting predictions on a")
        print("# stale panel.")

    bad = sum(len(by.get(k, [])) for k in ("NO-VENDOR-DATA", "NO-ADVANCE", "FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
