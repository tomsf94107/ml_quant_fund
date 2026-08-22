#!/usr/bin/env python3
"""
reconcile_universe.py -- reconcile raw_bars membership against the universe files.

WHY (2026-08-22)
  Two controls guard feed freshness and NEITHER covers per-ticker x unenrolled:

    feed_freshness_check  registers raw_bars as ONE feed and checks table-level
                          MAX(d). On 2026-08-22 that was 2026-08-21, so it
                          passed -- while IBM and JPM sat 9 days stale.
    repair_stale_feeds    is per-ticker but scoped to tickers.txt u watchlist u
                          metadata - retired. It printed the gap on every run:
                          "not in universe or metadata (skipped): BNED, CYBR,
                          IBM, JPM, RC" -- and nothing read that line.

  Root cause: massive_client.download() write-through-caches to raw_bars via
  price_cache (monitor_ticker.py:4059). Any ad-hoc `monitor <TICKER>` accretes
  rows permanently. IBM/JPM/BNED/RC each held 1,977-2,533 rows while appearing
  in NO universe file. Handoff section 1's "0 stale feeds" was true only of
  enrolled names.

WHAT IT CHECKS
  1. orphans   = raw_bars tickers outside tickers.txt u watchlist u metadata
                 u retired. These are fed by nothing and protected by nothing.
  2. staleness = per-ticker MAX(d) older than --max-age, for EVERY raw_bars
                 ticker regardless of enrolment. Retired names are excluded --
                 they never refresh, so alerting forever is an alarm that can
                 never clear.
  3. drift     = TICKER_CONFIG keys (monitor_ticker.py, the daily monitor set)
                 outside the union, or intersecting retired. A retired ticker
                 left in TICKER_CONFIG would be re-cached into raw_bars on the
                 next monitor run, silently un-retiring it at the data layer.

USAGE
  python scripts/reconcile_universe.py
  python scripts/reconcile_universe.py --max-age 7
  python scripts/reconcile_universe.py --detail        # per-ticker listing

EXIT CODE
  0 = all clear    1 = at least one alert   -> suitable for cron alerting
"""
import argparse
import ast
import csv
import datetime as dt
import os
import sqlite3
import sys

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
DB = os.path.join(ROOT, "prices.db")


def _line_file(name):
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        return set()
    return {l.strip().upper() for l in open(p)
            if l.strip() and not l.lstrip().startswith("#")}


def _csv_col0(name):
    p = os.path.join(ROOT, name)
    if not os.path.isfile(p):
        return set()
    out = set()
    for i, r in enumerate(csv.reader(open(p, newline=""))):
        if i == 0 and r and r[0].strip().lower() in ("ticker", "symbol"):
            continue
        if r and r[0].strip():
            out.add(r[0].strip().upper())
    return out


def _ticker_config():
    """TICKER_CONFIG keys from monitor_ticker.py, parsed via AST.

    Regex on the source undercounts: nested keys share the same indentation
    and a leading-capital pattern picked up SEC form strings ("S-3", "S-8")."""
    p = os.path.join(ROOT, "scripts", "monitor_ticker.py")
    if not os.path.isfile(p):
        return None
    tree = ast.parse(open(p).read())
    found = None
    for n in ast.walk(tree):
        t = (n.target if isinstance(n, ast.AnnAssign)
             else (n.targets[0] if isinstance(n, ast.Assign) and n.targets else None))
        if (isinstance(t, ast.Name) and t.id == "TICKER_CONFIG"
                and isinstance(n.value, ast.Dict)):
            found = {k.value.upper() for k in n.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=int, default=4,
                    help="alert if a ticker's MAX(d) is older than this many days")
    ap.add_argument("--detail", action="store_true",
                    help="list every offending ticker, not just counts")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT, DB
    if args.root:
        ROOT = os.path.expanduser(args.root)
        DB = os.path.join(ROOT, "prices.db")

    if not os.path.isfile(DB):
        sys.exit(f"FATAL: {DB} not found")

    runner    = _line_file("tickers.txt")
    watchlist = _line_file("tickers_watchlist.txt")
    metadata  = _csv_col0("tickers_metadata.csv")
    retired   = _csv_col0("tickers_retired.csv")
    known     = runner | watchlist | metadata | retired

    if not runner or not metadata:
        sys.exit("FATAL: tickers.txt or tickers_metadata.csv empty/missing -- "
                 "refusing to report every ticker as an orphan")

    con = sqlite3.connect(DB, timeout=30)
    rows = con.execute("SELECT UPPER(ticker), MAX(d), COUNT(*) "
                       "FROM raw_bars GROUP BY UPPER(ticker)").fetchall()
    con.close()
    bars = {t: (mx, n) for t, mx, n in rows}

    cutoff = (dt.date.today() - dt.timedelta(days=args.max_age)).isoformat()
    print(f"# reconcile_universe  db={DB}")
    print(f"# raw_bars={len(bars)}  runner={len(runner)}  watchlist={len(watchlist)} "
          f" metadata={len(metadata)}  retired={len(retired)}  union={len(known)}")
    print(f"# staleness cutoff={cutoff} (--max-age {args.max_age})\n")

    alerts = 0

    orphans = sorted(set(bars) - known)
    if orphans:
        alerts += 1
        print(f"! ORPHAN  {len(orphans)} ticker(s) in raw_bars, in NO universe file:")
        for t in orphans:
            mx, n = bars[t]
            print(f"    {t:8s} last={mx}  rows={n}")
        print("  -> enrol in tickers_metadata.csv, or retire via ticker_lifecycle.py\n")
    else:
        print(f"  ORPHAN    none\n")

    stale = sorted(t for t, (mx, _) in bars.items()
                   if t not in retired and mx < cutoff)
    if stale:
        alerts += 1
        print(f"! STALE   {len(stale)} non-retired ticker(s) past cutoff:")
        for t in stale:
            mx, n = bars[t]
            where = "ENROLLED" if t in known else "ORPHAN"
            print(f"    {t:8s} last={mx}  rows={n}  [{where}]")
        print("  -> repair_stale_feeds.py --tickers <list>\n")
    else:
        print(f"  STALE     none\n")

    cfg = _ticker_config()
    if cfg is None:
        print("  DRIFT     SKIPPED (monitor_ticker.py not found)\n")
    else:
        unknown = sorted(cfg - known)
        zombie  = sorted(cfg & retired)
        if unknown or zombie:
            alerts += 1
            print(f"! DRIFT   TICKER_CONFIG={len(cfg)}")
            if unknown:
                print(f"    monitored but in no universe file: {', '.join(unknown)}")
            if zombie:
                print(f"    RETIRED but still in TICKER_CONFIG: {', '.join(zombie)}")
                print("    -> next monitor run re-caches these into raw_bars, "
                      "un-retiring them at the data layer")
            print()
        else:
            print(f"  DRIFT     none (TICKER_CONFIG={len(cfg)})\n")

    if alerts:
        print(f"# {alerts} ALERT(S)")
        return 1
    print("# all clear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
