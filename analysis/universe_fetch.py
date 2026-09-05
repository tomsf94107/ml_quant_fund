#!/usr/bin/env python3
"""
universe_fetch.py — pull full daily history for the screened universe.

Holds NO database connection of its own. features/price_cache.py persists every
bar it downloads, so this script only calls download and lets the cache write.

WHY THAT MATTERS
    A first version opened its own connection to prices.db and held it for the
    whole run while calling massive_client.download in a loop. price_cache opens
    a SECOND connection per download, and the long-lived transaction blocked it:
    every fetch logged "price_cache failed for X: database is locked" and wrote
    nothing. The run deadlocked against itself.

    That was the fourth lock incident of 2026-09-05. The others: a fetch running
    beside the h=40 yearly test corrupted one seed's panel (GEMI, AME, ZM failed
    mid-run and the results were discarded); a feature-build overlap left the
    eightk leave-one-out group measured on half-refreshed data; and a patch
    smoke test deadlocked against a fetch.

    THE RULE: price_cache writes to prices.db on EVERY feature build and EVERY
    bar download, so exactly one process may touch it at a time. Check
    `ps aux | grep -E "universe_fetch|etl_|builder"` and confirm zero before
    starting anything that builds features or fetches bars.

WHY A PRIOR RUN FETCHED NOTHING EVEN WITHOUT THE LOCK
    price_cache.cached_daily only fetches the FULL range when the cache holds
    nothing for it (`if have.empty`). Otherwise it heals forward from the last
    cached bar: gap_start = last + 1 day. It never fills history BEFORE what is
    cached.

    The liquidity scan (analysis/universe_scan.py) left ~60-90 day stubs for
    8,787 tickers as a side effect, so a 2016-2026 request found data, took the
    else branch, and healed only the frontier. Those stubs were deleted before
    this run so have.empty is true and the full range comes down.

SURVIVORSHIP, UNCHANGED
    universe_final.txt lists companies that exist TODAY. Everything that
    delisted 2016-2026 is absent, and a wider universe makes the tilt worse in
    absolute terms. Correct for a live universe, wrong for a backtest, and
    fixable only with delisted price history -- si_leg_decomp.py sized it for
    the SI brick and found it near-moot there; nobody has sized it for the h=40
    candidate.

    python analysis/universe_fetch.py --dry-run
    python analysis/universe_fetch.py --limit 20
    python analysis/universe_fetch.py
"""
import argparse
import os
import sqlite3
import sys
import time

UNIVERSE = "universe_final.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prices.db")
    ap.add_argument("--universe", default=UNIVERSE)
    ap.add_argument("--start", default="2016-07-18",
                    help="Massive's earliest available session")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-bars", type=int, default=500,
                    help="a ticker at or above this is considered complete "
                         "and skipped")
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.universe):
        raise SystemExit(f"{args.universe} not found -- run "
                         f"analysis/universe_scan.py --report first")
    want = [l.strip().upper() for l in open(args.universe) if l.strip()]

    # Read-only. The connection is CLOSED before any download begins.
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    have = {t: n for t, n in con.execute(
        "SELECT ticker, COUNT(*) FROM raw_bars GROUP BY ticker")}
    n0 = con.execute("SELECT COUNT(*) FROM raw_bars").fetchone()[0]
    con.close()

    todo = [t for t in want if have.get(t, 0) < args.min_bars]
    print(f"{args.universe}: {len(want):,} screened names")
    print(f"prices.db: {n0:,} rows, {len(have)} tickers")
    print(f"  {len(want) - len(todo):,} already at >= {args.min_bars} bars, "
          f"{len(todo):,} to fetch")

    if args.dry_run:
        print(f"\nDRY RUN -- nothing written.")
        print(f"  first 15: {', '.join(todo[:15])}")
        print(f"  range {args.start} onward")
        return

    others = [p for p in ("universe_scan", "etl_eightk", "feature_",
                          "h40_", "linear_baseline")
              if os.popen(f"pgrep -f {p} | head -1").read().strip()]
    if others:
        print(f"\n  WARNING: these look like they are running: "
              f"{', '.join(others)}")
        print("  price_cache writes to prices.db on every build and every")
        print("  download, so a concurrent run will deadlock or corrupt.")
        print("  Ctrl-C now if any of them touches prices.db.\n")
        time.sleep(5)

    sys.path.insert(0, ".")
    from features import massive_client as mc
    end = mc._last_completed_session().strftime("%Y-%m-%d")
    print(f"\nfetching {args.start} .. {end}\n")

    n_ok = n_thin = n_fail = 0
    first_err = None
    t0 = time.time()
    for i, tk in enumerate(todo, 1):
        if args.limit and i > args.limit:
            break
        try:
            # price_cache persists this to raw_bars. No connection is held here.
            df = mc.download(tk, start=args.start, end=end,
                             auto_adjust=True, progress=False)
            if df is None or len(df) < 250:
                n_thin += 1
            else:
                n_ok += 1
        except Exception as e:
            n_fail += 1
            if first_err is None:
                first_err = f"{tk}: {type(e).__name__}: {e}"
        if i % 50 == 0:
            el = time.time() - t0
            rate = i / max(el, 1)
            print(f"  {i:>5}/{len(todo)}  ok {n_ok:>5}  thin {n_thin:>4}  "
                  f"fail {n_fail:>4}  {el/60:.0f} min, "
                  f"~{(len(todo)-i)/max(rate,.01)/60:.0f} min left")
        time.sleep(args.sleep)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    n1, tk1 = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker) FROM raw_bars").fetchone()
    full = con.execute(
        "SELECT COUNT(*) FROM (SELECT ticker FROM raw_bars GROUP BY ticker "
        "HAVING COUNT(*) >= ?)", (args.min_bars,)).fetchone()[0]
    con.close()

    print(f"\n  {n_ok} fetched, {n_thin} thin (< 250 bars), {n_fail} failed")
    if first_err:
        print(f"  first failure: {first_err}")
    print(f"  raw_bars: {n0:,} -> {n1:,} rows (+{n1-n0:,}), {tk1} tickers, "
          f"{full} with >= {args.min_bars} bars")
    print("\n  NEXT: re-run analysis/universe_expand.py to screen on the full")
    print("  history, then compare the h=40 book on the current 415 names")
    print("  against the expanded set -- same seeds. That comparison is what")
    print("  says whether breadth sharpens the ranking or dilutes it, and it")
    print("  is the only thing that justifies the added cost: feature building")
    print("  goes from ~25 minutes to ~2 hours per pass at this size.")
    print("\n  The h=40 SHADOW BOOK stays on the current 415 names. It is")
    print("  frozen (sha 9ef0cdd954dfd6ad), its clock started 2026-09-05, and")
    print("  changing its universe would void every observation.")


if __name__ == "__main__":
    main()
