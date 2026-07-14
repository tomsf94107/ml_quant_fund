#!/usr/bin/env python3
"""
scripts/backfill_raw_bars.py — deepen raw_bars WITHOUT touching price_cache.

WHY THIS EXISTS
    features/price_cache.cached_daily() only ever fetches FORWARD:

        have = _read_raw(con, ticker, start_s, end_s)
        if have.empty:  fetch(start_s, end_s)
        else:
            last      = have.index.max()
            gap_start = last + 1 day        # <-- forward only
            fetch(gap_start, end_s)

    have.index.MIN() is never compared to start_s. So a request for deeper
    history than the cache holds returns the SHALLOW window, silently, with no
    error. That is why raw_bars stayed pinned at 2022-01-03 after the paid
    Massive key unlocked 10 years: every caller asked for more and got what was
    already there.

    (That is a real bug in cached_daily. It is NOT fixed here — price_cache is
    the hot path for the panel builder, the momentum signal, and the monitor.
    This script sidesteps it instead of risking it.)

SAFETY
    - INSERT OR IGNORE. Existing bars are NEVER overwritten.
      (Seam-checked 2026-07-14: old-key 2022+ bars and new-key 2016+ bars agree
       to the cent across 1,134 overlapping sessions on AAPL/NVDA/MSFT.
       MISMATCHES 0. But IGNORE anyway — cheap insurance.)
    - Fetches ONLY the backward gap [start, first_cached − 1]. Never re-fetches
      what is already there.
    - Refreshes splits per ticker first. raw_bars is UNADJUSTED and is
      back-adjusted on read from the splits table — a missing 2016-2021 split
      would render those bars as a phantom 50-75% crash.
    - --dry-run shows every ticker's gap and writes nothing.

USAGE
    python scripts/backfill_raw_bars.py --start 2016-01-01 --dry-run
    python scripts/backfill_raw_bars.py --start 2016-01-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "prices.db"
BASE = "https://api.polygon.io"


def _get(url: str, tries: int = 4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise
        except Exception:
            if i < tries - 1:
                time.sleep(1 + i)
                continue
            raise
    return None


def fetch_bars(ticker: str, start: str, end: str, key: str) -> list[tuple]:
    u = (f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
         f"?adjusted=false&sort=asc&limit=50000&apiKey={key}")
    r = _get(u)
    out = []
    for b in (r or {}).get("results", []) or []:
        d = dt.datetime.utcfromtimestamp(b["t"] / 1000).strftime("%Y-%m-%d")
        out.append((ticker, d, float(b["o"]), float(b["h"]),
                    float(b["l"]), float(b["c"]), float(b["v"])))
    return out


def fetch_splits(ticker: str, key: str) -> list[tuple]:
    u = (f"{BASE}/v3/reference/splits?ticker={ticker}&limit=1000"
         f"&order=asc&apiKey={key}")
    r = _get(u)
    return [(ticker, s["execution_date"],
             float(s["split_from"]), float(s["split_to"]))
            for s in (r or {}).get("results", []) or []
            if s.get("execution_date")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01",
                    help="target earliest bar (Massive serves ~10yr: 2016-07-18)")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tickers", default=None, help="comma list; default = all in raw_bars")
    a = ap.parse_args()

    key = os.environ.get("MASSIVE_API_KEY", "")
    if len(key) < 20:
        print("MASSIVE_API_KEY not set (did you `set -a && . ./.env && set +a`?)")
        return 1

    con = sqlite3.connect(str(DB))
    con.execute("PRAGMA busy_timeout=60000")

    if a.tickers:
        universe = [t.strip().upper() for t in a.tickers.split(",")]
        depth = {t: con.execute(
            "SELECT MIN(d) FROM raw_bars WHERE ticker=?", (t,)).fetchone()[0]
            for t in universe}
    else:
        depth = dict(con.execute(
            "SELECT ticker, MIN(d) FROM raw_bars GROUP BY ticker").fetchall())

    todo = {t: f for t, f in depth.items() if f and f > a.start}
    skip = len(depth) - len(todo)

    print(f"target start   : {a.start}")
    print(f"tickers         : {len(depth)}")
    print(f"already deep    : {skip}")
    print(f"to backfill     : {len(todo)}")
    if a.dry_run:
        print("\n[DRY RUN] gaps that would be fetched:")
        for i, (t, f) in enumerate(sorted(todo.items())[:15]):
            print(f"    {t:<6} cached from {f}  -> fetch [{a.start} .. {f}]")
        if len(todo) > 15:
            print(f"    ... and {len(todo)-15} more")
        con.close()
        return 0
    if not todo:
        print("nothing to do.")
        con.close()
        return 0

    n_bars = n_splits = n_fail = 0
    t0 = time.time()
    for i, (tk, first) in enumerate(sorted(todo.items()), 1):
        # splits FIRST -- raw_bars is unadjusted; a missing 2016-2021 split
        # renders those bars as a phantom crash on read.
        try:
            sp = fetch_splits(tk, key)
            if sp:
                cur = con.executemany(
                    "INSERT OR IGNORE INTO splits VALUES (?,?,?,?)", sp)
                n_splits += cur.rowcount if cur.rowcount > 0 else 0
        except Exception as e:
            print(f"  [{i:3}/{len(todo)}] {tk:<6} SPLIT FETCH FAILED: {e}")

        back_end = (dt.date.fromisoformat(first) - dt.timedelta(days=1)).isoformat()
        try:
            bars = fetch_bars(tk, a.start, back_end, key)
        except Exception as e:
            n_fail += 1
            print(f"  [{i:3}/{len(todo)}] {tk:<6} FAILED: {type(e).__name__} {e}")
            time.sleep(a.sleep)
            continue

        if bars:
            cur = con.executemany(
                "INSERT OR IGNORE INTO raw_bars VALUES (?,?,?,?,?,?,?)", bars)
            added = cur.rowcount if cur.rowcount > 0 else 0
            n_bars += added
            con.commit()
            eta = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f"  [{i:3}/{len(todo)}] {tk:<6} +{added:>5} bars "
                  f"(now from {bars[0][1]})    eta {eta:4.1f}m")
        else:
            print(f"  [{i:3}/{len(todo)}] {tk:<6} no bars before {first} "
                  f"(listed later — OK)")
        time.sleep(a.sleep)

    con.commit()
    print(f"\n  bars added   : {n_bars:,}")
    print(f"  splits added : {n_splits:,}")
    print(f"  failed       : {n_fail}")
    r = con.execute(
        "SELECT MIN(d), MAX(d), COUNT(*), COUNT(DISTINCT ticker) FROM raw_bars").fetchone()
    print(f"  raw_bars now : {r[0]} -> {r[1]}  |  {r[2]:,} rows  |  {r[3]} tickers")
    con.close()
    print("\n  NEXT: verify the seam and the splits before trusting the deep panel:")
    print("    sqlite3 prices.db \"SELECT substr(d,1,4) yr, COUNT(DISTINCT ticker) "
          "FROM raw_bars GROUP BY yr ORDER BY yr;\"")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
