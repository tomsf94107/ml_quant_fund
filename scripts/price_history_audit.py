#!/usr/bin/env python3
"""
scripts/price_history_audit.py -- READ ONLY. WRITES NOTHING.

Answers the three questions that must be settled before any backfill touches
prices.db.

  1. HOW DEEP does Massive actually serve?
     The 2022-01-03 floor in raw_bars is a CACHE artifact, not a known vendor
     limit. backfill_truncated.py hardcodes FLOOR="2022-01-03" and backfills TO
     it, never past it. Nobody has ever asked the API for 2016.

  2. WHICH tickers are TRUNCATED vs genuinely NEW?
     MSFT/NVDA/META hold 632 bars from 2024-01-02 and contributed ONE fold each
     to a nine-fold walk-forward -- they are broken. CAVA/ARM/OKLO are short
     because they IPO'd -- they are correct. These need opposite treatment and
     only the API can tell them apart.
     ROOT CAUSE (from backfill_truncated.py's own docstring): price_cache's gap
     logic is FORWARD-ONLY, gap_start = MAX(d)+1. Ask for an EARLIER start than
     what is cached and it never backfills -- it silently serves what it has.

  3. Do we have SPLITS deep enough to back-adjust new history?
     raw_bars is UNADJUSTED by design; price_cache adjusts BACKWARD on read from
     the splits table. Backfilled bars with no split coverage read as garbage --
     that is the phantom -51% HON bug, at scale.

USAGE
    python scripts/price_history_audit.py              # full, ~40 API calls
    python scripts/price_history_audit.py --no-api     # DB only, 0 API calls
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

ROOT     = Path(__file__).resolve().parent.parent
PRICES   = ROOT / "prices.db"
CUR_FLOOR = "2022-01-03"
END      = "2026-07-10"
PROBE_STARTS = ["2005-01-01", "2010-01-01", "2014-01-01",
                "2016-01-01", "2018-01-01", "2020-01-01"]
PROBE_TICKERS = ["AAPL", "SPY", "JNJ"]      # all listed well before 2005
SLEEP    = 0.35                              # be polite to the rate limiter


def _key() -> str:
    k = os.environ.get("MASSIVE_API_KEY")
    if not k:
        sys.exit("MASSIVE_API_KEY not set. Run:  set -a && . ./.env && set +a")
    return k


def _aggs(ticker: str, start: str, key: str) -> tuple[str | None, int, str]:
    """Return (first_bar_date, n_bars, status). Never raises."""
    u = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{END}"
         f"?adjusted=false&sort=asc&limit=50000&apiKey={key}")
    try:
        r = json.loads(urllib.request.urlopen(u, timeout=45).read())
    except urllib.error.HTTPError as e:
        return None, 0, f"HTTP {e.code}"
    except Exception as e:
        return None, 0, f"ERR {str(e)[:24]}"
    res = r.get("results") or []
    if not res:
        return None, 0, str(r.get("status", "EMPTY"))
    first = dt.datetime.utcfromtimestamp(res[0]["t"] / 1000).date().isoformat()
    return first, len(res), str(r.get("status", "OK"))


def probe_depth(key: str) -> str | None:
    print("=" * 78)
    print("  1. HOW DEEP DOES MASSIVE SERVE?")
    print("=" * 78)
    print("  The 2022-01-03 floor is a CACHE artifact. Nobody has asked for 2016.\n")
    deepest = None
    for t in PROBE_TICKERS:
        print(f"  {t}")
        for s in PROBE_STARTS:
            first, n, status = _aggs(t, s, key)
            mark = "" if first is None else ("  <-- SERVED" if first[:4] <= s[:4] else "")
            print(f"    ask {s}  ->  status={status:<12} n={n:>6}  "
                  f"first={first or '-'}{mark}")
            if first and (deepest is None or first < deepest):
                deepest = first
            time.sleep(SLEEP)
        print()
    if deepest:
        yrs = (dt.date.fromisoformat(END) - dt.date.fromisoformat(deepest)).days / 365.25
        print(f"  DEEPEST BAR SERVED: {deepest}   ({yrs:.1f} years)")
        print(f"  CURRENT CACHE FLOOR: {CUR_FLOOR}   "
              f"({(dt.date.fromisoformat(END) - dt.date.fromisoformat(CUR_FLOOR)).days/365.25:.1f} years)")
        if deepest < CUR_FLOOR:
            print(f"\n  >>> THE FLOOR IS ARBITRARY. Massive serves {deepest}.")
            print(f"  >>> Backfilling to {deepest} turns 9 walk-forward folds into ~{int(yrs*4)-8}.")
        else:
            print(f"\n  >>> The floor is a VENDOR LIMIT. Deeper history needs a "
                  f"different tier or vendor.")
    else:
        print("  NO DATA RETURNED FOR ANY PROBE. Check the key / tier.")
    return deepest


def db_state() -> list[tuple[str, str, int]]:
    con = sqlite3.connect(f"file:{PRICES}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT ticker, MIN(d) AS first_bar, COUNT(*) AS n "
        "FROM raw_bars GROUP BY ticker ORDER BY first_bar, ticker").fetchall()
    con.close()
    return rows


def classify(rows, key: str | None):
    print()
    print("=" * 78)
    print("  2. TRUNCATED (broken) vs GENUINELY NEW (correct)")
    print("=" * 78)
    at_floor = [r for r in rows if r[1] <= "2022-01-10"]
    suspect  = [r for r in rows if r[1] >  "2022-01-10"]
    print(f"  at the {CUR_FLOOR} floor : {len(at_floor):>3} tickers")
    print(f"  starting LATER           : {len(suspect):>3} tickers  <-- classify these\n")

    if key is None:
        print("  --no-api: cannot distinguish truncation from a real IPO. "
              "Listing suspects only.\n")
        for t, first, n in suspect:
            print(f"    {t:<7} first={first}  bars={n:>5}")
        return [], []

    print(f"  {'ticker':<8}{'our first':<12}{'API first':<12}{'API bars':>9}  verdict")
    print("  " + "-" * 62)
    truncated, genuine = [], []
    for t, first, n in suspect:
        api_first, api_n, status = _aggs(t, "2000-01-01", key)
        time.sleep(SLEEP)
        if api_first is None:
            print(f"  {t:<8}{first:<12}{'-':<12}{'-':>9}  API FAILED ({status})")
            continue
        # >30 calendar days of history the cache never fetched = TRUNCATED
        if api_first < first and \
           (dt.date.fromisoformat(first) - dt.date.fromisoformat(api_first)).days > 30:
            truncated.append((t, first, api_first, api_n))
            print(f"  {t:<8}{first:<12}{api_first:<12}{api_n:>9}  TRUNCATED  <<<")
        else:
            genuine.append((t, first, api_first, api_n))
            print(f"  {t:<8}{first:<12}{api_first:<12}{api_n:>9}  genuinely new")
    return truncated, genuine


def splits_state(key: str | None):
    print()
    print("=" * 78)
    print("  3. SPLITS COVERAGE  (raw_bars is UNADJUSTED; back-adjust needs these)")
    print("=" * 78)
    con = sqlite3.connect(f"file:{PRICES}?mode=ro", uri=True)
    tabs = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  tables in prices.db: {tabs}")
    if "splits" in tabs:
        cols = [r[1] for r in con.execute("PRAGMA table_info(splits)").fetchall()]
        print(f"  splits columns     : {cols}")
        dcol = next((c for c in cols if "date" in c.lower()), None)
        n = con.execute("SELECT COUNT(*) FROM splits").fetchone()[0]
        nt = con.execute("SELECT COUNT(DISTINCT ticker) FROM splits").fetchone()[0]
        print(f"  rows={n}  tickers={nt}")
        if dcol:
            lo, hi = con.execute(f"SELECT MIN({dcol}), MAX({dcol}) FROM splits").fetchone()
            print(f"  {dcol} range      : {lo} -> {hi}")
            if lo and lo > CUR_FLOOR:
                print(f"\n  >>> SPLITS ONLY GO BACK TO {lo}. Backfilled bars before that")
                print(f"  >>> would NOT be back-adjusted -> phantom split jumps in the panel.")
                print(f"  >>> The splits table MUST be extended with the bars.")
    else:
        print("  NO splits TABLE. price_cache must be creating it lazily.")
    con.close()

    if key:
        print("\n  Can we FETCH deep splits?")
        u = (f"https://api.polygon.io/v3/reference/splits?ticker=AAPL"
             f"&limit=100&apiKey={key}")
        try:
            r = json.loads(urllib.request.urlopen(u, timeout=30).read())
            res = r.get("results") or []
            print(f"    /v3/reference/splits AAPL -> status={r.get('status')} "
                  f"n={len(res)}")
            for s in res[:5]:
                print(f"      {s.get('execution_date')}  "
                      f"{s.get('split_from')}:{s.get('split_to')}")
            if not res:
                print("    >>> NO SPLITS RETURNED. AAPL split 4:1 in 2020 and 7:1 in "
                      "2014.\n    >>> If this endpoint is empty, deep backfill is UNSAFE.")
        except urllib.error.HTTPError as e:
            print(f"    /v3/reference/splits -> HTTP {e.code}")
            print("    >>> BLOCKED ON THIS TIER. Deep backfill is UNSAFE without splits.")
        except Exception as e:
            print(f"    /v3/reference/splits -> {str(e)[:50]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    key = None if a.no_api else _key()

    rows = db_state()
    print(f"\nraw_bars: {len(rows)} tickers, "
          f"{sum(r[2] for r in rows):,} bars, floor {CUR_FLOOR}\n")

    deepest = probe_depth(key) if key else None
    truncated, genuine = classify(rows, key)
    splits_state(key)

    print()
    print("=" * 78)
    print("  PLAN")
    print("=" * 78)
    if key:
        print(f"  TRUNCATED (backfill these)   : {len(truncated)}")
        if truncated:
            print(f"    {', '.join(t[0] for t in truncated)}")
        print(f"  GENUINELY NEW (leave alone)  : {len(genuine)}")
        if genuine:
            print(f"    {', '.join(t[0] for t in genuine)}")
        if deepest and deepest < CUR_FLOOR:
            n_t = len(rows)
            print(f"\n  DEEPEN THE FLOOR {CUR_FLOOR} -> {deepest}")
            print(f"    {n_t} tickers x 1 call = {n_t} API calls "
                  f"(quota 40,000/day)")
            print(f"    ~{n_t * 250 * ((dt.date.fromisoformat(CUR_FLOOR) - dt.date.fromisoformat(deepest)).days // 365):,} new bars, est.")
    print("\n  NOTHING WAS WRITTEN. This script is read-only.")
    print("=" * 78)


if __name__ == "__main__":
    main()
