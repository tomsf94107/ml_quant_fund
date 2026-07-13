#!/usr/bin/env python3
"""
heal_daily_prices.py -- per-ticker gap heal from raw_bars into daily_prices.

THE BUG THIS FIXES
    sync_prices_from_rawbars.py is GLOBAL-date incremental:
        dp_max = MAX(date) over ALL of daily_prices
        insert raw_bars rows WHERE d > dp_max
    Any ticker whose own coverage lags the global max NEVER heals:
        SPY/QQQ/XLK/... : 18 rows in daily_prices, 1,133 in raw_bars
        JPM  : stale since 2026-06-12 (bars exist in raw_bars)
        CYBR : stale since 2026-02-10
    This script goes PER TICKER: it inserts every raw_bars row whose
    (ticker, d) is absent from daily_prices. Same split-adjustment logic
    as the sync (backward: multiply bars BEFORE a split by from/to).

SAFETY
    - INSERT OR IGNORE only. No DELETE, no REPLACE, no UPDATE anywhere.
    - Dry-run by DEFAULT. Pass --write to commit.
    - Existing rows are never modified: if a (ticker,date) already exists,
      the incoming row is IGNORED.

CAVEAT (known, accepted)
    raw_bars close is unadjusted-for-dividends; deep yfinance-era rows in
    daily_prices are dividend-adjusted. Healed rows are split-adjusted only.
    This matches what the daily sync already writes -- no new inconsistency
    is introduced, but the splice exists table-wide for high-yield names.

USAGE
    python heal_daily_prices.py                    # dry run, all tickers
    python heal_daily_prices.py --tickers SPY,QQQ  # dry run, subset
    python heal_daily_prices.py --write            # commit
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--tickers", default=None,
                help="comma-separated subset; default = every ticker in raw_bars")
ap.add_argument("--write", action="store_true",
                help="actually insert (default is dry-run)")
a = ap.parse_args()

db = os.path.join(os.path.expanduser(a.root), "prices.db")
con = sqlite3.connect(db, timeout=60)
cur = con.cursor()

# ---- splits: per ticker [(exec_date, from/to factor)] -- same as the sync ----
sp = defaultdict(list)
for tk, ed, sf, st in cur.execute(
        "SELECT ticker, exec_date, split_from, split_to FROM splits "
        "WHERE split_from > 0 AND split_to > 0"):
    sp[tk].append((str(ed)[:10], float(sf) / float(st)))
print(f"splits loaded: {sum(len(v) for v in sp.values())} across {len(sp)} tickers")

# ---- per-ticker coverage gap ------------------------------------------------
want = None
if a.tickers:
    want = {t.strip().upper() for t in a.tickers.split(",")}

rb = {t: (lo, hi, n) for t, lo, hi, n in cur.execute(
    "SELECT ticker, MIN(d), MAX(d), COUNT(*) FROM raw_bars GROUP BY ticker")}
dp = {t: (lo, hi, n) for t, lo, hi, n in cur.execute(
    "SELECT ticker, MIN(date), MAX(date), COUNT(*) FROM daily_prices GROUP BY ticker")}

targets = []
for t, (rlo, rhi, rn) in sorted(rb.items()):
    if want and t not in want:
        continue
    dlo, dhi, dn = dp.get(t, (None, None, 0))
    # candidate if raw_bars has bars daily_prices lacks, at either end or inside
    if dn == 0 or rlo < (dlo or "9999") or rhi > (dhi or "0000") or rn > dn:
        targets.append((t, rlo, rhi, rn, dlo, dhi, dn))

print(f"\ntickers with raw_bars coverage beyond daily_prices: {len(targets)}")
print(f"{'ticker':<7}{'rb_first':<12}{'rb_n':>6}   {'dp_first':<12}{'dp_n':>6}")
for t, rlo, rhi, rn, dlo, dhi, dn in targets[:40]:
    print(f"{t:<7}{rlo:<12}{rn:>6}   {str(dlo):<12}{dn:>6}")
if len(targets) > 40:
    print(f"  ... and {len(targets)-40} more")

# ---- build the insert set ----------------------------------------------------
total_new = 0
batch = []
for t, *_ in targets:
    have = {d for (d,) in cur.execute(
        "SELECT date FROM daily_prices WHERE ticker=?", (t,))}
    rows = cur.execute(
        "SELECT d, close FROM raw_bars WHERE ticker=? AND close IS NOT NULL",
        (t,)).fetchall()
    for d, c in rows:
        if d in have:
            continue
        f = 1.0
        for ed, factor in sp.get(t, ()):
            if ed > d:                       # split AFTER this bar -> adjust back
                f *= factor
        batch.append((t, d, float(c) * f))
        total_new += 1

print(f"\nrows to insert: {total_new:,}")

if not a.write:
    # show a sample so the adjustment can be eyeballed
    for t, d, v in batch[:6]:
        print(f"  {t} {d}: {v:.2f}")
    print("\n[DRY RUN] nothing written. Re-run with --write to commit.")
    con.close()
    raise SystemExit

before = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
cur.executemany(
    "INSERT OR IGNORE INTO daily_prices (ticker,date,adj_close) VALUES (?,?,?)",
    batch)
con.commit()
after = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
print(f"daily_prices: {before:,} -> {after:,}  (+{after-before:,})")

# verify the ETFs specifically
for t in ("SPY", "QQQ", "XLK", "TLT"):
    r = cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices "
                    "WHERE ticker=?", (t,)).fetchone()
    print(f"  {t}: {r[0]} rows  {r[1]} -> {r[2]}")
con.close()
