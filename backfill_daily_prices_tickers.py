#!/usr/bin/env python3
"""
Fill daily_prices for NAMED tickers from raw_bars, without the global DELETE that
sync_prices_from_rawbars.py --rebuild-from performs.

WHY: sync is incremental on a GLOBAL watermark (dp_max = MAX(date) FROM daily_prices).
A newly added ticker whose history lies entirely below that watermark is never synced.
--rebuild-from picks it up but DELETEs a date range across the WHOLE table, and
daily_prices (1,592,158 rows) is LARGER than raw_bars (1,016,653): it carries pre-2016
history raw_bars cannot reproduce. A deep rebuild would destroy that permanently.

This never deletes. INSERT OR IGNORE leaves existing rows untouched.
Same backward split adjustment as sync v3:
  adj_close(d) = close(d) * PROD(split_from/split_to) for splits with exec_date > d
"""
import argparse, os, sqlite3, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--tickers", nargs="+", required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

db = os.path.join(a.root, "prices.db")
if not os.path.isfile(db):
    print("[STOP] prices.db not found at %s" % db); sys.exit(1)

con = sqlite3.connect(db, timeout=30); cur = con.cursor()
tks = [t.strip().upper() for t in a.tickers if t.strip()]
print("  tickers: %s" % ", ".join(tks))

sp = {}
for tk, ed, sf, st in cur.execute(
        "SELECT ticker, exec_date, split_from, split_to FROM splits "
        "WHERE split_from > 0 AND split_to > 0"):
    sp.setdefault(tk, []).append((ed, sf / st))
print("  splits loaded: %d across %d tickers" % (sum(len(v) for v in sp.values()), len(sp)))

before = {tk: cur.execute("SELECT COUNT(*) FROM daily_prices WHERE ticker=?", (tk,)).fetchone()[0]
          for tk in tks}
qm = ",".join("?" * len(tks))
rows = cur.execute("SELECT ticker, d, close FROM raw_bars WHERE ticker IN (%s) "
                   "AND close IS NOT NULL" % qm, tks).fetchall()
print("  raw_bars rows available: %d" % len(rows))

out = []; adjusted = 0
for tk, d, c in rows:
    f = 1.0
    for ed, factor in sp.get(tk, ()):
        if ed > d: f *= factor
    if f != 1.0: adjusted += 1
    out.append((tk, d, c * f))
print("  rows needing split adjustment: %d" % adjusted)

if a.dry_run:
    print("  DRY-RUN: no writes")
    for tk in tks:
        print("    %-6s daily_prices now %6d | raw_bars candidates %6d"
              % (tk, before[tk], sum(1 for x in out if x[0] == tk)))
    con.close(); sys.exit(0)

cur.executemany("INSERT OR IGNORE INTO daily_prices (ticker,date,adj_close) VALUES (?,?,?)", out)
con.commit()
for tk in tks:
    aft, mn, mx = cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices "
                              "WHERE ticker=?", (tk,)).fetchone()
    print("    %-6s %6d -> %6d rows | %s .. %s" % (tk, before[tk], aft, mn, mx))
con.close()
