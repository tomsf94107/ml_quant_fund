#!/usr/bin/env python3
"""
sync_prices_from_rawbars.py -- keep daily_prices current, SPLIT-ADJUSTED, no network.

BUG THIS FIXES (mine, Jul 11 2026):
  v1 copied raw_bars.close -> daily_prices.adj_close. But price_cache fetches with
  auto_adjust=FALSE: raw_bars is UNADJUSTED and massive_client applies the split
  adjustment ON READ. So v1 wrote unadjusted prices into an ADJUSTED column ->
  CRWD (4:1 on 2026-07-02) became 772.74 -> 193.98, a FAKE -75% crash in the table
  every validation reads.

  v2 called massive_client.download() per ticker -- correct, but ~400 API calls and
  Polygon throttles it into the ground.

  v3 (this) applies the SAME backward split adjustment LOCALLY:
      adj_close(d) = close(d) * PROD(split_from / split_to) for splits with exec_date > d
  Verified against massive_client: CRWD 06-29 raw 742.91 /4 = 185.73 (matches);
  HON 06-24 = 227.42 (matches). Zero API calls. Instant.

ADDITIVE: INSERT OR IGNORE -- the deep 2008-2021 history is never touched.
"""
import argparse, os, sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--rebuild-from", default=None)
a = ap.parse_args()

db = os.path.join(os.path.expanduser(a.root), "prices.db")
con = sqlite3.connect(db, timeout=60); cur = con.cursor()

if a.rebuild_from and not a.dry_run:
    n = cur.execute("DELETE FROM daily_prices WHERE date >= ?", (a.rebuild_from,)).rowcount
    con.commit(); print(f"  rebuild: deleted {n:,} rows >= {a.rebuild_from}")

# splits, per ticker: [(exec_date, from/to factor)]
sp = defaultdict(list)
for tk, ed, sf, st in cur.execute(
        "SELECT ticker, exec_date, split_from, split_to FROM splits "
        "WHERE split_from > 0 AND split_to > 0"):
    sp[tk].append((str(ed)[:10], float(sf) / float(st)))
print(f"  splits loaded: {sum(len(v) for v in sp.values())} across {len(sp)} tickers")

dp_max = cur.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0] or "0000-00-00"
rb_max = cur.execute("SELECT MAX(d) FROM raw_bars").fetchone()[0]
before = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
print(f"  daily_prices max: {dp_max}  ({before:,} rows)")
print(f"  raw_bars    max: {rb_max}")
if not rb_max or rb_max <= dp_max:
    print("  nothing to add."); con.close(); raise SystemExit

rows = cur.execute(
    "SELECT ticker, d, close FROM raw_bars WHERE d > ? AND close IS NOT NULL",
    (dp_max,)).fetchall()
print(f"  candidate rows: {len(rows):,}")

out, adjusted = [], 0
for tk, d, c in rows:
    f = 1.0
    for ed, factor in sp.get(tk, ()):
        if ed > d: f *= factor          # split is AFTER this bar -> adjust it back
    if f != 1.0: adjusted += 1
    out.append((tk, d, float(c) * f))

print(f"  rows needing split adjustment: {adjusted:,}")
if a.dry_run:
    for tk, d, v in out:
        if tk in ("CRWD", "HON") and d <= "2026-07-02":
            print(f"    {tk} {d}: {v:.2f}")
    print("  [DRY RUN] nothing written."); con.close(); raise SystemExit

cur.executemany("INSERT OR IGNORE INTO daily_prices (ticker,date,adj_close) VALUES (?,?,?)", out)
con.commit()
after = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
mx = cur.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
print(f"  ADDED {after-before:,} rows | daily_prices now {after:,} | max {mx}")
con.close()
