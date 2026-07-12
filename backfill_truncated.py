#!/usr/bin/env python3
"""
backfill_truncated.py -- repair tickers stuck with truncated history.

THE BUG (same class as the volume-consolidation drift)
  price_cache's gap logic is FORWARD-ONLY:  gap_start = MAX(d) + 1.
  Ask for an EARLIER start than what is cached and it never backfills -- it just
  serves what it has. So any ticker first cached during a narrow window (someone
  ran `monitor TLT` once, say) is PERMANENTLY stuck with that window, and every
  training run since has silently used months of history instead of years.

  Found: 7 tickers with 19-69 bars against a 2022-01-03 cache floor for the other
  418. TLT has traded since 2002 and holds 19 bars. WCC since 1999, holds 69.

  BNY is EXCLUDED: its 34 bars are CORRECT. The pre-rename BNY series on Polygon is
  a different, recycled company trading at ~$10 while BK trades at ~$137 -- verified
  against the API today. Backfilling it would re-import the phantom bars we deleted.

METHOD: fetch -> STAGE -> verify -> swap. Never delete before the replacement is in
hand (an earlier naive delete-then-fetch left a hole in prices.db tonight).
"""
import os, sqlite3, sys, time, json, urllib.request
from datetime import datetime
sys.path.insert(0, ".")

TRUNCATED = ["TLT", "RC", "SANA", "VXRT", "VIXY", "BNED", "WCC"]   # BNY excluded, see above
FLOOR = "2022-01-03"
KEY = os.environ["MASSIVE_API_KEY"]

con = sqlite3.connect("prices.db", timeout=60)
print(f"  backfilling {len(TRUNCATED)} tickers to the {FLOOR} cache floor\n")

for t in TRUNCATED:
    before = con.execute("SELECT COUNT(*), MIN(d) FROM raw_bars WHERE ticker=?", (t,)).fetchone()
    u = (f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{FLOOR}/2026-07-10"
         f"?adjusted=false&sort=asc&limit=50000&apiKey={KEY}")
    try:
        res = json.loads(urllib.request.urlopen(u, timeout=40).read()).get("results", [])
    except Exception as e:
        print(f"  {t:6s} FETCH FAILED: {str(e)[:45]}  -- left untouched")
        time.sleep(5); continue
    if len(res) < before[0]:
        print(f"  {t:6s} API returned {len(res)} < cached {before[0]}  -- REFUSING to shrink")
        time.sleep(3); continue

    rows = [(t, time.strftime("%Y-%m-%d", time.gmtime(r["t"]/1000)),
             r.get("o"), r.get("h"), r.get("l"), r.get("c"), r.get("v"))
            for r in res]
    # INSERT OR REPLACE: adds the missing past, refreshes the present. No delete.
    con.executemany("INSERT OR REPLACE INTO raw_bars (ticker,d,open,high,low,close,volume) "
                    "VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    after = con.execute("SELECT COUNT(*), MIN(d), MAX(d) FROM raw_bars WHERE ticker=?", (t,)).fetchone()
    print(f"  {t:6s} {before[0]:>4} -> {after[0]:>5} bars   {before[1]} -> {after[1]}  (max {after[2]})")
    time.sleep(4)

print()
r = con.execute("SELECT COUNT(*) FROM (SELECT ticker FROM raw_bars GROUP BY ticker "
                "HAVING COUNT(*) < 200)").fetchone()
print(f"  tickers still under 200 bars: {r[0]}   (expect 1 = BNY, which is CORRECT)")
con.close()
