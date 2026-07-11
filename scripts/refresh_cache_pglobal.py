#!/usr/bin/env python3
"""
scripts/refresh_cache_pglobal.py
Merge-refresh: inject prob_global into the existing signals_cache.json from the
predictions table, WITHOUT a model run. Preserves all generator-computed fields
(price, targets, sharpe, etc.) — only adds/updates prob_global per ticker+horizon.

WHY MERGE not rebuild: predictions table lacks current_price/targets/atr/metrics
(those are generator-time only). A full DB rebuild would blank them. This patches
just the one field, so the Prob Global column fills while everything else stays
as of the last daily_runner.

USAGE: python scripts/refresh_cache_pglobal.py
"""
import json, sqlite3, datetime, shutil, sys
from pathlib import Path

CACHE = Path("data/signals_cache.json")
DB = Path("accuracy.db")

if not CACHE.exists():
    print("ERROR: cache not found"); sys.exit(1)

cache = json.loads(CACHE.read_text())
sigs = cache.get("signals", [])
if not sigs:
    print("ERROR: no signals in cache"); sys.exit(1)

# Pull latest prob_up_global per (ticker, horizon) from predictions
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
latest = con.execute("SELECT MAX(prediction_date) FROM predictions").fetchone()[0]
rows = con.execute(
    "SELECT ticker, horizon, prob_up_global FROM predictions WHERE prediction_date=?",
    (latest,)).fetchall()
con.close()

pg = {(t, h): v for (t, h, v) in rows}
print(f"predictions latest={latest}, {len(pg)} (ticker,horizon) global scores")

# Merge: inject prob_global into each cache record, preserve everything else
hit = miss = 0
for s in sigs:
    key = (s.get("ticker"), s.get("horizon"))
    if key in pg and pg[key] is not None:
        s["prob_global"] = round(float(pg[key]), 4)
        hit += 1
    else:
        s.setdefault("prob_global", None)
        miss += 1

cache["signals"] = sigs
cache["pglobal_refreshed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
cache["pglobal_source_date"] = latest

shutil.copy2(CACHE, str(CACHE) + ".bak.pgref." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
CACHE.write_text(json.dumps(cache, indent=2))
print(f"OK — injected prob_global into {hit} records ({miss} without a match), cache rewritten")
