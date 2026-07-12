#!/usr/bin/env python3
"""
backfill_vix.py -- a WORKING volatility-index feed, stored once, read by everyone.

WHY
  risk_gate.py:113 calls safe_yf_download(["^VIX"]). yfinance is XProtect-BLOCKED on
  this machine, so that call returns EMPTY every single time, the spike loop never
  executes, and risk_prev_1d has never once recorded a VIX spike. Same for
  fear_greed (stuck at 0.5) and vix_term_structure (stuck at 1.0) -- both are
  fallback constants. The ENTIRE volatility-regime axis of this system is dead, in
  training AND in production, and it has been failing silently.

  Polygon/Massive returns 403 on I:VIX (index data is a higher tier) and empty on
  ^VIX / VIX. But VIXY -- the VIX short-term futures ETF -- comes through cleanly.

THE CALIBRATION TRAP (why this is not a one-line swap)
  VIXY tracks VIX FUTURES, not spot VIX. Futures are much less volatile: a 20% spot
  move might be an 8-10% VIXY move. risk_gate's VIX_SPIKE_PCT = 0.20 would therefore
  NEVER fire on VIXY -- 0 of the last 48 days even cleared 8%. Swapping the symbol
  and keeping the threshold would leave the gate dead in a new way, which is worse
  than leaving it obviously broken.

  Fix: a SELF-CALIBRATING percentile. A spike = a day in the top 1% of |daily move|
  over the trailing year. No knowledge of the VIX/VIXY beta required, and it adapts
  if the feed ever changes.

ARCHITECTURE
  Fetch ONCE into accuracy.db.vix_history. risk_gate reads the table.
  builder.py calls build_risk_features() once per ticker -- ~400 times per pipeline
  run. Hitting the API 400 times for the same market-wide series is how you get
  429'd. One fetch, one table, many readers.
"""
import sqlite3, sys
from datetime import datetime
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from features.massive_client import download

con = sqlite3.connect("accuracy.db", timeout=60)
con.execute("""CREATE TABLE IF NOT EXISTS vix_history (
    date        TEXT PRIMARY KEY,
    close       REAL NOT NULL,   -- VIXY close (futures ETF, NOT spot VIX)
    ret         REAL,            -- daily pct change
    abs_ret     REAL,            -- |daily pct change| -- the spike input
    spike_p99   REAL,            -- trailing-252d 99th pctile of abs_ret (self-calibrating)
    is_spike    INTEGER,         -- abs_ret >= spike_p99
    source      TEXT,
    fetched_at  TEXT)""")
con.commit()

print("  fetching VIXY (Massive). I:VIX is 403 on this tier; ^VIX/VIX return empty.")
d = download("VIXY", start="2018-01-01", end=datetime.now().strftime("%Y-%m-%d"),
             auto_adjust=True)
if d is None or d.empty:
    print("  FATAL: VIXY returned empty."); sys.exit(1)

d = d.sort_index()
px = d["Close"].astype(float)
ret = px.pct_change()
absr = ret.abs()
# self-calibrating: a spike is a top-1% day IN THIS SERIES' OWN history. No need to
# know how a VIXY move maps to a spot-VIX move.
p99 = absr.rolling(252, min_periods=120).quantile(0.99)
spike = (absr >= p99).astype(int)

rows = [(str(i.date()), float(px.loc[i]),
         None if pd.isna(ret.loc[i]) else float(ret.loc[i]),
         None if pd.isna(absr.loc[i]) else float(absr.loc[i]),
         None if pd.isna(p99.loc[i]) else float(p99.loc[i]),
         int(spike.loc[i]) if not pd.isna(p99.loc[i]) else 0,
         "VIXY/massive", datetime.utcnow().isoformat(timespec="seconds"))
        for i in px.index]
con.executemany("INSERT OR REPLACE INTO vix_history VALUES (?,?,?,?,?,?,?,?)", rows)
con.commit()

r = con.execute("SELECT COUNT(*), MIN(date), MAX(date), SUM(is_spike) FROM vix_history").fetchone()
print(f"  vix_history: {r[0]:,} rows / {r[1]} .. {r[2]}")
print(f"  spike days (top 1% of |move|): {r[3]}  ({100*r[3]/max(r[0],1):.1f}%)")
last = con.execute("SELECT date, close, abs_ret, spike_p99, is_spike FROM vix_history "
                   "ORDER BY date DESC LIMIT 3").fetchall()
print("  latest:")
for x in last:
    print(f"    {x[0]}  close={x[1]:.2f}  |ret|={100*(x[2] or 0):.1f}%  "
          f"p99={100*(x[3] or 0):.1f}%  spike={x[4]}")
con.close()
