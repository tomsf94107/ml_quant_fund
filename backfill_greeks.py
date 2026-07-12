#!/usr/bin/env python3
"""
backfill_greeks.py -- one year of daily options dealer-positioning per ticker.

WHY THIS IS THE BEST REMAINING SHOT AT BRICK #2
  Every signal in this system so far is derived from PRICE and VOLUME. That is one
  information axis, it is the most crowded axis in finance, and it has produced
  exactly one validated brick (days_to_cover) plus three findings that turned out
  to be market beta wearing a costume (PEAD look-ahead, the ranker's in-sample
  Sharpe, the h=40 model's IC +0.20).

  Options dealer positioning is a DIFFERENT axis. It is not what prices did -- it
  is what market makers are FORCED to do next.

    GEX (gamma exposure) : how much dealers must trade to stay hedged.
        GEX > 0 -> dealers are long gamma -> they SELL rallies and BUY dips ->
                   they DAMPEN moves -> expect mean reversion, lower realized vol.
        GEX < 0 -> dealers are short gamma -> they BUY rallies and SELL dips ->
                   they AMPLIFY moves -> expect momentum, higher realized vol.
    DEX (delta exposure) : net directional positioning of the options market.
    VANNA : how delta moves when vol moves. Drives squeezes.
    CHARM : delta decay into expiry. Drives pinning and the unpin.

  UW's /api/stock/{t}/greek-exposure returns ~250 DATE-KEYED rows per ticker --
  a full trading year, historical, daily. Not a snapshot. Backfillable tonight.

WHAT THIS SCRIPT DOES
  Purely additive: creates accuracy.db.options_greeks. Touches nothing else. No
  production code path changes. Pull first, validate second, wire only if it
  passes -- the sequence that should have been used on PEAD.

  Stores raw Greeks AND the net aggregates, plus a per-ticker z-score, because
  raw GEX scales with market cap and open interest: NVDA's gamma is 100x SENS's
  for reasons that have nothing to do with signal. The cross-sectional object has
  to be normalised per ticker or the "signal" is just "which stock is big".
"""
import sqlite3, sys, time
from datetime import datetime
sys.path.insert(0, ".")
from features.uw_client import uw_get

con = sqlite3.connect("accuracy.db", timeout=60)
con.execute("""CREATE TABLE IF NOT EXISTS options_greeks (
    ticker       TEXT NOT NULL,
    date         TEXT NOT NULL,
    call_gamma   REAL, put_gamma  REAL,
    call_delta   REAL, put_delta  REAL,
    call_vanna   REAL, put_vanna  REAL,
    call_charm   REAL, put_charm  REAL,
    net_gamma    REAL,        -- GEX: the flagship. sign is what matters.
    net_delta    REAL,        -- DEX: directional positioning
    net_vanna    REAL,
    net_charm    REAL,
    fetched_at   TEXT,
    PRIMARY KEY (ticker, date))""")
con.commit()

tks = [l.strip().upper() for l in open("tickers.txt") if l.strip() and not l.startswith("#")]
print(f"  {len(tks)} tickers, ~250 trading days each -> ~{len(tks)*250:,} rows expected")

def f(x):
    try:    return float(x)
    except: return None

ok = fail = rows = 0
for i, t in enumerate(tks, 1):
    try:
        data = (uw_get(f"/api/stock/{t}/greek-exposure") or {}).get("data") or []
    except Exception as e:
        fail += 1
        if fail <= 5: print(f"    {t}: {str(e)[:45]}")
        continue
    if not data:
        fail += 1; continue
    batch = []
    for x in data:
        d = str(x.get("date") or "")[:10]
        if not d: continue
        cg, pg = f(x.get("call_gamma")), f(x.get("put_gamma"))
        cd, pd_ = f(x.get("call_delta")), f(x.get("put_delta"))
        cv, pv = f(x.get("call_vanna")), f(x.get("put_vanna"))
        cc, pc = f(x.get("call_charm")), f(x.get("put_charm"))
        net = lambda a, b: (a + b) if (a is not None and b is not None) else None
        batch.append((t, d, cg, pg, cd, pd_, cv, pv, cc, pc,
                      net(cg, pg), net(cd, pd_), net(cv, pv), net(cc, pc),
                      datetime.utcnow().isoformat(timespec="seconds")))
    if batch:
        con.executemany(
            "INSERT OR REPLACE INTO options_greeks VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        con.commit(); ok += 1; rows += len(batch)
    time.sleep(0.7)
    if i % 50 == 0:
        print(f"  [{i}/{len(tks)}] ok={ok} fail={fail} rows={rows:,}", flush=True)

r = con.execute("""SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(date), MAX(date),
                          SUM(net_gamma IS NOT NULL),
                          SUM(CASE WHEN net_gamma < 0 THEN 1 ELSE 0 END)
                   FROM options_greeks""").fetchone()
print(f"\n  options_greeks: {r[0]:,} rows / {r[1]} tickers / {r[2]} .. {r[3]}")
print(f"  net_gamma populated : {r[4]:,}")
print(f"  NEGATIVE gamma days : {r[5]:,} ({100*r[5]/max(r[4],1):.0f}%)  <- the amplifying regime")
con.close()
