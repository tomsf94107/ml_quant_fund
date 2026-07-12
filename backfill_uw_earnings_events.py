#!/usr/bin/env python3
"""
backfill_uw_earnings_events.py -- store UW's earnings-event history so
expected_move_perc can be a TRAINING feature, not a production-only one.

WHY: builder.py:1334 gates load_uw_earnings_features behind `if not training_mode`
because it makes a LIVE API call, which can't be point-in-time for a historical date.
Correct instinct. But UW's /api/earnings/{ticker} returns ~119 HISTORICAL rows per
ticker, each stamped with its own report_date. Read from a table instead of the wire
and the feature becomes PIT-honest and trainable.

Today the model trains with expected_move_perc = 0 and then receives live values it
has never learned to use. Wasted feature, not a leak.

PIT rule: expected_move is the options-implied move BEFORE the print, so it is
knowable from (report_date - 30d) up to report_date. Usable on any date in that
window. After the announcement it is stale and must not carry forward.
"""
import sqlite3, sys, time
from datetime import datetime
sys.path.insert(0, ".")
from features.uw_client import uw_get

con = sqlite3.connect("earnings.db", timeout=60)
con.execute("""CREATE TABLE IF NOT EXISTS uw_earnings_events (
    ticker              TEXT NOT NULL,
    report_date         TEXT NOT NULL,
    report_time         TEXT,
    ending_fiscal_quarter TEXT,
    expected_move       REAL,
    expected_move_perc  REAL,
    street_mean_est     REAL,
    actual_eps          REAL,
    pre_earnings_move_1d  REAL,
    pre_earnings_move_3d  REAL,
    post_earnings_move_1d REAL,
    post_earnings_move_3d REAL,
    post_earnings_move_1w REAL,
    fetched_at          TEXT,
    PRIMARY KEY (ticker, report_date))""")
con.commit()

tks = [l.strip().upper() for l in open("tickers.txt") if l.strip() and not l.startswith("#")]
print(f"  {len(tks)} tickers")
ok = fail = rows = 0
for i, t in enumerate(tks, 1):
    try:
        data = (uw_get(f"/api/earnings/{t}") or {}).get("data") or []
    except Exception as e:
        fail += 1; continue
    def f(x):
        try:    return float(x)
        except: return None
    batch = [(t, str(x.get("report_date"))[:10], x.get("report_time"),
              x.get("ending_fiscal_quarter"),
              f(x.get("expected_move")), f(x.get("expected_move_perc")),
              f(x.get("street_mean_est")), f(x.get("actual_eps")),
              f(x.get("pre_earnings_move_1d")), f(x.get("pre_earnings_move_3d")),
              f(x.get("post_earnings_move_1d")), f(x.get("post_earnings_move_3d")),
              f(x.get("post_earnings_move_1w")),
              datetime.utcnow().isoformat(timespec="seconds"))
             for x in data if x.get("report_date")]
    if batch:
        con.executemany("INSERT OR REPLACE INTO uw_earnings_events VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        con.commit(); ok += 1; rows += len(batch)
    time.sleep(0.7)
    if i % 50 == 0: print(f"  [{i}/{len(tks)}] ok={ok} fail={fail} rows={rows:,}", flush=True)

r = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(report_date), MAX(report_date), "
                "SUM(expected_move_perc IS NOT NULL) FROM uw_earnings_events").fetchone()
print(f"\n  uw_earnings_events: {r[0]:,} rows / {r[1]} tickers / {r[2]} .. {r[3]}")
print(f"  expected_move_perc populated: {r[4]:,} ({100*r[4]/max(r[0],1):.0f}%)")
con.close()
