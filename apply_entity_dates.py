#!/usr/bin/env python3
"""
Purge rows that predate a ticker's CURRENT entity.

WHY: some tickers are recycled -- a fourth corporate-action class after
delist/rename/merge, and the only one that does not announce itself. The vendor
serves one continuous series per symbol, so history splices two companies.
  SPCX: AXS SPAC and New Issue ETF until Apr 2026, then SpaceX Class A.
  BETR: Aurora Acquisition SPAC until the Aug 2023 merger.

NOT a one-time cleanup. finra_short_interest.py uses REPLACE INTO, so any
fetch_log clear restores the purged rows -- demonstrated 2026-08-31, same
session that purged them. RUN THIS AFTER EVERY SI FETCH.

Registry: ticker_entity_dates.csv (ticker, entity_start, reason)
"""
import argparse, csv, os, sqlite3, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

reg = os.path.join(a.root, "ticker_entity_dates.csv")
if not os.path.isfile(reg):
    print("[STOP] %s not found" % reg); sys.exit(1)
with open(reg, newline="") as f:
    rows = [r for r in csv.DictReader(f) if r.get("ticker", "").strip()]
print("  registry: %d ticker(s)" % len(rows))

targets = [
    (os.path.join(a.root, "short_interest.db"), "short_interest", "settlement_date"),
    (os.path.join(a.root, "prices.db"), "daily_prices", "date"),
    (os.path.join(a.root, "prices.db"), "raw_bars", "d"),
]
total = 0
for db, table, datecol in targets:
    if not os.path.isfile(db): continue
    con = sqlite3.connect(db, timeout=30); cur = con.cursor()
    try:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if not cur.fetchone(): continue
        for r in rows:
            tk = r["ticker"].strip().upper(); start = r["entity_start"].strip()
            n = cur.execute("SELECT COUNT(*) FROM %s WHERE ticker=? AND %s < ?"
                            % (table, datecol), (tk, start)).fetchone()[0]
            if not n: continue
            total += n
            print("    %-16s %-6s %5d row(s) before %s" % (table, tk, n, start))
            if not a.dry_run:
                cur.execute("DELETE FROM %s WHERE ticker=? AND %s < ?"
                            % (table, datecol), (tk, start))
        if not a.dry_run: con.commit()
    finally:
        con.close()
print("  DRY-RUN: no writes" if a.dry_run else "  purged %d row(s)" % total)
