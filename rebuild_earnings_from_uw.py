#!/usr/bin/env python3
"""
rebuild_earnings_from_uw.py -- rebuild earnings events keyed on REAL announcement dates.

BUG IT FIXES
  earnings_surprises.report_date holds the FISCAL PERIOD END, not the announcement.
  75% of rows land on a quarter-end; 35% on a weekend. MU's row says 2026-05-31 but
  MU announced 2026-06-24. PEAD enters day +2 from that column -> 15-27 days of
  phantom foreknowledge on every event. builder.py's created_at = fiscal_end + 2BD
  inherits the same leak.

VERIFIED
  UW's report_date == the SEC 8-K Item 2.02 filing date, 8/8 exact matches against
  our own eightk_items table. That IS the announcement date.

WRITES  earnings.db.earnings_events   (new table; the old one is NOT touched)
  quarterly only. Rows with no announce_date are SKIPPED, never back-filled with the
  fiscal end -- that back-fill is the bug.
"""
import argparse, os, sqlite3, sys, time
from datetime import datetime, timezone

def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.6)
    a = ap.parse_args()
    root = os.path.expanduser(a.root); sys.path.insert(0, root)
    from features.uw_client import uw_get

    uni = os.path.join(root, "tickers.txt")
    if not os.path.isfile(uni):
        print(f"[STOP] {uni} not found"); sys.exit(1)
    tickers = [l.strip().upper() for l in open(uni) if l.strip() and not l.startswith("#")]
    if a.limit: tickers = tickers[:a.limit]
    print(f"  universe: {len(tickers)} tickers")

    con = sqlite3.connect(os.path.join(root, "earnings.db"), timeout=60)
    con.execute("""CREATE TABLE IF NOT EXISTS earnings_events (
        ticker TEXT NOT NULL, announce_date TEXT NOT NULL, fiscal_end TEXT,
        eps_actual REAL, eps_estimate REAL, eps_surprise REAL,
        report_type TEXT, source TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY (ticker, announce_date))""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ee ON earnings_events(ticker, announce_date)")
    con.commit()

    def f(x):
        try: return float(x)
        except (TypeError, ValueError): return None

    ins = ann = nodate = failed = 0
    for i, tk in enumerate(tickers, 1):
        try:
            rows = (uw_get(f"/api/stock/{tk}/earnings") or {}).get("data") or []
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {tk}: FAIL {e}"); failed += 1
            time.sleep(a.sleep); continue
        n = 0
        for x in rows:
            if (x.get("report_type") or "").lower() != "quarterly":
                ann += 1; continue
            ad = x.get("report_date")
            if not ad:
                nodate += 1; continue
            act, est = f(x.get("reported_eps")), f(x.get("estimated_eps"))
            sur = f(x.get("surprise"))
            if sur is None and act is not None and est is not None: sur = act - est
            con.execute("""INSERT OR REPLACE INTO earnings_events
                (ticker,announce_date,fiscal_end,eps_actual,eps_estimate,
                 eps_surprise,report_type,source,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (tk, str(ad)[:10], str(x.get("fiscal_date_ending") or "")[:10] or None,
                 act, est, sur, "quarterly", "uw", now_iso()))
            n += 1; ins += 1
        if i % 25 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {tk}: +{n}   total={ins:,}")
        con.commit(); time.sleep(a.sleep)

    q = con.execute("""SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(announce_date),
        MAX(announce_date), SUM(eps_actual IS NOT NULL AND eps_estimate IS NOT NULL)
        FROM earnings_events""").fetchone()
    print("\n" + "=" * 70)
    print(f"  written={ins:,}  annual skipped={ann:,}  no-date skipped={nodate:,}  failed={failed}")
    print(f"  earnings_events: {q[0]:,} rows | {q[1]} tickers | {q[2]} .. {q[3]}")
    print(f"  usable (actual+estimate): {q[4]:,}      [old table gave 863]")
    print("\n  SANITY -- an announcement date is NOT a period end:")
    for lbl, sql in [
        ("on a quarter-end", "SELECT COUNT(*) FROM earnings_events WHERE substr(announce_date,6,5) IN ('03-31','06-30','09-30','12-31')"),
        ("on a weekend",     "SELECT COUNT(*) FROM earnings_events WHERE CAST(strftime('%w',announce_date) AS INT) IN (0,6)")]:
        c = con.execute(sql).fetchone()[0]
        print(f"    {lbl:18s}: {c:,} ({100*c/max(1,q[0]):.1f}%)   <- should be ~0   [old: 75% / 35%]")
    print("=" * 70)
    con.close()

if __name__ == "__main__":
    main()
