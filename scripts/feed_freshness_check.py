#!/usr/bin/env python3
"""
scripts/feed_freshness_check.py
Feed-freshness monitor. Checks MAX(date) per source table against a per-feed
staleness budget matched to that feed's REAL cadence; alarms (desktop notif +
nonzero exit) only on genuine deviation.

WHY: 2026-06-27 found institutional feed silently dead 37d, economic_calendar
stale 5wk (401 rate-limit, handled 'gracefully' -> silent). Four staleness
events found by accident. This turns 'found by luck' into 'alarmed'.

Per-feed thresholds are essential: short_interest is FINRA bi-monthly (3-4wk
gaps NORMAL), institutional should never exceed ~4d. A generic threshold would
false-alarm daily on short_interest.

USAGE: python scripts/feed_freshness_check.py [--quiet] [--json]
EXIT:  0=all fresh, 1=stale (details on stderr + desktop notification)
"""
from __future__ import annotations
import argparse, datetime as dt, json, sqlite3, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FEEDS = [
    ("institutional_trades", "institutional_trades.db",    "institutional_trades", "trade_date",      4,  "daily (Pipeline A Stage 2.5)"),
    ("institutional_duckdb", "institutional_trades.duckdb", "institutional_trades", "trade_date",      4,  "daily mirror (sync_inst_to_duckdb)"),
    ("options_skew_history", "accuracy.db",                 "options_skew_history", "date",            4,  "daily"),
    ("insider_flows",        "insider_trades.db",           "insider_flows",        "date",            5,  "daily (Pipeline A Stage 1)"),
    ("economic_calendar",    "accuracy.db",                 "economic_calendar",    "event_date",      6,  "M/W/F refresh"),
    ("short_interest",       "short_interest.db",           "short_interest",       "settlement_date", 35, "FINRA bi-monthly (3-4wk gaps NORMAL)"),
    ("daily_prices",         "prices.db",                   "daily_prices",         "date",            10, "PEAD cache (fetch_and_pead, NOT daily cron; momentum_shadow reads this)"),
    # Added Jul 12 2026 after walk_forward_history was found DEAD since Jun 29 and
    # nothing alerted. Root causes: cron said `* * 1` (Monday) not `* * 0` (Sunday);
    # it fired 01:00-03:00 VN while the Mac wakes at 06:50; and `timeout 3600` killed
    # every run at exactly 60 min. Three failures, zero alarms, two weeks of silence.
    # The point of a freshness check is to watch the TABLE, not trust the job.
    ("walk_forward_history", "accuracy.db",                 "walk_forward_history", "run_date",       10, "weekly Sun 08/10/12 VN — the honest-OOS harness"),
    ("predictions",          "accuracy.db",                 "predictions",          "prediction_date", 4, "daily (Pipeline B Stage 3)"),
    ("outcomes",             "accuracy.db",                 "outcomes",             "prediction_date", 9, "daily reconcile (h=5 needs 5 sessions to mature)"),
    ("raw_bars",             "prices.db",                   "raw_bars",             "d",               4, "daily (price_cache via Pipeline A/B)"),
    # GEX: UW serves a ROLLING ~250-day window and cannot be backfilled further.
    # A gap here is permanent -- the data is simply gone. Tighter budget than most.
    ("options_greeks",       "accuracy.db",                 "options_greeks",       "date",            4, "daily GEX pull (Tue-Sat 05:30 VN) -- CANNOT be backfilled, gaps are PERMANENT"),
]

def _max_duckdb(p, table, col):
    try:
        import duckdb
    except ImportError:
        return "__SKIP__"
    try:
        c = duckdb.connect(str(p), read_only=True)
        r = c.execute(f"SELECT MAX({col}) FROM {table}").fetchone(); c.close()
        return str(r[0])[:10] if r and r[0] is not None else None
    except Exception as e:
        return f"__ERR__:{e}"

def _max_sqlite(p, table, col):
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        r = conn.execute(f"SELECT MAX({col}) FROM {table}").fetchone(); conn.close()
        return str(r[0])[:10] if r and r[0] is not None else None
    except Exception as e:
        return f"__ERR__:{e}"

def check_feeds():
    today = dt.date.today(); out = []
    for name, dbf, table, col, budget, cadence in FEEDS:
        p = ROOT / dbf
        rec = {"feed": name, "db": dbf, "table": table, "stale_days": budget,
               "cadence": cadence, "latest": None, "age_days": None, "status": None}
        if not p.exists():
            rec["status"] = "MISSING_DB"; out.append(rec); continue
        latest = _max_duckdb(p, table, col) if dbf.endswith(".duckdb") else _max_sqlite(p, table, col)
        if latest == "__SKIP__":
            rec["status"] = "SKIP_NO_DUCKDB"; out.append(rec); continue
        if latest is None or (isinstance(latest, str) and latest.startswith("__ERR__")):
            rec["status"] = "ERROR"; rec["latest"] = latest; out.append(rec); continue
        rec["latest"] = latest
        try:
            age = (today - dt.date.fromisoformat(latest)).days
            rec["age_days"] = age
            rec["status"] = "STALE" if age > budget else "OK"
        except Exception:
            rec["status"] = "ERROR"
        out.append(rec)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = check_feeds()
    stale = [r for r in results if r["status"] in ("STALE", "ERROR", "MISSING_DB")]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        w = max(len(r["feed"]) for r in results)
        for r in results:
            age = f"{r['age_days']}d" if r["age_days"] is not None else "-"
            mark = {"OK":"OK ","STALE":"STALE","ERROR":"ERR","MISSING_DB":"NODB",
                    "SKIP_NO_DUCKDB":"skip","SKIP":"skip"}.get(r["status"],"??")
            print(f"[{mark:>5}] {r['feed']:<{w}}  latest={r['latest'] or '-'}  age={age:<5} budget={r['stale_days']}d  [{r['cadence']}]")
    if stale:
        names = ", ".join(f"{r['feed']}({r['age_days']}d)" if r['age_days'] is not None else r['feed'] for r in stale)
        msg = f"STALE FEED(S): {names}"
        if not args.quiet:
            try:
                subprocess.run(["osascript","-e",f'display notification "{msg}" with title "ML Quant Fund — Feed Freshness"'], check=False, capture_output=True)
            except Exception:
                pass
        print(f"\n>>> {msg}", file=sys.stderr)
        sys.exit(1)
    print("\n>>> all feeds fresh")
    sys.exit(0)

if __name__ == "__main__":
    main()
