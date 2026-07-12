#!/usr/bin/env python3
"""
scripts/feed_freshness_check.py
Feed-freshness monitor. Checks MAX(date) per source table against a per-feed
staleness budget matched to that feed's REAL cadence; alarms (desktop notif +
nonzero exit) only on genuine deviation.

Per-feed thresholds are essential: short_interest is FINRA bi-monthly (3-4wk
gaps NORMAL), institutional should never exceed ~4d.

2026-07-13 fixes:
  - The 7-orphan block was PASTED TWICE. Every one printed twice and appeared
    twice in the STALE summary.
  - prediction_features / portfolio_returns_ab had date_col='date'. Neither
    table HAS a 'date' column (both use 'prediction_date'), so both ERRORed
    every run and were effectively UNWATCHED. prediction_features is the
    model's input snapshot. An entry that ERRORs every run is not "watched";
    it is unwatched with extra steps.
  - today was dt.date.today() = the VIETNAM date, compared against US market
    dates. VN is ET+11, so age was inflated ~1d. Now America/New_York.
  - walk_forward_history budget 10d -> 8d. It runs WEEKLY (Sun), so a healthy
    table never exceeds 7d. At 10d, ONE missed Sunday stays silent.
"""
from __future__ import annotations
import argparse, datetime as dt, json, sqlite3, subprocess, sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

FEEDS = [
    ("institutional_trades", "institutional_trades.db",     "institutional_trades", "trade_date",      4,  "daily (Pipeline A Stage 2.5)"),
    ("institutional_duckdb", "institutional_trades.duckdb", "institutional_trades", "trade_date",      4,  "daily mirror (sync_inst_to_duckdb)"),
    ("options_skew_history", "accuracy.db",                 "options_skew_history", "date",            4,  "daily"),
    ("insider_flows",        "insider_trades.db",           "insider_flows",        "date",            5,  "daily (Pipeline A Stage 1)"),
    ("economic_calendar",    "accuracy.db",                 "economic_calendar",    "event_date",      6,  "M/W/F refresh"),
    ("short_interest",       "short_interest.db",           "short_interest",       "settlement_date", 35, "FINRA bi-monthly (3-4wk gaps NORMAL)"),
    ("daily_prices",         "prices.db",                   "daily_prices",         "date",            10, "PEAD cache (momentum_shadow reads this)"),
    ("walk_forward_history", "accuracy.db",                 "walk_forward_history", "run_date",        8,  "weekly Sun 08/10/12 VN -- the honest-OOS harness"),
    ("predictions",          "accuracy.db",                 "predictions",          "prediction_date", 4,  "daily (Pipeline B Stage 3)"),
    ("outcomes",             "accuracy.db",                 "outcomes",             "prediction_date", 9,  "daily reconcile (h=5 needs 5 sessions to mature)"),
    ("raw_bars",             "prices.db",                   "raw_bars",             "d",               4,  "daily (price_cache via Pipeline A/B)"),
    ("options_greeks",       "accuracy.db",                 "options_greeks",       "date",            4,  "daily GEX pull -- CANNOT be backfilled, gaps are PERMANENT"),
    ("vix_history",          "accuracy.db",                 "vix_history",          "date",            4,  "VIXY via Massive -- risk_gate spike detector reads this"),
    ("momentum_shadow",      "accuracy.db",                 "momentum_shadow_predictions", "prediction_date", 4, "Pipeline C stage 3 -- momentum promotion evidence"),
    ("prediction_features",  "accuracy.db",                 "prediction_features",  "prediction_date", 4,  "feature snapshot per prediction"),
    ("dark_pool_history",    "accuracy.db",                 "dark_pool_history",    "date",            4,  "UW dark pool (Pipeline A/C)"),
    ("institutional_history","accuracy.db",                 "institutional_history","date",            4,  "UW institutional flow"),
    ("analyst_cache",        "accuracy.db",                 "analyst_cache",        "date",            7,  "analyst ratings (weekly-ish)"),
    ("ftd_cache",            "accuracy.db",                 "ftd_cache",            "date",            7,  "fails-to-deliver (SEC, lagged)"),
    ("wiki_pageviews_cache", "accuracy.db",                 "wiki_pageviews_cache", "date",            7,  "wikipedia attention proxy"),
    ("portfolio_returns_ab", "accuracy.db",                 "portfolio_returns_ab", "prediction_date", 14, "REC% A/B framework -- STALE since 2026-05-29, may be retired"),
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
    today = dt.datetime.now(ET).date()
    out, seen = [], set()
    for name, dbf, table, col, budget, cadence in FEEDS:
        if name in seen:
            continue
        seen.add(name)
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
            mark = {"OK": "OK ", "STALE": "STALE", "ERROR": "ERR", "MISSING_DB": "NODB",
                    "SKIP_NO_DUCKDB": "skip", "SKIP": "skip"}.get(r["status"], "??")
            print(f"[{mark:>5}] {r['feed']:<{w}}  latest={r['latest'] or '-'}  "
                  f"age={age:<5} budget={r['stale_days']}d  [{r['cadence']}]")
    if stale:
        names = ", ".join(
            f"{r['feed']}({r['age_days']}d)" if r['age_days'] is not None else r['feed']
            for r in stale)
        msg = f"STALE FEED(S): {names}"
        if not args.quiet:
            try:
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{msg}" with title "ML Quant Fund -- Feed Freshness"'],
                    check=False, capture_output=True)
            except Exception:
                pass
        print(f"\n>>> {msg}", file=sys.stderr)
        sys.exit(1)
    print("\n>>> all feeds fresh")
    sys.exit(0)


if __name__ == "__main__":
    main()
