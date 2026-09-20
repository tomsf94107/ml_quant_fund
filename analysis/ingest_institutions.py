#!/usr/bin/env python3
"""
ingest_institutions.py — 13F holdings and activity for concentrated managers.

Builds institutions.db: roster, per-quarter holdings, per-quarter activity.

WHY THIS ONE IS DIFFERENT FROM THE CONGRESS INGEST
    That source had four quarters of history and no backfill, so it could only
    accumulate forward and nothing was testable on arrival. This reaches 2013 --
    Third Point returns rows for every quarter from 2013-12-31 to 2026-06-30,
    around 50 quarters. Prices for the same span are already in raw_bars. So a
    claim can be tested the day the data lands rather than in a year.

THE ROSTER IS DEFINED BY TRUNCATION, NOT BY A FLAG
    UW's is_hedge_fund is unreliable -- it marks T. Rowe, Dimensional and
    AllianceBernstein as hedge funds. share_holdings is a SHARE COUNT, not a
    position count: T. Rowe shows 7.16 billion.

    So the roster is built by probing: request 500 holdings for a quarter and
    see how many come back. Exactly 500 means the book is truncated and its
    position count, sector weights and concentration changes are all artifacts
    of the cap. Under 500 means the whole book is visible.

    Measured 2026-09-19 across 500 institutions: 127 fully visible. Citadel,
    Millennium and ARK all hit the cap. That filter is doing real work -- a
    manager whose entire 13F fits under 500 names is one making concentrated
    bets, which is where a thesis shift is legible at all. A multi-strat
    holding 3,000 names has no thesis to read.

WHAT 13F DOES AND DOES NOT SHOW
    Long US equity positions at quarter end, filed up to 45 days later. It does
    NOT show shorts, most derivatives, cash, bonds, or anything private. A
    manager reported as "flat" may simply hold nothing reportable -- Thiel
    Macro showed zero positions for two quarters in 2025, which means no long
    US equities, not no exposure.

    The 45-day lag is structural and it is the reason this is a THESIS tracker
    rather than a copy-trading signal. What survives the lag is the shape of a
    book: a manager who liquidates everything and rebuilds into a different
    sector two quarters later is making a statement that does not decay in six
    weeks. Their individual AMZN position does.

    avg_price in the feed is DERIVED by the vendor from filing history, not a
    reported cost basis. Direction is reliable, the exact number is not.

USAGE
    python analysis/ingest_institutions.py --roster        # build/refresh roster
    python analysis/ingest_institutions.py --backfill      # all quarters, roster
    python analysis/ingest_institutions.py                 # latest quarter only
    python analysis/ingest_institutions.py --status
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse as up
import urllib.request as ureq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.getenv("INSTITUTIONS_DB_PATH", ROOT / "institutions.db"))
API = "https://api.unusualwhales.com"
CAP = 500          # the vendor's hard page size; 500 back means truncated

DDL = """
CREATE TABLE IF NOT EXISTS inst_roster (
    name        TEXT PRIMARY KEY,
    cik         TEXT,
    is_hedge    INTEGER,
    positions   INTEGER,   -- at the probe quarter; < CAP means fully visible
    total_value REAL,
    tags        TEXT,
    people      TEXT,
    description TEXT,
    probed_at   TEXT
);
CREATE TABLE IF NOT EXISTS inst_holdings (
    name        TEXT NOT NULL,
    report_date TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    units       REAL,
    value       REAL,
    avg_price   REAL,   -- vendor-derived, not a reported cost basis
    close       REAL,
    sector      TEXT,
    security_type TEXT,
    full_name   TEXT,
    PRIMARY KEY (name, report_date, ticker, security_type)
);
CREATE TABLE IF NOT EXISTS inst_activity (
    name         TEXT NOT NULL,
    report_date  TEXT NOT NULL,
    filing_date  TEXT,
    ticker       TEXT NOT NULL,
    units        REAL,
    units_change REAL,
    avg_price    REAL,
    security_type TEXT,
    PRIMARY KEY (name, report_date, ticker, security_type)
);
CREATE INDEX IF NOT EXISTS idx_ih_q  ON inst_holdings(report_date);
CREATE INDEX IF NOT EXISTS idx_ih_tk ON inst_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_ia_q  ON inst_activity(report_date);
"""


def quarters(start="2013-12-31", end="2026-06-30"):
    out, y = [], int(start[:4])
    while y <= int(end[:4]):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            d = f"{y}-{md}"
            if start <= d <= end:
                out.append(d)
        y += 1
    return sorted(out, reverse=True)


def get(ep, key, tries=3):
    for i in range(tries):
        try:
            r = ureq.urlopen(ureq.Request(
                API + ep, headers={"Authorization": f"Bearer {key}",
                                   "Accept": "application/json"}), timeout=30)
            return json.load(r).get("data") or []
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return []


def build_roster(con, key, probe_q="2026-06-30"):
    """One call per institution. Position count at probe_q decides inclusion."""
    insts = get("/api/institutions?limit=500", key)
    print(f"  probing {len(insts)} institutions at {probe_q} ...")
    rows, full = [], 0
    for i, x in enumerate(insts, 1):
        nm = x["name"]
        try:
            h = get(f"/api/institution/{up.quote(nm)}/holdings"
                    f"?limit={CAP}&date={probe_q}", key)
            n = len(h)
        except Exception:
            continue
        if 3 <= n < CAP:
            full += 1
        rows.append((nm, x.get("cik"), 1 if x.get("is_hedge_fund") else 0, n,
                     float(x.get("total_value") or 0),
                     json.dumps(x.get("tags") or []),
                     json.dumps(x.get("people") or [])[:2000],
                     (x.get("description") or "")[:1000], probe_q))
        if i % 100 == 0:
            print(f"    {i}/{len(insts)}")
        time.sleep(0.04)
    con.executemany("INSERT OR REPLACE INTO inst_roster VALUES "
                    "(?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    print(f"  {full} of {len(rows)} fully visible (< {CAP} positions)")


def tracked(con):
    return [r[0] for r in con.execute(
        f"SELECT name FROM inst_roster WHERE positions >= 3 AND positions < {CAP} "
        f"ORDER BY total_value DESC")]


def pull(con, key, names, qs):
    nh = na = 0
    for i, nm in enumerate(names, 1):
        enc = up.quote(nm)
        for q in qs:
            try:
                h = get(f"/api/institution/{enc}/holdings?limit={CAP}&date={q}", key)
            except Exception:
                h = []
            if h:
                con.executemany(
                    "INSERT OR REPLACE INTO inst_holdings VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(nm, q, r.get("ticker"), _f(r.get("units")),
                      _f(r.get("value")), _f(r.get("avg_price")),
                      _f(r.get("close")), r.get("sector"),
                      r.get("security_type"), r.get("full_name")) for r in h
                     if r.get("ticker")])
                nh += len(h)
            try:
                a = get(f"/api/institution/{enc}/activity?limit={CAP}&date={q}", key)
            except Exception:
                a = []
            if a:
                con.executemany(
                    "INSERT OR REPLACE INTO inst_activity VALUES (?,?,?,?,?,?,?,?)",
                    [(nm, q, r.get("filing_date"), r.get("ticker"),
                      _f(r.get("units")), _f(r.get("units_change")),
                      _f(r.get("avg_price")), r.get("security_type"))
                     for r in a if r.get("ticker")])
                na += len(a)
            time.sleep(0.04)
        con.commit()
        print(f"  [{i}/{len(names)}] {nm[:40]:42} holdings {nh:,}  activity {na:,}")
    return nh, na


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def status(con):
    for t in ("inst_roster", "inst_holdings", "inst_activity"):
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:16} {n:>10,}")
        except Exception:
            print(f"  {t:16}  (absent)")
    r = con.execute("SELECT MIN(report_date), MAX(report_date), "
                    "COUNT(DISTINCT report_date), COUNT(DISTINCT name) "
                    "FROM inst_holdings").fetchone()
    if r[0]:
        print(f"  quarters {r[0]} .. {r[1]}  ({r[2]} quarters, {r[3]} managers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap managers, for a test run")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("UW_API_KEY")
    con = sqlite3.connect(DB, timeout=60)
    con.executescript(DDL)

    if a.status:
        status(con); con.close(); return
    if not key:
        print("UW_API_KEY not set"); return

    if a.roster or not con.execute(
            "SELECT COUNT(*) FROM inst_roster").fetchone()[0]:
        build_roster(con, key)

    names = tracked(con)
    if a.limit:
        names = names[:a.limit]
    qs = quarters() if a.backfill else ["2026-06-30"]
    print(f"\n{len(names)} managers x {len(qs)} quarters = "
          f"{len(names)*len(qs)*2:,} calls (daily limit 20,000)\n")
    if len(names) * len(qs) * 2 > 18000:
        print("  that exceeds the safe daily budget -- use --limit, or run")
        print("  --backfill across several days; INSERT OR REPLACE makes")
        print("  repeated runs idempotent.\n")
    nh, na = pull(con, key, names, qs)
    print(f"\nstored {nh:,} holdings rows, {na:,} activity rows")
    status(con)
    con.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume, writes are idempotent.")
