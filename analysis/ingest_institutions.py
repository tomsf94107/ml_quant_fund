#!/usr/bin/env python3
"""
ingest_institutions.py — 13F holdings for concentrated managers.

Builds institutions.db: inst_roster + inst_holdings.

WHAT THE FIRST VERSION GOT WRONG, AND WHY THE ACTIVITY ENDPOINT IS GONE
    v1 pulled two endpoints per manager per quarter. The /activity endpoint
    does NOT filter to the requested quarter -- it returns a WINDOW of roughly
    five quarters. Verified on Berkshire at date=2026-06-30: the 500 rows
    carried report_date values of 2026-03-31 (29), 2025-06-30 (18),
    2025-12-31 (16), 2026-06-30 (15) and 2025-09-30 (13), each correctly paired
    with its own filing date (2026-05-15, 2025-08-14, 2026-02-17, 2026-08-14,
    2025-11-14 -- the real 13F deadlines).

    v1 stored the REQUESTED date as report_date and discarded each row's own.
    Two consequences: every quarter label was wrong, and because the primary
    key was (name, report_date, ticker, security_type), five quarters of the
    same ticker collapsed to one row and INSERT OR REPLACE kept only the last.
    About 80% of every response was silently thrown away.

    /holdings does NOT have this problem -- the same probe returned 30 rows all
    at date=2026-06-30. One quarter per call, labels correct.

    So activity is dropped entirely. holdings already carries units_change,
    which was the only reason to call activity at all. Half the calls, no
    corruption: 127 managers x 51 quarters = 6,477 instead of 12,954.

FIELDS v1 THREW AWAY AND THIS ONE KEEPS
    perc_of_total          the position as a share of the WHOLE book. This is
                           the conviction measure. Ranked by dollars, Citadel's
                           $6.2B AAPL put line -- 0.7% of its book, dealer
                           inventory -- outranks Thiel's $59M VST at 14.1%,
                           which is a conviction bet. Dollar rank inverts them.
    perc_of_share_value    same, against the equity book only.
    put_call               "call" / "put" / NULL. WITHOUT IT a crowding count
                           is meaningless: Citadel's 2026-06-30 AAPL book is a
                           $9.0B call line, a $6.2B PUT line and $2.1B of
                           shares -- three positions, three directions.
    first_buy              when the manager first held the name. A NEW position
                           is a stronger statement than an add.
    units_change           vendor-computed QoQ change; handles splits.
    avg_price              vendor-DERIVED cost basis, not a reported figure.
                           Direction is reliable, the number is not.
    shares_outstanding     lets a position be read as % of the company.

ROSTER BY TRUNCATION, NOT BY FLAG
    is_hedge_fund marks T. Rowe, Dimensional and AllianceBernstein as hedge
    funds. share_holdings is a SHARE COUNT, not a position count -- T. Rowe
    shows 7.16 billion. So the roster is built by probing: ask for 500 holdings
    and count what returns. Exactly 500 means truncated. 127 of 500 probed are
    fully visible; Citadel exceeds 3,148 tickers over 12 pages and 68% of its
    rows are Options -- dealer inventory, not a view.

USAGE
    python analysis/ingest_institutions.py --roster
    python analysis/ingest_institutions.py --backfill
    python analysis/ingest_institutions.py --status
"""
import argparse
import json
import os
import sqlite3
import time
import urllib.parse as up
import urllib.request as ureq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.getenv("INSTITUTIONS_DB_PATH", ROOT / "institutions.db"))
API = "https://api.unusualwhales.com"
CAP = 500

DDL = """
CREATE TABLE IF NOT EXISTS inst_roster (
    name TEXT PRIMARY KEY, cik TEXT, is_hedge INTEGER, positions INTEGER,
    total_value REAL, tags TEXT, people TEXT, description TEXT, probed_at TEXT
);
CREATE TABLE IF NOT EXISTS inst_holdings (
    name          TEXT NOT NULL,
    report_date   TEXT NOT NULL,   -- the ROW's own date, never the requested one
    ticker        TEXT NOT NULL,
    security_type TEXT NOT NULL,
    put_call      TEXT,            -- in the PK: a call and a put on one name
                                   -- are different positions, often opposite
    units         REAL,
    units_change  REAL,
    value         REAL,
    pct_total     REAL,            -- perc_of_total: the conviction measure
    pct_share     REAL,
    avg_price     REAL,            -- vendor-derived, not a reported basis
    close         REAL,
    shares_out    REAL,
    first_buy     TEXT,
    sector        TEXT,
    full_name     TEXT,
    PRIMARY KEY (name, report_date, ticker, security_type, put_call)
);
CREATE INDEX IF NOT EXISTS idx_ih_q  ON inst_holdings(report_date);
CREATE INDEX IF NOT EXISTS idx_ih_tk ON inst_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_ih_nm ON inst_holdings(name);
"""


def quarters(start="2013-12-31", end="2026-06-30"):
    out = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            d = f"{y}-{md}"
            if start <= d <= end:
                out.append(d)
    return sorted(out, reverse=True)


def get(ep, key, tries=3):
    for i in range(tries):
        try:
            r = ureq.urlopen(ureq.Request(
                API + ep, headers={"Authorization": f"Bearer {key}",
                                   "Accept": "application/json"}), timeout=30)
            return json.load(r).get("data") or []
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return []


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def rows_for(nm, payload):
    """Map a holdings response to storage rows.

    report_date comes from the ROW (x["date"]), never from the requested
    quarter -- that substitution is what corrupted v1.
    """
    out = []
    for x in payload:
        tk = x.get("ticker")
        if not tk:
            continue
        out.append((
            nm, str(x.get("date"))[:10], tk,
            x.get("security_type") or "Share", x.get("put_call"),
            _f(x.get("units")), _f(x.get("units_change")), _f(x.get("value")),
            _f(x.get("perc_of_total")), _f(x.get("perc_of_share_value")),
            _f(x.get("avg_price")), _f(x.get("close")),
            _f(x.get("shares_outstanding")), x.get("first_buy"),
            x.get("sector"), x.get("full_name"),
        ))
    return out


def build_roster(con, key, probe_q="2026-06-30"):
    insts = get("/api/institutions?limit=500", key)
    print(f"  probing {len(insts)} institutions at {probe_q} ...")
    rows, full = [], 0
    for i, x in enumerate(insts, 1):
        nm = x["name"]
        try:
            h = get(f"/api/institution/{up.quote(nm)}/holdings"
                    f"?limit={CAP}&date={probe_q}", key)
        except Exception:
            continue
        n = len(h)
        if 3 <= n < CAP:
            full += 1
        # KEEP THE PROBE'S PAYLOAD. v1 ran this 500-call probe twice and stored
        # only the row counts, discarding every holding it had already paid for.
        if h:
            con.executemany("INSERT OR REPLACE INTO inst_holdings VALUES "
                            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows_for(nm, h))
        rows.append((nm, x.get("cik"), 1 if x.get("is_hedge_fund") else 0, n,
                     _f(x.get("total_value")) or 0.0,
                     json.dumps(x.get("tags") or []),
                     json.dumps(x.get("people") or [])[:2000],
                     (x.get("description") or "")[:1000], probe_q))
        if i % 100 == 0:
            con.commit()
            print(f"    {i}/{len(insts)}")
        time.sleep(0.04)
    con.executemany("INSERT OR REPLACE INTO inst_roster VALUES (?,?,?,?,?,?,?,?,?)",
                    rows)
    con.commit()
    print(f"  {full} of {len(rows)} fully visible (< {CAP} positions)")


def tracked(con):
    return [r[0] for r in con.execute(
        f"SELECT name FROM inst_roster WHERE positions >= 3 AND positions < {CAP} "
        f"ORDER BY total_value DESC")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(DB, timeout=60)
    con.executescript(DDL)

    if a.status:
        for t in ("inst_roster", "inst_holdings"):
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:16}{n:>12,}")
        r = con.execute("SELECT MIN(report_date), MAX(report_date), "
                        "COUNT(DISTINCT report_date), COUNT(DISTINCT name) "
                        "FROM inst_holdings").fetchone()
        if r[0]:
            print(f"  {r[0]} .. {r[1]}  {r[2]} quarters, {r[3]} managers")
        con.close(); return

    key = os.environ.get("UW_API_KEY")
    if not key:
        print("UW_API_KEY not set"); return

    if a.roster or not con.execute(
            "SELECT COUNT(*) FROM inst_roster").fetchone()[0]:
        build_roster(con, key)

    names = tracked(con)
    if a.limit:
        names = names[:a.limit]
    qs = quarters() if a.backfill else ["2026-06-30"]

    # RESUME. Skip managers already carrying most of the expected quarters, so
    # a crash costs one manager's work rather than the whole run.
    done = {r[0] for r in con.execute(
        "SELECT name FROM inst_holdings GROUP BY name "
        "HAVING COUNT(DISTINCT report_date) >= ?", (max(1, len(qs) - 8),))}
    if done and a.backfill:
        names = [n for n in names if n not in done]
        print(f"  resuming: {len(done)} complete, {len(names)} to go")

    print(f"\n{len(names)} managers x {len(qs)} quarters = "
          f"{len(names)*len(qs):,} calls\n")
    total = 0
    for i, nm in enumerate(names, 1):
        enc = up.quote(nm)
        for q in qs:
            try:
                h = get(f"/api/institution/{enc}/holdings?limit={CAP}&date={q}", key)
            except Exception as e:
                print(f"    {nm[:30]} {q}: {type(e).__name__}")
                continue
            if h:
                rr = rows_for(nm, h)
                con.executemany("INSERT OR REPLACE INTO inst_holdings VALUES "
                                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rr)
                total += len(rr)
            time.sleep(0.04)
        con.commit()
        print(f"  [{i}/{len(names)}] {nm[:40]:42} {total:,} rows")
    print(f"\nstored {total:,} rows")
    con.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun to resume, writes are idempotent.")
