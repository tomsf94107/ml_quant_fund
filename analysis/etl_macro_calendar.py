#!/usr/bin/env python3
"""
etl_macro_calendar.py — historical macro release dates, PIT-honest.

Writes accuracy.db.macro_events. Reads FRED for release history and a static
FOMC table. Touches nothing else.

WHY
    accuracy.db.economic_calendar holds 237 rows covering 2026-05-04 to
    2026-09-04 -- the next 45 days on a rolling basis, refreshed M/W/F. That is
    enough to know what is coming and nothing to train on.

    The literature is specific about which announcements matter. Savor & Wilson
    (2013) and Lucca & Moench (2015) study a canonical set -- FOMC, nonfarm
    payrolls, ISM, GDP, industrial production, personal income, housing starts,
    initial claims, PPI, CPI and consumer sentiment -- and find the returns
    arise MAINLY from FOMC. Pooling NFP, ISM, GDP and FOMC gives an average
    pre-announcement return of 5.66% annually.

    The cross-sectional hook is Savor & Wilson's security market line result:
    on announcement days its slope is a significant 6.81 basis points, against
    roughly flat otherwise. High-beta names earn more SPECIFICALLY on
    announcement days. That is what makes a market-wide calendar usable in a
    model that ranks stocks against each other -- days_to_event alone is
    identical for every ticker on a date and cannot separate them, but
    days_to_event x beta_60d differs per stock.

WHY THE EXISTING TITLES CANNOT BE USED DIRECTLY
    economic_calendar's `title` is vendor prose and duplicates the same release
    under several names: "Consumer price index" and "CPI year over year",
    "Initial jobless claims" and "Weekly Jobless Claims", "Minutes of Fed's July
    FOMC meeting" alongside "FOMC interest-rate decision". impact='High' also
    admits 20+ Fed speaker events, which are not scheduled data releases and
    have no consensus forecast.

    So this normalises to a small set of EVENT CODES and ignores the rest.

SOURCES AND PIT HONESTY
    FRED releases/dates gives the historical release calendar for every series
    it carries -- CPI, NFP, PPI, GDP, industrial production, retail sales, PCE
    -- back decades. A release date is knowable in advance: BLS and BEA publish
    their schedules a year ahead, so using it as a forward-looking countdown is
    not a look-ahead.

    FOMC is not in FRED. Its dates are published by the Fed in the summer BEFORE
    the calendar year, so a static table is both correct and PIT-honest. The
    2016-2026 dates are hardcoded below and must be extended each year.

    NOTE the distinction that matters: the DATE is knowable in advance, the
    OUTCOME is not. This ETL stores dates only. Any feature built on the
    surprise (actual minus consensus) is knowable only after release and must be
    lagged accordingly -- the same rule that governs eps_surprise.

    python analysis/etl_macro_calendar.py --dry-run
    python analysis/etl_macro_calendar.py
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import date

DB = "accuracy.db"
DDL = """
CREATE TABLE IF NOT EXISTS macro_events (
    event_date  TEXT NOT NULL,
    event_code  TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (event_date, event_code)
)
"""

# FRED release IDs for the canonical announcements. Release, not series: a
# release has a published date, a series has an observation date, and the two
# differ by weeks. Using the observation date would be a look-ahead.
FRED_RELEASES = {
    "CPI":   10,    # Consumer Price Index
    "NFP":   50,    # Employment Situation
    "PPI":   46,    # Producer Price Index
    "GDP":   53,    # Gross Domestic Product
    "IP":    13,    # Industrial Production and Capacity Utilization
    "RETAIL": 8,    # Advance Monthly Sales for Retail and Food Services
    "PCE":   54,    # Personal Income and Outlays
}

# FOMC decision dates, second day of each two-day meeting. Published a year in
# advance by the Federal Reserve. EXTEND THIS EACH YEAR.
FOMC = """
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-03-03 2020-03-15 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-11-04 2026-12-16
""".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"))
    args = ap.parse_args()

    end = date.today().isoformat()
    print(f"macro calendar {args.start} .. {end}")
    print(f"  FOMC: {len(FOMC)} hardcoded dates "
          f"{FOMC[0]} .. {FOMC[-1]}")
    print(f"  FRED releases: {', '.join(FRED_RELEASES)}")

    if not args.api_key:
        print("\n  FRED_API_KEY not set. Source it from .env:")
        print("    set -a && . ./.env && set +a")
        if not args.dry_run:
            raise SystemExit(1)

    rows = [(d, "FOMC", "fed_calendar") for d in FOMC if d >= args.start]

    if not args.dry_run:
        for code, rid in FRED_RELEASES.items():
            try:
                q = urllib.parse.urlencode({
                    "release_id": rid, "api_key": args.api_key,
                    "file_type": "json", "realtime_start": args.start,
                    "realtime_end": end, "limit": 10000})
                u = f"https://api.stlouisfed.org/fred/release/dates?{q}"
                d = json.loads(urllib.request.urlopen(u, timeout=60).read())
                got = [r["date"] for r in d.get("release_dates", [])
                       if r["date"] >= args.start]
                rows += [(x, code, f"fred_release_{rid}") for x in got]
                print(f"  {code:<7}{len(got):>6} release dates "
                      f"{min(got) if got else '-'} .. {max(got) if got else '-'}")
            except Exception as e:
                print(f"  {code:<7}FAILED: {type(e).__name__}: {e}")
            time.sleep(0.3)

    if args.dry_run:
        print(f"\nDRY RUN -- {len(rows)} FOMC rows would be written; FRED not "
              f"queried.")
        print("  Re-run without --dry-run, with FRED_API_KEY set.")
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.execute(DDL)
    now = date.today().isoformat()
    con.executemany(
        "INSERT OR REPLACE INTO macro_events "
        "(event_date, event_code, source, created_at) VALUES (?,?,?,?)",
        [(d, c, s, now) for d, c, s in rows])
    con.commit()
    tot = con.execute(
        "SELECT event_code, COUNT(*), MIN(event_date), MAX(event_date) "
        "FROM macro_events GROUP BY event_code ORDER BY 2 DESC").fetchall()
    n = con.execute("SELECT COUNT(*) FROM macro_events").fetchone()[0]
    con.close()

    print(f"\n  macro_events: {n} rows")
    print(f"  {'code':<8}{'n':>6}  {'first':<12}{'last'}")
    for c, k, lo, hi in tot:
        print(f"  {c:<8}{k:>6}  {lo:<12}{hi}")
    print("\n  The DATE is knowable in advance -- BLS and BEA publish schedules")
    print("  a year ahead, the Fed publishes FOMC dates the summer before. The")
    print("  OUTCOME is not. This table stores dates only; any surprise feature")
    print("  is knowable only after release and must be lagged, the same rule")
    print("  that governs eps_surprise.")
    print("\n  EXTEND THE FOMC LIST each year -- it is hardcoded and will go")
    print("  stale silently, which is the failure shape this codebase keeps")
    print("  finding.")


if __name__ == "__main__":
    main()
