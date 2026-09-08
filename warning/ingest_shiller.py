#!/usr/bin/env python3
"""
ingest_shiller.py -- Shiller ie_data.xls -> data_vintages as SHILLER_CAPE.

WHY A SEPARATE INGEST RATHER THAN READING THE XLS IN THE BUILDER
    Every other builder reads data_vintages through pit.series_asof, which
    filters pub_date <= asof. A builder that opened the spreadsheet directly
    would see all 1,878 months at every evaluation date and bypass PIT
    entirely -- the class of leak that voided PEAD at IC +0.2612. The xls is a
    source file; data_vintages is the point-in-time record.

WHICH COLUMN
    Column 12 (header "P/E10 or CAPE"), not column 14 ("TR P/E10 or TR CAPE").
    Shiller added the total-return variant in 2018 to correct for the shift
    from dividends to buybacks. The registry pins the standard series: its
    historical_verdict_2000 is "44.19 Dec-99 (top of history)", and 44.19 is
    the standard reading. TR CAPE runs materially higher -- 43.81 against
    41.12 for 2026-08 -- so the choice changes every percentile.

THE DATE COLUMN IS NOT A NUMBER
    Shiller writes dates as 1871.01 .. 2026.09, where .1 means OCTOBER, not
    January: 2026.1 is 2026-10 and 2026.01 is 2026-01. Excel drops the
    trailing zero, so a naive float parse turns October into January and is
    wrong for three months of every year. Parsed as text, splitting on the
    decimal point.

THE LAST MONTH IS INCOMPLETE
    Shiller's price is a monthly average of DAILY CLOSES, so the newest row is
    a partial month -- 2026.09 held about two sessions when this was written.
    Comparing a two-day average against 154 years of full months is not a
    valuation reading. The trailing month is dropped unless --include-partial.
    Same discipline as DECISIONS.md D3: months must be fully published before
    they count.

PUB_DATE
    month_end + PUB_LAG_DAYS. The registry gives S13 a publication lag of
    about one month, which is what the partial-month rule above implements on
    the source side; the lag here is the additional delay before Shiller posts
    an updated file. Deriving it reproduces what an ALFRED-style vintage would
    return rather than stamping the pull date, which would make every
    historical point-in-time read return NA.

THE SOURCE URL ROTS
    Yale's copy at econ.yale.edu/~shiller/data/ie_data.xls is FROZEN at
    2023-08 -- it downloads cleanly, parses cleanly, and is three years stale.
    The live file is a GoDaddy CDN blob linked from shillerdata.com with a
    ?ver= query parameter that will change. --max-age-days fails loudly rather
    than silently ingesting a dead mirror, because a stale file that parses is
    more dangerous than one that does not.

USAGE
    python warning/ingest_shiller.py --xls data/raw/shiller/ie_data.xls --dry-run
    python warning/ingest_shiller.py --xls data/raw/shiller/ie_data.xls
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, timedelta

SERIES_ID = "SHILLER_CAPE"
CAPE_COL = 12
DATE_COL = 0
HEADER_ROW = 7            # data begins at row 8 (1871.01)
PUB_LAG_DAYS = 5
DEFAULT_MAX_AGE_DAYS = 120


def month_end(y: int, m: int) -> date:
    return (date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1))


def parse_shiller_date(raw) -> tuple[int, int] | None:
    """1871.01 -> (1871, 1); 2026.1 -> (2026, 10). Text, never float."""
    s = str(raw).strip()
    if "." not in s:
        return None
    y, _, frac = s.partition(".")
    if not y.isdigit() or not frac.isdigit():
        return None
    # Excel drops the trailing zero: ".1" is October, ".10" is also October,
    # ".01" is January. A single digit is therefore the TENS place.
    month = int(frac) if len(frac) >= 2 else int(frac) * 10
    if not 1 <= month <= 12:
        return None
    return int(y), month


def load_rows(xls_path: str):
    import pandas as pd
    df = pd.ExcelFile(xls_path).parse("Data", header=None)
    out = []
    for _, row in df.iloc[HEADER_ROW + 1:].iterrows():
        ym = parse_shiller_date(row[DATE_COL])
        if ym is None:
            continue
        v = row[CAPE_COL]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v:                      # NaN: CAPE is undefined before ~1881
            continue
        out.append((ym[0], ym[1], v))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", default="data/raw/shiller/ie_data.xls")
    ap.add_argument("--db", default=os.environ.get("WARNING_DB",
                                               "warning.db"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-partial", action="store_true",
                    help="keep the trailing incomplete month (default: drop)")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help="refuse if the newest observation is older than this")
    args = ap.parse_args()

    if not os.path.exists(args.xls):
        sys.exit(f"{args.xls} not found. Download from shillerdata.com -- the "
                 f"econ.yale.edu copy is frozen at 2023-08.")

    rows = load_rows(args.xls)
    if not rows:
        sys.exit("no CAPE observations parsed; check the sheet layout.")

    dropped = None
    if not args.include_partial:
        dropped = rows[-1]
        rows = rows[:-1]

    newest = month_end(rows[-1][0], rows[-1][1])
    age = (date.today() - newest).days
    if age > args.max_age_days:
        sys.exit(f"newest complete observation is {newest} ({age} days old, "
                 f"limit {args.max_age_days}). This is what a frozen mirror "
                 f"looks like -- re-download from shillerdata.com. Refusing "
                 f"rather than ingesting stale data that parses cleanly.")

    print(f"{len(rows)} observations  {rows[0][0]}-{rows[0][1]:02d} .. "
          f"{rows[-1][0]}-{rows[-1][1]:02d}  (newest {age}d old)")
    if dropped:
        print(f"dropped trailing partial month {dropped[0]}-{dropped[1]:02d} "
              f"(CAPE {dropped[2]:.2f}) -- monthly average of daily closes, "
              f"incomplete. --include-partial to keep it.")
    print(f"latest complete CAPE: {rows[-1][2]:.2f}")

    if args.dry_run:
        print(f"\nDRY RUN. {len(rows)} rows would be written as {SERIES_ID}.")
        return

    con = sqlite3.connect(args.db)
    n = 0
    for y, m, v in rows:
        obs = month_end(y, m)
        pub = obs + timedelta(days=PUB_LAG_DAYS)
        con.execute("INSERT OR REPLACE INTO data_vintages "
                    "(series_id, obs_date, pub_date, value, source) "
                    "VALUES (?,?,?,?,?)",
                    (SERIES_ID, obs.isoformat(), pub.isoformat(), v, "Shiller"))
        n += 1
    con.commit()
    print(f"\nwrote {n} {SERIES_ID} rows (pub = month_end + {PUB_LAG_DAYS}d).")


if __name__ == "__main__":
    main()
