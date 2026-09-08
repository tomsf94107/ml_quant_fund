#!/usr/bin/env python3
"""
ingest_finra_margin.py -- FINRA margin statistics -> data_vintages.

SERIES
    FINRA_MARGIN_DEBIT   debit balances in customers' securities margin
                         accounts, $ millions, monthly, 1997-01 onward.
    That is the only column S10 needs. The two free-credit columns are
    ingested as well because they are free, but nothing reads them yet.

PUB_DATE IS THE WHOLE POINT OF THIS SIGNAL
    FINRA's own page: "FINRA generally publishes updates to the Margin
    Statistics on the third week of the month following the reference month."
    The registry gives S10 a publication lag of ~3 weeks and its formula says
    the trigger is dated "as of publication date".

    That lag is not an inconvenience to be minimised -- it is why the signal is
    honest. The registry's own verdicts depend on it: the March 2000 peak was
    not knowable until late April, so the trigger fires ~Jun-Jul 2000 with 96%
    of the decline still ahead. Stamping obs_date instead would fabricate a
    signal that fired months before anyone could have seen the data.

    Implemented as the 21st of the following month. FINRA says "third week"
    rather than a fixed day, so this is a within-week approximation of a
    within-week statement, not a precision claim.

DEFINITION BREAK AT FEBRUARY 2010
    Before Feb-2010 NYSE and FINRA collected margin data separately from their
    own members and combined free credit balances in cash and margin accounts
    into ONE figure -- which is why column 3 is empty for those rows. Debit
    balances are continuous across the break and are what S10 reads. The free
    credit columns are ingested as they stand, with the break unpatched: any
    future signal using them must handle it rather than inherit a silent
    splice.

SOURCE URL
    https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx
    Reachable from Vietnam without a VPN as of 2026-09-08, contrary to a note
    in the crontab. Stable path with no version parameter, unlike the Shiller
    blob. --max-age-days still fails loudly: the Shiller lesson is that a
    frozen mirror downloads and parses without complaint.

USAGE
    python warning/ingest_finra_margin.py --dry-run
    python warning/ingest_finra_margin.py
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, timedelta

DEBIT_SERIES = "FINRA_MARGIN_DEBIT"
CREDIT_CASH_SERIES = "FINRA_FREE_CREDIT_CASH"
CREDIT_MARGIN_SERIES = "FINRA_FREE_CREDIT_MARGIN"
COLS = {1: DEBIT_SERIES, 2: CREDIT_CASH_SERIES, 3: CREDIT_MARGIN_SERIES}

PUB_DAY = 21                      # "third week of the following month"
DEFAULT_MAX_AGE_DAYS = 75         # monthly + ~3wk lag; 75 catches a skipped release


def month_end(y: int, m: int) -> date:
    return date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1)


def pub_date_for(y: int, m: int) -> date:
    return date(y + (m == 12), (m % 12) + 1, PUB_DAY)


def load_rows(path: str):
    import pandas as pd
    df = pd.ExcelFile(path).parse("Customer Margin Balances", header=None)
    out = []
    for _, row in df.iloc[1:].iterrows():
        ym = str(row[0]).strip()[:7]
        if len(ym) != 7 or ym[4] != "-":
            continue
        try:
            y, m = int(ym[:4]), int(ym[5:7])
        except ValueError:
            continue
        if not 1 <= m <= 12:
            continue
        vals = {}
        for col, sid in COLS.items():
            try:
                v = float(row[col])
            except (TypeError, ValueError, KeyError):
                continue
            if v == v:                       # skip NaN (pre-2010 free credit)
                vals[sid] = v
        if DEBIT_SERIES in vals:
            out.append((y, m, vals))
    out.sort(key=lambda r: (r[0], r[1]))     # file is descending; store ascending
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/raw/finra/margin-statistics.xlsx")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    args = ap.parse_args()

    if not os.path.exists(args.xlsx):
        sys.exit(f"{args.xlsx} not found. Download from "
                 f"https://www.finra.org/sites/default/files/2021-03/"
                 f"margin-statistics.xlsx")

    rows = load_rows(args.xlsx)
    if not rows:
        sys.exit("no observations parsed; check the sheet layout.")

    y, m, _ = rows[-1]
    newest = month_end(y, m)
    age = (date.today() - newest).days
    if age > args.max_age_days:
        sys.exit(f"newest observation is {newest} ({age} days old, limit "
                 f"{args.max_age_days}). Either a release was skipped or the "
                 f"file is stale. Refusing rather than ingesting silently.")

    print(f"{len(rows)} months  {rows[0][0]}-{rows[0][1]:02d} .. "
          f"{y}-{m:02d}  (newest {age}d old, publishes {pub_date_for(y, m)})")
    last = rows[-1][2][DEBIT_SERIES]
    prior = next((r[2][DEBIT_SERIES] for r in reversed(rows[:-1])
                  if r[0] == y - 1 and r[1] == m), None)
    if prior:
        print(f"debit balances: {last:,.0f}M, YoY {100*(last/prior - 1):+.1f}%")

    if args.dry_run:
        n = sum(len(r[2]) for r in rows)
        print(f"\nDRY RUN. {n} rows across {len(COLS)} series would be written.")
        return

    con = sqlite3.connect(args.db)
    n = 0
    for y_, m_, vals in rows:
        obs = month_end(y_, m_).isoformat()
        pub = pub_date_for(y_, m_).isoformat()
        for sid, v in vals.items():
            con.execute("INSERT OR REPLACE INTO data_vintages "
                        "(series_id, obs_date, pub_date, value, source) "
                        "VALUES (?,?,?,?,?)", (sid, obs, pub, v, "FINRA"))
            n += 1
    con.commit()
    print(f"\nwrote {n} rows (pub = the 21st of the following month).")


if __name__ == "__main__":
    main()
