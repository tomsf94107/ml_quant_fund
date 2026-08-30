#!/usr/bin/env python3
"""
ingest_short_interest.py — per-ticker short interest into warning.db.

WHY PER-TICKER AND NOT A PRE-AGGREGATED TOTAL
    S9 needs an aggregate, but the aggregate must be built from a POINT-IN-TIME
    panel. Selecting "tickers present on every settlement date" uses today's
    knowledge of which names survived -- look-ahead. The builder therefore needs
    the per-ticker rows so it can choose the panel from what was visible at each
    evaluation date.

    Why that matters, measured on the real data 2026-08-30:
        naive sum over all names   4.42bn -> 9.88bn   (+124%)
        fixed 362-name panel       4.40bn -> 7.88bn   (+79%)
    Roughly 45 percentage points of the apparent surge was coverage expansion,
    not shorting. An aggregate over a drifting universe measures the ingest
    history.

PUBLICATION LAG IS APPLIED HERE
    FINRA publishes short interest roughly 8 BUSINESS days after the settlement
    date (registry S9: publication_lag "~8 bus days"). obs_date is the settlement
    date; pub_date is settlement + 8 business days, so a point-in-time read
    cannot see a settlement before it was published. Without this the signal
    would act on positioning data a week and a half early.

SERIES NAMING
    "SI:<TICKER>". The colon keeps them out of any LIKE 'X_%' pattern -- SQL
    treats '_' as a single-character wildcard, which already caused a
    miscounted total once in this project (parse_cfe, 2026-08-30).

USAGE
    python warning/ingest_short_interest.py --src short_interest.db --db warning.db
    python warning/ingest_short_interest.py --src short_interest.db --db warning.db --dry-run
"""
import argparse
import sqlite3
from datetime import date, timedelta

PUB_LAG_BDAYS = 8


def plus_business_days(iso: str, n: int) -> str:
    d = date.fromisoformat(iso)
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="short_interest.db")
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = sqlite3.connect(f"file:{args.src}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT ticker, settlement_date, current_short FROM short_interest "
        "WHERE current_short IS NOT NULL ORDER BY settlement_date, ticker"
    ).fetchall()
    src.close()

    dates = sorted({r[1] for r in rows})
    tickers = sorted({r[0] for r in rows})
    print(f"source: {len(rows)} rows, {len(tickers)} tickers, "
          f"{len(dates)} settlement dates {dates[0]}..{dates[-1]}")
    print(f"publication lag applied: settlement + {PUB_LAG_BDAYS} business days")
    print(f"  e.g. {dates[-1]} -> published "
          f"{plus_business_days(dates[-1], PUB_LAG_BDAYS)}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    con = sqlite3.connect(args.db)
    before = con.execute("SELECT COUNT(*) FROM data_vintages "
                         "WHERE series_id LIKE 'SI:%'").fetchone()[0]
    pub_cache = {}
    n = 0
    for tkr, settle, short in rows:
        if settle not in pub_cache:
            pub_cache[settle] = plus_business_days(settle, PUB_LAG_BDAYS)
        con.execute("INSERT OR IGNORE INTO data_vintages "
                    "(series_id, obs_date, pub_date, value, source) "
                    "VALUES (?,?,?,?,?)",
                    (f"SI:{tkr}", settle, pub_cache[settle], float(short),
                     "short_interest.db"))
        n += 1
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM data_vintages "
                        "WHERE series_id LIKE 'SI:%'").fetchone()[0]
    con.close()
    print(f"\nwrote {n} rows; data_vintages SI:* {before} -> {after}")
    print("Re-runs are idempotent (INSERT OR IGNORE on the full key).")


if __name__ == "__main__":
    main()
