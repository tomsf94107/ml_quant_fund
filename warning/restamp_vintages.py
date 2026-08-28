#!/usr/bin/env python3
"""
restamp_vintages.py — correct pub_date for NON-REVISABLE series.

THE PROBLEM
    fetch_free_history.py stamped pub_date = pull date on every FRED row. For a
    2000 observation pulled in 2026 that is 26 years wrong, so every historical
    point-in-time read returns NA and no macro signal can be validated against
    the registry's documented 2000/2008 verdicts.

THE CORRECTION
    For series that are NEVER REVISED (see series_meta.py), the value published
    at obs_date + publication_lag is final, so pub_date can be derived exactly.
    This UPDATES the pub_date stamp in place. It does NOT change any value, does
    NOT delete rows, and does NOT re-download anything. `pulled_at` still records
    when we actually fetched, so provenance is preserved.

    REVISABLE series are refused. Their true vintages exist only in ALFRED; a
    derived stamp there would invent a first print that never existed.

SAFETY
    - Dry-run by default. --apply is required to write.
    - Refuses any series not explicitly declared non-revisable.
    - Skips rows whose pub_date is already earlier than the derived date (never
      moves a stamp later, which could hide data that was genuinely available).
    - Reports a before/after row count; totals must match (this is UPDATE, not
      INSERT -- no row is created or destroyed).

USAGE
    python warning/restamp_vintages.py --db warning.db
    python warning/restamp_vintages.py --db warning.db --apply
"""
import argparse, os, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from series_meta import SERIES_META, derivable_pub_date, pub_lag_days, note  # noqa: E402


def plan(con):
    rows = []
    for series in sorted(SERIES_META):
        cur = con.execute(
            "SELECT COUNT(*), MIN(obs_date), MAX(obs_date), MIN(pub_date), MAX(pub_date) "
            "FROM data_vintages WHERE series_id=?", (series,)).fetchone()
        n = cur[0]
        if not n:
            continue
        if not derivable_pub_date(series):
            rows.append((series, n, "REFUSED (revisable -- ALFRED required)", cur))
            continue
        n_change = con.execute(
            "SELECT COUNT(*) FROM data_vintages "
            "WHERE series_id=? AND pub_date > date(obs_date, ?)",
            (series, f"+{pub_lag_days(series)} day")).fetchone()[0]
        rows.append((series, n, f"restamp {n_change}", cur))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--apply", action="store_true", help="write (default is dry-run)")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)

    before_total = con.execute("SELECT COUNT(*) FROM data_vintages").fetchone()[0]
    print(f"data_vintages rows before: {before_total}\n")

    print(f"{'series':<14}{'rows':>8}  {'action':<40} obs range")
    print("-" * 100)
    for series, n, action, cur in plan(con):
        print(f"{series:<14}{n:>8}  {action:<40} {cur[1]}..{cur[2]}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to commit.")
        return

    print("\napplying...")
    changed = 0
    for series in sorted(SERIES_META):
        if not derivable_pub_date(series):
            continue
        lag = f"+{pub_lag_days(series)} day"
        cur = con.execute(
            "UPDATE data_vintages SET pub_date = date(obs_date, ?) "
            "WHERE series_id = ? AND pub_date > date(obs_date, ?)",
            (lag, series, lag))
        if cur.rowcount:
            print(f"  {series:<14} restamped {cur.rowcount:>7}   ({note(series)})")
            changed += cur.rowcount
    con.commit()

    after_total = con.execute("SELECT COUNT(*) FROM data_vintages").fetchone()[0]
    print(f"\nrestamped {changed} rows")
    print(f"data_vintages rows after:  {after_total}")
    if after_total != before_total:
        print("!! ROW COUNT CHANGED -- this should be impossible for an UPDATE. Investigate.")
        sys.exit(1)
    print("row count unchanged, as expected for an in-place UPDATE.")


if __name__ == "__main__":
    main()
