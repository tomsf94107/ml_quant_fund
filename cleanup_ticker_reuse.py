#!/usr/bin/env python3
"""
cleanup_ticker_reuse.py — remove outcomes computed against the wrong company.

THE ROWS
    32 outcomes carry returns of +420% to +4862% because the ticker referred to
    a different instrument on the prediction date:

      AI    9 rows  2020-10-19..23  C3.ai listed 2020-12-09
      META  9 rows  2022-01-24..28  "META" was Meta Materials until mid-2022
      S     9 rows  2020-03-25..31  "S" was Sprint; SentinelOne listed 2021
      FIG   5 rows  2025-05-19..23  Figma IPO'd 2025-07-31

    These are not repairable. There is no correct price for "AI" in October 2020
    because the instrument did not exist; the stored return was computed against
    C3.ai's later prices. Unlike the BYND split, nothing can be recomputed.

WHAT IS DELIBERATELY KEPT
    GME +788% (Jan-2021 squeeze), AMC +570%, BNED +495%, SMMT +431%, QURE +330%,
    QUBT +301%. These are REAL moves. Deleting them would bias the outcome record
    against exactly the events a short-interest strategy targets -- and the
    fund's one validated brick is a short-interest signal.

    BNED is likely a 1-for-10 reverse split; the patched writer will adjust it
    correctly if it recomputes, so it is left for that rather than deleted here.

SAFETY
    Dry-run by default. Deletes only the four named ticker/date-range pairs --
    never a blanket threshold, because a threshold is exactly what would have
    taken GME with it.
"""
import argparse
import sqlite3

# (ticker, first_bad_date, last_bad_date, why)
BAD = [
    ("AI",   "2020-01-01", "2020-12-08", "C3.ai listed 2020-12-09"),
    ("META", "2022-01-01", "2022-06-08", "'META' was Meta Materials until mid-2022"),
    ("S",    "2020-01-01", "2021-06-29", "'S' was Sprint; SentinelOne listed 2021-06-30"),
    ("FIG",  "2025-01-01", "2025-07-30", "Figma IPO'd 2025-07-31"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)

    total = 0
    print("rows to remove (ticker reuse -- not repairable):\n")
    for tkr, lo, hi, why in BAD:
        rows = con.execute(
            "SELECT COUNT(*), MIN(prediction_date), MAX(prediction_date), "
            "ROUND(MAX(ABS(actual_return))*100,0) FROM outcomes "
            "WHERE ticker=? AND prediction_date BETWEEN ? AND ?",
            (tkr, lo, hi)).fetchone()
        n = rows[0] or 0
        total += n
        print(f"  {tkr:<6} {n:>3} rows  {rows[1]} .. {rows[2]}  "
              f"max {rows[3] or 0:.0f}%   {why}")

    print("\nKEPT (real moves, deliberately not deleted):")
    for r in con.execute(
            "SELECT ticker, COUNT(*), ROUND(MAX(ABS(actual_return))*100,0) "
            "FROM outcomes WHERE ABS(actual_return) > 3.0 "
            "AND ticker NOT IN ('AI','META','S','FIG') GROUP BY ticker "
            "ORDER BY 3 DESC"):
        print(f"  {r[0]:<6} {r[1]:>3} rows  max {r[2]:.0f}%")

    if not args.apply:
        print(f"\nDRY RUN -- {total} rows would be deleted. Re-run with --apply.")
        return

    deleted = 0
    for tkr, lo, hi, _ in BAD:
        cur = con.execute(
            "DELETE FROM outcomes WHERE ticker=? AND prediction_date BETWEEN ? AND ?",
            (tkr, lo, hi))
        deleted += cur.rowcount
    con.commit()
    left = con.execute("SELECT COUNT(*) FROM outcomes "
                       "WHERE ABS(actual_return) > 3.0").fetchone()[0]
    print(f"\ndeleted {deleted} rows")
    print(f"rows still above 300%: {left}  (expected: the real squeezes)")
    con.close()


if __name__ == "__main__":
    main()
