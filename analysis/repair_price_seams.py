#!/usr/bin/env python3
"""
repair_price_seams.py — halve (or rescale) prices on the wrong side of a split.

DRY RUN BY DEFAULT. Backs up the database before any write.

THE DEFECT (2026-09-03)
    APH executed a 1-for-2 forward split on 2026-09-03. The provider adjusted
    2026-08-31 onward and left everything from 2026-08-28 back on the old
    basis:

        2026-09-02    80.04   new basis
        2026-08-31    79.275  new basis
        2026-08-28   157.74   OLD basis   <- the seam
        2026-08-27   161.38   OLD basis

    si_positions_live.py marks the SI book against daily_prices.adj_close, so a
    position entered before the seam and marked after it reads as a ~48.6% loss.
    APH was reported as the book's worst position and dragged the total to
    -12.14%. Nothing was actually lost: after a 1-for-2 split you hold twice the
    shares at half the price.

WHAT THE REPAIR DOES
    Multiplies every price BEFORE the seam date by the split factor, so the
    series is continuous across it. For a 1-for-2 forward split the factor is
    0.5 and pre-seam prices halve; for a reverse split it is greater than 1 and
    they rise.

    This is a REPORTING repair. No money moved. The position was never down 48%
    -- only the number was.

SAFETY
    - dry run unless --apply
    - full file copy of the database before writing
    - only rows for the named ticker before the named date are touched
    - the post-repair series is re-checked for a remaining seam, and the run
      reports it rather than declaring success blindly

WHY NOT REPAIR EVERYTHING AUTOMATICALLY
    A seam and a real 50% move look similar in isolation. The detector
    classifies, a human confirms, and only then is a specific ticker repaired by
    name. Sweeping every candidate would eventually halve a genuine crash.

    python analysis/repair_price_seams.py --ticker APH --before 2026-08-31 --factor 0.5
    python analysis/repair_price_seams.py --ticker APH --before 2026-08-31 --factor 0.5 --apply
"""
import argparse
import os
import shutil
import sqlite3
from datetime import datetime


def show(con, ticker, around, n=4):
    """Rows either side of the seam, split EXACTLY as the UPDATE splits them.

    The UPDATE rescales `date < around`, so the preview must use `<` too. An
    earlier version used `<=`, which listed the seam date itself as rescaled and
    then computed a bogus post-repair ratio of 0.486 -- a false alarm that would
    have talked the operator out of a correct repair.
    """
    rows = con.execute(
        "SELECT date, adj_close FROM daily_prices WHERE ticker=? "
        "AND date < ? ORDER BY date DESC LIMIT ?", (ticker, around, n)).fetchall()
    after = con.execute(
        "SELECT date, adj_close FROM daily_prices WHERE ticker=? "
        "AND date >= ? ORDER BY date LIMIT ?", (ticker, around, n)).fetchall()
    return list(reversed(rows)), after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prices.db")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--before", required=True,
                    help="rows with date < this are rescaled (the seam date)")
    ap.add_argument("--factor", type=float, required=True,
                    help="split_from/split_to. 1-for-2 forward = 0.5; "
                         "1-for-30 reverse = 30")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    before, after = show(con, args.ticker, args.before)
    if not before or not after:
        raise SystemExit(f"not enough {args.ticker} rows around {args.before}")

    print(f"{args.ticker} around the seam at {args.before}, factor {args.factor:g}\n")
    print(f"  {'date':<12}{'current':>10}{'after repair':>14}")
    for d, p in before:
        print(f"  {d:<12}{p:>10.4f}{p * args.factor:>14.4f}   <- rescaled")
    for d, p in after:
        print(f"  {d:<12}{p:>10.4f}{p:>14.4f}")

    gap_old = before[-1][1] / after[0][1] if after[0][1] else 0
    gap_new = (before[-1][1] * args.factor) / after[0][1] if after[0][1] else 0
    print(f"\n  ratio across the seam: {gap_old:.3f} now -> {gap_new:.3f} after")
    if abs(gap_new - 1.0) > 0.25:
        print(f"  !! {gap_new:.3f} is still far from 1.0 -- the factor or the "
              f"seam date may be wrong. Check before applying.")

    n_rows = con.execute(
        "SELECT COUNT(*) FROM daily_prices WHERE ticker=? AND date < ?",
        (args.ticker, args.before)).fetchone()[0]
    print(f"\n  {n_rows} rows would be rescaled "
          f"({args.ticker}, date < {args.before})")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        con.close()
        return

    bak = f"{args.db}.bak.{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(args.db, bak)
    print(f"\nbackup {bak} ({os.path.getsize(bak)/1e6:.0f} MB)")

    con.execute("UPDATE daily_prices SET adj_close = adj_close * ? "
                "WHERE ticker=? AND date < ?",
                (args.factor, args.ticker, args.before))
    con.commit()

    before2, after2 = show(con, args.ticker, args.before)
    gap_final = before2[-1][1] / after2[0][1] if after2[0][1] else 0
    print(f"repaired. ratio across the seam is now {gap_final:.3f}")
    if abs(gap_final - 1.0) > 0.25:
        print(f"!! still {gap_final:.3f} -- restore from {bak} and re-check "
              f"the factor.")
    else:
        print("series is continuous across the seam.")
    con.close()
    print("\nRe-run the SI cycle to see the corrected book:\n"
          "  python si_positions_live.py")


if __name__ == "__main__":
    main()
