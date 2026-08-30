#!/usr/bin/env python3
"""
cleanup_zero_predictions.py — deal with rows where a failed generator wrote 0.0.

THE ROWS
    60 rows (CBRS 33 from 2026-08-14, SPCX 27 from 2026-08-17) carry
    prob_up = prob_raw = 0.0 with every multiplier NULL and overlay_reason NULL.
    That is the signature of signals/generator.py's error path: the generator
    raised "No saved model for <ticker> horizon=<h>d", returned a placeholder
    0.0, and daily_runner logged it without the error string.

    0.0 is a VALID probability. Any accuracy query treats these as
    maximum-confidence DOWN calls, and they will be wrong roughly half the time
    by construction -- dragging down exactly the metric being investigated.

TWO MODES

  mark (default)
    Sets overlay_reason='NO_MODEL_ARTIFACT' and gate_block=1, leaving the rows
    in place. Preserves the audit trail that these tickers WERE evaluated, and
    makes them filterable. Cost: a naive query that does not filter still counts
    them, so every accuracy query must add
        WHERE overlay_reason IS NULL
    or the poisoning persists in a quieter form.

  delete
    Removes them. Any query is then correct by default. Cost: the record that
    the system tried and failed on these dates is gone -- though the failure is
    still evident from the absence of rows on days their peers have them.

    NOTE: prob_up is declared NOT NULL, so setting it to NULL is not an option;
    those are the only two choices the schema permits.

RECOMMENDATION
    `mark` if you will maintain the filter discipline, `delete` if you will not.
    Marking a row and then forgetting the filter is worse than either, because
    the data looks clean and is not.

SAFETY
    Dry-run by default. Only touches rows matching the exact error signature:
    prob_up = 0.0 AND prob_raw = 0.0 AND risk_mult IS NULL AND
    overlay_reason IS NULL. A genuine 0.0 from a working model would have
    multipliers populated and is not matched.

USAGE
    python cleanup_zero_predictions.py
    python cleanup_zero_predictions.py --mode mark   --apply
    python cleanup_zero_predictions.py --mode delete --apply
"""
import argparse
import sqlite3

SIGNATURE = ("prob_up = 0.0 AND prob_raw = 0.0 "
             "AND risk_mult IS NULL AND overlay_reason IS NULL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--mode", choices=["mark", "delete"], default="mark")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)

    rows = con.execute(
        f"SELECT ticker, COUNT(*), MIN(prediction_date), MAX(prediction_date) "
        f"FROM predictions WHERE {SIGNATURE} GROUP BY ticker").fetchall()
    total = sum(r[1] for r in rows)
    print(f"rows matching the error signature: {total}\n")
    for t, n, lo, hi in rows:
        print(f"  {t:<8} {n:>4} rows   {lo} .. {hi}")

    guard = con.execute(
        "SELECT COUNT(*) FROM predictions "
        "WHERE prob_up = 0.0 AND risk_mult IS NOT NULL").fetchone()[0]
    print(f"\nsanity: rows with prob_up=0.0 but multipliers PRESENT "
          f"(genuine, not matched): {guard}")

    if not total:
        print("\nnothing to do.")
        return

    if not args.apply:
        print(f"\nDRY RUN, mode={args.mode} -- nothing written. "
              f"Re-run with --apply.")
        return

    if args.mode == "mark":
        cur = con.execute(
            f"UPDATE predictions SET overlay_reason='NO_MODEL_ARTIFACT', "
            f"gate_block=1 WHERE {SIGNATURE}")
        con.commit()
        print(f"\nmarked {cur.rowcount} rows.")
        print("EVERY accuracy query must now filter:  WHERE overlay_reason IS NULL")
    else:
        cur = con.execute(f"DELETE FROM predictions WHERE {SIGNATURE}")
        con.commit()
        print(f"\ndeleted {cur.rowcount} rows.")

    left = con.execute(
        f"SELECT COUNT(*) FROM predictions WHERE {SIGNATURE}").fetchone()[0]
    print(f"rows still matching the signature: {left}")
    con.close()


if __name__ == "__main__":
    main()
