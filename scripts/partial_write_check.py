#!/usr/bin/env python3
"""PARTIAL-WRITE CHECK -- is the latest write COMPLETE, not just RECENT?

freshness_check answers "is it recent?"   This answers "is it whole?"

v2 (2026-07-13):
  - outcomes is checked PER HORIZON. It matures over 5 sessions, so the newest
    prediction_date can only hold h=1. Comparing that against a 3-horizon
    median produced a permanent 33% FALSE POSITIVE, every day, forever.
  - prediction_features date column: 'date' -> 'prediction_date'. It has no
    'date' column, so it errored every run and was effectively UNWATCHED.
  - A split_col feed is checked independently per group, which also catches
    "h=3 reconcile broke but h=1 is fine".
"""
import sqlite3
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FEEDS = [
    ("raw_bars",              "prices.db",   "raw_bars",                    "d",               None),
    ("daily_prices",          "prices.db",   "daily_prices",                "date",            None),
    ("predictions",           "accuracy.db", "predictions",                 "prediction_date", None),
    ("outcomes",              "accuracy.db", "outcomes",                    "prediction_date", "horizon"),
    ("prediction_features",   "accuracy.db", "prediction_features",         "prediction_date", None),
    ("momentum_shadow",       "accuracy.db", "momentum_shadow_predictions", "prediction_date", None),
    ("options_greeks",        "accuracy.db", "options_greeks",              "date",            None),
    ("dark_pool_history",     "accuracy.db", "dark_pool_history",           "date",            None),
    ("institutional_history", "accuracy.db", "institutional_history",       "date",            None),
    ("vix_history",           "accuracy.db", "vix_history",                 "date",            None),
]

THRESHOLD = 0.50
THIN = 0.80


def date_counts(con, tbl, dcol, where_sql="", params=()):
    sql = (f"SELECT {dcol} AS d, COUNT(*) AS n FROM {tbl} {where_sql} "
           f"GROUP BY {dcol} ORDER BY {dcol} DESC LIMIT 11")
    return con.execute(sql, params).fetchall()


def evaluate(label, rows, bad):
    if len(rows) < 4:
        print(f"  {label:<26}{'--':<12}{'too few dates':>25}")
        return
    latest_d, latest_n = rows[0]
    prior = [n for _, n in rows[1:]]
    med = st.median(prior)
    ratio = latest_n / med if med else 0
    if ratio < THRESHOLD:
        status = "PARTIAL WRITE  <<<"
        bad.append((label, latest_n, int(med)))
    elif ratio < THIN:
        status = "thin"
    else:
        status = "OK"
    print(f"  {label:<26}{str(latest_d)[:10]:<12}{latest_n:>8,}{int(med):>9,}"
          f"{ratio:>7.0%}   {status}")


def main():
    bad = []
    print("=" * 84)
    print("  PARTIAL-WRITE CHECK -- is the latest write COMPLETE, not just RECENT?")
    print("=" * 84)
    print(f"  {'feed':<26}{'latest':<12}{'rows':>8}{'median':>9}{'ratio':>8}   status")
    print("  " + "-" * 78)

    for label, db, tbl, dcol, split in FEEDS:
        try:
            con = sqlite3.connect(f"file:{ROOT/db}?mode=ro", uri=True, timeout=20)
        except Exception as e:
            print(f"  {label:<26}ERROR {str(e)[:44]}")
            continue
        try:
            if split is None:
                evaluate(label, date_counts(con, tbl, dcol), bad)
            else:
                vals = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {split} FROM {tbl} ORDER BY {split}").fetchall()]
                for v in vals:
                    rows = date_counts(con, tbl, dcol, f"WHERE {split} = ?", (v,))
                    evaluate(f"{label}[{split}={v}]", rows, bad)
        except Exception as e:
            print(f"  {label:<26}ERROR {str(e)[:44]}")
        finally:
            con.close()

    print()
    print("=" * 84)
    if bad:
        msg = ", ".join(f"{n} ({a} vs {m} normal)" for n, a, m in bad)
        print(f"  {len(bad)} PARTIAL WRITE(S): {msg}")
        print()
        print("  The feed is RECENT but INCOMPLETE. feed_freshness_check would report OK.")
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{len(bad)} feed(s) wrote partial data" '
             f'with title "ML Quant -- PARTIAL WRITE"'],
            check=False, capture_output=True)
        sys.exit(1)

    print("  All feeds wrote a COMPLETE latest batch.")
    print()
    print("  Freshness    = is it recent?  (feed_freshness_check.py)")
    print("  Completeness = is it whole?   (this script)")
    print("  outcomes matures over 5 sessions -> checked PER HORIZON.")
    print("  A whole-table check on it fires a permanent false 33%.")


if __name__ == "__main__":
    main()
