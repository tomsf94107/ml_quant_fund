#!/usr/bin/env python3
"""
recovery_tracker.py — did the 2026-09-14 macro fix change anything?

READ-ONLY. Reads logged predictions and outcomes. Writes nothing.

WHAT IS BEING TRACKED
    On 2026-09-14 three macro inputs were repaired. All three had been feeding
    the model a constant where a live value belonged:

        vix_ret    0.0 in 100% of stored rows Jul-Sep, 56% of Jun, 3% of May
        dxy_ret    0.0 in 98% of May rows, 97% of Jun, 100% from Jul
        regime VIX pinned at the dataclass default 20, so every logged regime
                   read NEUTRAL / confidence 0.5 / signal_multiplier 1.0

    Root cause was one thing: index symbols route to yfinance, XProtect 5347
    SIGKILLs the process on exec, download() returns an empty frame, and every
    consumer silently takes its fallback. Fixed in features/massive_client.py
    by routing nine index symbols to FRED.

    Over those same months h=5 prob>=0.60 hit rate fell 61.9% -> 37.4% and mean
    return +2.68% -> -0.92%. The timing fits. That is ALL it is: correlation on
    one sequence of events, with no counterfactual. This script exists to find
    out, not to confirm.

WHY THE BENCHMARK COLUMN IS THE POINT
    docs/MASTER_TODO_LIST.md section 1.1b is the cautionary tale:

        "5d: BUY 57.8% vs HOLD 59.8% = -2.0pp (NEGATIVE) ... The 58% is the
         market's rising tide (57-60% of EVERYTHING went up this window), not
         skill."

    A raw hit rate is uninterpretable. 58% is excellent against a 50% base and
    negative against a 60% one. Every row here therefore reports the selection
    AND the rest of the same day's universe, and the edge between them. The
    absolute numbers will move with the market; the edge is what should respond
    to a fix.

WHAT WOULD COUNT AS RECOVERY
    Not a good month. October alone proves nothing -- July was +2.23pp on the
    decile spread at t=+2.99 and August fell straight back to +0.23.

    Recovery is the EDGE column rising and staying up across several months,
    while the pre-fix months remain visible above it for comparison. The table
    prints every month precisely so a single good one cannot be read in
    isolation.

    Conversely if the edge keeps falling with correct inputs, the macro skew
    was not the cause and the decay has another explanation that is still
    unfound. That outcome is as useful as the other and should not be treated
    as a failure of the fix.

A LIMIT WORTH STATING
    The most recent h-day window cannot have outcomes yet: an h=5 prediction
    made four days ago has no realised return. The final month is therefore
    biased toward its early dates and understated in count. The `mature`
    column shows what fraction of that month's predictions have resolved, so a
    partial month is visible rather than mistaken for a level.

USAGE
    python analysis/recovery_tracker.py
    python analysis/recovery_tracker.py --horizon 5 --gate 0.70
"""
import argparse
import sqlite3
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--gate", type=float, default=0.70,
                    help="the production BUY threshold; 0.60 is the wider gate "
                         "where the decay was sharpest")
    ap.add_argument("--fix-date", default="2026-09-14")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    rows = list(con.execute("""
        SELECT strftime('%Y-%m', p.prediction_date) AS m,
               p.prob_up, o.actual_up, o.actual_return
        FROM predictions p
        JOIN outcomes o ON o.ticker = p.ticker
                       AND o.prediction_date = p.prediction_date
                       AND o.horizon = p.horizon
        WHERE p.horizon = ? AND p.prob_up IS NOT NULL
          AND o.actual_up IS NOT NULL
    """, (a.horizon,)))

    # Maturity: predictions logged vs predictions with outcomes, per month.
    logged = dict(con.execute("""
        SELECT strftime('%Y-%m', prediction_date), COUNT(*)
        FROM predictions WHERE horizon = ? GROUP BY 1
    """, (a.horizon,)))
    con.close()

    if not rows:
        print("no rows -- check --db and --horizon")
        return

    hi = defaultdict(list)
    lo = defaultdict(list)
    for m, prob, up, ret in rows:
        (hi if float(prob) >= a.gate else lo)[m].append((int(up), float(ret)))

    print(f"\nRECOVERY TRACKER — h={a.horizon}, gate prob_up >= {a.gate:.2f}")
    print(f"Macro inputs repaired {a.fix_date}. Months before it are the "
          f"broken-input period.")
    print("Every selection is measured against the REST of the same day's "
          "universe, never in isolation.\n")

    hdr = (f"{'month':9}{'n':>6}{'mature':>8}   "
           f"{'sel hit':>8}{'rest':>8}{'EDGE':>9}   "
           f"{'sel ret':>9}{'rest ret':>10}{'EDGE':>9}")
    print(hdr)
    print("-" * len(hdr))

    for m in sorted(set(list(hi) + list(lo))):
        h, l = hi.get(m, []), lo.get(m, [])
        if len(h) < 5 or len(l) < 5:
            print(f"{m:9}{len(h):>6}{'':>8}   too few to benchmark")
            continue
        h_hit = sum(x[0] for x in h) / len(h)
        l_hit = sum(x[0] for x in l) / len(l)
        h_ret = sum(x[1] for x in h) / len(h)
        l_ret = sum(x[1] for x in l) / len(l)
        n_out = len(h) + len(l)
        mat = n_out / logged.get(m, n_out) if logged.get(m) else 1.0
        mark = "  <-- post-fix" if m >= a.fix_date[:7] else ""
        part = "*" if mat < 0.90 else " "
        print(f"{m:9}{len(h):>6}{mat:>7.0%}{part}  "
              f"{h_hit:>8.1%}{l_hit:>8.1%}{h_hit-l_hit:>+9.1%}   "
              f"{h_ret:>9.2%}{l_ret:>10.2%}{h_ret-l_ret:>+9.2%}{mark}")

    print()
    print("  sel  = predictions at or above the gate")
    print("  rest = every other prediction that day, the benchmark")
    print("  EDGE = sel minus rest. This is the only column that means "
          "anything on its own.")
    print("  *    = under 90% of that month's predictions have matured; an "
          "h-day window")
    print("         cannot resolve until h days have passed, so the newest "
          "month is partial.")
    print()
    print("  One good month is not recovery. July 2026 read +2.23pp on the")
    print("  decile spread at t=+2.99 and August fell back to +0.23. Read the")
    print("  EDGE column as a trend across months, with the pre-fix rows above")
    print("  it for scale.")
    print()
    print("  If EDGE keeps falling on correct inputs, the macro skew was not "
          "the cause")
    print("  and the decay has another explanation that is still unfound. That "
          "is a")
    print("  result, not a failure.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
