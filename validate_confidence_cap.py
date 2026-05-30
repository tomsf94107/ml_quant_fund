#!/usr/bin/env python3
"""
Sprint W1 — confidence-cap validation.

signals/generator.py caps effective confidence at 0.65 for h=3 and h=5
(INVERSION_HORIZONS). Justification, per the code comment:
  "h=3 and h=5 measured INVERTED at prob_up >= 0.70 per May 7 SHAP analysis.
   Mid-confidence (40-60%) wins more than high-confidence (>70%)."

PROBLEM: that May 7 analysis ran BEFORE the May 4 outcomes-reconciliation
fix and the May 12 validator fix. Outcomes were wiped and cleanly
re-reconciled May 12 (10,697 rows). So the inversion claim may be an
artifact of the old bugged labels — the same bug that made AUC look like
0.520 when the clean number is 0.486.

THIS SCRIPT re-tests the inversion on CLEAN labels. For h=1 (control),
h=3, h=5 it buckets predictions by prob_up and reports realized win rate
(actual_return > 0) per bucket, with Wilson CIs.

READ:
  - If high-confidence (>=0.70) win rate < mid (0.40-0.60) for h3/h5
    => inversion is REAL on clean data => cap is justified, keep it.
  - If high-confidence win rate >= mid (the normal monotone pattern)
    => inversion was a bugged-label artifact => the cap is THROTTLING
       good signals and should be removed or raised.
  - h=1 is the control — it was never capped; it should look monotone.

Run:  python validate_confidence_cap.py
"""
import sqlite3
import math

DB = "accuracy.db"
WINDOW_START = "2026-04-15"
WINDOW_END = "2026-05-14"

# prob_up confidence buckets
BUCKETS = [
    ("low      <0.40",  0.00, 0.40),
    ("mid 0.40-0.60",   0.40, 0.60),
    ("high0.60-0.70",   0.60, 0.70),
    ("vhigh   >=0.70",  0.70, 1.01),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


def main():
    con = sqlite3.connect(DB)

    print("=" * 72)
    print(" Confidence-cap validation — does the h3/h5 inversion survive")
    print(f" clean labels?   window {WINDOW_START}..{WINDOW_END}")
    print("=" * 72)

    for horizon in (1, 3, 5):
        tag = "CONTROL — never capped" if horizon == 1 else "CAPPED at 0.65"
        print(f"\n{'='*72}\n HORIZON {horizon}   ({tag})\n{'='*72}")

        rows = con.execute("""
            SELECT p.prob_up, o.actual_return
            FROM predictions p
            JOIN outcomes o
              ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
             AND p.horizon=o.horizon
            WHERE p.horizon = ?
              AND p.prediction_date BETWEEN ? AND ?
              AND p.prob_up IS NOT NULL
              AND o.actual_return IS NOT NULL
        """, (horizon, WINDOW_START, WINDOW_END)).fetchall()

        print(f"  {'bucket':<16}{'n':>6}{'wins':>7}{'win_rate':>10}"
              f"{'CI_lo':>8}{'CI_hi':>8}")
        bucket_stats = {}
        for label, lo, hi in BUCKETS:
            sub = [r for r in rows if lo <= r[0] < hi]
            n = len(sub)
            wins = sum(1 for _, ret in sub if ret > 0)
            p, clo, chi = wilson(wins, n)
            bucket_stats[label] = (n, p, clo, chi)
            print(f"  {label:<16}{n:>6}{wins:>7}{p*100:>9.1f}%"
                  f"{clo*100:>7.0f}%{chi*100:>7.0f}%")

        # the inversion test: vhigh vs mid
        mid = bucket_stats["mid 0.40-0.60"]
        vhigh = bucket_stats["vhigh   >=0.70"]
        print()
        if mid[0] < 8 or vhigh[0] < 8:
            print("  -> INSUFFICIENT DATA in mid or vhigh bucket "
                  f"(mid n={mid[0]}, vhigh n={vhigh[0]}). Verdict deferred.")
        else:
            mid_p, vhigh_p = mid[1], vhigh[1]
            # CIs: do they even separate?
            sep = "non-overlapping" if (vhigh[3] < mid[2] or mid[3] < vhigh[2]) \
                  else "overlapping (not statistically distinct)"
            print(f"  mid win_rate   = {mid_p*100:.1f}%  (n={mid[0]})")
            print(f"  vhigh win_rate = {vhigh_p*100:.1f}%  (n={vhigh[0]})")
            print(f"  CIs are {sep}.")
            if vhigh_p < mid_p:
                print("  -> INVERSION PRESENT: high confidence wins LESS than")
                print("     mid. IF CIs are non-overlapping, the cap is")
                print("     justified on clean data — keep it.")
            else:
                print("  -> NO INVERSION: high confidence wins >= mid (normal")
                print("     monotone pattern). The May 7 inversion was a")
                print("     bugged-label artifact. The 0.65 cap is now")
                print("     THROTTLING good h={0} signals.".format(horizon))

    con.close()

    print("\n" + "=" * 72)
    print(" WHAT TO DO")
    print("=" * 72)
    print("  h=1 is the control: it should look monotone (more confidence")
    print("  -> higher win rate). If h=1 looks monotone but h=3/h=5 also")
    print("  look monotone, the inversion is GONE on clean labels and the")
    print("  cap should be removed (or raised well above observed prob_eff).")
    print()
    print("  If h=3/h=5 STILL invert with non-overlapping CIs, the cap is")
    print("  doing real risk control — keep it, and note it survived a")
    print("  clean-label re-test.")
    print()
    print("  Either way: do NOT change signals/generator.py this week —")
    print("  a generator change confounds Friday's Pipeline B institutional")
    print("  -features read. Record the verdict; ship the cap change (if")
    print("  any) alongside the Task D fitness-gate commit post-Friday.")


if __name__ == "__main__":
    main()
