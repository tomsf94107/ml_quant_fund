#!/usr/bin/env python3
"""
highconf_sample_check.py — is August's high-confidence sample even complete?

READ-ONLY. Writes nothing. ONE question, deliberately.

WHY ONLY ONE
    Six hypotheses have already been tested against the May-to-August change:
    model degradation (walk-forward AUC flat, 0.5347 -> 0.5362), market
    direction (edge is measured over each month's own base rate), dispersion
    collapse (~10%, not the ~93% needed), concentration, contributor
    persistence, and volatility regime. Each was eliminated or confounded --
    the vol split turned out to be month composition wearing a regime label,
    since May was 20/20 low-vol days and good while August was 21/21 low-vol
    and bad.

    Continuing to slice is now itself a multiple-comparisons problem: run
    enough splits on four months and one will separate by chance. So this
    script asks the last question that could change the interpretation, and
    then stops.

THE QUESTION
    High-confidence h=5 predictions with outcomes fell from 332 in July to 78
    in August. Two very different explanations:

      (a) The model genuinely became less confident -- the probability
          distribution shifted down, so fewer predictions cleared 0.70. This is
          already partly established: mean prob_up went 0.525 (May) to 0.463
          (August).

      (b) The predictions were MADE but their outcomes are missing -- unscored,
          still pending, or dropped by the outcome writer. If August's scored
          subset is not a random sample of August's predictions, then the 48.7%
          is measuring something other than what it appears to.

    (b) matters because the outcome writer was repaired mid-session on
    2026-08-30: split adjustment, a sanity bound, a listing guard, and flat
    closes retained. Any of those changes what gets scored. If August is
    partially scored and the unscored part is not random, the comparison to
    July is not like-for-like.

WHAT IS CHECKED
    1. Predictions made vs predictions scored, per month, at the gate.
    2. The scored fraction -- a fall in August is the warning sign.
    3. Whether unscored predictions differ systematically from scored ones in
       probability, since a bias there biases the accuracy directly.
    4. How many predictions are legitimately still pending (h=5 needs five
       sessions to resolve) versus missing for another reason.

    python analysis/highconf_sample_check.py --db accuracy.db
"""
import argparse
import math
import sqlite3
from collections import defaultdict


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.70)
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    made = defaultdict(int)
    scored = defaultdict(int)
    prob_scored = defaultdict(list)
    prob_unscored = defaultdict(list)
    last_pred = {}

    rows = con.execute("""
        SELECT p.prediction_date, p.prob_up, o.actual_up
        FROM predictions p
        LEFT JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.horizon=? AND p.prob_up >= ?
          AND p.prediction_date >= '2026-05-01'
    """, (args.horizon, args.thresh)).fetchall()

    for d, p, y in rows:
        m = d[:7]
        made[m] += 1
        last_pred[m] = max(last_pred.get(m, ""), d)
        if y is None:
            prob_unscored[m].append(p)
        else:
            scored[m] += 1
            prob_scored[m].append(p)

    print(f"h={args.horizon}, prob_up >= {args.thresh}\n")
    print(f"  {'month':<9}{'made':>7}{'scored':>8}{'unscored':>10}"
          f"{'scored %':>10}{'mean p scored':>15}{'mean p unscored':>17}")
    for m in sorted(made):
        mk, sc = made[m], scored[m]
        un = mk - sc
        ps = sum(prob_scored[m]) / len(prob_scored[m]) if prob_scored[m] else float("nan")
        pu = sum(prob_unscored[m]) / len(prob_unscored[m]) if prob_unscored[m] else float("nan")
        print(f"  {m:<9}{mk:>7}{sc:>8}{un:>10}{100*sc/max(mk,1):>9.1f}%"
              f"{ps:>15.3f}{pu:>17.3f}")

    print("\n  A scored%% that falls sharply in August means the sample is "
          "incomplete.\n  A mean probability that differs between scored and "
          "unscored means the\n  scored subset is BIASED, and its accuracy is "
          "not the month's accuracy.")

    # how much of August's shortfall is legitimately pending?
    latest_outcome = con.execute(
        "SELECT MAX(outcome_date) FROM outcomes WHERE horizon=?",
        (args.horizon,)).fetchone()[0]
    latest_pred = con.execute(
        "SELECT MAX(prediction_date) FROM predictions WHERE horizon=?",
        (args.horizon,)).fetchone()[0]
    print(f"\n  latest prediction date: {latest_pred}")
    print(f"  latest outcome date:    {latest_outcome}")
    print(f"  An h={args.horizon} prediction needs {args.horizon} sessions to "
          f"resolve, so the last\n  ~{args.horizon} trading days of "
          f"predictions are legitimately unscored.")

    # scored accuracy by half-month, to see whether late-August is the gap
    print(f"\n  scored accuracy by half-month (is the shortfall at the end?):")
    half = defaultdict(lambda: [0, 0])
    for d, p, y in rows:
        if y is None:
            continue
        key = d[:7] + ("a" if int(d[8:10]) <= 15 else "b")
        half[key][0] += 1
        half[key][1] += y
    print(f"  {'period':<10}{'n':>6}{'acc':>8}{'95% CI':>18}")
    for k in sorted(half):
        n, hits = half[k]
        if n < 8:
            print(f"  {k:<10}{n:>6}   too few")
            continue
        lo, hi = wilson(hits, n)
        print(f"  {k:<10}{n:>6}{100*hits/n:>7.1f}%   [{lo:>5.1f}, {hi:>5.1f}]")

    con.close()
    print("\n  If August's scored fraction is normal and the unscored rows are "
          "just the\n  last few sessions pending, then n=78 is simply what the "
          "model emitted --\n  a real drop in confidence, not a measurement "
          "artifact. That closes the\n  investigation: August was one weak "
          "month on a small sample, and the\n  model's measured ceiling was "
          "2-4pp all along.")


if __name__ == "__main__":
    main()
