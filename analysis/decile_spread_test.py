#!/usr/bin/env python3
"""
decile_spread_test.py — is the direction model still INVERTED at the extremes?

READ-ONLY. Reads logged predictions and outcomes. Trains nothing, writes nothing.

WHY THIS EXISTS
    signals/generator.py:996 forces every BUY to HOLD and has done since
    2026-05-31. Its stated justification, from the comment and from
    docs/MASTER_TODO_LIST.md section 1.1(b):

        "INVERTED AT EXTREMES, worst at 5d: decile spread t=-2.29"

    That number is not reproducible. No script in the repo computes a decile
    spread, no saved output records one, and the measurement survives only as a
    sentence in a code comment and a line in a to-do list. A switch that has
    suppressed every BUY for three and a half months rests on a figure nobody
    can re-run.

    This reconstructs it. If the spread is still negative at around t = -2.3,
    the switch is correct and the matter is settled. If it has turned positive,
    the justification has expired and lifting the switch becomes defensible on
    the same evidence type that set it.

WHAT A DECILE SPREAD IS
    On each date, rank every prediction by prob_up. Take the mean forward
    return of the top decile and subtract the mean of the bottom decile. A
    working direction model gives a POSITIVE spread: the names it likes beat
    the names it dislikes. A negative spread means the ranking is backwards at
    the extremes -- the model's confident calls are its worst.

    Reported with a Newey-West t across dates, lag = horizon, because forward
    windows of length H overlap on consecutive dates and the raw t would be
    inflated.

WHAT THIS CANNOT SETTLE
    The original t=-2.29 was measured on a 90-day window ending around
    2026-05-31, and predictions before the 2026-05-07 three-way temporal split
    (the Sprint 1 leak fix) were scored by a model fitted on contaminated data
    -- the leak was worth roughly 0.30 AUC. So the pre-switch period here is
    NOT the same estimator as the post-switch period, and a sign change across
    the boundary may be the leak fix landing rather than the market changing.
    The monthly table is printed so that is visible rather than assumed.

    If the pre-switch window does NOT reproduce something near -2.29, the
    original figure came from data or a construction not present in
    predictions/outcomes, and that is itself worth knowing.

THE BENCHMARK PROBLEM THIS AVOIDS
    docs/MASTER_TODO_LIST.md section 1.1b is emphatic that a raw BUY hit rate
    means nothing: "5d: BUY 57.8% vs HOLD 59.8% = -2.0pp (NEGATIVE) ... The 58%
    is the market's rising tide, not skill." A decile spread is internally
    benchmarked -- top decile against bottom decile on the SAME date -- so the
    rising tide cancels. That is why it was the chosen statistic.

USAGE
    python analysis/decile_spread_test.py
    python analysis/decile_spread_test.py --horizon 5 --split 2026-05-31
"""
import argparse
import math
import sqlite3
import sys
from collections import defaultdict

import numpy as np


def nw_t(x, lag):
    """Newey-West t on the mean of x, `lag` overlapping periods."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if s <= 0:
        return float("nan")
    return x.mean() / math.sqrt(s / n)


def spreads_for(rows, n_bucket, min_names):
    """rows: {date: [(prob, fwd_ret), ...]} -> {date: spread}"""
    out = {}
    for d, vals in rows.items():
        if len(vals) < min_names:
            continue
        vals = sorted(vals, key=lambda x: x[0])
        k = max(1, len(vals) // n_bucket)
        lo = [v[1] for v in vals[:k]]
        hi = [v[1] for v in vals[-k:]]
        out[d] = (sum(hi) / len(hi)) - (sum(lo) / len(lo))
    return out


def report(label, sp, lag):
    if len(sp) < 5:
        print(f"  {label:22} {len(sp)} dates -- too few to report")
        return None
    v = np.array([sp[d] for d in sorted(sp)], dtype=float)
    t = nw_t(v, lag)
    pos = float((v > 0).mean())
    print(f"  {label:22} {len(v):>4} dates   mean {v.mean():+.4%}   "
          f"NW t {t:+.2f}   {pos:.0%} of dates > 0")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--buckets", type=int, default=10, help="10 = deciles")
    ap.add_argument("--min-names", type=int, default=50,
                    help="skip dates thinner than this; the thinnest date in "
                         "predictions carries 101 names, so a decile is 10")
    ap.add_argument("--split", default="2026-05-31",
                    help="the STEP 1 kill-switch date")
    a = ap.parse_args()
    H = a.horizon

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    q = """
        SELECT p.prediction_date, p.prob_up, o.actual_return
        FROM predictions p
        JOIN outcomes o ON o.ticker = p.ticker
                       AND o.prediction_date = p.prediction_date
                       AND o.horizon = p.horizon
        WHERE p.horizon = ? AND p.prob_up IS NOT NULL
          AND o.actual_return IS NOT NULL
    """
    byd = defaultdict(list)
    for d, prob, ret in con.execute(q, (H,)):
        byd[d].append((float(prob), float(ret)))
    con.close()

    if not byd:
        print("no rows -- check --db and --horizon")
        return

    sp = spreads_for(byd, a.buckets, a.min_names)
    dates = sorted(sp)
    print(f"\nDECILE SPREAD, h={H} -- top {a.buckets}-ile minus bottom, per date")
    print(f"{len(byd)} dates with outcomes, {len(sp)} usable "
          f"(>= {a.min_names} names), {dates[0]} .. {dates[-1]}")
    print(f"Newey-West lag {H} (forward windows overlap on consecutive dates)\n")

    print("OVERALL")
    report("all dates", sp, H)

    print(f"\nSPLIT AT THE KILL-SWITCH DATE ({a.split})")
    pre = {d: v for d, v in sp.items() if d < a.split}
    post = {d: v for d, v in sp.items() if d >= a.split}
    t_pre = report("before", pre, H)
    t_post = report("after", post, H)

    print("\nBY MONTH")
    bym = defaultdict(dict)
    for d, v in sp.items():
        bym[d[:7]][d] = v
    for m in sorted(bym):
        report(m, bym[m], H)

    print("\n" + "=" * 70)
    print("READ")
    print("=" * 70)
    if t_pre is not None:
        print(f"  The switch was set on 'decile spread t=-2.29' at h=5, measured")
        print(f"  on a 90-day window ending around 2026-05-31. This computes")
        print(f"  t={t_pre:+.2f} on the pre-switch dates available here.")
        if t_pre > -1.0:
            print("  That does NOT reproduce -2.29. The original figure came from")
            print("  a different window, universe or construction than")
            print("  predictions/outcomes hold, so the two are not comparable and")
            print("  neither confirms nor refutes the other.")
    if t_post is not None:
        if t_post > 2.0:
            print(f"\n  POST-SWITCH t={t_post:+.2f}: the spread is positive and")
            print("  significant. The inversion the switch was set on is not")
            print("  present in the current data. That is an argument for")
            print("  revisiting it -- on the same evidence type that set it.")
        elif t_post < -2.0:
            print(f"\n  POST-SWITCH t={t_post:+.2f}: still inverted. The switch is")
            print("  doing its job and should stay. Settled.")
        else:
            print(f"\n  POST-SWITCH t={t_post:+.2f}: neither clearly inverted nor")
            print("  clearly working. No case for lifting, no case for a new")
            print("  diagnosis. The honest answer is that it is not measurable")
            print("  yet at this sample size.")
    print("\n  Caveat that applies to every number above: predictions before the")
    print("  2026-05-07 leak fix were scored by a model fitted on contaminated")
    print("  data (~0.30 AUC of leak). A sign change across the split may be")
    print("  that fix landing rather than the market or the signal changing.")
    print("  The monthly rows are printed so that is visible.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
