#!/usr/bin/env python3
"""
calibration_window_test.py — which fit window gives a stable, accurate map?

READ-ONLY. Writes nothing.

THE QUESTION
    The backfill run on 2026-09-01 showed k oscillating badly between monthly
    refits on a 180-day window:
        h=1   0.235 -> 0.028 -> 0.039
        h=3   0.020 -> 0.028 -> 0.125   (June's fit was mildly INVERTED)
        h=5   0.195 -> 0.135 -> 0.239
    The calibration literature warns about exactly this: refitting too often on
    too few samples makes the map fluctuate. A longer window steadies k at the
    cost of adapting more slowly to a genuine regime change.

    This measures the trade-off instead of guessing at it.

WHAT IS REPORTED PER WINDOW LENGTH
    ECE      out-of-sample calibration error, walk-forward, size-weighted.
             LOWER is better -- this is the thing calibration is for.
    k drift  mean absolute change in k between consecutive refits.
             LOWER is more stable. A map that jumps around is one whose output
             is not comparable month to month, which defeats the purpose of
             having a scale.
    k range  min..max across refits, so a single wild fit is visible rather
             than averaged away.

    The right choice is the shortest window whose k is stable, not the one with
    the lowest ECE -- a map that fits each month beautifully and disagrees with
    itself is worse than a slightly stale one that holds.

    python analysis/calibration_window_test.py --db accuracy.db
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MIN_FIT = 500
K_FLOOR = 0.02
N_BINS = 10
WINDOWS = [90, 180, 270, 365, 9999]      # 9999 = expanding (all prior data)


def ece(pairs, n_bins=N_BINS):
    d = sorted(pairs)
    size = max(1, len(d) // n_bins)
    tot, n = 0.0, len(d)
    for b in range(n_bins):
        c = d[b * size:(b + 1) * size] if b < n_bins - 1 else d[b * size:]
        if not c:
            continue
        tot += len(c) * abs(sum(x[0] for x in c) / len(c)
                            - sum(x[1] for x in c) / len(c))
    return tot / n


def fit_one(pairs):
    base = sum(y for _, y in pairs) / len(pairs)
    d = sorted(pairs)
    s = max(1, len(d) // N_BINS)
    lo_p = sum(x[0] for x in d[:s]) / s
    hi_p = sum(x[0] for x in d[-s:]) / s
    lo_r = sum(x[1] for x in d[:s]) / s
    hi_r = sum(x[1] for x in d[-s:]) / s
    raw_k = (hi_r - lo_r) / (hi_p - lo_p) if (hi_p - lo_p) else 1.0
    return base, max(K_FLOOR, min(1.0, raw_k)), raw_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.horizon, p.prediction_date, o.outcome_date, p.prob_raw, o.actual_up
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.prob_raw IS NOT NULL AND o.actual_up IS NOT NULL
          AND o.outcome_date IS NOT NULL
    """).fetchall()
    con.close()

    by_h = defaultdict(list)
    for h, pd_, od, pr, y in rows:
        by_h[h].append((pd_, od, float(pr), int(y)))

    for h in sorted(by_h):
        recs = by_h[h]
        months = sorted({r[0][:7] for r in recs})
        print(f"\n{'=' * 72}\nHORIZON {h}d   n={len(recs)}   "
              f"{months[0]}..{months[-1]}\n{'=' * 72}")
        print(f"  {'window':>8}{'refits':>8}{'ECE':>9}{'k drift':>10}"
              f"{'k range':>18}")
        for w in WINDOWS:
            held, ks = [], []
            for m in months[1:]:
                start = f"{m}-01"
                cut = ("0000-00-00" if w == 9999 else
                       (date.fromisoformat(start) - timedelta(days=w)).isoformat())
                fit = [(r[2], r[3]) for r in recs
                       if r[1][:10] < start and r[0] >= cut]
                test = [(r[2], r[3]) for r in recs if r[0][:7] == m]
                if len(fit) < MIN_FIT or not test:
                    continue
                base, k, _raw = fit_one(fit)
                ks.append(k)
                held.append([(base + k * (p - base), y) for p, y in test])
            if not held:
                print(f"  {('exp' if w == 9999 else w):>8}{'0':>8}"
                      f"{'--':>9}{'--':>10}{'--':>18}")
                continue
            tot = sum(len(x) for x in held)
            e = sum(ece(x) * len(x) for x in held) / tot
            drift = (sum(abs(ks[i] - ks[i - 1]) for i in range(1, len(ks)))
                     / max(len(ks) - 1, 1))
            print(f"  {('exp' if w == 9999 else w):>8}{len(ks):>8}{e:>9.4f}"
                  f"{drift:>10.4f}   {min(ks):.3f}..{max(ks):.3f}")

    print("\nPick the SHORTEST window whose k is stable. A map that fits each "
          "month\nbeautifully but disagrees with itself month to month is worse "
          "than a\nslightly stale one that holds -- the whole point is a scale "
          "that means\nthe same thing over time.")


if __name__ == "__main__":
    main()
