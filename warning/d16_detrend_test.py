#!/usr/bin/env python3
"""
d16_detrend_test.py — is S9's expanding linear detrend actually misspecified?

THE CLAIM (D16)
    S9 fits an EXPANDING linear OLS to log(aggregate short interest) and
    z-scores the residual. If the underlying growth is ACCELERATING, a straight
    line always lags a convex curve, so the most recent point sits above the fit
    by construction -- biasing the endpoint residual positive and pushing the
    signal toward firing.

    I recorded that as an open ruling and said it could not be tested without
    FINRA history back to 2014. That was wrong: the bias is a property of the
    FIT, not of the sample's length, so it is measurable on the 2021-2026 data
    directly.

THE TEST
    Walk every settlement date. At each, run S9's own expanding OLS over the
    history visible to that date and record the SIGN of the final residual.

    Under a correctly specified trend the endpoint residual is a mean-zero
    fluctuation: positive about half the time. Under a linear fit to convex
    growth it is positive far more often. A binomial interval on the observed
    fraction says whether the difference is real.

    Also reported: the fitted slope over time. A slope that rises monotonically
    is direct evidence of acceleration, independent of the residual argument.

    python warning/d16_detrend_test.py --db warning.db
"""
import argparse
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders.s9_short_interest import _ols_residual_last, MIN_TREND_OBS


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    rows = con.execute(
        "SELECT series_id, obs_date, value FROM data_vintages "
        "WHERE series_id LIKE 'SI:%' ORDER BY obs_date").fetchall()
    con.close()
    by_date = {}
    for sid, d, v in rows:
        by_date.setdefault(d, {})[sid] = v
    dates = sorted(by_date)

    # Fixed panel across the whole sample: this test is about the FIT, so the
    # panel is held constant to keep universe drift out of the measurement.
    panel = set(by_date[dates[0]])
    for d in dates[1:]:
        panel &= set(by_date[d])
    agg = [(d, math.log(sum(by_date[d][s] for s in panel))) for d in dates]
    print(f"{len(dates)} settlement dates {dates[0]}..{dates[-1]}, "
          f"fixed panel {len(panel)} names\n")

    pos = n = 0
    slopes = []
    for i in range(MIN_TREND_OBS, len(agg) + 1):
        ys = [v for _, v in agg[:i]]
        resid, _a, b = _ols_residual_last(ys)
        if resid is None:
            continue
        n += 1
        if resid[-1] > 0:
            pos += 1
        slopes.append((agg[i - 1][0], b))

    lo, hi = wilson(pos, n)
    print(f"endpoint residual POSITIVE on {pos}/{n} evaluation dates "
          f"({100.0*pos/n:.1f}%)")
    print(f"  95% interval [{100*lo:.1f}%, {100*hi:.1f}%]")
    print(f"  a correctly specified trend gives ~50%")
    verdict = ("MISSPECIFIED -- the fit is biased upward"
               if lo > 0.5 else
               "biased DOWNWARD" if hi < 0.5 else
               "not distinguishable from 50%: D16's mechanism is NOT confirmed")
    print(f"  -> {verdict}\n")

    print("fitted slope over time (acceleration shows as a rising slope):")
    step = max(1, len(slopes) // 10)
    for d, b in slopes[::step]:
        print(f"  {d}  {b:+.5f} per obs")
    if len(slopes) > 1:
        first, last = slopes[0][1], slopes[-1][1]
        print(f"\n  first {first:+.5f} -> last {last:+.5f}  "
              f"({'RISING: growth is accelerating' if last > first * 1.2 else 'roughly stable'})")


if __name__ == "__main__":
    main()
