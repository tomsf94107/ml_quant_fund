#!/usr/bin/env python3
"""
rate_beta_regime_test.py — is rate_beta alpha, or just duration exposure?

READ-ONLY. Writes nothing.

THE QUESTION
    rate_beta produced the cleanest result in this whole sequence: predictive IC
    t=+2.09 at h=5 and t=+1.84 at h=20 with clean nulls (+0.15, +0.92), and a
    10.6pp spread in unconditional up-rates across terciles -- 47.0% for the
    most rate-sensitive names against 57.6% for the least.

    But the sample is 2021-2026, during which the 10-year went from roughly 1%
    to 4.7%. In a monotone rising-rate regime, rate-sensitive names MUST
    underperform -- that is what rate sensitivity means. The result could be
    pure factor exposure, rediscovered.

    A shuffle null cannot detect this. Shuffling tests for leakage; it does not
    test whether a real relationship is regime-contingent. Only splitting by
    regime does.

THE TEST
    Split evaluation dates by the trailing 20-day change in the 10-year yield:

      RISING    yields up over the trailing 20 days
      FALLING   yields down

    If rate_beta is DURATION EXPOSURE, its IC flips sign between the two: high
    beta names lose when rates rise and win when rates fall, mechanically. The
    strategy would be a levered bet on rates, not alpha, and would have lost
    money in 2020 or any easing cycle.

    If the IC has the SAME SIGN in both regimes, something else is going on --
    the characteristic is picking up quality, leverage or fragility that
    persists regardless of rate direction. That would be worth pursuing.

    Also reported: the unconditional up-rate by tercile WITHIN each regime. If
    the 47.0%/57.6% spread reverses when yields fall, that settles it visually
    without needing the IC at all.

WHAT WOULD MAKE THIS CONCLUSIVE, AND WHY IT IS NOT
    A clean test needs a full easing cycle. This sample has stretches of falling
    yields inside a rising trend, which is weaker evidence: a 20-day dip within
    a multi-year rise is not the same as 2019 or 2020. The honest ceiling here
    is "consistent with" or "contradicts", not "proves".

    Reported alongside: how many dates and what yield range each regime covers,
    so the asymmetry is visible rather than buried.

    python analysis/rate_beta_regime_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict


def spearman(pairs):
    n = len(pairs)
    if n < 10:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for m in range(i, j + 1):
                r[o[m]] = a
            i = j + 1
        return r
    rx = rank([p[0] for p in pairs]); ry = rank([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def nw_t(s, lag):
    n = len(s)
    if n < 10:
        return None
    m = sum(s) / n
    d = [x - m for x in s]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--warning-db", default="warning.db")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--min-names", type=int, default=25)
    args = ap.parse_args()
    HOR = (5, 20)

    wc = sqlite3.connect(f"file:{args.warning_db}?mode=ro", uri=True)
    y = {str(d)[:10]: v for d, v in wc.execute(
        "SELECT obs_date, value FROM data_vintages WHERE series_id='DGS10' "
        "AND obs_date >= ? ORDER BY obs_date", (args.start,))}
    wc.close()
    yd = sorted(y)
    dy = {yd[i]: y[yd[i]] - y[yd[i - 1]] for i in range(1, len(yd))}
    dy20 = {yd[i]: y[yd[i]] - y[yd[i - 20]] for i in range(20, len(yd))}
    print(f"DGS10 {yd[0]}..{yd[-1]}: {y[yd[0]]:.2f}% -> {y[yd[-1]]:.2f}%")

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close = defaultdict(dict)
    for t, d, c in px.execute(
            "SELECT ticker, d, close FROM raw_bars WHERE d >= ? AND close>0",
            (args.start,)):
        close[t][d] = c
    px.close()

    fwd = {h: {} for h in HOR}
    beta = {}
    for t, s in close.items():
        ds = sorted(s)
        rets = {}
        for i in range(1, len(ds)):
            a, b = s[ds[i - 1]], s[ds[i]]
            if a and abs((b - a) / a) < 0.5:
                rets[ds[i]] = (b - a) / a
        for h in HOR:
            for i in range(len(ds) - h):
                a, b = s[ds[i]], s[ds[i + h]]
                if a and b and abs((b - a) / a) < 0.8:
                    fwd[h][(t, ds[i])] = (b - a) / a
        rd = sorted(rets)
        for i in range(args.window, len(rd)):
            w = rd[i - args.window:i]
            pair = [(dy[d], rets[d]) for d in w if d in dy]
            if len(pair) < args.window * 0.6:
                continue
            mx = sum(p[0] for p in pair) / len(pair)
            my = sum(p[1] for p in pair) / len(pair)
            sxx = sum((p[0] - mx) ** 2 for p in pair)
            if sxx <= 0:
                continue
            beta[(t, rd[i])] = sum((p[0] - mx) * (p[1] - my)
                                   for p in pair) / sxx

    dates = sorted({d for t in close for d in close[t] if d >= args.start})[::5]
    regimes = {"RISING": [], "FALLING": []}
    for d in dates:
        c = dy20.get(d)
        if c is None:
            continue
        regimes["RISING" if c > 0 else "FALLING"].append(d)
    for r, ds_ in regimes.items():
        if ds_:
            print(f"  {r:<8}{len(ds_):>4} dates   "
                  f"20d yield change {st.mean(dy20[d] for d in ds_):+.3f}pp avg")

    rnd = random.Random(23)
    print(f"\nIC BY REGIME\n")
    print(f"  {'regime':<9}{'h':>4}{'dates':>7}{'mean IC':>10}{'NW t':>8}"
          f"{'null t':>9}")
    ic_by = {}
    for reg, ds_ in regimes.items():
        for h in HOR:
            ics, nl = [], []
            for d in ds_:
                obs = [(beta[(t, d)], fwd[h][(t, d)]) for t in close
                       if (t, d) in beta and (t, d) in fwd[h]]
                if len(obs) < args.min_names:
                    continue
                r = spearman(obs)
                if r is not None:
                    ics.append(r)
                ys = [o[1] for o in obs]
                rnd.shuffle(ys)
                rn = spearman([(obs[i][0], ys[i]) for i in range(len(ys))])
                if rn is not None:
                    nl.append(rn)
            ic_by[(reg, h)] = ics
            if len(ics) < 15:
                print(f"  {reg:<9}{h:>4}{len(ics):>7}   too few dates")
                continue
            print(f"  {reg:<9}{h:>4}{len(ics):>7}{st.mean(ics):>+10.4f}"
                  f"{(nw_t(ics,h) or 0):>+8.2f}{(nw_t(nl,h) or 0):>+9.2f}")
        print()

    print("UP-RATE BY TERCILE, WITHIN EACH REGIME\n")
    print(f"  {'regime':<9}{'h':>4}{'LOW beta':>11}{'MID':>9}{'HIGH':>9}"
          f"{'spread':>10}")
    for reg, ds_ in regimes.items():
        for h in HOR:
            buckets = {"LOW": [], "MID": [], "HIGH": []}
            for d in ds_:
                obs = [(beta[(t, d)], fwd[h][(t, d)]) for t in close
                       if (t, d) in beta and (t, d) in fwd[h]]
                if len(obs) < args.min_names:
                    continue
                sv = sorted(obs)
                q = len(sv) // 3
                buckets["LOW"] += [x[1] for x in sv[:q]]
                buckets["MID"] += [x[1] for x in sv[q:2 * q]]
                buckets["HIGH"] += [x[1] for x in sv[2 * q:]]
            if min(len(v) for v in buckets.values()) < 100:
                print(f"  {reg:<9}{h:>4}   too few observations")
                continue
            up = {k: 100.0 * sum(1 for x in v if x > 0) / len(v)
                  for k, v in buckets.items()}
            print(f"  {reg:<9}{h:>4}{up['LOW']:>10.1f}%{up['MID']:>8.1f}%"
                  f"{up['HIGH']:>8.1f}%{up['HIGH']-up['LOW']:>+9.1f}pp")
        print()

    print("  LOW beta = MOST rate-sensitive (most negative). In the full sample "
          "the\n  spread was +10.6pp (47.0% vs 57.6%).\n")
    print("  IF THE SPREAD REVERSES IN FALLING-YIELD PERIODS, rate_beta is "
          "DURATION\n  EXPOSURE -- a levered bet on rates, not alpha, and it "
          "would have lost in\n  any easing cycle. If the spread holds the same "
          "sign in both regimes, the\n  characteristic is picking up something "
          "that is not rate direction.\n")
    print("  Caveat: FALLING periods here are dips inside a multi-year rise, "
          "not a true\n  easing cycle. This can be consistent-with or "
          "contradict; it cannot prove.")


if __name__ == "__main__":
    main()
