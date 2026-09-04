#!/usr/bin/env python3
"""
pct7_gauntlet.py — can PCT7 survive the tests that killed everything else?

READ-ONLY. Writes nothing.

WHAT PCT7 HAS ALREADY PASSED
    Scored 2026-09-05 after fourteen unread weeks in shadow mode, on a model
    trained 2026-05-25 and never retrained -- so the entire record is
    out-of-sample:

        threshold  n      dates  tickers  hit    base   lift
        0.20       3,506  65     189      22.8%  10.8%  +12.0pp   CI [21.5,24.2]
        0.30       1,105  64     116      29.0%  10.8%  +18.2pp
        0.50          66  14      37      57.6%  10.8%  +46.8pp

    Monotone in threshold. Calibration monotone through all six buckets. And
    within-date, which removes market direction, it beat its own session's base
    rate on 48 of 65 days -- z ~ 3.8 against a 50/50 null.

WHY A GAUNTLET ANYWAY
    Every other candidate this project has tested died at one of these steps,
    usually the last two. The SI brick passed them and is the fund's only
    validated edge. Applying the same bar is what makes the comparison mean
    anything.

    1. NULL CONTROL. Shuffle the outcome within each date and the lift must
       vanish. Mandatory here: this project has had a false positive survive to
       "confirmed" before (the PEAD result of 2026-06-25, overturned on audit).

    2. PER-DATE INDEPENDENCE. 3,506 stock-days are not 3,506 observations -- a
       market-wide move correlates every name on a date. The unit is the DATE.
       Newey-West on the per-date lift series, at lag = horizon - 1 for the
       overlapping 5-day windows.

    3. REGIME STABILITY. Split by VIX and by market direction. A signal that
       only works in one regime is a regime bet. Note the recent sessions
       already hint at this: on 2026-08-25/26/27 the universe base rate was
       2.6-3.6% and PCT7's lift was ~0 -- it needs large moves to exist before
       it can find them.

    4. YEAR/MONTH CONSISTENCY. Lift by month. Concentration in one month is
       the shape of a false positive.

    5. ECONOMIC VALUE NET OF COST. Hit rate is not money. The realised mean
       return of selected names, minus a cost ladder, decides whether this is
       tradeable or merely true. A +7% target implies wide dispersion, so the
       LOSING tail matters as much as the hit rate: a 22.8% hit rate with a -8%
       average miss loses money.

    python analysis/pct7_gauntlet.py
    python analysis/pct7_gauntlet.py --thresh 0.30
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def nw_t(series, lag):
    n = len(series)
    if n < 10:
        return None
    m = sum(series) / n
    d = [x - m for x in series]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--thresh", type=float, default=0.20)
    ap.add_argument("--target", type=float, default=0.07)
    args = ap.parse_args()
    H = 5

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.prediction_date, p.ticker, p.prob_pct7, o.actual_return
        FROM predictions p JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        WHERE p.horizon=5 AND p.prob_pct7 IS NOT NULL
          AND o.actual_return IS NOT NULL
    """).fetchall()
    con.close()
    print(f"PCT7 gauntlet — threshold {args.thresh}, "
          f"target fwd 5d >= {args.target:+.0%}")
    print(f"{len(rows):,} scored predictions, "
          f"{len({r[0] for r in rows})} dates\n")

    byd = defaultdict(list)
    for d, t, p, r in rows:
        byd[d].append((t, p, r))

    # ---------- 1. per-date lift, and the null ----------
    lifts, nulls, sel_n = [], [], []
    rnd = random.Random(7)
    for d in sorted(byd):
        v = byd[d]
        sel = [x for x in v if x[1] >= args.thresh]
        if len(sel) < 5:
            continue
        base = sum(1 for x in v if x[2] >= args.target) / len(v)
        hit = sum(1 for x in sel if x[2] >= args.target) / len(sel)
        lifts.append(100 * (hit - base))
        sel_n.append(len(sel))
        # shuffle the OUTCOME within the date, keeping the selection
        outs = [x[2] for x in v]
        rnd.shuffle(outs)
        sh = outs[:len(sel)]
        nulls.append(100 * (sum(1 for r in sh if r >= args.target) / len(sel)
                            - base))

    print("1. PER-DATE LIFT, Newey-West (the date is the unit, not the row)")
    t_ = nw_t(lifts, H - 1)
    tn = nw_t(nulls, H - 1)
    print(f"   dates {len(lifts)}   mean lift {st.mean(lifts):+.2f}pp   "
          f"NW t {t_ if t_ else 0:+.2f}")
    print(f"   NULL: mean {st.mean(nulls):+.2f}pp   NW t {tn if tn else 0:+.2f}")
    print(f"   positive on {sum(1 for x in lifts if x > 0)}/{len(lifts)} dates")
    if tn and abs(tn) > 1.5:
        print("   !! the null is large -- the result is NOT trustworthy")
    print()

    # ---------- 2. by month ----------
    print("2. BY MONTH (concentration is the shape of a false positive)")
    bym = defaultdict(list)
    for i, d in enumerate(sorted(byd)):
        v = byd[d]
        sel = [x for x in v if x[1] >= args.thresh]
        if len(sel) < 5:
            continue
        base = sum(1 for x in v if x[2] >= args.target) / len(v)
        hit = sum(1 for x in sel if x[2] >= args.target) / len(sel)
        bym[d[:7]].append((100 * (hit - base), len(sel)))
    print(f"   {'month':<9}{'dates':>7}{'n':>7}{'mean lift':>12}")
    for m in sorted(bym):
        v = bym[m]
        print(f"   {m:<9}{len(v):>7}{sum(x[1] for x in v):>7}"
              f"{st.mean(x[0] for x in v):>+11.1f}pp")
    print()

    # ---------- 3. regime ----------
    print("3. BY REGIME — universe base rate that day as the proxy for how")
    print("   many large moves were available at all")
    buck = defaultdict(list)
    for d in sorted(byd):
        v = byd[d]
        sel = [x for x in v if x[1] >= args.thresh]
        if len(sel) < 5:
            continue
        base = sum(1 for x in v if x[2] >= args.target) / len(v)
        hit = sum(1 for x in sel if x[2] >= args.target) / len(sel)
        k = ("quiet  (base <5%)" if base < 0.05 else
             "normal (5-12%)" if base < 0.12 else "active (>12%)")
        buck[k].append(100 * (hit - base))
    print(f"   {'regime':<20}{'dates':>7}{'mean lift':>12}")
    for k in ("quiet  (base <5%)", "normal (5-12%)", "active (>12%)"):
        if k in buck:
            print(f"   {k:<20}{len(buck[k]):>7}"
                  f"{st.mean(buck[k]):>+11.1f}pp")
    print()

    # ---------- 4. economic value ----------
    print("4. ECONOMIC VALUE — hit rate is not money")
    sel = [r for r in rows if r[2] >= args.thresh]
    allr = [r[3] for r in rows]
    sr = [r[3] for r in sel]
    wins = [x for x in sr if x >= args.target]
    loss = [x for x in sr if x < args.target]
    print(f"   selected {len(sel):,}   mean return {100*st.mean(sr):+.2f}%   "
          f"median {100*st.median(sr):+.2f}%")
    print(f"   universe               mean return {100*st.mean(allr):+.2f}%   "
          f"median {100*st.median(allr):+.2f}%")
    print(f"   hits  n={len(wins):>5}  mean {100*st.mean(wins):+.2f}%")
    print(f"   misses n={len(loss):>5}  mean {100*st.mean(loss):+.2f}%")
    print(f"\n   {'cost/leg':>9}{'net mean':>11}{'vs universe':>13}")
    for c in (0, 5, 10, 20, 40):
        net = 100 * st.mean(sr) - c / 100.0
        base_net = 100 * st.mean(allr)
        print(f"   {c:>7}bps{net:>10.2f}%{net - base_net:>+12.2f}pp")
    print("\n   A +7% target implies wide dispersion, so the LOSING tail "
          "matters as\n   much as the hit rate. Mean return of selected names "
          "against the\n   universe -- net of cost -- is what decides whether "
          "this is tradeable\n   or merely true.")


if __name__ == "__main__":
    main()
