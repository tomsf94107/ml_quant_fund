#!/usr/bin/env python3
"""
prob_threshold_honest_test.py — does prob_up >= 0.70 at h=3/5 survive honest weighting?

READ-ONLY. Reads logged predictions and outcomes. Trains nothing.

WHAT PROMPTED THIS
    A pooled query on 2026-09-07 over logged predictions, restricted to tickers
    whose current backtest sharpe is >= 0.2, found the hit rate and mean return
    both rising monotonically with the probability threshold:

        h=5   all             22,961   52.7% up   +0.727%
              prob >= 0.55     7,303   55.1%      +1.121%
              prob >= 0.60     4,219   57.3%      +1.454%
              prob >= 0.70       974   59.4%      +1.912%

        h=3   all             22,990   51.4%      +0.427%
              prob >= 0.70       697   58.1%      +1.120%

    That is the "hockey stick" shape -- conviction concentrated in a small top
    slice -- and it sits awkwardly against this fund's recorded position that
    the h=1/3/5 direction model runs at ~0.51 AUC with no validated directional
    edge, and against the STEP 1 kill switch that has forced every BUY to HOLD
    since 2026-05-31.

    Before any of that is revisited, the number has to survive the three things
    that have killed every similar result in this system.

THE THREE PROBLEMS WITH THE POOLED FIGURE

    1. IT IS POOLED, NOT DAY-WEIGHTED. PCT7 showed +1.46% pooled and -1.97%
       day-weighted on the same data. A pooled mean weights each NAME equally
       and so describes a portfolio holding every selection regardless of which
       day it fired -- not a strategy, since daily counts vary and you cannot
       know the future distribution in order to size it. Weighting each DAY once
       is what a book actually earns.

    2. THERE IS NO BENCHMARK. +1.912% over five days means nothing until it is
       compared with what the universe did on those same days. Base rates here
       run 50.4% / 51.3% / 52.7% at h=1/3/5 -- stocks drift up, and a long-only
       selection inherits that.

    3. THE SHARPE FILTER IS A LOOK-AHEAD. signals_cache.json holds TODAY'S
       backtest sharpe per ticker, applied backwards to historical predictions.
       A ticker only has sharpe >= 0.2 today partly because it went up during
       the sample. This script therefore reports WITH and WITHOUT that filter,
       and the unfiltered version is the honest one.

ALSO CHECKED
    Whether the effect is concentrated in a handful of dates. PCT7's entire
    result came from 6 days of 64, with 38 losing -- which the mean concealed
    and the median exposed.

    And whether it survives the kill-switch boundary: predictions after
    2026-05-31 are HOLD-converted in production but still logged, so the model
    kept scoring. Splitting on that date shows whether the effect is a pre-May
    artifact.

    python analysis/prob_threshold_honest_test.py
"""
import argparse
import json
import math
import os
import sqlite3
import statistics as st
from collections import defaultdict

THRESHOLDS = (0.55, 0.60, 0.70)


def nw_t(v, lag):
    n = len(v)
    if n < 10:
        return None
    m = sum(v) / n
    d = [x - m for x in v]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-names", type=int, default=15,
                    help="minimum predictions on a date for it to count")
    ap.add_argument("--kill-date", default="2026-05-31")
    args = ap.parse_args()

    sharpe = {}
    p = "data/signals_cache.json"
    if os.path.exists(p):
        try:
            for s in json.load(open(p)).get("signals", []):
                t, h, sh = s.get("ticker"), s.get("horizon"), s.get("sharpe")
                if t and sh is not None:
                    sharpe[(t, int(h))] = float(sh)
        except Exception as e:
            print(f"  signals_cache unreadable ({e}); running unfiltered only")

    con = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.ticker, p.prediction_date, p.horizon, p.prob_up,
               o.actual_return
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.horizon IN (1,3,5)""").fetchall()
    con.close()
    print(f"{len(rows):,} matched prediction-outcome pairs\n")

    for use_sharpe in (False, True):
        if use_sharpe and not sharpe:
            continue
        lab = ("WITH the sharpe>=0.2 filter (LOOK-AHEAD: today's sharpe "
               "applied to past predictions)" if use_sharpe
               else "NO sharpe filter -- the honest version")
        print("=" * 74)
        print(lab)
        print("=" * 74)

        byd = defaultdict(lambda: defaultdict(list))
        for t, d, h, pu, r in rows:
            if use_sharpe and sharpe.get((t, int(h)), -9) < 0.2:
                continue
            byd[int(h)][d].append((pu, r))

        for h in (1, 3, 5):
            dates = {d: v for d, v in byd[h].items()
                     if len(v) >= args.min_names}
            if len(dates) < 20:
                print(f"  h={h}: only {len(dates)} usable dates")
                continue
            print(f"  h={h} — {len(dates)} dates, "
                  f"{sum(len(v) for v in dates.values()):,} predictions")
            print(f"    {'selection':<14}{'day-wtd':>10}{'universe':>10}"
                  f"{'excess':>10}{'NW t':>8}{'median':>9}{'win dy':>9}"
                  f"{'n/day':>7}")
            for th in THRESHOLDS:
                ex, sel_n, wins = [], [], 0
                for d, v in dates.items():
                    sel = [r for pu, r in v if pu >= th]
                    if not sel:
                        continue
                    mkt = st.mean(r for _, r in v)
                    e = st.mean(sel) - mkt
                    ex.append(e)
                    sel_n.append(len(sel))
                    if e > 0:
                        wins += 1
                if len(ex) < 20:
                    print(f"    {'prob>='+str(th):<14}  fires on only "
                          f"{len(ex)} dates")
                    continue
                uni = st.mean(st.mean(r for _, r in v)
                              for d, v in dates.items())
                selmean = st.mean(ex) + uni
                t_ = nw_t(ex, max(1, h)) or 0.0
                print(f"    {'prob>='+str(th):<14}{100*selmean:>+9.3f}%"
                      f"{100*uni:>+9.3f}%{100*st.mean(ex):>+9.3f}pp"
                      f"{t_:>+8.2f}{100*st.median(ex):>+8.3f}pp"
                      f"{wins:>5}/{len(ex)}{st.mean(sel_n):>7.1f}")
            print()

    # concentration and the kill-switch boundary, on the honest (unfiltered) set
    print("=" * 74)
    print("CONCENTRATION and the KILL-SWITCH BOUNDARY (no sharpe filter)")
    print("=" * 74)
    for h in (3, 5):
        byd = defaultdict(list)
        for t, d, hh, pu, r in rows:
            if int(hh) == h:
                byd[d].append((pu, r))
        dates = {d: v for d, v in byd.items() if len(v) >= args.min_names}
        ex = {}
        for d, v in dates.items():
            sel = [r for pu, r in v if pu >= 0.70]
            if sel:
                ex[d] = st.mean(sel) - st.mean(r for _, r in v)
        if len(ex) < 20:
            continue
        vals = sorted(ex.values(), reverse=True)
        tot = sum(vals)
        top5 = sum(vals[:5])
        neg = sum(1 for x in vals if x < 0)
        print(f"  h={h}, prob>=0.70 — {len(vals)} dates")
        print(f"    top 5 dates carry {100*top5/tot:.0f}% of the total excess"
              if tot else "    total excess is zero")
        print(f"    {neg}/{len(vals)} dates negative "
              f"({100*neg/len(vals):.0f}%)")
        pre = [v for d, v in ex.items() if d < args.kill_date]
        post = [v for d, v in ex.items() if d >= args.kill_date]
        if len(pre) >= 10 and len(post) >= 10:
            print(f"    before {args.kill_date}: {100*st.mean(pre):+.3f}pp "
                  f"on {len(pre)} dates")
            print(f"    after  {args.kill_date}: {100*st.mean(post):+.3f}pp "
                  f"on {len(post)} dates")
        print()

    print("  The DAY-WEIGHTED excess against the universe is the number that")
    print("  matters. PCT7 showed +1.46% pooled and -1.97% day-weighted on this")
    print("  same data, and the difference is entirely the weighting.\n")
    print("  Read the median beside the mean: PCT7's whole result came from 6")
    print("  days of 64 while 38 lost money, which the mean concealed.\n")
    print("  Bar is NW t > 3.0 per Harvey, Liu & Zhu, and this is a filter")
    print("  chosen AFTER seeing the pooled table -- a selection effect the")
    print("  hurdle exists to absorb.")


if __name__ == "__main__":
    main()
