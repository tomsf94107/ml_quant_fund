#!/usr/bin/env python3
"""
decile_monotonicity.py -- the metric the alpha gate is missing.

THE PROBLEM (2026-08-21)
  alpha_fitness scores every alpha on rank_ic, ic_t, sharpe, turnover, fitness.
  1,222 of 9,626 alphas pass |ic_t| > 3. The top 20 look like this:

    pc_ratio_snap__ts_decay_linear__w10  h=5  IC +0.0416  t +9.02  Sharpe -0.51

  Strongly significant IC, NEGATIVE Sharpe. That is not a contradiction -- it is
  a documented pattern. RankIC measures the FULL cross-section; Sharpe measures
  the EXTREME DECILES you actually trade. They can disagree, and when they do the
  diagnostic is DECILE MONOTONICITY: the Spearman correlation between decile index
  (1..10) and each decile's realised mean return.

    mono ~ +1  ranking is monotone -> IC converts to money, tradeable
    mono ~  0  IC lives in the middle of the book -> untradeable at the extremes
    mono ~ -1  the tails INVERT -> the signal is right on average and wrong where
               you trade it

  This matters here more than usual because the SAME signature appears three
  separate times in this system:
    - direction model: "inverted at the confident extremes", DOWN-calls right
      40% at h=5 (MASTER_TODO 1.1b / 1.2)
    - alpha panel:     IC +0.04 with Sharpe -0.51 (above)
    - momentum:        18yr backtest Sharpe +1.53, live shadow edge -10.95pp
  Three measurements, one candidate mechanism. Nothing currently tests for it.

WHAT IT DOES
  Ranks names into deciles by a signal on each date, computes each decile's mean
  forward return, then reports:
    - the decile return ladder (so you can SEE where it breaks)
    - monotonicity (Spearman decile-index vs decile-return)
    - top-minus-bottom spread, gross and net of cost
    - a per-date t-stat on the spread (Newey-West, lag = horizon)

  Signals come from accuracy.db (prob_up, prob_raw, ...) or any column you name;
  forward returns come from outcomes (h=1/3/5) so they match production exactly.

USAGE
  python scripts/decile_monotonicity.py --horizon 5
  python scripts/decile_monotonicity.py --horizon 3 --prob-col prob_raw --days 365
  python scripts/decile_monotonicity.py --horizon 5 --deciles 5 --cost-bps 10
"""
import argparse
import math
import os
import sqlite3
import sys
from collections import defaultdict

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))
ACC = os.path.join(ROOT, "accuracy.db")


def spearman(a, b):
    def rk(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    x, y = rk(a), rk(b)
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / math.sqrt(vx * vy)


def nw_t(xs, lag):
    n = len(xs)
    if n < 5:
        return None
    mu = sum(xs) / n
    e = [v - mu for v in xs]
    s = sum(v * v for v in e) / n
    for l in range(1, min(lag, n - 1) + 1):
        g = sum(e[t] * e[t + l] for t in range(n - l)) / n
        s += 2.0 * (1.0 - l / (lag + 1.0)) * g
    if s <= 0:
        return None
    return mu / math.sqrt(s / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--prob-col", default="prob_up")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--min-names", type=int, default=30)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--include-watchlist", action="store_true")
    ap.add_argument("--db")
    args = ap.parse_args()

    dbp = args.db or ACC
    if not os.path.isfile(dbp):
        sys.exit(f"FATAL: {dbp} not found")
    con = sqlite3.connect(dbp, timeout=30)
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
    if args.prob_col not in cols:
        sys.exit(f"FATAL: --prob-col '{args.prob_col}' not in predictions. "
                 f"Available: {', '.join(cols)}")
    wl = "" if args.include_watchlist or "is_watchlist" not in cols \
        else " AND COALESCE(p.is_watchlist,0)=0"

    rows = con.execute(
        f'SELECT p.prediction_date, p.ticker, p."{args.prob_col}", o.actual_return '
        f'FROM predictions p JOIN outcomes o '
        f'  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date '
        f'  AND p.horizon=o.horizon '
        f'WHERE p.horizon=? AND o.actual_return IS NOT NULL '
        f'  AND p."{args.prob_col}" IS NOT NULL' + wl +
        f"  AND p.prediction_date >= date('now', ?)",
        (args.horizon, f"-{args.days} days")).fetchall()
    con.close()
    if not rows:
        sys.exit("FATAL: no scored rows in range")

    by_date = defaultdict(list)
    for d, tk, p, r in rows:
        by_date[d].append((tk, float(p), float(r)))
    dates = sorted(d for d, v in by_date.items() if len(v) >= args.min_names)
    if len(dates) < 10:
        sys.exit(f"FATAL: only {len(dates)} usable dates")

    D = args.deciles
    bucket_rets = defaultdict(list)     # decile -> per-date mean return
    spreads = []
    for d in dates:
        recs = sorted(by_date[d], key=lambda x: x[1])   # ascending by signal
        n = len(recs)
        per = [[] for _ in range(D)]
        for i, (_tk, _p, r) in enumerate(recs):
            per[min(D - 1, i * D // n)].append(r)
        means = [sum(b) / len(b) if b else None for b in per]
        if means[0] is None or means[-1] is None:
            continue
        for k, m in enumerate(means):
            if m is not None:
                bucket_rets[k].append(m)
        spreads.append(means[-1] - means[0])

    print(f"# decile_monotonicity  signal='{args.prob_col}'  h={args.horizon}  "
          f"deciles={D}")
    print(f"# {len(dates)} dates, {len(rows)} scored rows, watchlist "
          f"{'INCLUDED' if args.include_watchlist else 'EXCLUDED'}")
    print()
    print(f"  {'decile':<8}{'mean ret %':>12}{'n dates':>10}   (1 = lowest signal)")
    print("  " + "-" * 42)
    dec_means = []
    for k in range(D):
        v = bucket_rets.get(k, [])
        if not v:
            continue
        m = sum(v) / len(v)
        dec_means.append(m)
        bar = "#" * min(40, int(abs(m) * 4000))
        print(f"  {k+1:<8}{m*100:>11.4f}%{len(v):>10}   {bar}")

    mono = spearman(list(range(1, len(dec_means) + 1)), dec_means)
    mu_s = sum(spreads) / len(spreads) if spreads else 0.0
    t_s = nw_t(spreads, args.horizon)
    # top-minus-bottom is a two-sided book: both legs turn over
    net = mu_s - 2 * args.cost_bps / 10000.0

    print()
    print(f"  MONOTONICITY (Spearman decile-index vs decile-return): "
          f"{mono:+.3f}" if mono is not None else "  MONOTONICITY: n/a")
    print(f"  TOP-BOTTOM spread : {mu_s*100:+.4f}% per {args.horizon}d   "
          f"NW-t {t_s:+.2f}" if t_s is not None else f"  TOP-BOTTOM: {mu_s*100:+.4f}%")
    print(f"  NET of {args.cost_bps:.0f}bps x2 : {net*100:+.4f}%")
    print()
    print("  HOW TO READ")
    print("    mono ~ +1  ranking is monotone -> IC converts to money at the tails")
    print("    mono ~  0  IC lives mid-book -> not tradeable where you trade")
    print("    mono ~ -1  TAILS INVERT -> right on average, wrong at the extremes")
    print("    A high |IC| with mono <= 0 is the exact pattern behind this system's")
    print("    'inverted at the confident extremes' finding. The ladder above shows")
    print("    WHERE it breaks -- look for the decile that does not fit the trend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
