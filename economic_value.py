#!/usr/bin/env python3
"""
economic_value.py — does the daily model's ordering survive costs?

READ-ONLY. Standard library only.

    python economic_value.py > econ_$(date +%Y%m%d).txt 2>&1

WHY THIS EXISTS
    pooled_accuracy.py established that prob_up orders outcomes: AUC 0.5155 to
    0.5261 at n~26k, CIs clear of 0.5, shuffle null clean at 0.500. That is
    statistically real. It says nothing about whether it is worth trading.

    AUC is rank-based and scale-free. A model can rank correctly while the
    return difference between its confident and unconfident names is smaller
    than the spread you cross to act on it. This script converts the ordering
    into basis points and applies a cost ladder.

METHOD
    For each horizon, sort predictions by prob_up, split into deciles, and
    measure the MEAN REALIZED RETURN per decile (outcomes.actual_return is
    already there). The long-short spread is decile 10 minus decile 1. That is
    the quantity a trade would actually capture.

    Costs are applied as a ladder rather than a single assumption, because the
    honest answer depends on execution: 5bps is optimistic for this universe,
    10bps is the project's own gauntlet standard, 20bps is conservative.

    A long-short spread pays costs on BOTH legs, so the round-trip charge is
    doubled. A long-only variant is reported too, since the fund's one validated
    brick is long-only and the recorded finding is that long-only construction
    converts cross-sectional signal into a market bet -- so the long-only column
    is shown net of the universe mean, not raw.

STATISTICS
    Returns overlap: an h=5 prediction made today shares four days with
    tomorrow's. Overlapping windows inflate t-statistics badly, so the daily
    spread series gets a Newey-West correction with lag = horizon - 1. The naive
    t is printed alongside so the size of the inflation is visible.

    The shuffle null is mandatory, as in pooled_accuracy.py: prob_up is permuted
    within each date and the spread recomputed. If the shuffled spread is not
    ~0, the decile construction is leaking.

WHAT THIS CANNOT ANSWER
    It measures the spread available in-sample on predictions already made. It
    is not a backtest: no position sizing, no capacity, no borrow, no slippage
    model, no survivorship handling beyond whatever the prediction universe
    already had. Treat a positive result as "worth testing properly", never as
    a strategy result.
"""
import math
import random
import sqlite3
from collections import defaultdict

DB = "accuracy.db"
COST_LADDER_BPS = [0, 5, 10, 20]
N_PERMUTATIONS = 200
N_DECILES = 10


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def newey_west_t(series, lag):
    """t-stat of the mean with a Newey-West HAC correction.

    Overlapping horizons make consecutive observations dependent; the naive t
    treats them as independent and overstates significance, often by a lot.
    """
    n = len(series)
    if n < 3:
        return None, None
    m = mean(series)
    dev = [x - m for x in series]
    gamma0 = sum(d * d for d in dev) / n
    var = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = sum(dev[i] * dev[i - L] for i in range(L, n)) / n
        var += 2.0 * w * cov
    if var <= 0:
        return None, None
    se_nw = math.sqrt(var / n)
    se_naive = stdev(series) / math.sqrt(n)
    return (m / se_nw if se_nw else None,
            m / se_naive if se_naive else None)


def winsorize_by_date(rows, pct=0.01):
    """Clip returns at the 1st/99th percentile WITHIN each date.

    Found 2026-08-30: raw decile means are dominated by extreme values -- one
    decile showed +1.29% mean daily return against ~0.10% for its neighbours,
    and deciles were non-monotonic. That is the signature of unadjusted splits
    or bad prices in outcomes.actual_return, not of signal. Clipping within date
    keeps the cross-sectional comparison intact while stopping a handful of rows
    from setting the mean.

    Clipping is reported, never silent: the caller prints how many rows moved.
    """
    by_date = defaultdict(list)
    for d, p, r in rows:
        by_date[d].append((p, r))
    out, clipped = [], 0
    for d, day in by_date.items():
        rets = sorted(r for _, r in day)
        n = len(rets)
        if n < 20:
            out.extend((d, p, r) for p, r in day)
            continue
        lo = rets[int(pct * n)]
        hi = rets[int((1 - pct) * n)]
        for p, r in day:
            r2 = min(max(r, lo), hi)
            if r2 != r:
                clipped += 1
            out.append((d, p, r2))
    return out, clipped


def median(xs):
    if not xs:
        return 0.0
    v = sorted(xs)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def deciles_by_date(rows, n_dec=N_DECILES):
    """rows: [(date, prob, ret)] -> per-date decile assignment.

    Ranking WITHIN each date is essential: ranking pooled across dates would let
    a market-wide up day populate the top decile and manufacture a spread that
    is really just market beta.
    """
    by_date = defaultdict(list)
    for d, p, r in rows:
        by_date[d].append((p, r))
    out = defaultdict(list)          # decile -> [(date, ret)]
    daily_spread = []                # [(date, top_mean - bottom_mean)]
    daily_long = []                  # [(date, top_mean - universe_mean)]
    for d in sorted(by_date):
        day = sorted(by_date[d])
        n = len(day)
        if n < n_dec * 2:
            continue                 # too thin to decile meaningfully
        size = n / n_dec
        buckets = []
        for k in range(n_dec):
            lo = int(round(k * size))
            hi = int(round((k + 1) * size)) if k < n_dec - 1 else n
            chunk = day[lo:hi]
            buckets.append(chunk)
            for _, r in chunk:
                out[k].append((d, r))
        top = mean([r for _, r in buckets[-1]])
        bot = mean([r for _, r in buckets[0]])
        uni = mean([r for _, r in day])
        daily_spread.append((d, top - bot))
        daily_long.append((d, top - uni))
    return out, daily_spread, daily_long


def shuffle_null_spread(rows, n_perm=N_PERMUTATIONS, seed=11):
    """Permute prob_up WITHIN each date; the spread must collapse to ~0."""
    rnd = random.Random(seed)
    by_date = defaultdict(list)
    for d, p, r in rows:
        by_date[d].append((p, r))
    spreads = []
    for _ in range(n_perm):
        shuffled = []
        for d, day in by_date.items():
            probs = [p for p, _ in day]
            rets = [r for _, r in day]
            rnd.shuffle(probs)
            shuffled.extend((d, p, r) for p, r in zip(probs, rets))
        _, ds, _ = deciles_by_date(shuffled)
        if ds:
            spreads.append(mean([s for _, s in ds]))
    if not spreads:
        return None, None, None
    spreads.sort()
    return (mean(spreads), spreads[int(0.025 * len(spreads))],
            spreads[int(0.975 * len(spreads))])


def report_horizon(h, rows, winsorized=False, clipped=0):
    tag = "WINSORIZED 1/99 within date" if winsorized else "RAW"
    print(f"\n{'-' * 78}\nHORIZON {h}d   n={len(rows)}   "
          f"dates={len(set(r[0] for r in rows))}   [{tag}]")
    if winsorized:
        print(f"  {clipped} rows clipped ({100*clipped/max(len(rows),1):.2f}%)")
    else:
        rets = [r for _, _, r in rows]
        ext = sum(1 for r in rets if abs(r) > 0.5)
        print(f"  return range [{min(rets)*100:+.1f}%, {max(rets)*100:+.1f}%]   "
              f"|ret|>50%: {ext} rows")
    dec, daily_spread, daily_long = deciles_by_date(rows)
    if not daily_spread:
        print("  too few names per date to decile")
        return

    print(f"\n  {'decile':>7} {'n':>7} {'mean ret':>10} {'median':>10}")
    for k in sorted(dec):
        rets = [r for _, r in dec[k]]
        print(f"  {k+1:>7} {len(rets):>7} {mean(rets)*100:>9.4f}% "
              f"{median(rets)*100:>9.4f}%")
    print("  (median is the robust read: a mean that disagrees with its median "
          "is being set by outliers)")

    sp = [s for _, s in daily_spread]
    lo = [s for _, s in daily_long]
    t_nw, t_naive = newey_west_t(sp, lag=max(h - 1, 1))
    lt_nw, lt_naive = newey_west_t(lo, lag=max(h - 1, 1))

    print(f"\n  LONG-SHORT (decile 10 - decile 1), per {h}d holding period")
    print(f"    mean spread      {mean(sp)*100:+.4f}%   over {len(sp)} rebalance dates")
    print(f"    t-stat  NW({max(h-1,1)})   {t_nw:+.2f}" if t_nw else "    t-stat NW: n/a")
    print(f"    t-stat  naive    {t_naive:+.2f}   "
          f"(inflation factor {abs(t_naive/t_nw):.2f}x)"
          if t_nw and t_naive else "")

    print(f"\n  LONG-ONLY vs universe mean (decile 10 - all names)")
    print(f"    mean excess      {mean(lo)*100:+.4f}%")
    print(f"    t-stat  NW({max(h-1,1)})   {lt_nw:+.2f}" if lt_nw else "")

    print(f"\n  COST LADDER — long-short pays the round trip on BOTH legs")
    print(f"    {'cost/leg':>10} {'net/period':>12} {'net ann.':>10}  verdict")
    for c in COST_LADDER_BPS:
        net = mean(sp) - 2 * c / 10000.0
        ann = ((1 + net) ** (252 / h) - 1) * 100 if abs(net) < 0.5 else float('nan')
        verdict = "positive" if net > 0 else "NEGATIVE"
        print(f"    {c:>8}bps {net*100:>11.4f}% {ann:>9.1f}%  {verdict}")

    pos = sum(1 for s_ in sp if s_ > 0)
    print(f"\n  SIGN TEST: {pos}/{len(sp)} dates positive ({100*pos/len(sp):.1f}%)"
          f"   -- a real edge shows up in the hit rate, not just the mean")

    m, nlo, nhi = shuffle_null_spread(rows)
    if m is not None:
        obs = mean(sp)
        # CORRECT TEST: is the OBSERVED spread outside the null's 95% band?
        # An earlier version compared |null| to a fraction of |observed|, which
        # divides by a near-zero denominator whenever the real spread is ~0 and
        # therefore flagged a perfectly good null as broken.
        inside = nlo <= obs <= nhi
        print(f"\n  SHUFFLE NULL (prob permuted within date)")
        print(f"    null mean {m*100:+.4f}%   null 95% "
              f"[{nlo*100:+.4f}%, {nhi*100:+.4f}%]")
        print(f"    observed  {obs*100:+.4f}%   -> "
              + ("INSIDE the null band: NOT DISTINGUISHABLE FROM CHANCE"
                 if inside else
                 "OUTSIDE the null band: distinguishable"))


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    print("ECONOMIC VALUE OF THE DAILY MODEL'S ORDERING")
    print("AUC said the ordering is real. This asks what it is worth.\n")
    print("Not a backtest: no sizing, capacity, borrow or slippage model.")
    print("A positive result means 'worth testing properly', not 'strategy'.")

    rows_all = con.execute("""
        SELECT p.horizon, p.prediction_date, p.prob_up, o.actual_return
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.prob_up IS NOT NULL AND o.actual_return IS NOT NULL
    """).fetchall()
    con.close()

    by_h = defaultdict(list)
    for h, d, p, r in rows_all:
        by_h[h].append((d, p, r))
    for h in sorted(by_h):
        report_horizon(h, by_h[h])
        w, clipped = winsorize_by_date(by_h[h])
        report_horizon(h, w, winsorized=True, clipped=clipped)

    print("\n\nEND")


if __name__ == "__main__":
    main()
