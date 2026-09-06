#!/usr/bin/env python3
"""
announcement_beta_test.py — does beta pay on announcement days? (Savor-Wilson)

READ-ONLY. Writes nothing. Uses no model.

THE PUBLISHED CLAIM
    Savor & Wilson (2014) show the security market line -- the cross-sectional
    relationship between beta and return -- has a significantly POSITIVE slope
    on macroeconomic announcement days, estimated at 6.81 basis points, against
    roughly flat on non-announcement days. Lucca & Moench (2015) find the effect
    concentrates in FOMC. Pooling NFP, ISM, GDP and FOMC, the average
    pre-announcement return is 5.66% annually.

    Ai & Bansal frame it as a risk premium: investors demand compensation for
    holding market exposure across a scheduled resolution of uncertainty, and
    high-beta names carry more of it.

WHY TEST THIS BEFORE BUILDING ANY FEATURE
    days_to_fomc is IDENTICAL for every ticker on a given date. A cross-
    sectional model that ranks 1,920 names against each other cannot use a
    constant. The only way a market-wide calendar enters a ranking model is
    through an interaction with something stock-specific -- and Savor-Wilson
    says that something is beta.

    So the whole case for days_to_event x beta_60d rests on this effect being
    present in THIS universe over THIS period. If beta does not pay on
    announcement days here, the interaction has no mechanism and building it
    would be fitting a feature to a story.

    That ordering matters. On 2026-09-05 a "short covering" mechanism was
    invented to explain a seed-1 dark-pool result that the other two seeds then
    reversed. Test the premise, then build.

WHAT IS MEASURED
    No model, no training. Per date, cross-sectionally:

      Spearman(beta_60d, forward return)

    then split those per-date ICs by whether the date was an announcement day,
    and by which announcement. Newey-West on the per-date series.

    Reported at h=1 (the horizon the literature measures), and at h=3, h=5 and
    h=40 to see how far it carries. Also reported: the plain return spread
    between the top and bottom beta quintile, which is the economically legible
    version of the same thing.

READING IT
    IC positive and larger on announcement days than other days -> replicates.
    The interaction has a mechanism and days_to_event x beta_60d is worth
    building.
    No difference -> the effect is absent here. Do not build the feature.
    NEGATIVE on announcement days -> something is inverted; check the beta sign
    and the date join before concluding anything.

    Bar is NW t > 3.0 per Harvey, Liu & Zhu on any NEW claim. Replicating a
    published result is a lower bar -- the direction was predicted in advance --
    but the magnitude should still be read against the noise.

    python analysis/announcement_beta_test.py
"""
import argparse
import math
import sqlite3
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

HORIZONS = (1, 3, 5, 40)
T_HURDLE = 3.0


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


def spearman(x, y):
    n = len(x)
    if n < 8:
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
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=300,
                    help="beta is cheap to compute, so use a wide sample")
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--beta-window", type=int, default=60)
    ap.add_argument("--min-names", type=int, default=40)
    ap.add_argument("--universe", default="tickers_expanded.txt")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    con = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    try:
        ev = defaultdict(set)
        for d, c in con.execute(
                "SELECT event_date, event_code FROM macro_events"):
            ev[str(d)[:10]].add(c)
    except Exception as e:
        con.close()
        raise SystemExit(f"macro_events unavailable ({e}) -- run "
                         f"analysis/etl_macro_calendar.py first")
    con.close()
    print(f"macro_events: {len(ev)} distinct dates, "
          f"{sum(len(v) for v in ev.values())} events")

    import random
    uni = [l.strip().upper() for l in open(args.universe) if l.strip()]
    random.Random(args.seed).shuffle(uni)
    uni = uni[:args.tickers]

    con = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    bars = defaultdict(list)
    q = ("SELECT ticker, d, close FROM raw_bars WHERE close > 0 AND d >= ? "
         "AND ticker IN (%s)" % ",".join("?" * len(uni)))
    for t, d, c in con.execute(q, [args.start] + uni):
        bars[t].append((str(d)[:10], float(c)))
    spy = {}
    for d, c in con.execute(
            "SELECT d, close FROM raw_bars WHERE ticker='SPY' AND close>0 "
            "AND d >= ?", (args.start,)):
        spy[str(d)[:10]] = float(c)
    con.close()
    for t in bars:
        bars[t].sort()
    print(f"prices: {len(bars)} tickers, SPY {len(spy)} bars\n")
    if len(spy) < 500:
        raise SystemExit("SPY history missing -- beta cannot be computed")

    sdates = sorted(spy)
    sret = {sdates[i]: (spy[sdates[i]] - spy[sdates[i-1]]) / spy[sdates[i-1]]
            for i in range(1, len(sdates)) if spy[sdates[i-1]]}

    # beta_60d per ticker per date, and forward returns at each horizon
    panel = defaultdict(dict)      # date -> ticker -> beta
    fwd = {h: defaultdict(dict) for h in HORIZONS}
    W = args.beta_window
    for t, b in bars.items():
        ds = [x[0] for x in b]
        cl = [x[1] for x in b]
        rets = [0.0] + [(cl[i] - cl[i-1]) / cl[i-1] if cl[i-1] else 0.0
                        for i in range(1, len(cl))]
        for i in range(W, len(ds)):
            win = [(rets[k], sret.get(ds[k])) for k in range(i - W, i)]
            win = [(a, m) for a, m in win if m is not None]
            if len(win) < W * 0.7:
                continue
            ma = sum(m for _, m in win) / len(win)
            aa = sum(a for a, _ in win) / len(win)
            cov = sum((a - aa) * (m - ma) for a, m in win) / len(win)
            var = sum((m - ma) ** 2 for _, m in win) / len(win)
            if var <= 0:
                continue
            panel[ds[i]][t] = cov / var
            for h in HORIZONS:
                if i + h < len(cl) and cl[i]:
                    r = (cl[i + h] - cl[i]) / cl[i]
                    if abs(r) < 1.5:
                        fwd[h][ds[i]][t] = r

    print(f"  {'horizon':<9}{'group':<14}{'dates':>7}{'IC':>10}{'NW t':>8}"
          f"{'Q5-Q1 ret':>12}")
    for h in HORIZONS:
        buckets = defaultdict(list)
        spreads = defaultdict(list)
        for d in sorted(panel):
            names = [t for t in panel[d] if t in fwd[h].get(d, {})]
            if len(names) < args.min_names:
                continue
            bs = [panel[d][t] for t in names]
            rs = [fwd[h][d][t] for t in names]
            ic = spearman(bs, rs)
            if ic is None:
                continue
            order = sorted(range(len(names)), key=lambda i: bs[i])
            k = max(1, len(order) // 5)
            sp = (st.mean(rs[i] for i in order[-k:])
                  - st.mean(rs[i] for i in order[:k]))
            codes = ev.get(d, set())
            key = "announcement" if codes else "other"
            buckets[key].append(ic)
            spreads[key].append(sp)
            if "FOMC" in codes:
                buckets["  FOMC"].append(ic); spreads["  FOMC"].append(sp)
            if "CPI" in codes:
                buckets["  CPI"].append(ic); spreads["  CPI"].append(sp)
            if "NFP" in codes:
                buckets["  NFP"].append(ic); spreads["  NFP"].append(sp)
        for key in ("announcement", "other", "  FOMC", "  CPI", "  NFP"):
            v = buckets.get(key, [])
            if len(v) < 10:
                continue
            t_ = nw_t(v, max(1, h // 2)) or 0.0
            print(f"  {'h='+str(h):<9}{key:<14}{len(v):>7}{st.mean(v):>+9.4f}"
                  f"{t_:>+8.2f}{100*st.mean(spreads[key]):>+11.3f}pp"
                  + ("  t>3" if abs(t_) > T_HURDLE else ""))
        print()

    print("  Savor & Wilson (2014) estimate the security market line slope at")
    print("  6.81bp on announcement days against roughly flat otherwise, and")
    print("  Lucca & Moench (2015) find the effect concentrates in FOMC.")
    print("  'announcement' IC materially above 'other' replicates that.\n")
    print("  If it does NOT replicate, days_to_event x beta_60d has no")
    print("  mechanism in this universe and should not be built. A market-wide")
    print("  countdown is constant across tickers on any date and cannot enter")
    print("  a cross-sectional ranking except through such an interaction.")


if __name__ == "__main__":
    main()
