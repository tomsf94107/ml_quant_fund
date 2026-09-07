#!/usr/bin/env python3
"""
pth_winner_test.py — 52-week high, conditional on being a past winner.

READ-ONLY. Trains nothing. Reads prices only.

THE PUBLISHED CLAIM
    George & Hwang (2004) find that "nearness to the 52-week high dominates and
    improves upon the forecasting power of past returns (both individual and
    industry returns) for future returns", and that high price-to-52-week-high
    stocks earn high future returns. Subsequent work reports the q-factor model
    captures the anomaly and that PTH's relations with future profitability and
    future investment growth are both significantly positive.

    A 2026 refinement (Chen et al., Journal of Banking and Finance) sharpens it
    into an INTERACTION rather than a standalone: "the stock returns for
    past-winner stocks with a high PTA strongly outperform past-winner stocks
    with a low PTA", and the patterns are "stronger during better sentiment
    times".

    So the testable statement is not "high PTH predicts returns" -- it is "among
    past winners, high PTH separates the ones that keep working."

WHY THIS IS WORTH A LOOK HERE
    high_52w_ratio and low_52w_ratio are already in the feature set. But
    tonight's leave-one-out put the [52week] group at +0.437pp -- meaning the
    model did BETTER without it -- on 1 of 2 seeds, which is noise but is also
    the opposite of what the literature predicts. Either the feature is being
    used wrongly, or the effect is absent here, or the univariate form is the
    wrong construction.

    The interaction has never been tested in this system. return_20d and
    return_60d supply the winner/loser split; high_52w_ratio supplies the anchor.

METHOD
    Per date, cross-sectionally, on the top-400 universe:
      1. split names into past-winner and past-loser halves by trailing return
      2. within EACH half, Spearman(pth, forward return)
      3. and the plain long-short: high-PTH winners minus low-PTH winners

    Reported at h=20 and h=40 -- this fund's own horizon sweep found the feature
    set carries 40-day information and essentially nothing at 5, and the
    momentum literature works at monthly frequency, so a 1-day test would be
    testing the wrong horizon for both reasons.

    Newey-West on the per-date series, and a within-date shuffle null.

    NOTE the formation window uses a SKIP. The standard construction is t-12 to
    t-2 because "the one-month skip between the formation period and the holding
    period avoids contamination from short-term reversal" (Jegadeesh 1990 finds
    the prior month reverses at roughly 2.49%/month). Skipping is not optional;
    without it the winner split is measuring last month's bounce.

THE BAR
    NW t > 3.0 per Harvey, Liu & Zhu. This is a published effect, so the
    direction is predicted rather than searched -- better provenance than a
    swept result -- but the magnitude still has to clear the noise.

    python analysis/pth_winner_test.py
"""
import argparse
import math
import sqlite3
import statistics as st
from collections import defaultdict

HORIZONS = (20, 40)
T_HURDLE = 3.0


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
    ap.add_argument("--universe", default="tickers_top400.txt")
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--form", type=int, default=252,
                    help="formation lookback in sessions (12 months)")
    ap.add_argument("--skip", type=int, default=21,
                    help="sessions skipped between formation and holding")
    ap.add_argument("--min-names", type=int, default=40)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--state-lag", type=int, default=45,
                    help="sessions between the end of the state window and the "
                         "formation date; must exceed the forward horizon so "
                         "state and outcome do not share calendar")
    args = ap.parse_args()

    uni = [l.strip().upper() for l in open(args.universe) if l.strip()]
    con = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    q = ("SELECT ticker, d, close FROM raw_bars WHERE close > 0 AND d >= ? "
         "AND ticker IN (%s)" % ",".join("?" * len(uni)))
    bars = defaultdict(list)
    for t, d, c in con.execute(q, [args.start] + uni):
        bars[t].append((str(d)[:10], float(c)))
    con.close()
    for t in bars:
        bars[t].sort()
    have = [t for t in bars if len(bars[t]) > args.form + max(HORIZONS) + 40]
    print(f"{len(have)} of {len(uni)} names with enough history\n")

    # date -> ticker -> (pth, formation return)
    panel = defaultdict(dict)
    fwd = {h: defaultdict(dict) for h in HORIZONS}
    F, S = args.form, args.skip
    for t in have:
        b = bars[t]
        cl = [x[1] for x in b]
        ds = [x[0] for x in b]
        for i in range(F + S, len(b) - max(HORIZONS), args.step):
            win = cl[i - F: i]
            if not win:
                continue
            hi = max(win)
            if hi <= 0 or not cl[i]:
                continue
            pth = cl[i] / hi
            # formation return with a SKIP: t-F to t-S, not t-F to t
            a, z = cl[i - F], cl[i - S]
            if not a:
                continue
            form = (z - a) / a
            panel[ds[i]][t] = (pth, form)
            for h in HORIZONS:
                if i + h < len(cl) and cl[i]:
                    r = (cl[i + h] - cl[i]) / cl[i]
                    if abs(r) < 1.5:
                        fwd[h][ds[i]][t] = r

    # Market state, for the rebound split. "Momentum Crashes and the 52-Week
    # High" argues stocks FAR from their 52-week highs draw speculative demand
    # in rebounds -- investors read them as having room to run -- and surge in
    # the month of a market rebound. That is a mechanism for a NEGATIVE PTH IC,
    # and this sample (2016-2026) contains the COVID crash and recovery, the
    # 2022 bear and the 2023 rebound. If the negative IC concentrates in
    # rebounds and flattens elsewhere, the mechanism is confirmed and the result
    # stops being anomalous.
    con = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    spy = {}
    for d, c in con.execute(
            "SELECT d, close FROM raw_bars WHERE ticker='SPY' AND close>0 "
            "AND d >= ?", (args.start,)):
        spy[str(d)[:10]] = float(c)
    con.close()
    sd = sorted(spy)
    state = {}
    LAG = args.state_lag
    for i, d in enumerate(sd):
        if i < 126 + LAG:
            continue
        # STRICTLY BACKWARD-LOOKING. A first version classified the state from
        # SPY's trailing 21 sessions ENDING ON THE FORMATION DATE, which at h=20
        # and h=40 overlaps the forward window being predicted -- the state and
        # the outcome were measured over the same calendar. Not a look-ahead in
        # the classic sense, since only past prices are used, but the two are
        # not independent and a "selloff" label partly describes the very drop
        # the forward return then records.
        #
        # Here the window ENDS `--state-lag` sessions BEFORE the formation date,
        # so the state is fully knowable and fully disjoint from the outcome.
        # The first version put winners/selloff at IC -0.1394, NW t -4.73 at
        # h=40, the only cell in the test clearing t>3. Whether it survives this
        # is the whole question.
        j = i - LAG
        r21 = (spy[sd[j]] - spy[sd[j - 21]]) / spy[sd[j - 21]]
        hi = max(spy[sd[k]] for k in range(j - 126, j))
        dd = spy[sd[j]] / hi - 1.0
        if r21 > 0.02 and dd < -0.03:
            state[d] = "rebound"
        elif r21 < -0.02:
            state[d] = "selloff"
        else:
            state[d] = "normal"
    from collections import Counter
    print("  market state days: " + ", ".join(
        f"{k} {v}" for k, v in Counter(state.values()).most_common()) + "\n")

    import random
    rnd = random.Random(args.seed)
    print(f"  {'h':>3}  {'group':<16}{'dates':>7}{'IC':>10}{'NW t':>8}"
          f"{'null':>9}{'Q5-Q1':>11}")
    for h in HORIZONS:
        res = defaultdict(list)
        spread = defaultdict(list)
        nulls = defaultdict(list)
        for d, mm in panel.items():
            fr = fwd[h].get(d, {})
            names = [t for t in mm if t in fr]
            if len(names) < args.min_names:
                continue
            forms = sorted(mm[t][1] for t in names)
            med = forms[len(forms) // 2]
            for lab, sel in (
                    ("all", names),
                    ("past winners", [t for t in names if mm[t][1] >= med]),
                    ("past losers", [t for t in names if mm[t][1] < med])):
                if len(sel) < 15:
                    continue
                xs = [mm[t][0] for t in sel]
                ys = [fr[t] for t in sel]
                r = spearman(xs, ys)
                if r is not None:
                    res[lab].append(r)
                sh = ys[:]
                rnd.shuffle(sh)
                rn = spearman(xs, sh)
                if rn is not None:
                    nulls[lab].append(rn)
                o = sorted(range(len(sel)), key=lambda i: xs[i])
                k = max(1, len(o) // 5)
                spread[lab].append(st.mean(ys[i] for i in o[-k:])
                                   - st.mean(ys[i] for i in o[:k]))
        for lab in ("all", "past winners", "past losers"):
            v = res.get(lab, [])
            if len(v) < 20:
                continue
            t_ = nw_t(v, max(1, h // 10)) or 0.0
            print(f"  {h:>3}  {lab:<16}{len(v):>7}{st.mean(v):>+9.4f}"
                  f"{t_:>+8.2f}{st.mean(nulls[lab]):>+9.4f}"
                  f"{100*st.mean(spread[lab]):>+10.3f}pp"
                  + ("   PASSES t>3" if abs(t_) > T_HURDLE else ""))
        # the rebound split, on past winners only -- the cell the interaction
        # says should be strongest
        bys = defaultdict(list)
        for d, mm in panel.items():
            fr = fwd[h].get(d, {})
            names = [t for t in mm if t in fr]
            if len(names) < args.min_names:
                continue
            forms = sorted(mm[t][1] for t in names)
            med = forms[len(forms) // 2]
            sel = [t for t in names if mm[t][1] >= med]
            if len(sel) < 15:
                continue
            r = spearman([mm[t][0] for t in sel], [fr[t] for t in sel])
            if r is not None:
                bys[state.get(d, "normal")].append(r)
        for lab in ("rebound", "normal", "selloff"):
            v = bys.get(lab, [])
            if len(v) < 15:
                continue
            t2 = nw_t(v, max(1, h // 10)) or 0.0
            print(f"  {h:>3}    winners/{lab:<8}{len(v):>7}{st.mean(v):>+9.4f}"
                  f"{t2:>+8.2f}")
        print()

    print("  George & Hwang: nearness to the 52-week high DOMINATES past")
    print("  returns as a predictor. Chen et al. (2026) sharpen it to an")
    print("  interaction -- high PTH among PAST WINNERS is where it works.")
    print("  So 'past winners' should read materially stronger than 'all'.\n")
    print("  Formation uses a 21-session SKIP. Jegadeesh (1990) finds the prior")
    print("  month reverses at roughly 2.49%/month, so without the skip the")
    print("  winner split measures last month's bounce rather than momentum.\n")
    print(f"  Bar is NW t > {T_HURDLE}. The direction is PREDICTED by published")
    print("  work rather than searched, which is better provenance than a swept")
    print("  result -- but the magnitude still has to clear the noise.")


if __name__ == "__main__":
    main()
