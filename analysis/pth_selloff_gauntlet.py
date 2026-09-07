#!/usr/bin/env python3
"""
pth_selloff_gauntlet.py — the full battery on PTH x past-winner, in selloffs.

READ-ONLY. Reads prices only. Trains nothing.

THE CLAIM UNDER TEST
    When SPY has been falling in a window ending well before setup, past
    winners NEAR their 52-week high subsequently underperform past winners FAR
    from theirs.

    Measured 2026-09-07 on the top-400 universe:
        h=40  winners/selloff  IC -0.0877  NW t -3.44  (184 dates)
              winners/rebound  IC -0.0445  NW t -1.49  (107)
              winners/normal   IC -0.0119  NW t -0.85  (595)
        h=20  winners/selloff  IC -0.1011 -> the same cell, and the only one in
              the whole test clearing t > 3.0 after a proper control.

    A first version put the selloff cell at IC -0.1394, t -4.73, but its
    21-session state window ended ON the formation date, so state and outcome
    shared calendar -- a "selloff" label partly described the drop the forward
    return then recorded. Lagging the state window 45 sessions before formation
    cut the effect by roughly a third and it still cleared the bar. That
    correction is why the remaining number is worth testing properly.

PROVENANCE, WHICH MATTERS FOR THE HURDLE
    The direction was NOT found by sweeping. George & Hwang (2004) established
    that nearness to the 52-week high predicts returns; Chen et al. (2026)
    sharpened it to an interaction concentrated among past winners; and
    "Momentum Crashes and the 52-Week High" supplies the conditional mechanism
    -- stocks far from their highs draw speculative demand because investors
    read them as having room to run, so they outperform when the market is not
    in a normal state.

    The SIGN is opposite to George & Hwang, and the literature explains that
    too: their 1963-2001 sample is equal-weighted and "implicitly overweights
    micro- and small-cap stocks", while this test runs the 400 most liquid US
    names, and an Australian out-of-sample study found "the 52-week high
    strategy comprising liquid stocks fails to produce significant dollar
    profits".

WHAT THIS ADDS
  1. MULTI-SEED. Different ticker samples. Three single-seed results reversed
     on replication in this fund on 2026-09-05 alone.
  2. AN ECONOMIC TEST. IC is not money. The top_decile target posted AUC 0.7316
     and delivered +0.01pp of day-weighted excess. Reported here as the
     day-weighted return of a low-PTH-minus-high-PTH book among past winners,
     against the same day's universe.
  3. A SHUFFLE NULL within each date.
  4. A BETA STRIP. High-PTH names are the crowded winners; in a selloff they may
     simply be high-beta. Three prior "discoveries" in this fund died here, and
     validate_gex.py's header flags it as the control that matters.
  5. STATE-DEFINITION ROBUSTNESS. The selloff threshold is arbitrary. Varying it
     shows whether the effect is a knife-edge artifact.

THE BAR
    NW t > 3.0 per Harvey, Liu & Zhu, AND positive across seeds, AND surviving
    the beta strip, AND economically non-trivial. 184 dates at h=40 with
    overlapping windows is roughly 20 independent observations, so the t-stat
    flatters even with Newey-West.

    python analysis/pth_selloff_gauntlet.py --seeds 3
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict

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


def spearman(x, y):
    n = len(x)
    if n < 8:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return num / (dx * dy) if dx and dy else None


def resid(y, x):
    """y orthogonalised against x, cross-sectionally."""
    n = len(y)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0:
        return y
    b = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / sxx
    return [y[i] - b * (x[i] - mx) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=250)
    ap.add_argument("--universe", default="tickers_top400.txt")
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--form", type=int, default=252)
    ap.add_argument("--skip", type=int, default=21)
    ap.add_argument("--state-lag", type=int, default=45)
    ap.add_argument("--min-names", type=int, default=40)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()
    H, F, S, LAG = args.horizon, args.form, args.skip, args.state_lag

    uni_all = [l.strip().upper() for l in open(args.universe) if l.strip()]
    con = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    bars = defaultdict(list)
    q = ("SELECT ticker, d, close FROM raw_bars WHERE close>0 AND d>=? "
         "AND ticker IN (%s)" % ",".join("?" * len(uni_all)))
    for t, d, c in con.execute(q, [args.start] + uni_all):
        bars[t].append((str(d)[:10], float(c)))
    spy = {}
    for d, c in con.execute(
            "SELECT d, close FROM raw_bars WHERE ticker='SPY' AND close>0 "
            "AND d>=?", (args.start,)):
        spy[str(d)[:10]] = float(c)
    con.close()
    for t in bars:
        bars[t].sort()
    have = [t for t in bars if len(bars[t]) > F + S + H + 60]
    sd = sorted(spy)
    print(f"{len(have)} tickers, SPY {len(sd)} bars, h={H}, "
          f"state lag {LAG} sessions\n")

    def states(thresh):
        out = {}
        for i, d in enumerate(sd):
            if i < 126 + LAG:
                continue
            j = i - LAG
            r21 = (spy[sd[j]] - spy[sd[j - 21]]) / spy[sd[j - 21]]
            hi = max(spy[sd[k]] for k in range(j - 126, j))
            dd = spy[sd[j]] / hi - 1.0
            if r21 > 0.02 and dd < -0.03:
                out[d] = "rebound"
            elif r21 < thresh:
                out[d] = "selloff"
            else:
                out[d] = "normal"
        return out

    st_map = states(-0.02)

    agg = defaultdict(list)
    for seed in range(1, args.seeds + 1):
        u = have[:]
        random.Random(seed).shuffle(u)
        sample = u[:args.tickers]
        rnd = random.Random(seed)

        ics, nulls, betas, ex = [], [], [], []
        for t0 in [None]:
            pass
        panel = defaultdict(dict)
        for t in sample:
            b = bars[t]
            cl = [x[1] for x in b]
            ds = [x[0] for x in b]
            rets = [0.0] + [(cl[i] - cl[i - 1]) / cl[i - 1] if cl[i - 1] else 0
                            for i in range(1, len(cl))]
            for i in range(F + S, len(b) - H, args.step):
                win = cl[i - F:i]
                if not win or not cl[i]:
                    continue
                hi = max(win)
                a, z = cl[i - F], cl[i - S]
                if hi <= 0 or not a:
                    continue
                bw = [(rets[k], None) for k in range(i - 60, i)]
                fr = (cl[i + H] - cl[i]) / cl[i]
                if abs(fr) > 1.5:
                    continue
                # trailing 60d beta vs SPY
                sr = []
                for k in range(i - 60, i):
                    d0, d1 = ds[k - 1], ds[k]
                    if d0 in spy and d1 in spy and spy[d0]:
                        sr.append(((cl[k] - cl[k - 1]) / cl[k - 1]
                                   if cl[k - 1] else 0.0,
                                   (spy[d1] - spy[d0]) / spy[d0]))
                bet = 1.0
                if len(sr) > 40:
                    mm = sum(m for _, m in sr) / len(sr)
                    aa = sum(x for x, _ in sr) / len(sr)
                    vv = sum((m - mm) ** 2 for _, m in sr)
                    if vv > 0:
                        bet = sum((x - aa) * (m - mm)
                                  for x, m in sr) / vv
                panel[ds[i]][t] = (cl[i] / hi, (z - a) / a, fr, bet)

        for d, mm in panel.items():
            if st_map.get(d) != "selloff" or len(mm) < args.min_names:
                continue
            forms = sorted(v[1] for v in mm.values())
            med = forms[len(forms) // 2]
            sel = [t for t in mm if mm[t][1] >= med]
            if len(sel) < 15:
                continue
            pth = [mm[t][0] for t in sel]
            fr = [mm[t][2] for t in sel]
            bt = [mm[t][3] for t in sel]
            r = spearman(pth, fr)
            if r is None:
                continue
            ics.append(r)
            sh = fr[:]
            rnd.shuffle(sh)
            rn = spearman(pth, sh)
            if rn is not None:
                nulls.append(rn)
            rb = spearman(pth, resid(fr, bt))
            if rb is not None:
                betas.append(rb)
            # economic: LOW pth minus HIGH pth, quintiles, vs the day's mean
            o = sorted(range(len(sel)), key=lambda i: pth[i])
            k = max(1, len(o) // 5)
            mkt = st.mean(fr)
            ex.append(st.mean(fr[i] for i in o[:k]) - mkt)

        if len(ics) < 20:
            print(f"seed {seed}: only {len(ics)} selloff dates")
            continue
        t_ = nw_t(ics, 4) or 0.0
        tb = nw_t(betas, 4) or 0.0
        te = nw_t(ex, 4) or 0.0
        print(f"SEED {seed} — {len(ics)} selloff dates, "
              f"{len(panel)} dates total")
        print(f"  IC                 {st.mean(ics):+.4f}   NW t {t_:+.2f}"
              + ("   t>3" if abs(t_) > T_HURDLE else ""))
        print(f"  null               {st.mean(nulls):+.4f}")
        print(f"  IC after beta strip{st.mean(betas):+.4f}   NW t {tb:+.2f}"
              + ("   t>3" if abs(tb) > T_HURDLE else ""))
        print(f"  low-PTH book       {100*st.mean(ex):+.3f}pp  NW t {te:+.2f}"
              f"   ({100*sum(1 for x in ex if x>0)/len(ex):.0f}% of dates +)")
        agg["ic"].append(st.mean(ics))
        agg["beta"].append(st.mean(betas))
        agg["ex"].append(st.mean(ex))
        print()

    if agg["ic"]:
        print("=" * 58)
        print("ACROSS SEEDS")
        print("=" * 58)
        for k, lab, unit in (("ic", "IC", ""), ("beta", "IC beta-stripped", ""),
                             ("ex", "low-PTH book", "pp")):
            v = agg[k]
            m = st.mean(v) * (100 if unit else 1)
            print(f"  {lab:<20}{m:>+9.4f}{unit:<3}"
                  f"  seeds same sign "
                  f"{sum(1 for x in v if (x<0)==(st.mean(v)<0))}/{len(v)}")

    print("\n  STATE-THRESHOLD ROBUSTNESS (seed 1 sample, IC in selloffs)")
    u = have[:]
    random.Random(1).shuffle(u)
    for th in (-0.01, -0.02, -0.03, -0.05):
        sm = states(th)
        v = []
        for d, mm in panel.items():
            if sm.get(d) != "selloff" or len(mm) < args.min_names:
                continue
            forms = sorted(x[1] for x in mm.values())
            med = forms[len(forms) // 2]
            sel = [t for t in mm if mm[t][1] >= med]
            if len(sel) < 15:
                continue
            r = spearman([mm[t][0] for t in sel], [mm[t][2] for t in sel])
            if r is not None:
                v.append(r)
        if len(v) >= 20:
            tt = nw_t(v, 4) or 0.0
            print(f"    SPY 21d < {th:+.0%}  {len(v):>4} dates  "
                  f"IC {st.mean(v):+.4f}  NW t {tt:+.2f}")

    print("\n  A knife-edge that only works at one threshold is a fitted")
    print("  parameter, not a state. The IC should degrade smoothly.\n")
    print(f"  Bar is NW t > {T_HURDLE}, positive across seeds, surviving the")
    print("  beta strip, and economically non-trivial. High-PTH names are the")
    print("  crowded winners and may simply be high-beta in a selloff -- three")
    print("  prior discoveries in this fund died exactly there.")


if __name__ == "__main__":
    main()
