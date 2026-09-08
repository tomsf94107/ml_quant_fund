#!/usr/bin/env python3
"""
pth_beta_quintile.py — the one test that settles the PTH axis.

READ-ONLY. Reads prices only. Trains nothing.

WHY THIS EXISTS
    pth_selloff_gauntlet.py passed four of five controls and died on the fifth:

        raw IC              -0.0991   3/3 seeds, seed 3 at NW t -3.64
        shuffle null        +0.0038   clean
        threshold sweep     smooth from -1% to -5%, no knife edge
        low-PTH book        +2.284pp  3/3 seeds, ~60% of dates positive
        BETA STRIP          -0.0389   -61%, per-seed t -1.60 and -2.33

    High-PTH names are the crowded winners, crowded winners are high beta, and
    high beta falls harder in a selloff. Three prior discoveries in this fund
    died at that control; PTH was the fourth.

    Post-hoc orthogonalisation removes only the LINEAR component of beta, and
    only in-sample on each date. This ranks PTH WITHIN beta buckets, so the
    comparison is high-PTH against low-PTH at the same beta, by construction
    rather than by regression. If the effect survives, it is not beta. If it
    collapses, it was beta all along and the axis closes.

CONSTRUCTION
    Identical to the gauntlet in every other respect -- same panel, same
    52-week-high ratio, same 21-session formation skip per Jegadeesh (1990),
    same state definition with the 45-session lag that keeps state and outcome
    disjoint, same trailing 60-day beta, same seeds, same step, same filters.
    Only the cross-sectional comparison changes. A reimplementation that
    changed anything else could not be set against -0.0991 and -0.0389.

    Per date, among past winners: sort by beta, cut into N equal buckets, and
    compute the PTH-vs-forward-return Spearman WITHIN each bucket. The date's
    IC is the n-weighted mean across buckets. The book is the bottom-k PTH
    names within each bucket measured against THAT BUCKET's own mean return,
    averaged equally across buckets -- so it is long-short at constant beta
    rather than long low-beta against a market that includes high-beta names.

COMPARABILITY
    Bucketing needs more names per date than the flat test, so some dates drop
    out. Raw IC and post-hoc beta-strip IC are therefore recomputed on the
    RETAINED dates and printed alongside. Comparing a bucketed number on 120
    dates against a flat number on 184 would confound the construction with
    the sample.

PRE-REGISTERED ACCEPTANCE — three outcomes, not two
    REAL          beta-neutral IC at NW t > 3.0, same sign in every seed, AND a
                  book that clears cost after the haircut. -> shadow book.
    BETA ALL ALONG effect collapses. -> close the axis.
    SIGNED BUT THIN sign holds but t < 3.0, or the book does not clear cost.
                  -> CLOSE THE AXIS. This is the likely outcome and the one the
                  source memo did not name. Raw was ~60% beta, so the neutral
                  book is plausibly half of +2.284pp or less, at a t already
                  marginal at +2.5 to +2.8. A consistently-signed residual is
                  not a promotion.

    python analysis/pth_beta_quintile.py --seeds 3
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict

T_HURDLE = 3.0
COST_BPS = 20.0          # round-trip, the bar mom_12_1 had to clear


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
    """y orthogonalised against x, cross-sectionally. The post-hoc control."""
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
    ap.add_argument("--buckets", type=int, default=5,
                    help="beta buckets. 5 = quintiles; 3 if dates are thin.")
    ap.add_argument("--min-per-bucket", type=int, default=12)
    args = ap.parse_args()
    H, F, S, LAG, B = (args.horizon, args.form, args.skip,
                       args.state_lag, args.buckets)

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
          f"state lag {LAG} sessions, {B} beta buckets\n")

    st_map = {}
    for i, d in enumerate(sd):
        if i < 126 + LAG:
            continue
        j = i - LAG
        r21 = (spy[sd[j]] - spy[sd[j - 21]]) / spy[sd[j - 21]]
        hi = max(spy[sd[k]] for k in range(j - 126, j))
        dd = spy[sd[j]] / hi - 1.0
        if r21 > 0.02 and dd < -0.03:
            st_map[d] = "rebound"
        elif r21 < -0.02:
            st_map[d] = "selloff"
        else:
            st_map[d] = "normal"

    agg = defaultdict(list)
    for seed in range(1, args.seeds + 1):
        u = have[:]
        random.Random(seed).shuffle(u)
        sample = u[:args.tickers]
        rnd = random.Random(seed)

        panel = defaultdict(dict)
        for t in sample:
            b = bars[t]
            cl = [x[1] for x in b]
            ds = [x[0] for x in b]
            for i in range(F + S, len(b) - H, args.step):
                win = cl[i - F:i]
                if not win or not cl[i]:
                    continue
                hi = max(win)
                a, z = cl[i - F], cl[i - S]
                if hi <= 0 or not a:
                    continue
                fr = (cl[i + H] - cl[i]) / cl[i]
                if abs(fr) > 1.5:
                    continue
                sr = []
                for k in range(i - 60, i):
                    d0, d1 = ds[k - 1], ds[k]
                    if d0 in spy and d1 in spy and spy[d0]:
                        sr.append(((cl[k] - cl[k - 1]) / cl[k - 1]
                                   if cl[k - 1] else 0.0,
                                   (spy[d1] - spy[d0]) / spy[d0]))
                bet = 1.0
                if len(sr) > 40:
                    mm_ = sum(m for _, m in sr) / len(sr)
                    aa = sum(x for x, _ in sr) / len(sr)
                    vv = sum((m - mm_) ** 2 for _, m in sr)
                    if vv > 0:
                        bet = sum((x - aa) * (m - mm_) for x, m in sr) / vv
                panel[ds[i]][t] = (cl[i] / hi, (z - a) / a, fr, bet)

        q_ic, q_null, q_ex = [], [], []
        raw_ic, strip_ic, flat_ex = [], [], []
        bucket_betas = [[] for _ in range(B)]
        n_selloff = n_kept = 0

        for d, mm in panel.items():
            if st_map.get(d) != "selloff" or len(mm) < args.min_names:
                continue
            n_selloff += 1
            forms = sorted(v[1] for v in mm.values())
            med = forms[len(forms) // 2]
            sel = [t for t in mm if mm[t][1] >= med]
            if len(sel) < max(15, B * args.min_per_bucket):
                continue
            n_kept += 1

            pth = [mm[t][0] for t in sel]
            fr = [mm[t][2] for t in sel]
            bt = [mm[t][3] for t in sel]

            # Same-date flat comparisons, on the RETAINED dates only.
            r = spearman(pth, fr)
            if r is not None:
                raw_ic.append(r)
            rb = spearman(pth, resid(fr, bt))
            if rb is not None:
                strip_ic.append(rb)
            o_flat = sorted(range(len(sel)), key=lambda i: pth[i])
            kf = max(1, len(o_flat) // 5)
            flat_ex.append(st.mean(fr[i] for i in o_flat[:kf]) - st.mean(fr))

            # --- beta buckets ---
            order = sorted(range(len(sel)), key=lambda i: bt[i])
            size = len(order) // B
            ics_w, ws, ex_b, null_w = [], [], [], []
            for q in range(B):
                lo = q * size
                hi_ = (q + 1) * size if q < B - 1 else len(order)
                idx = order[lo:hi_]
                if len(idx) < 8:
                    continue
                p_ = [pth[i] for i in idx]
                f_ = [fr[i] for i in idx]
                bucket_betas[q].append(st.mean(bt[i] for i in idx))
                rr = spearman(p_, f_)
                if rr is None:
                    continue
                ics_w.append(rr)
                ws.append(len(idx))
                sh = f_[:]
                rnd.shuffle(sh)
                rn = spearman(p_, sh)
                if rn is not None:
                    null_w.append(rn)
                # book: bottom-k PTH vs THIS bucket's own mean -> beta-neutral
                ob = sorted(range(len(idx)), key=lambda i: p_[i])
                kb = max(1, len(ob) // 5)
                ex_b.append(st.mean(f_[i] for i in ob[:kb]) - st.mean(f_))
            if not ics_w:
                continue
            tot = sum(ws)
            q_ic.append(sum(ics_w[i] * ws[i] for i in range(len(ws))) / tot)
            if null_w:
                q_null.append(st.mean(null_w))
            if ex_b:
                q_ex.append(st.mean(ex_b))

        if len(q_ic) < 20:
            print(f"seed {seed}: only {len(q_ic)} usable dates "
                  f"({n_selloff} selloff, {n_kept} with >= {B*args.min_per_bucket} "
                  f"winners). Try --buckets 3.")
            continue

        t_q = nw_t(q_ic, 4) or 0.0
        t_e = nw_t(q_ex, 4) or 0.0
        t_r = nw_t(raw_ic, 4) or 0.0
        t_s = nw_t(strip_ic, 4) or 0.0
        print(f"SEED {seed} — {n_selloff} selloff dates, {n_kept} kept "
              f"({100*n_kept/max(n_selloff,1):.0f}%)")
        print(f"  [same dates] raw IC        {st.mean(raw_ic):+.4f}"
              f"   NW t {t_r:+.2f}")
        print(f"  [same dates] beta-stripped {st.mean(strip_ic):+.4f}"
              f"   NW t {t_s:+.2f}")
        print(f"  WITHIN-BETA IC             {st.mean(q_ic):+.4f}"
              f"   NW t {t_q:+.2f}"
              + ("   t>3" if abs(t_q) > T_HURDLE else "   UNDER BAR"))
        print(f"  within-beta null           {st.mean(q_null):+.4f}")
        print(f"  within-beta book      {100*st.mean(q_ex):+.3f}pp"
              f"  NW t {t_e:+.2f}"
              f"   ({100*sum(1 for x in q_ex if x>0)/len(q_ex):.0f}% dates +)")
        print(f"  flat book (same dates){100*st.mean(flat_ex):+.3f}pp")
        print("  bucket betas  " + "  ".join(
            f"Q{i+1} {st.mean(v):.2f}" for i, v in enumerate(bucket_betas)
            if v))
        agg["q_ic"].append(st.mean(q_ic))
        agg["q_ex"].append(st.mean(q_ex))
        agg["raw"].append(st.mean(raw_ic))
        agg["strip"].append(st.mean(strip_ic))
        agg["t_q"].append(t_q)
        agg["t_e"].append(t_e)
        print()

    if not agg["q_ic"]:
        return

    print("=" * 60)
    print("ACROSS SEEDS")
    print("=" * 60)
    for k, lab, unit in (("raw", "raw IC (same dates)", ""),
                         ("strip", "post-hoc beta strip", ""),
                         ("q_ic", "WITHIN-BETA IC", ""),
                         ("q_ex", "within-beta book", "pp")):
        v = agg[k]
        m = st.mean(v) * (100 if unit else 1)
        same = sum(1 for x in v if (x < 0) == (st.mean(v) < 0))
        print(f"  {lab:<22}{m:>+9.4f}{unit:<3}  seeds same sign "
              f"{same}/{len(v)}")

    ic_m = st.mean(agg["q_ic"])
    ex_m = st.mean(agg["q_ex"]) * 100
    t_max = max(abs(x) for x in agg["t_q"])
    same_ic = sum(1 for x in agg["q_ic"] if (x < 0) == (ic_m < 0))
    clears_t = t_max > T_HURDLE
    clears_cost = abs(ex_m) > COST_BPS / 100.0
    all_seeds = same_ic == len(agg["q_ic"])

    print("\n" + "=" * 60)
    print("PRE-REGISTERED VERDICT")
    print("=" * 60)
    print(f"  NW t > {T_HURDLE} in any seed        {'YES' if clears_t else 'NO'}"
          f"   (max |t| {t_max:.2f})")
    print(f"  same sign in every seed     {'YES' if all_seeds else 'NO'}"
          f"   ({same_ic}/{len(agg['q_ic'])})")
    print(f"  book clears {COST_BPS:.0f}bp round-trip  "
          f"{'YES' if clears_cost else 'NO'}   ({ex_m:+.3f}pp)")
    print()
    if clears_t and all_seeds and clears_cost:
        print("  REAL. The effect survives beta by construction, not by")
        print("  regression. Merits a shadow book. Secondary checks before")
        print("  deployment: sector-neutral rerun, since financials and REITs")
        print("  carry structurally high beta that is not crowding; turnover")
        print("  and cost at 20bp; and whether the low-PTH long leg is")
        print("  separable from the SI brick's low-DTC long leg, which may be")
        print("  holding the same names.")
    elif abs(ic_m) < 0.01:
        print("  BETA ALL ALONG. The effect collapses once the comparison is")
        print("  held at constant beta. CLOSE THE AXIS.")
    else:
        print("  SIGNED BUT THIN. The residual keeps its direction and does")
        print("  not clear the bar. This is the third outcome the source memo")
        print("  did not name, and it is a CLOSE, not a shadow book. A")
        print("  consistently-signed sub-bar residual is what a partially")
        print("  removed beta exposure looks like. CLOSE THE AXIS.")


if __name__ == "__main__":
    main()
