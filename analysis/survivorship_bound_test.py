#!/usr/bin/env python3
"""
survivorship_bound_test.py — bound the survivorship exposure of the h=40 book.

READ-ONLY. Trains throwaway copies. Writes nothing.

WHAT THIS CAN AND CANNOT DO
    It CANNOT measure survivorship bias. That needs delisted price history,
    which this fund does not have: prices.db holds the names that exist today,
    and everything that failed between 2016 and 2026 is absent.

    It BOUNDS the exposure, by comparing the same strategy on universe slices
    with different attrition rates. Large, liquid names get acquired but rarely
    go to zero; small, illiquid, high-volatility names do both. If the edge is
    much larger on the high-attrition slice, some of that excess is names that
    survived where their peers did not.

    The comparison is INDIRECT and confounded: size also brings liquidity,
    analyst coverage, lower volatility and different investor bases. A gap could
    come from any of those. This gives a rough bound, not a measurement, and it
    should be read as "how nervous should I be" rather than "the bias is X".

WHY BOUND AT ALL RATHER THAN JUST BUY THE DATA
    Shumway (1997, Journal of Finance) found that even CRSP -- the academic gold
    standard -- carries a delisting bias: delists for bankruptcy and other
    negative reasons are generally surprises, correct delisting returns are NOT
    available for most negatively-delisted stocks since 1962, and the omitted
    returns are large. So buying delisted history does not fully close the hole
    either; the terminal wipeout is often the missing piece.

    Published magnitudes vary by source, and the split is worth noting:
      CRSP index 1926-2001            1.6pp/year
      Elton, Gruber & Blake (funds)   0.9-1.5%/year
      backtesting vendor blogs        3-8%, 4-6%, "5 to 15%"
    The peer-reviewed figures are the LOW ones. The alarming numbers come from
    platform marketing.

    Against the h=40 book's ~+10-11% annualised excess, a 1.6pp haircut leaves
    ~9% and a 4pp haircut leaves ~6-7%. Both remain worth having, which is why
    bounding first is cheaper than subscribing first.

    Kothari, Shanken & Sloan augmented COMPUSTAT with delisted firms and found
    that adding non-survivors ATTENUATED the coefficients on book-to-market,
    earnings yield and cash flow while leaving them significant. Shrinkage
    rather than disappearance is the expected shape.

METHOD
    Four slices of the same universe, ranked by 20-day dollar ADV, each run
    through the identical h=40 book:

      mega    top 100 by ADV      lowest attrition -- these are acquired, not
                                  wiped out
      large   ranks 100-400
      mid     ranks 400-900
      small   ranks 900+          highest attrition

    Same seeds, same folds, same model. Only the pool changes.

    Reported per slice: cap-3 and prob>=0.70 excess, plus the median ADV so the
    slices can be sanity-checked.

READING IT
    edge FLAT across slices        survivorship is unlikely to be driving the
                                   result; the mechanism is not size-dependent
    edge RISES sharply into small  the excess in the small slice is suspect --
                                   that is where failed peers are missing
    edge FALLS into small          survivorship is not the story; something
                                   else (liquidity, coverage) favours large

    None of these is proof. The SI brick has a stronger argument available to
    it: si_leg_decomp.py showed 79% of its edge sits in low days-to-cover names,
    which STRUCTURALLY rarely delist. That is mechanistic. The h=40 book has no
    such argument, which is exactly why it needs bounding.

    python analysis/survivorship_bound_test.py --seeds 3
"""
import argparse
import math
import os
import sqlite3
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

SLICES = (("mega", 0, 100), ("large", 100, 400),
          ("mid", 400, 900), ("small", 900, 10000))
CAPS = (3,)
THRESH = 0.70


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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=70,
                    help="sampled from EACH slice, so slices are comparable")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--universe", default="tickers_expanded.txt")
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    uni = [l.strip().upper() for l in open(args.universe) if l.strip()]
    con = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    adv = {}
    for t, a in con.execute(
            "SELECT ticker, AVG(close*volume) FROM raw_bars "
            "WHERE d >= date('now','-60 days') AND close > 0 AND volume > 0 "
            "GROUP BY ticker"):
        if a:
            adv[t] = float(a)
    con.close()
    ranked = [t for t in sorted(adv, key=lambda x: -adv[x]) if t in set(uni)]
    print(f"{len(ranked)} of {len(uni)} universe names have recent ADV\n")

    print(f"  {'slice':<8}{'names':>7}{'median ADV':>14}")
    pools = {}
    for name, lo, hi in SLICES:
        pool = ranked[lo:hi]
        if len(pool) < 30:
            continue
        pools[name] = pool
        print(f"  {name:<8}{len(pool):>7}"
              f"{'$'+format(st.median(adv[t] for t in pool)/1e6, '.1f')+'M':>14}")
    print()

    agg = defaultdict(lambda: defaultdict(list))
    for seed in range(1, args.seeds + 1):
        for sname, pool in pools.items():
            u = pool[:]
            random.Random(seed).shuffle(u)
            sample = u[:args.tickers]
            X, fwd = {}, {}
            for t in sample:
                try:
                    df = build_feature_dataframe(t, start_date=args.start,
                                                 training_mode=True)
                    if df is None or len(df) < 400 or "close" not in df.columns:
                        continue
                    num = df.select_dtypes("number")
                    num = num.drop(columns=[c for c in num.columns
                                            if c.startswith("target_")],
                                   errors="ignore")
                    ds = [str(d)[:10] for d in df["date"]]
                    cl = list(df["close"])
                    for j in range(20, len(ds) - H):
                        a, b = cl[j], cl[j + H]
                        if not a or not b:
                            continue
                        r = (b - a) / a
                        if abs(r) > 1.5:
                            continue
                        X[(t, ds[j])] = [float(v) if v == v else float("nan")
                                         for v in num.iloc[j].tolist()]
                        fwd[(t, ds[j])] = r
                except Exception:
                    continue
            if len(X) < 12000:
                print(f"  seed {seed} {sname}: only {len(X)} rows, skipped")
                continue

            dates = sorted({k[1] for k in X})
            months = sorted({d[:7] for d in dates})
            anchors = months[int(len(months) * 0.55)::3]
            cap_ex, thr_ex = [], []
            for i in range(len(anchors) - 1):
                tr, te = anchors[i] + "-01", anchors[i + 1] + "-01"
                ktr = [k for k in X if k[1] < tr]
                kte = [k for k in X if tr <= k[1] < te]
                if len(ktr) < 4000 or len(kte) < 800:
                    continue
                m = XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss",
                                  verbosity=0)
                m.fit([X[k] for k in ktr],
                      [1 if fwd[k] > 0 else 0 for k in ktr])
                p = [float(v) for v in
                     m.predict_proba([X[k] for k in kte])[:, 1]]
                byd = defaultdict(list)
                for z, k in enumerate(kte):
                    byd[k[1]].append((p[z], fwd[k]))
                for d, v in byd.items():
                    if len(v) < args.min_names:
                        continue
                    v.sort(reverse=True)
                    mkt = st.mean(r for _, r in v)
                    cap_ex.append(st.mean(r for _, r in v[:3]) - mkt)
                    hi = [r for pr, r in v if pr >= THRESH]
                    if hi:
                        thr_ex.append(st.mean(hi) - mkt)
            if cap_ex:
                agg[sname]["cap3"].append(st.mean(cap_ex))
                agg[sname]["t"].append(nw_t(cap_ex, 3) or 0.0)
            if thr_ex:
                agg[sname]["thr"].append(st.mean(thr_ex))
        print(f"  seed {seed} done")

    print("\n" + "=" * 62)
    print("SURVIVORSHIP BOUND — same book, slices by liquidity")
    print("=" * 62)
    print(f"  {'slice':<8}{'cap-3':>11}{'seeds +':>9}{'prob>=0.70':>13}"
          f"{'NW t':>8}")
    for name, _, _ in SLICES:
        a = agg.get(name)
        if not a or not a["cap3"]:
            continue
        c, tr = a["cap3"], a.get("thr", [])
        print(f"  {name:<8}{100*st.mean(c):>+10.3f}pp"
              f"{sum(1 for x in c if x > 0):>5}/{len(c)}"
              + (f"{100*st.mean(tr):>+12.3f}pp" if tr else f"{'-':>13}")
              + f"{st.mean(a['t']):>+8.2f}")

    if agg.get("mega", {}).get("cap3") and agg.get("small", {}).get("cap3"):
        gap = st.mean(agg["small"]["cap3"]) - st.mean(agg["mega"]["cap3"])
        print(f"\n  small minus mega: {100*gap:+.3f}pp per 40-day period")
        print(f"  annualised at ~6.3 periods: {100*gap*6.3:+.2f}pp")
        print("\n  That gap is an UPPER bound on survivorship exposure and a")
        print("  loose one -- size also brings liquidity, coverage, lower")
        print("  volatility and a different investor base, any of which could")
        print("  produce it. Published estimates for reference: CRSP index")
        print("  1.6pp/year, Elton-Gruber-Blake 0.9-1.5%/year peer-reviewed;")
        print("  3-8% from vendor blogs. If this gap is near or below the")
        print("  peer-reviewed range, survivorship is not the main worry.")


if __name__ == "__main__":
    main()
