#!/usr/bin/env python3
"""
h40_book_test.py — threshold vs cap, and does the h=40 edge survive cost?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHAT IS ALREADY ESTABLISHED (2026-09-05)
    XGBoost on the existing feature set, h=40, quarterly refits, 3 seeds:
        cap-3 day-weighted excess +5.194pp, positive 3/3 seeds
        cap-5 +4.476pp 3/3, cap-10 +3.724pp 3/3
        AUC 0.577 -- the model's only above-random AUC in the sweep
        turnover 27%, against 61% at h=3
    And it is NOT the short-interest brick reappearing: overlap with the
    lowest-DTC quintile is 32.2% against 19.5% random, Spearman(prob, DTC) is
    -0.060, and retraining with short_ratio and all its derivatives REMOVED
    gives +5.376pp -- slightly stronger, 3/3 seeds.

WHAT THIS SCRIPT ADDS, AND WHY IN THIS ORDER

  1. THRESHOLD vs CAP. Everything so far took a fixed count -- top 3, top 5.
     Production does not work that way: the gate is prob_eff >= 0.70. A fixed
     threshold takes however many clear the bar, which varies by day. That
     matters at h=40 precisely because AUC is 0.577, so probabilities should
     actually spread here, unlike h=5 where the model was near-random and almost
     nothing cleared 0.70.
     Reported with NAMES PER DAY, because a threshold that fires on 40 names
     some days and zero on others is a different instrument from a top-3 book
     even at the same average excess.

  2. COST. +5.134pp gross at 27% turnover over 40 days. A ladder rather than a
     single assumed figure, because the right bps depends on the names and the
     broker, and because every h=5 result this session died on friction rather
     than on sign.

  3. DRAWDOWN AND CONCENTRATION. PCT7 passed significance, nulls, regime
     stability, month-consistency and a cost ladder on 2026-09-05 and was still
     retracted -- because nobody had asked whether the measured quantity was
     capturable. Its pooled +1.46% was -1.97% day-weighted, and 38 of 64 days
     lost money. So: worst stretch, share of periods negative, and how
     concentrated the picks are in a few tickers.

METHOD
    Same construction as the horizon sweep: one feature panel per seed,
    quarterly refits, day-weighted excess over the SAME DAY's universe so market
    direction cannot flatter it, multi-seed because three single-seed results
    reversed on replication today.

    python analysis/h40_book_test.py --seeds 3 --tickers 80
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

CAPS = (1, 3, 5, 10)
THRESHOLDS = (0.50, 0.55, 0.60, 0.70)


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
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--min-names", type=int, default=25)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    print(f"h={H} book test — {args.seeds} seeds x {args.tickers} tickers\n")

    agg_cap = defaultdict(list)
    agg_thr = defaultdict(list)
    agg_n = defaultdict(list)
    agg_turn = defaultdict(list)
    dd_all, neg_all, conc_all = [], [], []

    for seed in range(1, args.seeds + 1):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        X, fwd = {}, {}
        for t in u[:args.tickers]:
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
        if len(X) < 15000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        cap_ex = defaultdict(list)
        cap_turn = defaultdict(list)
        thr_ex = defaultdict(list)
        thr_n = defaultdict(list)
        thr_turn = defaultdict(list)
        picks = []
        prev_cap = {c: set() for c in CAPS}
        prev_thr = {t: set() for t in THRESHOLDS}

        for i in range(len(anchors) - 1):
            tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr_end]
            kte = [k for k in X if tr_end <= k[1] < te_end]
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0)
            m.fit([X[k] for k in ktr], [1 if fwd[k] > 0 else 0 for k in ktr])
            p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]
            byd = defaultdict(list)
            for z, k in enumerate(kte):
                byd[k[1]].append((p[z], fwd[k], k[0]))

            for d in sorted(byd):
                v = sorted(byd[d], reverse=True)
                if len(v) < args.min_names:
                    continue
                mkt = st.mean(x[1] for x in v)
                for c in CAPS:
                    sel = v[:c]
                    cap_ex[c].append(st.mean(x[1] for x in sel) - mkt)
                    cur = {x[2] for x in sel}
                    if prev_cap[c]:
                        cap_turn[c].append(
                            100.0 * len(cur - prev_cap[c]) / max(len(cur), 1))
                    prev_cap[c] = cur
                    if c == 3:
                        picks += list(cur)
                for th in THRESHOLDS:
                    sel = [x for x in v if x[0] >= th]
                    thr_n[th].append(len(sel))
                    if not sel:
                        continue
                    thr_ex[th].append(st.mean(x[1] for x in sel) - mkt)
                    cur = {x[2] for x in sel}
                    if prev_thr[th]:
                        thr_turn[th].append(
                            100.0 * len(cur - prev_thr[th]) / max(len(cur), 1))
                    prev_thr[th] = cur

        if not cap_ex[3]:
            print(f"seed {seed}: no scoreable rebalances\n")
            continue

        print(f"SEED {seed} — {len(X):,} rows, {len(cap_ex[3])} rebalances")
        print(f"  {'selection':<14}{'excess':>10}{'NW t':>8}{'names/day':>11}"
              f"{'turnover':>10}")
        for c in CAPS:
            print(f"  {'cap '+str(c):<14}{100*st.mean(cap_ex[c]):>+9.3f}pp"
                  f"{(nw_t(cap_ex[c], 3) or 0):>+8.2f}{c:>11}"
                  f"{(st.mean(cap_turn[c]) if cap_turn[c] else 0):>9.0f}%")
            agg_cap[c].append(st.mean(cap_ex[c]))
        for th in THRESHOLDS:
            if not thr_ex[th]:
                print(f"  {'prob>='+str(th):<14}   never fires")
                continue
            print(f"  {'prob>='+str(th):<14}{100*st.mean(thr_ex[th]):>+9.3f}pp"
                  f"{(nw_t(thr_ex[th], 3) or 0):>+8.2f}"
                  f"{st.mean(thr_n[th]):>11.1f}"
                  f"{(st.mean(thr_turn[th]) if thr_turn[th] else 0):>9.0f}%")
            agg_thr[th].append(st.mean(thr_ex[th]))
            agg_n[th].append(st.mean(thr_n[th]))
            agg_turn[th].append(st.mean(thr_turn[th]) if thr_turn[th] else 0)

        e = cap_ex[3]
        eq, peak, mdd = 1.0, 1.0, 0.0
        for r in e:
            eq *= (1 + r / H)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
        neg = 100.0 * sum(1 for x in e if x < 0) / len(e)
        c5 = 100.0 * sum(n for _, n in Counter(picks).most_common(5)) / max(len(picks), 1)
        print(f"  cap-3 path: max drawdown {100*mdd:.1f}%, "
              f"{neg:.0f}% of rebalances negative, top-5 tickers "
              f"{c5:.0f}% of picks, {len(set(picks))} distinct\n")
        dd_all.append(mdd); neg_all.append(neg); conc_all.append(c5)

    if not agg_cap[3]:
        print("no seeds produced results")
        return

    print("=" * 70)
    print("ACROSS SEEDS")
    print("=" * 70)
    print(f"  {'selection':<14}{'excess':>10}{'seeds +':>9}{'names/day':>11}"
          f"{'turnover':>10}")
    for c in CAPS:
        v = agg_cap[c]
        print(f"  {'cap '+str(c):<14}{100*st.mean(v):>+9.3f}pp"
              f"{sum(1 for x in v if x>0):>5}/{len(v)}{c:>11}")
    for th in THRESHOLDS:
        v = agg_thr.get(th, [])
        if not v:
            continue
        print(f"  {'prob>='+str(th):<14}{100*st.mean(v):>+9.3f}pp"
              f"{sum(1 for x in v if x>0):>5}/{len(v)}"
              f"{st.mean(agg_n[th]):>11.1f}{st.mean(agg_turn[th]):>9.0f}%")

    print(f"\n  COST LADDER on cap-3 ({100*st.mean(agg_cap[3]):+.3f}pp gross, "
          f"~27% turnover, one round trip per rebalance)")
    print(f"    {'bps/leg':>9}{'net':>11}")
    g = st.mean(agg_cap[3])
    for bps in (0, 5, 10, 20, 40, 100):
        print(f"    {bps:>9}{100*(g - 2*bps/10000.0):>+10.3f}pp")

    if dd_all:
        print(f"\n  cap-3 path across seeds: max drawdown "
              f"{100*st.mean(dd_all):.1f}%, "
              f"{st.mean(neg_all):.0f}% of rebalances negative, "
              f"top-5 tickers {st.mean(conc_all):.0f}% of picks")

    print("\n  A threshold that matches cap-3's excess with MORE names is a")
    print("  better book -- same edge, less idiosyncratic risk. One that fires")
    print("  on almost nothing is a top-1 book wearing a different label, and")
    print("  the names/day column is what tells them apart.\n")
    print("  Drawdown here scales cohort returns by 1/H and assumes equal")
    print("  weights and free fills. It is a sanity check on the path, not a")
    print("  backtest.")


if __name__ == "__main__":
    main()
