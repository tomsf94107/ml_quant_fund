#!/usr/bin/env python3
"""
h40_yearly_test.py — is the h=40 edge carried by one period?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHY THIS IS THE LAST CHECK BEFORE WRITE-UP
    The h=40 signal has passed everything else on 2026-09-05:

      horizon      per-day cap-3 excess rises 0.023pp at h=3 to 0.130pp at h=40,
                   turnover falls 61% to 27%, positive 3/3 seeds
      separate     not the SI brick -- retraining with short_ratio and all its
                   derivatives REMOVED gives +5.376pp, slightly stronger
      book shape   prob>=0.70 is the strongest cell and is the EXISTING
                   production gate: +3.671pp on 6.8 names/day, 3/3 seeds
      longer sample 2016-2026, 1,065 rebalances, ~26 independent periods,
                   every cell positive 3/3
      cost         survives 100bps

    What has NOT been checked is WHEN the return arrives. That distinction
    retracted PCT7 earlier the same day: it passed significance, nulls, regime
    stability, month-consistency and a cost ladder, and then 6 days of 64 turned
    out to carry the entire result while 38 lost money.

    The SI brick got exactly this treatment today and it is what separated
    "record overstated" from "signal decaying" -- half-samples matched, per-year
    IC was stable 2021-2025, so the answer was the record, not the signal.

    44% of h=40 rebalances are negative and the window contains COVID. If 2020
    carries it, this is one extraordinary period, not an edge.

WHAT IS MEASURED
    Per calendar year, for cap-3 and prob>=0.70:
      mean excess, share of rebalances positive, count
    Then the same with 2020 EXCLUDED, because a model trained through 2019 and
    tested in March 2020 produces extreme numbers in both directions and one
    year should not decide this.

    Multi-seed, since seed dispersion at h=40 is wide -- prob>=0.70 ranged
    2.14pp to 5.46pp across three seeds on the 2016 sample.

INTERPRETATION
    Positive in most years, with 2020 removed leaving the result broadly intact
    -> a real edge, and the magnitude is roughly the ex-2020 figure.
    Carried by 2020, or negative in half the years -> one period, and the
    aggregate is not a description of what the strategy does.

    python analysis/h40_yearly_test.py --seeds 3 --tickers 80
"""
import argparse
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--thresh", type=float, default=0.70)
    ap.add_argument("--cap", type=int, default=3)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random
    import os
    NULL = os.environ.get("ML_QUANT_NULL") == "1"
    DUMP = os.environ.get("ML_QUANT_DUMP")
    _dump = open(DUMP, "w") if DUMP else None
    if _dump is not None:
        _dump.write("seed,date,year,n,prob_n,cap3_excess,prob_excess\n")

    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    print(f"h={H} yearly breakdown — {args.seeds} seeds x {args.tickers} "
          f"tickers, from {args.start}\n")

    # (seed, year) -> list of per-rebalance excess
    by_year_cap = defaultdict(lambda: defaultdict(list))
    by_year_thr = defaultdict(lambda: defaultdict(list))

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
        if len(X) < 20000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.35)::3]

        for i in range(len(anchors) - 1):
            tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr_end]
            kte = [k for k in X if tr_end <= k[1] < te_end]
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0)
            y_tr = [1 if fwd[k] > 0 else 0 for k in ktr]
            if NULL:
                random.Random(hash((seed, tr_end)) & 0xFFFFFFFF).shuffle(y_tr)
            m.fit([X[k] for k in ktr], y_tr)
            p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]
            byd = defaultdict(list)
            for z, k in enumerate(kte):
                byd[k[1]].append((p[z], fwd[k]))
            for d, v in byd.items():
                if len(v) < args.min_names:
                    continue
                v = sorted(v, reverse=True)
                mkt = st.mean(x[1] for x in v)
                yr = d[:4]
                if _dump is not None:
                    _c3 = st.mean(x[1] for x in v[:3]) - mkt
                    _hi = [x[1] for x in v if x[0] >= 0.70]
                    _pe = (st.mean(_hi) - mkt) if _hi else float("nan")
                    _dump.write(f"{seed},{d},{yr},{len(v)},{len(_hi)},{_c3:.6f},{_pe:.6f}\n")
                by_year_cap[seed][yr].append(
                    st.mean(x[1] for x in v[:args.cap]) - mkt)
                sel = [x for x in v if x[0] >= args.thresh]
                if sel:
                    by_year_thr[seed][yr].append(
                        st.mean(x[1] for x in sel) - mkt)

        if not by_year_cap[seed]:
            print(f"seed {seed}: no scoreable rebalances\n")
            continue
        print(f"SEED {seed}")
        print(f"  {'year':<7}{'n':>5}{'cap-'+str(args.cap):>12}{'pos':>7}"
              f"{'prob>='+str(args.thresh):>13}{'pos':>7}")
        for yr in sorted(by_year_cap[seed]):
            c = by_year_cap[seed][yr]
            t_ = by_year_thr[seed].get(yr, [])
            cp = 100.0 * sum(1 for x in c if x > 0) / len(c)
            tp = (100.0 * sum(1 for x in t_ if x > 0) / len(t_)) if t_ else 0
            print(f"  {yr:<7}{len(c):>5}{100*st.mean(c):>+11.2f}pp{cp:>6.0f}%"
                  + (f"{100*st.mean(t_):>+12.2f}pp{tp:>6.0f}%" if t_
                     else f"{'—':>12}{'—':>7}"))
        print()

    print("=" * 68)
    print("POOLED ACROSS SEEDS, BY YEAR")
    print("=" * 68)
    years = sorted({y for s in by_year_cap for y in by_year_cap[s]})
    print(f"  {'year':<7}{'n':>6}{'cap-'+str(args.cap):>12}{'seeds +':>9}"
          f"{'prob>='+str(args.thresh):>13}{'seeds +':>9}")
    cap_ex2020, thr_ex2020 = [], []
    cap_all, thr_all = [], []
    for yr in years:
        cs = [st.mean(by_year_cap[s][yr]) for s in by_year_cap
              if by_year_cap[s].get(yr)]
        ts = [st.mean(by_year_thr[s][yr]) for s in by_year_thr
              if by_year_thr[s].get(yr)]
        n = sum(len(by_year_cap[s].get(yr, [])) for s in by_year_cap)
        if not cs:
            continue
        print(f"  {yr:<7}{n:>6}{100*st.mean(cs):>+11.2f}pp"
              f"{sum(1 for x in cs if x>0):>5}/{len(cs)}"
              + (f"{100*st.mean(ts):>+12.2f}pp"
                 f"{sum(1 for x in ts if x>0):>5}/{len(ts)}" if ts
                 else f"{'—':>12}{'—':>9}"))
        cap_all += cs
        thr_all += ts
        if yr != "2020":
            cap_ex2020 += cs
            thr_ex2020 += ts

    print(f"\n  {'all years':<20}cap-{args.cap} {100*st.mean(cap_all):>+7.2f}pp"
          + (f"   prob>={args.thresh} {100*st.mean(thr_all):>+7.2f}pp"
             if thr_all else ""))
    if cap_ex2020:
        print(f"  {'EXCLUDING 2020':<20}cap-{args.cap} "
              f"{100*st.mean(cap_ex2020):>+7.2f}pp"
              + (f"   prob>={args.thresh} "
                 f"{100*st.mean(thr_ex2020):>+7.2f}pp" if thr_ex2020 else ""))
        pos = sum(1 for yr in years if yr != "2020"
                  and [st.mean(by_year_cap[s][yr]) for s in by_year_cap
                       if by_year_cap[s].get(yr)]
                  and st.mean([st.mean(by_year_cap[s][yr]) for s in by_year_cap
                               if by_year_cap[s].get(yr)]) > 0)
        print(f"  positive years excluding 2020: {pos}/"
              f"{len([y for y in years if y != '2020'])}")

    print("\n  A model trained through 2019 and tested in March 2020 produces")
    print("  extreme numbers in both directions. If removing 2020 leaves the")
    print("  result broadly intact, the edge is real and the ex-2020 figure is")
    print("  the honest magnitude. If it collapses, this is one period.")
    print("\n  Each year holds only ~6 rebalances at h=40, so per-year figures")
    print("  are indicative. The count of POSITIVE YEARS is the robust read.")


if __name__ == "__main__":
    main()
