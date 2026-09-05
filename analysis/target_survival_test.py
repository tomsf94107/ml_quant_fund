#!/usr/bin/env python3
"""
target_survival_test.py — does the +0.39pp survive costs, capping and concentration?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

WHY
    target_comparison_test.py (2026-09-05) trained four label definitions on the
    same features and scored each by the day-weighted forward return of its own
    top decile:

        target            AUC      vs universe   win days
        any_positive      0.5100      +0.26pp     182/339
        top_decile        0.7316      +0.01pp     179/339
        pct7              0.7670      +0.39pp     187/339
        triple_barrier    0.5593      -0.30pp     159/339

    That already confirmed the AUC trap: top_decile posts 0.7316 -- the A8
    claim the roadmap has carried since May -- and converts to +0.01pp.

    But +0.39pp per 5-day cohort is roughly 50 cohorts a year. Dismissing that
    as "not significant" would be wrong. It is small, not nothing, and it
    deserves the three tests that decide whether small survives.

    The caution is specific, not generic. PCT7 showed +1.46% pooled and -1.97%
    day-weighted on the same data; and capping its positions turned +1.46% into
    -2.38%. A positive headline has already collapsed once today under exactly
    these checks.

THE THREE TESTS

  1. COSTS. +0.39pp is gross. A 5-day hold is one round trip: entry and exit.
     At 10bps a leg that is -0.20pp, at 20bps -0.40pp. Reported as a ladder
     rather than a single assumption, because the right figure depends on the
     names and the broker.

  2. CAPPING. The top decile is ~6 names of 60 per day. A real book takes fewer.
     If the return lives in the tail of the decile rather than its head, capping
     destroys it -- which is precisely what happened to PCT7, where the top 3
     returned -2.38% while all 1,105 returned +1.46%.

  3. CONCENTRATION. 187/339 win days is 55%, barely above a coin flip. If the
     +0.39pp comes from a handful of sessions, it is not holdable. Reported as
     the return with the best N days removed, which is the blunt version of the
     question.

    Only any_positive and pct7 are tested -- the two that showed positive lift.
    Spending compute on the two that did not would be searching for a result.

    python analysis/target_survival_test.py --tickers 60
"""
import argparse
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    universe = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    random.Random(args.seed).shuffle(universe)
    universe = universe[:args.tickers]
    print(f"building {len(universe)} tickers -- same seed and split as "
          f"target_comparison_test\n")

    X, fwd = {}, {}
    for i, t in enumerate(universe, 1):
        try:
            df = build_feature_dataframe(t, start_date=args.start,
                                         training_mode=True)
            if df is None or len(df) < 300 or "close" not in df.columns:
                continue
            num = df.select_dtypes("number")
            num = num.drop(columns=[c for c in num.columns
                                    if c.startswith("target_")],
                           errors="ignore")
            ds = [str(d)[:10] for d in df["date"]]
            cl = list(df["close"])
            for j in range(20, len(ds) - H):
                a = cl[j]
                if not a or cl[j + H] is None:
                    continue
                r = (cl[j + H] - a) / a
                if abs(r) > 0.8:
                    continue
                X[(t, ds[j])] = [float(v) if v == v else float("nan")
                                 for v in num.iloc[j].tolist()]
                fwd[(t, ds[j])] = r
            if i % 20 == 0:
                print(f"  ...{i} tickers, {len(X):,} rows")
        except Exception:
            continue
    print(f"\n{len(X):,} panel rows\n")
    if len(X) < 5000:
        print("too few rows")
        return

    dates = sorted({k[1] for k in X})
    cut = dates[int(len(dates) * 0.70)]
    keys_tr = [k for k in X if k[1] < cut]
    keys_te = [k for k in X if k[1] >= cut]
    Xtr = [X[k] for k in keys_tr]
    Xte = [X[k] for k in keys_te]

    bydate = defaultdict(list)
    for k in X:
        bydate[k[1]].append(k)

    uni = defaultdict(list)
    for k in keys_te:
        uni[k[1]].append(fwd[k])
    uni_day = st.mean(st.mean(v) for v in uni.values())
    print(f"train < {cut}, test >= {cut}, universe day-weighted "
          f"{100*uni_day:+.2f}%\n")

    for name in ("any_positive", "pct7"):
        if name == "any_positive":
            lab = {k: (1 if fwd[k] > 0 else 0) for k in X}
        else:
            lab = {k: (1 if fwd[k] >= 0.07 else 0) for k in X}
        ytr = [lab[k] for k in keys_tr]
        yte = [lab[k] for k in keys_te]
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr)
        p = [float(v) for v in m.predict_proba(Xte)[:, 1]]
        byd = defaultdict(list)
        for i, k in enumerate(keys_te):
            byd[k[1]].append((p[i], fwd[k]))
        days = {d: sorted(v, reverse=True) for d, v in byd.items()
                if len(v) >= 10}

        print(f"{'='*66}\n{name}\n{'='*66}")

        print("  2. CAPPING — take the top N by prediction each day")
        print(f"     {'cap':>6}{'mean ret':>11}{'vs universe':>13}"
              f"{'win days':>11}")
        for cap in (1, 2, 3, 5, 10, None):
            rs = []
            for d, v in days.items():
                n = cap if cap else max(1, len(v) // 10)
                rs.append(st.mean(r for _, r in v[:n]))
            w = sum(1 for d, v in days.items()
                    if st.mean(r for _, r in
                               v[:(cap if cap else max(1, len(v)//10))])
                    > st.mean(r for _, r in v))
            lbl = str(cap) if cap else "decile"
            print(f"     {lbl:>6}{100*st.mean(rs):>10.2f}%"
                  f"{100*(st.mean(rs)-uni_day):>+12.2f}pp"
                  f"{w:>7}/{len(days)}")

        dec = [st.mean(r for _, r in v[:max(1, len(v)//10)])
               for v in days.values()]
        gross = st.mean(dec)
        print(f"\n  1. COSTS on the decile — 5-day hold is one round trip")
        print(f"     {'bps/leg':>9}{'net ret':>10}{'vs universe':>13}")
        for c in (0, 5, 10, 20, 40):
            net = gross - 2 * c / 10000.0
            print(f"     {c:>9}{100*net:>9.2f}%"
                  f"{100*(net-uni_day):>+12.2f}pp")

        print(f"\n  3. CONCENTRATION — drop the best N days")
        s = sorted(dec, reverse=True)
        print(f"     {'drop':>6}{'mean ret':>11}{'vs universe':>13}")
        for k in (0, 1, 3, 5, 10):
            if k >= len(s):
                continue
            v = st.mean(s[k:])
            print(f"     {k:>6}{100*v:>10.2f}%{100*(v-uni_day):>+12.2f}pp")
        print()

    print("  A result that survives all three is worth building on, however")
    print("  small. One that needs the full decile, zero costs and its best")
    print("  days intact is not a strategy -- it is a measurement.")


if __name__ == "__main__":
    main()
