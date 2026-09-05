#!/usr/bin/env python3
"""
consensus_test.py — does 2-of-3 or 3-of-3 model agreement beat 1-of-3?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

THE HYPOTHESIS
    If three models independently flag the same ticker, that agreement should
    carry more information than any one model's call. That is the standard
    intuition behind consensus, and it is worth testing rather than arguing.

WHY IT MIGHT NOT WORK HERE, STATED UP FRONT
    Consensus adds information only when the models make INDEPENDENT errors.
    These three train on the same 119 features, the same target, the same folds.
    Two measurements from 2026-09-05 bear on this:

      - The existing XGB + LightGBM ensemble gains +0.002 AUC (t=+0.63). Two
        gradient-boosted tree libraries agree mostly about the same noise.
      - blend50 -- a 50/50 average, i.e. the 2-of-2 agreement case -- scored
        AUC 0.4932, BELOW RANDOM, while posting the best cap returns of any
        model tested. Agreement did not improve the ranking.

    But blending PROBABILITIES and COUNTING VOTES are different operations. An
    average is dragged by whichever model is most confident; a vote is not. A
    consensus filter can work where averaging does not, so this tests the vote.

WHAT IS TESTED
    Three models on identical folds:
      linear     L2-logistic at C=1e-05
      xgboost    the production estimator
      lightgbm   the current ensemble partner

    Each date, each model nominates its top decile. Names are then bucketed by
    how many models nominated them -- 1, 2 or 3 -- and the day-weighted forward
    return of each bucket is compared against the same day's universe.

    THE COMPARISON THAT MATTERS is 3-of-3 against 1-of-3. If unanimous names
    outperform singly-nominated ones, agreement carries information. If the
    buckets are indistinguishable, the models are agreeing about noise.

    Also reported: how OFTEN they agree. If 3-of-3 is 90% of nominations, the
    models are near-identical and consensus is not a filter at all -- it is the
    same list with extra steps.

    Multi-seed, because two single-seed results today reversed on replication:
    a top-N book at +1.17pp went to -0.81pp, and a linear economic edge at
    +0.645pp went to -0.021pp.

    python analysis/consensus_test.py --seeds 3 --tickers 100
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")


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


def build(universe, start, H):
    from features.builder import build_feature_dataframe
    X, fwd = {}, {}
    for t in universe:
        try:
            df = build_feature_dataframe(t, start_date=start,
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
        except Exception:
            continue
    return X, fwd


def run_fold(X, fwd, tr_end, te_end, C, H, min_names, top_frac):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None

    ktr = [k for k in X if k[1] < tr_end]
    kte = [k for k in X if tr_end <= k[1] < te_end]
    if len(ktr) < 5000 or len(kte) < 1000:
        return None
    ytr = [1 if fwd[k] > 0 else 0 for k in ktr]
    Xtr = [X[k] for k in ktr]
    Xte = [X[k] for k in kte]

    models = {
        "linear": Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                            ("lr", LogisticRegression(C=C, max_iter=2000))]),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8,
                                 eval_metric="logloss", verbosity=0),
    }
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05,
                                            subsample=0.8,
                                            colsample_bytree=0.8,
                                            verbose=-1)

    preds = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds[name] = [float(v) for v in m.predict_proba(Xte)[:, 1]]

    byd = defaultdict(list)
    for i, k in enumerate(kte):
        byd[k[1]].append(i)

    votes_ret = defaultdict(list)      # nvotes -> per-day mean excess
    counts = defaultdict(int)
    for d, idxs in byd.items():
        if len(idxs) < min_names:
            continue
        n_top = max(1, int(len(idxs) * top_frac))
        nominated = defaultdict(int)
        for name in models:
            ranked = sorted(idxs, key=lambda i: -preds[name][i])[:n_top]
            for i in ranked:
                nominated[i] += 1
        mkt = st.mean(fwd[kte[i]] for i in idxs)
        for v in (1, 2, 3):
            sel = [i for i, c in nominated.items() if c == v]
            if sel:
                votes_ret[v].append(st.mean(fwd[kte[i]] for i in sel) - mkt)
                counts[v] += len(sel)
    return votes_ret, counts, len(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=100)
    ap.add_argument("--C", type=float, default=1e-5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--top-frac", type=float, default=0.10,
                    help="each model nominates this fraction as its top names")
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    import random
    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    print(f"consensus test — {args.seeds} seeds x {args.tickers} tickers, "
          f"h={H}, each model nominates top {args.top_frac:.0%}\n")

    agg = defaultdict(list)
    tot_counts = defaultdict(int)
    for seed in range(1, args.seeds + 1):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        X, fwd = build(u[:args.tickers], args.start, H)
        if len(X) < 20000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue
        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        seed_ret = defaultdict(list)
        seed_cnt = defaultdict(int)
        nmod = 0
        for i in range(len(anchors) - 1):
            r = run_fold(X, fwd, anchors[i] + "-01", anchors[i + 1] + "-01",
                         args.C, H, args.min_names, args.top_frac)
            if not r:
                continue
            vr, ct, nmod = r
            for v in vr:
                seed_ret[v] += vr[v]
            for v in ct:
                seed_cnt[v] += ct[v]
        if not seed_ret:
            print(f"seed {seed}: no scoreable refits\n")
            continue

        print(f"SEED {seed} — {len(X):,} rows, {nmod} models")
        print(f"  {'votes':>6}{'names':>8}{'share':>8}{'excess/day':>13}"
              f"{'NW t':>8}")
        total = sum(seed_cnt.values())
        for v in sorted(seed_ret):
            m = st.mean(seed_ret[v])
            t_ = nw_t(seed_ret[v], H - 1) or 0.0
            print(f"  {v:>6}{seed_cnt[v]:>8}{100*seed_cnt[v]/total:>7.0f}%"
                  f"{100*m:>+12.3f}pp{t_:>+8.2f}")
            agg[v].append(m)
            tot_counts[v] += seed_cnt[v]
        print()

    print("=" * 60)
    print("ACROSS SEEDS")
    print("=" * 60)
    total = sum(tot_counts.values())
    print(f"  {'votes':>6}{'share':>8}{'mean':>13}{'min':>13}{'max':>13}"
          f"{'seeds +':>9}")
    for v in sorted(agg):
        y = agg[v]
        print(f"  {v:>6}{100*tot_counts[v]/total:>7.0f}%{100*st.mean(y):>+12.3f}pp"
              f"{100*min(y):>+12.3f}pp{100*max(y):>+12.3f}pp"
              f"{sum(1 for x in y if x > 0):>5}/{len(y)}")

    if 3 in agg and 1 in agg:
        d = st.mean(agg[3]) - st.mean(agg[1])
        print(f"\n  3-of-3 minus 1-of-3: {100*d:+.3f}pp")
        print("  Positive and consistent across seeds means agreement carries")
        print("  information. Near zero means the models agree about noise.")
    print("\n  Watch the SHARE column too. If 3-of-3 is most of the")
    print("  nominations, the models are near-identical and consensus is not a")
    print("  filter -- it is the same list with extra steps.")


if __name__ == "__main__":
    main()
