#!/usr/bin/env python3
"""
linear_multiseed_test.py — does the linear result survive different samples?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

WHY THIS IS MANDATORY BEFORE ANYTHING IS BUILT
    linear_economic_test.py (2026-09-05, seed 5, 120 tickers) found the linear
    model's AUC edge converting to day-weighted excess return, while XGBoost on
    identical folds went negative:

        cap    linear      xgboost
        1      +1.283pp    -0.660pp
        3      +0.645pp    -0.467pp  (t=-2.01)
        5      +0.357pp    -0.311pp
        decile +0.180pp    -0.105pp

    Monotone at the head for linear -- the signature of a real ranking, and the
    exact place PCT7 failed. Turnover also lower: 42% at cap 3 against 70%.

    But that is ONE seed and ONE split, and this same session already produced
    the counter-example: a top-N book measured +1.17pp on seed 5 and gave
    +0.14 / -0.32 / -0.81pp on seeds 1, 2 and 3. The single-split figure was a
    draw, not a finding, and only the multi-seed walk-forward revealed it.

    Linear cap-3 runs t=+1.48 on 339 overlapping days -- roughly 68 independent
    5-day periods. That does not clear any bar on its own. So the question is
    entirely whether it repeats.

WHAT THIS RUNS
    The same economic measurement across N seeds, each drawing a different
    ticker sample, and within each seed a walk-forward with quarterly refits
    rather than a single 70/30 boundary. One split is one regime transition;
    several refits average over more.

    Both models on every seed, so the linear-minus-xgboost difference is
    measured on identical data rather than compared across runs.

    Reported per seed and then pooled, with min and max across seeds. A result
    that holds is worth building on. One that swings sign is a draw.

WHAT WOULD MAKE THIS ACTIONABLE
    Linear positive at cap 3-5 in EVERY seed, and beating XGBoost in every seed.
    Anything less -- and especially a seed where linear goes negative -- means
    the seed-5 numbers were a sample artifact, exactly like the top-N book.

    python analysis/linear_multiseed_test.py --seeds 3 --tickers 100
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

CAPS = (1, 3, 5, 10)


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


def evaluate(X, fwd, tr_end, te_end, C, H, min_names):
    """Fit both models on dates < tr_end, score [tr_end, te_end)."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    ktr = [k for k in X if k[1] < tr_end]
    kte = [k for k in X if tr_end <= k[1] < te_end]
    if len(ktr) < 5000 or len(kte) < 1000:
        return None
    ytr = [1 if fwd[k] > 0 else 0 for k in ktr]
    Xtr = [X[k] for k in ktr]
    Xte = [X[k] for k in kte]

    out = {}
    for name, m in (
        ("linear", Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("lr", LogisticRegression(C=C, max_iter=2000))])),
        ("xgboost", XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8,
                                  eval_metric="logloss", verbosity=0)),
    ):
        m.fit(Xtr, ytr)
        p = [float(v) for v in m.predict_proba(Xte)[:, 1]]
        byd = defaultdict(list)
        for i, k in enumerate(kte):
            byd[k[1]].append((p[i], fwd[k]))
        days = {d: sorted(v, reverse=True) for d, v in byd.items()
                if len(v) >= min_names}
        if len(days) < 15:
            return None
        res = {}
        for cap in CAPS:
            ex = []
            for d, v in days.items():
                sel = v[:cap]
                ex.append(st.mean(r for _, r in sel)
                          - st.mean(r for _, r in v))
            res[cap] = (st.mean(ex), nw_t(ex, H - 1) or 0.0, len(days))
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=100)
    ap.add_argument("--C", type=float, default=1e-5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--min-names", type=int, default=25)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    import random
    universe_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]

    print(f"linear vs xgboost, {args.seeds} seeds x {args.tickers} tickers, "
          f"h={H}, C={args.C}\n")
    agg = defaultdict(lambda: defaultdict(list))

    for seed in range(1, args.seeds + 1):
        u = universe_all[:]
        random.Random(seed).shuffle(u)
        X, fwd = build(u[:args.tickers], args.start, H)
        if len(X) < 20000:
            print(f"seed {seed}: only {len(X)} rows, skipped")
            continue
        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        seed_res = defaultdict(lambda: defaultdict(list))
        for i in range(len(anchors) - 1):
            r = evaluate(X, fwd, anchors[i] + "-01", anchors[i + 1] + "-01",
                         args.C, H, args.min_names)
            if not r:
                continue
            for mdl in r:
                for cap in CAPS:
                    seed_res[mdl][cap].append(r[mdl][cap][0])

        if not seed_res:
            print(f"seed {seed}: no scoreable refits\n")
            continue
        nref = len(seed_res["linear"][CAPS[0]])
        print(f"SEED {seed} — {len(X):,} rows, {nref} quarterly refits")
        print(f"  {'cap':>5}{'linear':>11}{'xgboost':>11}{'diff':>11}")
        for cap in CAPS:
            L = st.mean(seed_res["linear"][cap])
            Xg = st.mean(seed_res["xgboost"][cap])
            print(f"  {cap:>5}{100*L:>+10.3f}pp{100*Xg:>+10.3f}pp"
                  f"{100*(L-Xg):>+10.3f}pp")
            agg["linear"][cap].append(L)
            agg["xgboost"][cap].append(Xg)
        print()

    print("=" * 60)
    print("ACROSS SEEDS")
    print("=" * 60)
    print(f"  {'model':<9}{'cap':>5}{'mean':>11}{'min':>11}{'max':>11}"
          f"{'seeds +':>9}")
    for mdl in ("linear", "xgboost"):
        for cap in CAPS:
            v = agg[mdl][cap]
            if len(v) < 2:
                continue
            print(f"  {mdl:<9}{cap:>5}{100*st.mean(v):>+10.3f}pp"
                  f"{100*min(v):>+10.3f}pp{100*max(v):>+10.3f}pp"
                  f"{sum(1 for x in v if x > 0):>5}/{len(v)}")
        print()

    print("  Positive at cap 3-5 in EVERY seed, and beating xgboost in every")
    print("  seed, is what would make this actionable. A seed where linear goes")
    print("  negative means the seed-5 result was a sample artifact -- which is")
    print("  exactly what happened to the top-N book earlier today (+1.17pp on")
    print("  seed 5, then +0.14 / -0.32 / -0.81pp on seeds 1-3).")


if __name__ == "__main__":
    main()
