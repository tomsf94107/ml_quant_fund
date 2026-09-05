#!/usr/bin/env python3
"""
blend_multiseed_test.py — does linear + XGBoost beat either alone?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

THE CASE FOR TRYING
    The existing XGB + LightGBM ensemble was measured on 2026-09-05 at +0.002
    AUC (t=+0.63, better on 13 of 30 ticker-horizon pairs). Its header claims
    "+1-3%". Two gradient-boosted tree libraries on one feature set make
    CORRELATED errors, so averaging them cancels almost nothing.

    Linear and trees are genuinely different classes -- linear captures monotone
    global structure, trees capture interactions and thresholds -- so their
    errors should decorrelate, which is what ensembling actually requires.

    And there is a measured asymmetry to exploit: on 246 tickers the linear
    model reached test AUC 0.5756 with a train-test gap of 0.046, against
    XGBoost's ~0.50 and gap ~0.25, holding at h=3 as well (0.5592).

WHAT IS ALSO KNOWN, AND WHY EXPECTATIONS SHOULD BE LOW
    AUC and return have decoupled in EVERY test this session:

      top_decile target   AUC 0.7316  ->  +0.01pp day-weighted
      PCT7                real within-date edge  ->  -1.97% day-weighted
      linear alone        AUC 0.5756  ->  -0.021pp at cap 3 across 3 seeds
      XGB+LGB ensemble    +0.002 AUC

    So a blend is expected to improve the thing that already works (ranking the
    full cross-section) and leave untouched the thing that does not (which few
    names to actually buy). This runs it because the test is cheap and the
    harness exists, not because the prior is good.

WHAT IS TESTED
    Four models on identical folds, every seed:
      linear      L2-logistic at C=1e-05, the plateau of the regularisation
                  sweep, where coefficients are shrunk nearly to zero and the
                  model approaches an equal-weight z-score composite
      xgboost     the production estimator
      blend50     an even average of the two probabilities
      blend_opt   weight chosen on a VALIDATION slice carved out of training,
                  never on test -- mirroring models/ensemble.py::_optimal_weights

    Scored on day-weighted excess return over the same day's universe at several
    caps, NOT on AUC, plus AUC alongside so the decoupling is visible in one
    table.

    MULTI-SEED IS NON-NEGOTIABLE HERE. A top-N book measured +1.17pp on seed 5
    and +0.14 / -0.32 / -0.81pp on seeds 1-3, and the linear economic result
    measured +0.645pp at cap 3 on seed 5 and -0.021pp across seeds 1-3. Both
    were single-seed artifacts. Any number below that does not repeat across
    seeds should be read as a draw.

    python analysis/blend_multiseed_test.py --seeds 3 --tickers 100
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

CAPS = (3, 5, 10)


def auc_of(scores, labels):
    n = len(scores)
    pos = sum(labels)
    if not pos or pos == n:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        a = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = a
        i = j + 1
    rs = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (rs - pos * (pos + 1) / 2.0) / (pos * (n - pos))


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


def fit_and_score(X, fwd, tr_end, te_end, C, H, min_names):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    ktr_all = sorted([k for k in X if k[1] < tr_end], key=lambda k: k[1])
    kte = [k for k in X if tr_end <= k[1] < te_end]
    if len(ktr_all) < 5000 or len(kte) < 1000:
        return None

    # carve a VALIDATION slice off the END of training for the blend weight.
    # Never touches test -- same discipline as models/ensemble.py.
    vcut = int(len(ktr_all) * 0.85)
    ktr, kva = ktr_all[:vcut], ktr_all[vcut:]
    if len(kva) < 500:
        return None

    ytr = [1 if fwd[k] > 0 else 0 for k in ktr]
    yva = [1 if fwd[k] > 0 else 0 for k in kva]
    yte = [1 if fwd[k] > 0 else 0 for k in kte]

    lin = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("lr", LogisticRegression(C=C, max_iter=2000))])
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", verbosity=0)
    Xtr = [X[k] for k in ktr]
    lin.fit(Xtr, ytr)
    xgb.fit(Xtr, ytr)

    Xva = [X[k] for k in kva]
    pl_va = [float(v) for v in lin.predict_proba(Xva)[:, 1]]
    px_va = [float(v) for v in xgb.predict_proba(Xva)[:, 1]]
    best_w, best = 0.5, -1.0
    for w in [i / 10 for i in range(0, 11)]:
        a = auc_of([w * pl_va[i] + (1 - w) * px_va[i]
                    for i in range(len(yva))], yva)
        if a is not None and a > best:
            best, best_w = a, w

    Xte = [X[k] for k in kte]
    pl = [float(v) for v in lin.predict_proba(Xte)[:, 1]]
    px = [float(v) for v in xgb.predict_proba(Xte)[:, 1]]
    preds = {
        "linear": pl,
        "xgboost": px,
        "blend50": [0.5 * pl[i] + 0.5 * px[i] for i in range(len(pl))],
        "blend_opt": [best_w * pl[i] + (1 - best_w) * px[i]
                      for i in range(len(pl))],
    }

    out = {"_w": best_w}
    for name, p in preds.items():
        byd = defaultdict(list)
        for i, k in enumerate(kte):
            byd[k[1]].append((p[i], fwd[k]))
        days = {d: sorted(v, reverse=True) for d, v in byd.items()
                if len(v) >= min_names}
        if len(days) < 15:
            return None
        res = {"auc": auc_of(p, yte)}
        for cap in CAPS:
            ex = [st.mean(r for _, r in v[:cap]) - st.mean(r for _, r in v)
                  for v in days.values()]
            res[cap] = st.mean(ex)
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
    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    names = ("linear", "xgboost", "blend50", "blend_opt")

    print(f"linear / xgboost / blends — {args.seeds} seeds x {args.tickers} "
          f"tickers, h={H}, C={args.C}\n")
    agg = defaultdict(lambda: defaultdict(list))
    ws = []

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

        sr = defaultdict(lambda: defaultdict(list))
        for i in range(len(anchors) - 1):
            r = fit_and_score(X, fwd, anchors[i] + "-01",
                              anchors[i + 1] + "-01", args.C, H,
                              args.min_names)
            if not r:
                continue
            ws.append(r["_w"])
            for m in names:
                sr[m]["auc"].append(r[m]["auc"])
                for cap in CAPS:
                    sr[m][cap].append(r[m][cap])
        if not sr["linear"]["auc"]:
            print(f"seed {seed}: no scoreable refits\n")
            continue

        print(f"SEED {seed} — {len(X):,} rows, "
              f"{len(sr['linear']['auc'])} refits")
        print(f"  {'model':<11}{'AUC':>8}" +
              "".join(f"{'cap '+str(c):>11}" for c in CAPS))
        for m in names:
            print(f"  {m:<11}{st.mean(sr[m]['auc']):>8.4f}" +
                  "".join(f"{100*st.mean(sr[m][c]):>+10.3f}pp" for c in CAPS))
            agg[m]["auc"].append(st.mean(sr[m]["auc"]))
            for c in CAPS:
                agg[m][c].append(st.mean(sr[m][c]))
        print()

    print("=" * 66)
    print("ACROSS SEEDS")
    print("=" * 66)
    print(f"  {'model':<11}{'AUC':>8}" +
          "".join(f"{'cap '+str(c):>11}{'+':>4}" for c in CAPS))
    for m in names:
        if not agg[m]["auc"]:
            continue
        row = f"  {m:<11}{st.mean(agg[m]['auc']):>8.4f}"
        for c in CAPS:
            v = agg[m][c]
            row += f"{100*st.mean(v):>+10.3f}pp{sum(1 for x in v if x>0):>2}/{len(v)}"
        print(row)
    if ws:
        print(f"\n  blend_opt weight on linear: mean {st.mean(ws):.2f}, "
              f"range {min(ws):.1f}-{max(ws):.1f}")

    print("\n  The AUC column is where a blend should help, and where every")
    print("  model this session has already looked fine. The cap columns are")
    print("  what decides anything, and the '+' counts how many seeds were")
    print("  positive. A blend that lifts AUC while the cap columns stay near")
    print("  zero is the same decoupling seen all day, not a new result.")


if __name__ == "__main__":
    main()
