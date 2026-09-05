#!/usr/bin/env python3
"""
horizon_sweep_test.py — does a longer horizon help? h=3 / 5 / 20 / 40.

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

WHY LONGER
    Everything measured on 2026-09-05 at h=3/5 landed near zero in money:
    six insider constructions, four label definitions, linear vs XGBoost,
    ensembles, and consensus voting. Meanwhile the fund's ONE validated edge --
    short interest, low days-to-cover -- runs at h=40 with IC -0.037, NW-t
    -3.12, surviving FINRA's 8-business-day publication lag.

    Two reasons that is not a coincidence:

    1. SIGNAL-TO-NOISE SCALES WITH HORIZON. Daily equity returns have an SNR
       near 0.8% (return sd ~3.49% against a mean ~0.028%), versus ~100% for
       image classification. Noise grows with the square root of time while a
       persistent drift grows linearly, so the ratio improves as the horizon
       extends. That is also why the C=1e-05 sweep won by shrinking almost to a
       constant: at this SNR the optimal action is close to predicting the mean.

    2. TURNOVER. Every h=5 result today died on friction, not on sign. The
       top-N book was +0.2pp gross at 52-61% turnover, which is a round trip per
       position per week. At h=40 a book rebalances roughly 6 times a year.

WHAT IS MEASURED
    The same feature panel and the same models at four horizons. For each,
    day-weighted excess return over the same day's universe at several caps,
    plus AUC alongside so the decoupling stays visible, plus turnover -- which
    is the whole point of going longer.

    Both linear (C=1e-05, the plateau of the regularisation sweep) and XGBoost,
    on identical folds.

    MULTI-SEED. Three single-seed results reversed on replication today: a
    top-N book at +1.17pp went to -0.81pp, a linear economic edge at +0.645pp
    went to -0.021pp, and 3-of-3 model consensus came in at -0.173pp against
    1-of-3. Any number that does not repeat across seeds is a draw.

WHAT WOULD BE A REAL RESULT
    Excess return rising with horizon AND positive in every seed AND turnover
    falling enough that the gross number survives cost. Anything less is the
    same near-zero seen all day, measured over a longer window.

    Note h=40 needs ~2 extra months of forward data per observation, so the
    usable sample shrinks at the long end -- reported as the date count.

    python analysis/horizon_sweep_test.py --seeds 3 --tickers 80
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

HORIZONS = (3, 5, 20, 40)
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


def build(universe, start, max_h):
    """One panel, forward returns at every horizon, so the models see the
    identical feature rows and only the target changes."""
    from features.builder import build_feature_dataframe
    X = {}
    fwd = {h: {} for h in HORIZONS}
    for t in universe:
        try:
            df = build_feature_dataframe(t, start_date=start,
                                         training_mode=True)
            if df is None or len(df) < 400 or "close" not in df.columns:
                continue
            num = df.select_dtypes("number")
            num = num.drop(columns=[c for c in num.columns
                                    if c.startswith("target_")],
                           errors="ignore")
            ds = [str(d)[:10] for d in df["date"]]
            cl = list(df["close"])
            for j in range(20, len(ds) - max_h):
                a = cl[j]
                if not a:
                    continue
                ok = False
                for h in HORIZONS:
                    b = cl[j + h] if j + h < len(cl) else None
                    if b:
                        r = (b - a) / a
                        if abs(r) < 1.5:
                            fwd[h][(t, ds[j])] = r
                            ok = True
                if ok:
                    X[(t, ds[j])] = [float(v) if v == v else float("nan")
                                     for v in num.iloc[j].tolist()]
        except Exception:
            continue
    return X, fwd


def score(X, fwdh, tr_end, te_end, C, h, min_names):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    ktr = [k for k in X if k[1] < tr_end and k in fwdh]
    kte = [k for k in X if tr_end <= k[1] < te_end and k in fwdh]
    if len(ktr) < 4000 or len(kte) < 800:
        return None
    ytr = [1 if fwdh[k] > 0 else 0 for k in ktr]
    yte = [1 if fwdh[k] > 0 else 0 for k in kte]
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
            byd[k[1]].append((p[i], fwdh[k], k[0]))
        days = {d: sorted(v, reverse=True) for d, v in byd.items()
                if len(v) >= min_names}
        if len(days) < 12:
            return None
        res = {"auc": auc_of(p, yte), "days": len(days)}
        for cap in CAPS:
            ex, turn, prev = [], [], set()
            for d in sorted(days):
                v = days[d]
                sel = v[:cap]
                ex.append(st.mean(x[1] for x in sel)
                          - st.mean(x[1] for x in v))
                cur = {x[2] for x in sel}
                if prev:
                    turn.append(100.0 * len(cur - prev) / max(len(cur), 1))
                prev = cur
            res[cap] = (st.mean(ex), st.mean(turn) if turn else 0.0)
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--C", type=float, default=1e-5)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--min-names", type=int, default=25)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import random
    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    print(f"horizon sweep — {args.seeds} seeds x {args.tickers} tickers, "
          f"h={HORIZONS}, C={args.C}\n")

    agg = defaultdict(lambda: defaultdict(list))
    for seed in range(1, args.seeds + 1):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        X, fwd = build(u[:args.tickers], args.start, max(HORIZONS))
        if len(X) < 15000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue
        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        print(f"SEED {seed} — {len(X):,} feature rows")
        print(f"  {'model':<9}{'h':>4}{'AUC':>8}"
              + "".join(f"{'cap'+str(c):>10}{'trn':>6}" for c in CAPS)
              + f"{'dates':>7}")
        for h in HORIZONS:
            acc = defaultdict(lambda: defaultdict(list))
            for i in range(len(anchors) - 1):
                r = score(X, fwd[h], anchors[i] + "-01",
                          anchors[i + 1] + "-01", args.C, h, args.min_names)
                if not r:
                    continue
                for mdl in r:
                    acc[mdl]["auc"].append(r[mdl]["auc"])
                    acc[mdl]["days"].append(r[mdl]["days"])
                    for c in CAPS:
                        acc[mdl][c].append(r[mdl][c][0])
                        acc[mdl][("t", c)].append(r[mdl][c][1])
            for mdl in ("linear", "xgboost"):
                if not acc[mdl]["auc"]:
                    continue
                row = f"  {mdl:<9}{h:>4}{st.mean(acc[mdl]['auc']):>8.4f}"
                for c in CAPS:
                    row += (f"{100*st.mean(acc[mdl][c]):>+9.3f}pp"
                            f"{st.mean(acc[mdl][('t', c)]):>5.0f}%")
                row += f"{sum(acc[mdl]['days']):>7}"
                print(row)
                agg[(mdl, h)]["auc"].append(st.mean(acc[mdl]["auc"]))
                for c in CAPS:
                    agg[(mdl, h)][c].append(st.mean(acc[mdl][c]))
        print()

    print("=" * 72)
    print("ACROSS SEEDS")
    print("=" * 72)
    print(f"  {'model':<9}{'h':>4}{'AUC':>8}"
          + "".join(f"{'cap'+str(c):>11}{'+':>4}" for c in CAPS))
    for mdl in ("linear", "xgboost"):
        for h in HORIZONS:
            a = agg.get((mdl, h))
            if not a or not a["auc"]:
                continue
            row = f"  {mdl:<9}{h:>4}{st.mean(a['auc']):>8.4f}"
            for c in CAPS:
                v = a[c]
                row += (f"{100*st.mean(v):>+10.3f}pp"
                        f"{sum(1 for x in v if x > 0):>2}/{len(v)}")
            print(row)
        print()

    print("  'trn' is turnover -- the share of each period's names that are")
    print("  new. It is the whole reason to go longer: every h=5 result today")
    print("  died on friction rather than on sign.\n")
    print("  A real result is excess return RISING with horizon, positive in")
    print("  EVERY seed, with turnover low enough that the gross number")
    print("  survives cost. The fund's one working signal is the SI brick at")
    print("  h=40, which is consistent with that shape.")


if __name__ == "__main__":
    main()
