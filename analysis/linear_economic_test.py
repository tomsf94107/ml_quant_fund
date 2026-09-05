#!/usr/bin/env python3
"""
linear_economic_test.py — does the linear model's AUC edge convert to money?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

THE FINDING THIS TESTS
    models/linear_baseline.py had never run -- it crashed on NaN for every
    ticker, because LogisticRegression raises on NaN while XGBoost handles it
    natively. Fixed 2026-09-05 with a train-fold-only imputation pipeline.

    With it running, an L2 regularisation sweep on 10 tickers at h=5:

        C        train    test    gap
        0.1      0.757    0.559   +0.198
        0.001    0.662    0.564   +0.098
        1e-05    0.609    0.574   +0.034
        1e-06    0.608    0.575   +0.033      <- plateau

    Train falls, test RISES, gap collapses. Monotone over five steps. And on
    125 tickers at C=0.001 the effect is stronger, not weaker: test 0.5728,
    gap +0.1015, and 108 of 125 tickers above 0.52.

    XGBoost on the same features, folds and target: train 0.66-0.79, test ~0.50,
    gap ~0.25. The tree is leaving roughly 7 AUC points on the table.

    At the plateau the L2 penalty is so severe that coefficients are shrunk
    nearly to zero, so the model approaches a plain equal-weighted composite of
    standardised features. The reading is that the signal is broad and diffuse
    across many weak features, and that fitting weights to them destroys it.

WHY AUC IS NOT ENOUGH, MEASURED TODAY
    On 2026-09-05 the top_decile target posted AUC 0.7316 -- far above
    production's 0.5100 -- and converted to +0.01pp of day-weighted return.
    PCT7 posted a real within-date edge and -1.97% day-weighted. A top-N book
    looked like +1.17pp on one seed and -0.33pp across three.

    Every AUC-shaped result today failed the economic test. So this script does
    not report AUC at all.

WHAT IT MEASURES
    Per date, cross-sectionally: rank all tickers by the linear model's
    probability, take the top N, and record the equal-weighted forward return
    against the universe mean that same day.

      - DAY-WEIGHTED, not pooled. A pooled mean over selections is the return of
        a portfolio holding every selection, which is not a strategy when daily
        counts vary. This weights each day once.
      - Against the SAME DAY's universe, so market direction cannot flatter it.
      - At several caps, because a real ranking should be BETTER at the head.
        PCT7 failed exactly here: +1.46% over all selections, -2.38% at the top 3.
      - Turnover is reported, because at 5-day holds a book that replaces itself
        daily pays a round trip per position per week, and the previous top-N
        finding died on precisely that.

    XGBoost is run on the identical folds so the comparison is like-for-like.

    python analysis/linear_economic_test.py --tickers 120 --C 1e-05
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=120)
    ap.add_argument("--C", type=float, default=1e-5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--min-names", type=int, default=30)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    import random

    universe = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    random.Random(args.seed).shuffle(universe)
    universe = universe[:args.tickers]
    print(f"building {len(universe)} tickers from {args.start} -- slow\n")

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
            if i % 30 == 0:
                print(f"  ...{i} tickers, {len(X):,} rows")
        except Exception:
            continue
    print(f"\n{len(X):,} panel rows\n")
    if len(X) < 10000:
        print("too few rows")
        return

    dates = sorted({k[1] for k in X})
    cut = dates[int(len(dates) * 0.70)]
    ktr = [k for k in X if k[1] < cut]
    kte = [k for k in X if k[1] >= cut]
    ytr = [1 if fwd[k] > 0 else 0 for k in ktr]
    print(f"train < {cut} ({len(ktr):,} rows), test >= {cut} "
          f"({len(kte):,} rows)\n")

    models = {
        "linear": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=args.C, max_iter=2000)),
        ]),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8,
                                 eval_metric="logloss", verbosity=0),
    }

    uni = defaultdict(list)
    for k in kte:
        uni[k[1]].append(fwd[k])
    uni_day = st.mean(st.mean(v) for v in uni.values())
    print(f"universe day-weighted mean over the test period: "
          f"{100*uni_day:+.3f}%\n")

    for name, m in models.items():
        m.fit([X[k] for k in ktr], ytr)
        p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]
        byd = defaultdict(list)
        for i, k in enumerate(kte):
            byd[k[1]].append((p[i], fwd[k], k[0]))
        days = {d: sorted(v, reverse=True) for d, v in byd.items()
                if len(v) >= args.min_names}

        print(f"=== {name} (C={args.C})" if name == "linear"
              else f"=== {name}")
        print(f"  {'cap':>7}{'ret/day':>10}{'vs uni':>10}{'NW t':>8}"
              f"{'win days':>11}{'turnover':>10}")
        for cap in (1, 3, 5, 10, 20, None):
            rs, wins, turn = [], 0, []
            prev = set()
            for d in sorted(days):
                v = days[d]
                n = cap if cap else max(1, len(v) // 10)
                sel = v[:n]
                r = st.mean(x[1] for x in sel)
                rs.append(r - st.mean(x[1] for x in v))   # excess, same day
                if r > st.mean(x[1] for x in v):
                    wins += 1
                cur = {x[2] for x in sel}
                if prev:
                    turn.append(100.0 * len(cur - prev) / max(len(cur), 1))
                prev = cur
            t_ = nw_t(rs, H - 1) or 0.0
            lbl = str(cap) if cap else "decile"
            print(f"  {lbl:>7}{100*st.mean(rs):>+9.3f}%"
                  f"{100*st.mean(rs):>+9.3f}pp{t_:>+8.2f}"
                  f"{wins:>6}/{len(days)}"
                  f"{(st.mean(turn) if turn else 0):>9.0f}%")
        print()

    print("  'vs uni' is already the same-day excess, so the two columns match")
    print("  by construction -- market direction is removed per date.\n")
    print("  A real ranking is BETTER at the head. PCT7 failed exactly there:")
    print("  +1.46% over all its selections, -2.38% at the top 3. Watch whether")
    print("  cap 1-5 beats the decile, and watch turnover: at 5-day holds a book")
    print("  that replaces itself daily pays a round trip per position per week,")
    print("  which is what killed the earlier top-N result at +0.2pp gross.")


if __name__ == "__main__":
    main()
