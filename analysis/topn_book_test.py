#!/usr/bin/env python3
"""
topn_book_test.py — is a daily top-N cross-sectional book real?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

THE FINDING THIS TESTS
    target_survival_test.py (2026-09-05) found that the any_positive target --
    the one already in production -- has a ranking whose HEAD carries the
    return:

        cap 1    +1.17pp vs universe    197/339 win days
        cap 2    +0.78pp               194/339
        cap 3    +0.67pp               194/339
        cap 5    +0.42pp               190/339
        decile   +0.26pp               182/339

    Monotone: tighter selection, higher return, more win days. That is what a
    genuine ordering looks like, and it is the OPPOSITE of PCT7, whose cap-1
    was -0.17pp and cap-2 -0.66pp while its full decile was +0.39pp.

    So the production model's target was never the problem, and the kill switch
    was justified for the model AS CONFIGURED -- it emits BUYs at
    prob_eff >= 0.70 across the whole universe, which is a different strategy
    from a per-day top-N.

WHAT IS NOT YET ESTABLISHED, AND WHY THIS SCRIPT EXISTS
    A single train/test split on 60 tickers with one seed is one draw. Before
    anything is built on it, four things have to hold:

  1. SEED STABILITY. Different ticker samples must give a similar answer. If
     cap-1 swings from +1.17pp to negative on another 60 tickers, the finding is
     a sample artifact.

  2. WALK-FORWARD, not a single split. One boundary at 70% is one regime
     transition. Refit each quarter and score the next, so the result is an
     average over several out-of-sample periods rather than one.

  3. COST AND TURNOVER AT THE ACTUAL CAP. The cost ladder in the previous test
     was computed on the decile. At cap 1-3 the position count is small but
     turnover may be near total -- a different name every day is 100% turnover,
     and at a 5-day hold that is a full round trip per position per week.
     Measured here rather than assumed.

  4. CONCENTRATION IN NAMES, not just days. If cap-1 is the same three tickers
     for months, it is a bet on those tickers. Reported as the distinct-ticker
     count and the top-ticker share.

    A finding that survives all four is worth building. One that needs a
    particular seed, a particular split date, or zero turnover is not.

    python analysis/topn_book_test.py --tickers 60 --seeds 3
"""
import argparse
import statistics as st
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")


def build_panel(universe, start, H):
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


def score(X, fwd, dates, tr_end, te_end, caps):
    """Fit on dates < tr_end, score dates in [tr_end, te_end)."""
    from xgboost import XGBClassifier
    ktr = [k for k in X if k[1] < tr_end]
    kte = [k for k in X if tr_end <= k[1] < te_end]
    if len(ktr) < 3000 or len(kte) < 500:
        return None
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", verbosity=0)
    m.fit([X[k] for k in ktr], [1 if fwd[k] > 0 else 0 for k in ktr])
    p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]
    byd = defaultdict(list)
    for i, k in enumerate(kte):
        byd[k[1]].append((p[i], fwd[k], k[0]))
    days = {d: sorted(v, reverse=True) for d, v in byd.items()
            if len(v) >= 10}
    out = {}
    for cap in caps:
        rs, picks, wins = [], [], 0
        prev = set()
        turn = []
        for d in sorted(days):
            v = days[d][:cap]
            rs.append(st.mean(r for _, r, _ in v))
            names = {t for _, _, t in v}
            picks += list(names)
            if prev:
                turn.append(100.0 * len(names - prev) / max(len(names), 1))
            prev = names
            if st.mean(r for _, r, _ in v) > st.mean(r for _, r, _ in days[d]):
                wins += 1
        uni = st.mean(st.mean(r for _, r, _ in v) for v in days.values())
        out[cap] = {
            "ret": st.mean(rs), "uni": uni, "wins": wins, "days": len(days),
            "turnover": st.mean(turn) if turn else 0.0,
            "tickers": len(set(picks)),
            "top_share": (100.0 * Counter(picks).most_common(1)[0][1]
                          / max(len(picks), 1)) if picks else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    CAPS = (1, 2, 3, 5, 10)

    sys.path.insert(0, ".")
    import random
    universe_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]

    print(f"top-N book test — {args.seeds} seeds x {args.tickers} tickers, "
          f"h={args.horizon}\n")
    agg = defaultdict(list)
    for seed in range(1, args.seeds + 1):
        u = universe_all[:]
        random.Random(seed).shuffle(u)
        u = u[:args.tickers]
        X, fwd = build_panel(u, args.start, args.horizon)
        if len(X) < 5000:
            print(f"seed {seed}: only {len(X)} rows, skipped")
            continue
        dates = sorted({k[1] for k in X})

        # 1. single split, matching the previous test
        cut = dates[int(len(dates) * 0.70)]
        s = score(X, fwd, dates, cut, "9999", CAPS)
        print(f"SEED {seed} — {len(X):,} rows, single split at {cut}")
        print(f"  {'cap':>5}{'ret':>9}{'vs uni':>10}{'win days':>11}"
              f"{'turnover':>10}{'tickers':>9}{'top name':>10}")
        for c in CAPS:
            r = s[c]
            print(f"  {c:>5}{100*r['ret']:>8.2f}%"
                  f"{100*(r['ret']-r['uni']):>+9.2f}pp"
                  f"{r['wins']:>6}/{r['days']}{r['turnover']:>9.0f}%"
                  f"{r['tickers']:>9}{r['top_share']:>9.0f}%")
            agg[("single", c)].append(100 * (r["ret"] - r["uni"]))

        # 2. walk-forward: refit each quarter
        qs = sorted({d[:7] for d in dates})
        anchors = [q for q in qs if q >= qs[max(0, int(len(qs) * 0.5))]][::3]
        wf = defaultdict(list)
        for i in range(len(anchors) - 1):
            a, b = anchors[i] + "-01", anchors[i + 1] + "-01"
            s2 = score(X, fwd, dates, a, b, CAPS)
            if not s2:
                continue
            for c in CAPS:
                wf[c].append(100 * (s2[c]["ret"] - s2[c]["uni"]))
        if wf:
            print(f"  walk-forward, {len(wf[CAPS[0]])} quarterly refits:")
            for c in CAPS:
                v = wf[c]
                if v:
                    print(f"    cap {c:>2}  mean {st.mean(v):+.2f}pp   "
                          f"positive {sum(1 for x in v if x>0)}/{len(v)}")
                    agg[("wf", c)].append(st.mean(v))
        print()

    print("=" * 66)
    print("ACROSS SEEDS — the stability check")
    print("=" * 66)
    print(f"  {'method':<10}{'cap':>5}{'mean':>9}{'min':>9}{'max':>9}")
    for meth in ("single", "wf"):
        for c in CAPS:
            v = agg.get((meth, c), [])
            if len(v) < 2:
                continue
            print(f"  {meth:<10}{c:>5}{st.mean(v):>+8.2f}pp"
                  f"{min(v):>+8.2f}pp{max(v):>+8.2f}pp")
    print("\n  A finding that holds across seeds AND across quarterly refits is")
    print("  worth building on. One that needs a particular sample or split")
    print("  date is a draw, not a signal. Watch turnover too: at cap 1-3 a")
    print("  different name each day is 100% turnover and a full round trip per")
    print("  position per week.")


if __name__ == "__main__":
    main()
