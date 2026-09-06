#!/usr/bin/env python3
"""
group_model_test.py — one model for everything, or one per size bucket?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHY
    A 2025 paper on modelling choices in cross-sectional return prediction
    found that re-weighting observations by market-capitalisation GROUPS
    improves predictability and strategy performance by addressing data
    heterogeneity, and tested two implementations: training GROUP-SPECIFIC
    MODELS, and predicting RELATIVE returns. Both gave similar economic
    improvements over a model trained on the full cross-section, and
    value-weighted strategies benefited most -- which is the relevant case for
    a book that only buys liquid names.

    That is a different prescription from simply down-weighting small caps in
    the loss, which nothing in the literature supports. Gu, Kelly & Xiu train
    EQUAL-WEIGHTED and deliberately keep small stocks, noting that a larger
    sample raises the observation-to-parameter ratio and guards against
    overfitting -- while also reporting their results are "qualitatively
    identical and quantitatively unchanged" if those firms are filtered out.

    This fund's own survivorship slice test (2026-09-05) found the h=40 cap-3
    edge at +2.035pp for mega, +1.806pp large, +2.274pp mid and +3.779pp small,
    every slice 3/3 seeds. That gradient matches the documented finding that
    predictability concentrates in small illiquid stocks -- Chen reports the
    share of significant t-stats falling from ~33% equal-weighted to ~16%
    value-weighted for exactly that reason.

WHAT IS TESTED
    Four configurations on identical folds and seeds:

      pooled_all     one model trained on every name, buy top 3 from every name
                     -- the current design
      pooled_buy400  one model trained on every name, buy top 3 ONLY from the
                     400 most liquid -- separates the RANKING pool from the
                     BUYING pool, which is the practical question for a book
                     that will not hold illiquid names
      group_large    a model trained ONLY on the top 400, buying from them
      group_small    a model trained ONLY on the rest, buying from them

    pooled_buy400 against group_large is the direct test of whether the extra
    1,500 names help the model rank the 400 it actually buys, or just add noise.

    pooled_all against group_large + group_small answers the paper's question:
    does splitting by size beat pooling?

THE BAR
    NW t > 3.0 per Harvey, Liu & Zhu, not 2.0, given the number of
    configurations already tested on this data. And multi-seed: three
    single-seed results reversed on replication on 2026-09-05.

    python analysis/group_model_test.py --seeds 3
"""
import argparse
import math
import sqlite3
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

T_HURDLE = 3.0
CAPS = (3, 5)
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
    ap.add_argument("--tickers", type=int, default=140,
                    help="sampled per seed, split by liquidity rank")
    ap.add_argument("--liquid-n", type=int, default=400,
                    help="the BUY pool: this many most liquid names")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--min-names", type=int, default=20)
    ap.add_argument("--universe", default="tickers_expanded.txt")
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    uni = {l.strip().upper() for l in open(args.universe) if l.strip()}
    con = sqlite3.connect("file:prices.db?mode=ro", uri=True)
    adv = {t: float(a) for t, a in con.execute(
        "SELECT ticker, AVG(close*volume) FROM raw_bars "
        "WHERE d >= date('now','-60 days') AND close>0 AND volume>0 "
        "GROUP BY ticker") if a and t in uni}
    con.close()
    ranked = sorted(adv, key=lambda x: -adv[x])
    liquid = set(ranked[:args.liquid_n])
    print(f"{len(ranked)} names; BUY pool = top {args.liquid_n} by ADV "
          f"(floor ${adv[ranked[args.liquid_n-1]]/1e6:.0f}M)\n")

    agg = defaultdict(lambda: defaultdict(list))
    for seed in range(1, args.seeds + 1):
        rnd = random.Random(seed)
        big = ranked[:args.liquid_n][:]
        small = ranked[args.liquid_n:][:]
        rnd.shuffle(big)
        rnd.shuffle(small)
        # half the sample from each pool so both models see comparable data
        sample = big[:args.tickers // 2] + small[:args.tickers // 2]

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
        if len(X) < 20000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        res = defaultdict(lambda: defaultdict(list))
        for i in range(len(anchors) - 1):
            tr, te = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr]
            kte = [k for k in X if tr <= k[1] < te]
            if len(ktr) < 5000 or len(kte) < 1000:
                continue

            def fit_predict(train_keys, test_keys):
                if len(train_keys) < 3000 or len(test_keys) < 300:
                    return None
                m = XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8,
                                  eval_metric="logloss", verbosity=0)
                m.fit([X[k] for k in train_keys],
                      [1 if fwd[k] > 0 else 0 for k in train_keys])
                p = m.predict_proba([X[k] for k in test_keys])[:, 1]
                return {test_keys[z]: float(p[z])
                        for z in range(len(test_keys))}

            p_all = fit_predict(ktr, kte)
            ktr_b = [k for k in ktr if k[0] in liquid]
            kte_b = [k for k in kte if k[0] in liquid]
            ktr_s = [k for k in ktr if k[0] not in liquid]
            kte_s = [k for k in kte if k[0] not in liquid]
            p_big = fit_predict(ktr_b, kte_b)
            p_small = fit_predict(ktr_s, kte_s)

            def score(label, preds, restrict=None):
                if not preds:
                    return
                byd = defaultdict(list)
                for k, pv in preds.items():
                    if restrict and k[0] not in restrict:
                        continue
                    byd[k[1]].append((pv, fwd[k]))
                for d, v in byd.items():
                    if len(v) < args.min_names:
                        continue
                    v.sort(reverse=True)
                    mkt = st.mean(r for _, r in v)
                    for c in CAPS:
                        res[label][c].append(
                            st.mean(r for _, r in v[:c]) - mkt)
                    hi = [r for pv, r in v if pv >= THRESH]
                    if hi:
                        res[label]["thr"].append(st.mean(hi) - mkt)

            score("pooled_all", p_all)
            score("pooled_buy400", p_all, restrict=liquid)
            score("group_large", p_big)
            score("group_small", p_small)

        print(f"SEED {seed} — {len(X):,} rows")
        print(f"  {'config':<16}{'cap3':>10}{'cap5':>10}{'prob>=.70':>12}"
              f"{'NW t (cap3)':>13}{'n':>6}")
        for lab in ("pooled_all", "pooled_buy400", "group_large",
                    "group_small"):
            r = res.get(lab)
            if not r or not r.get(3):
                continue
            t_ = nw_t(r[3], 3) or 0.0
            print(f"  {lab:<16}{100*st.mean(r[3]):>+9.3f}pp"
                  f"{100*st.mean(r[5]):>+9.3f}pp"
                  + (f"{100*st.mean(r['thr']):>+11.3f}pp" if r.get("thr")
                     else f"{'-':>12}")
                  + f"{t_:>+13.2f}{len(r[3]):>6}"
                  + ("  t>3" if abs(t_) > T_HURDLE else ""))
            agg[lab]["cap3"].append(st.mean(r[3]))
            agg[lab]["cap5"].append(st.mean(r[5]))
            if r.get("thr"):
                agg[lab]["thr"].append(st.mean(r["thr"]))
        print()

    if not agg:
        print("no results")
        return
    print("=" * 66)
    print("ACROSS SEEDS")
    print("=" * 66)
    print(f"  {'config':<16}{'cap3':>11}{'+':>5}{'cap5':>11}{'+':>5}"
          f"{'prob>=.70':>12}{'+':>5}")
    for lab in ("pooled_all", "pooled_buy400", "group_large", "group_small"):
        a = agg.get(lab)
        if not a or not a["cap3"]:
            continue
        row = f"  {lab:<16}"
        for key in ("cap3", "cap5", "thr"):
            v = a.get(key, [])
            if v:
                row += (f"{100*st.mean(v):>+10.3f}pp"
                        f"{sum(1 for x in v if x>0):>2}/{len(v)}")
            else:
                row += f"{'-':>11}{'-':>5}"
        print(row)

    print("\n  pooled_buy400 vs group_large is the direct test: does training")
    print("  on the extra ~1,500 names help the model rank the 400 it actually")
    print("  buys, or just add noise? Same buy pool, different training set.\n")
    print("  pooled_all vs group_large+group_small answers the paper's")
    print("  question: does splitting by size beat pooling? It found")
    print("  group-specific models improve on full-cross-section training,")
    print("  with value-weighted strategies benefiting most.\n")
    print(f"  Bar is NW t > {T_HURDLE}, not 2.0 -- Harvey, Liu & Zhu, given how")
    print("  many configurations have now been tested on this data.")


if __name__ == "__main__":
    main()
