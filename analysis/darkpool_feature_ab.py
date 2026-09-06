#!/usr/bin/env python3
"""
darkpool_feature_ab.py — paired A/B: do the dark-pool COLUMNS help or hurt?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHY A PAIRED TEST AND NOT TWO RUNS
    A first attempt compared two separate runs of h40_book_test.py, one with
    ML_QUANT_NO_DARKPOOL=1. It reported cap-1 rising from +2.933pp to +4.264pp
    and prob>=0.70 from +2.629pp to +3.704pp when the columns were dropped.

    That comparison is not clean. The two runs were executed at different times,
    the first built its panels before data/sector_etf_sic.csv existed and the
    second read cached panels afterwards, and the cap-3 gap was +0.112pp --
    inside the noise. Seed dispersion on this test has run +/-1.5pp all session.

    An earlier attempt was worse: it returned figures IDENTICAL TO THREE
    DECIMALS because h40_book_test.py builds its matrix from
    df.select_dtypes("number"), not from FEATURE_COLUMNS, so the flag wired into
    models/classifier.py had no effect there at all.

    This runs BOTH ARMS INSIDE ONE PROCESS, on the SAME panel, fitting twice per
    fold on exactly the same training rows. The only difference is the column
    set. Paired differences are then tested directly, which is far more powerful
    than comparing two independent means.

WHAT IS MEASURED, AND WHY BOTH
    AUC        does the model rank the full cross-section better or worse?
               This is the "accuracy" question.
    Excess     does the BOOK earn more? Cap-1, cap-3, cap-5 and prob>=0.70,
               day-weighted against the same day's universe.

    They can disagree. On 2026-09-05 the top_decile target posted AUC 0.7316
    against production's 0.5100 and converted to +0.01pp of return. AUC measures
    the whole ranking; money comes from the extremes.

THE PRIOR
    Both columns are sparse: dark-pool history begins 2026-03-19, so
    dp_volume_share is NaN on roughly 80% of a 2016-start panel and boulton_cell
    on roughly 92% -- the latter also needs 60 observations of a 252-day rolling
    quantile before it produces anything.

    A tree offered a mostly-empty column will still split on it. So a small
    NEGATIVE effect is plausible. So is no measurable effect at all, which would
    mean the earlier +1.3pp reading was seed noise.

    Either answer leaves the Boulton finding intact. That was validated as an
    EXCLUSION FILTER -- +1.118pp at cap-3, 3/3 seeds, direction predicted by the
    paper before the test ran -- and a filter that removes names is a different
    mechanism from a column the model must learn to read.

    python analysis/darkpool_feature_ab.py --seeds 3 --tickers 80
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

DROP = ("dp_volume_share", "boulton_cell")
CAPS = (1, 3, 5)
THRESH = 0.70


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


def paired_t(diffs):
    """Paired t on the per-observation differences. Far more powerful than
    comparing two independent means, because fold-level and date-level noise is
    common to both arms and cancels."""
    n = len(diffs)
    if n < 6:
        return None
    m = sum(diffs) / n
    sd = st.pstdev(diffs)
    return m / (sd / math.sqrt(n)) if sd > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--universe", default="tickers_top400.txt")
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    uni_all = [l.strip().upper() for l in open(args.universe) if l.strip()]
    print(f"paired A/B on {DROP}")
    print(f"  {args.seeds} seeds x {args.tickers} tickers, h={H}, "
          f"{args.universe}\n")

    agg = defaultdict(list)
    for seed in range(1, args.seeds + 1):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        X, fwd, cols = {}, {}, None
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
                if cols is None:
                    cols = list(num.columns)
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
        if len(X) < 15000 or cols is None:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        present = [c for c in DROP if c in cols]
        if not present:
            print(f"seed {seed}: {DROP} not in the panel -- nothing to test")
            print(f"  columns include: {', '.join(cols[:6])} ...")
            return
        keep_idx = [i for i, c in enumerate(cols) if c not in DROP]
        all_idx = list(range(len(cols)))
        nn = sum(1 for k in X
                 if X[k][cols.index(present[0])] == X[k][cols.index(present[0])])
        print(f"SEED {seed} — {len(X):,} rows, {len(cols)} columns, "
              f"dropping {len(present)}")
        print(f"  {present[0]} non-NaN on {100*nn/len(X):.0f}% of rows")

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        d_auc, d_cap = [], defaultdict(list)
        d_thr = []
        with_v, without_v = defaultdict(list), defaultdict(list)

        for i in range(len(anchors) - 1):
            tr, te = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr]
            kte = [k for k in X if tr <= k[1] < te]
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            ytr = [1 if fwd[k] > 0 else 0 for k in ktr]
            yte = [1 if fwd[k] > 0 else 0 for k in kte]

            preds = {}
            for arm, idx in (("with", all_idx), ("without", keep_idx)):
                m = XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8,
                                  eval_metric="logloss", verbosity=0,
                                  random_state=42)
                m.fit([[X[k][j] for j in idx] for k in ktr], ytr)
                preds[arm] = [float(v) for v in m.predict_proba(
                    [[X[k][j] for j in idx] for k in kte])[:, 1]]

            aw = auc_of(preds["with"], yte)
            ao = auc_of(preds["without"], yte)
            if aw is not None and ao is not None:
                d_auc.append(aw - ao)

            byd = defaultdict(lambda: {"with": [], "without": [], "r": []})
            for z, k in enumerate(kte):
                byd[k[1]]["with"].append((preds["with"][z], fwd[k]))
                byd[k[1]]["without"].append((preds["without"][z], fwd[k]))
            for d, v in byd.items():
                if len(v["with"]) < args.min_names:
                    continue
                mkt = st.mean(r for _, r in v["with"])
                a_s = sorted(v["with"], reverse=True)
                b_s = sorted(v["without"], reverse=True)
                for c in CAPS:
                    ea = st.mean(r for _, r in a_s[:c]) - mkt
                    eb = st.mean(r for _, r in b_s[:c]) - mkt
                    d_cap[c].append(ea - eb)
                    with_v[c].append(ea)
                    without_v[c].append(eb)
                ha = [r for p, r in a_s if p >= THRESH]
                hb = [r for p, r in b_s if p >= THRESH]
                if ha and hb:
                    d_thr.append((st.mean(ha) - mkt) - (st.mean(hb) - mkt))
                    with_v["thr"].append(st.mean(ha) - mkt)
                    without_v["thr"].append(st.mean(hb) - mkt)

        if not d_cap[3]:
            print("  no scoreable folds\n")
            continue
        print(f"  {'metric':<12}{'with':>10}{'without':>10}{'diff':>11}"
              f"{'paired t':>10}{'n':>6}")
        if d_auc:
            t_ = paired_t(d_auc) or 0.0
            print(f"  {'AUC':<12}{'-':>10}{'-':>10}"
                  f"{st.mean(d_auc):>+10.5f}{t_:>10.2f}{len(d_auc):>6}")
        for key, lab in [(c, f"cap {c}") for c in CAPS] + [("thr",
                                                            "prob>=0.70")]:
            dv = d_cap[key] if key in d_cap else d_thr
            if key == "thr":
                dv = d_thr
            if len(dv) < 10:
                continue
            t2 = paired_t(dv) or 0.0
            print(f"  {lab:<12}{100*st.mean(with_v[key]):>+9.3f}pp"
                  f"{100*st.mean(without_v[key]):>+9.3f}pp"
                  f"{100*st.mean(dv):>+10.3f}pp{t2:>10.2f}{len(dv):>6}")
            agg[key].append(st.mean(dv))
        if d_auc:
            agg["auc"].append(st.mean(d_auc))
        print()

    if not agg:
        print("no results")
        return
    print("=" * 64)
    print("ACROSS SEEDS — 'with' minus 'without'")
    print("=" * 64)
    print(f"  {'metric':<12}{'mean diff':>13}{'seeds':>8}{'negative':>10}")
    for key, lab in ([("auc", "AUC")]
                     + [(c, f"cap {c}") for c in CAPS]
                     + [("thr", "prob>=0.70")]):
        v = agg.get(key, [])
        if not v:
            continue
        neg = sum(1 for x in v if x < 0)
        unit = "" if key == "auc" else "pp"
        val = st.mean(v) if key == "auc" else 100 * st.mean(v)
        print(f"  {lab:<12}{val:>+12.4f}{unit:<2}{len(v):>6}{neg:>8}/{len(v)}")

    print("\n  NEGATIVE means the columns HURT -- the book earned less with")
    print("  them. The paired t uses per-date differences on identical panels,")
    print("  so date and fold noise cancels; it is far more powerful than")
    print("  comparing two separate runs, which is what produced the earlier")
    print("  +1.3pp reading on a +0.112pp cap-3 gap.\n")
    print("  Either answer leaves the Boulton EXCLUSION FILTER intact:")
    print("  +1.118pp at cap-3, 3/3 seeds, direction predicted before testing.")
    print("  Removing names is not the same mechanism as learning a column.")


if __name__ == "__main__":
    main()
