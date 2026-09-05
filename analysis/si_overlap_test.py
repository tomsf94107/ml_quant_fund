#!/usr/bin/env python3
"""
si_overlap_test.py — is the h=40 model a second signal, or the SI brick again?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHY THIS GATES EVERYTHING ELSE
    On 2026-09-05 an XGBoost model on the existing feature set produced, at
    h=40, a day-weighted cap-3 excess of +5.194pp positive in 3 of 3 seeds, with
    turnover 27%. That is the first result of the session to pass every
    pre-registered condition.

    The fund's ONE validated edge -- short interest, low days-to-cover -- also
    runs at h=40, at IC -0.037 and NW-t -3.12 after FINRA's 8-business-day
    publication lag.

    Two signals at the same horizon on the same universe may be two edges or one
    edge seen twice. That distinction decides:

      - whether the fund has diversification it is currently assuming
      - whether a cost analysis of the h=40 model is measuring something new
      - whether the firm-characteristics build should target this model or the
        SI book

    Measuring cost or capacity before knowing this would be analysing a signal
    that may already be owned.

WHAT IS MEASURED

  1. NAME OVERLAP. Each rebalance, the model's top-N against the lowest-DTC
     quintile. Reported as the share of model picks that are also SI longs, and
     compared against what random selection would give -- because with a
     quintile long leg, ~20% overlap is the null, not zero.

  2. RANK CORRELATION. Spearman between the model's probability and days-to-cover
     across the whole cross-section, per date. Overlap counts the extremes; this
     asks whether the model is learning days-to-cover throughout.

  3. FEATURE DEPENDENCE. short_ratio IS days-to-cover in this feature set --
     features/builder.py's _load_short_interest_pit docstring says so, and it
     was revived on 2026-08-26 from a broadcast constant to a per-row PIT join,
     going from 0.000 to 5.119 importance. If the model leans on it, the two
     signals are related BY CONSTRUCTION. Measured by refitting with short_ratio
     and its derivatives removed and seeing what survives.

  4. RESIDUAL EDGE. The model's excess return on picks that are NOT in the SI
     long leg. If that collapses, the h=40 result is the SI brick wearing a
     different hat.

INTERPRETATION
    High overlap AND collapsing residual edge -> one signal, and the h=40
    finding is corroboration of the SI brick rather than an addition.
    Low overlap AND surviving residual edge -> genuinely separate, and the fund
    has a second candidate.
    Anything in between needs the feature-dependence test to adjudicate.

    python analysis/si_overlap_test.py --seeds 3 --tickers 80
"""
import argparse
import datetime
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

SI_DERIVED = ("short_ratio", "short_pct_float", "vol_x_short",
              "is_squeeze_setup", "is_squeeze_setup__ts_argmax__w20",
              "short_self_rank", "short_zscore_60d")


def spearman(x, y):
    n = len(x)
    if n < 8:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for m in range(i, j + 1):
                r[o[m]] = a
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--si-quantile", type=float, default=0.2)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random
    import sqlite3

    # ---- SI: days_to_cover per ticker, PIT on the settlement grid ----
    sc = sqlite3.connect("file:short_interest.db?mode=ro", uri=True)
    si_rows = sc.execute(
        "SELECT ticker, settlement_date, days_to_cover FROM short_interest "
        "WHERE days_to_cover IS NOT NULL AND days_to_cover <= 50").fetchall()
    sc.close()
    si_hist = defaultdict(list)
    for t, d, v in si_rows:
        si_hist[t.upper()].append((str(d)[:10], float(v)))
    for t in si_hist:
        si_hist[t].sort()

    def dtc_asof(t, d):
        """Latest settlement at least 12 days old -- FINRA disseminates ~8
        business days after settlement, so anything fresher was not public."""
        h = si_hist.get(t)
        if not h:
            return None
        cut = (datetime.date.fromisoformat(d)
               - datetime.timedelta(days=12)).isoformat()
        best = None
        for sd, v in h:
            if sd <= cut:
                best = v
            else:
                break
        return best

    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    print(f"SI overlap test — {args.seeds} seeds x {args.tickers} tickers, "
          f"h={H}, cap {args.cap}, SI long leg = lowest "
          f"{args.si_quantile:.0%} days-to-cover\n")

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
        if len(X) < 15000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        si_idx = [i for i, c in enumerate(cols) if c in SI_DERIVED]
        keep = [i for i in range(len(cols)) if i not in si_idx]
        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        ov, rnd_ov, rho, ex_all, ex_resid, ex_nosi = [], [], [], [], [], []
        n_resid = 0
        for i in range(len(anchors) - 1):
            tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr_end]
            kte = [k for k in X if tr_end <= k[1] < te_end]
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            ytr = [1 if fwd[k] > 0 else 0 for k in ktr]

            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0)
            m.fit([X[k] for k in ktr], ytr)
            p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]

            # same model minus every SI-derived feature
            m2 = XGBClassifier(n_estimators=200, max_depth=4,
                               learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="logloss",
                               verbosity=0)
            m2.fit([[X[k][j] for j in keep] for k in ktr], ytr)
            p2 = [float(v) for v in
                  m2.predict_proba([[X[k][j] for j in keep] for k in kte])[:, 1]]

            byd = defaultdict(list)
            for z, k in enumerate(kte):
                byd[k[1]].append((p[z], p2[z], fwd[k], k[0]))
            for d, v in byd.items():
                if len(v) < args.min_names:
                    continue
                dtcs = [(t, dtc_asof(t, d)) for _, _, _, t in v]
                dtcs = [(t, x) for t, x in dtcs if x is not None]
                if len(dtcs) < args.min_names // 2:
                    continue
                vals = sorted(x for _, x in dtcs)
                cutq = vals[max(0, int(len(vals) * args.si_quantile) - 1)]
                si_long = {t for t, x in dtcs if x <= cutq}

                mkt = st.mean(z[2] for z in v)
                top = sorted(v, reverse=True)[:args.cap]
                names = {z[3] for z in top}
                ov.append(len(names & si_long) / max(len(names), 1))
                rnd_ov.append(len(si_long) / len(dtcs))
                ex_all.append(st.mean(z[2] for z in top) - mkt)

                outside = [z for z in sorted(v, reverse=True)
                           if z[3] not in si_long][:args.cap]
                if outside:
                    ex_resid.append(st.mean(z[2] for z in outside) - mkt)
                    n_resid += len(outside)

                top2 = sorted(v, key=lambda z: -z[1])[:args.cap]
                ex_nosi.append(st.mean(z[2] for z in top2) - mkt)

                dmap = dict(dtcs)
                pair = [(z[0], dmap[z[3]]) for z in v if z[3] in dmap]
                if len(pair) >= 10:
                    r = spearman([q[0] for q in pair], [q[1] for q in pair])
                    if r is not None:
                        rho.append(r)

        if not ov:
            print(f"seed {seed}: no scoreable rebalances\n")
            continue
        print(f"SEED {seed} — {len(X):,} rows, {len(ov)} rebalances")
        print(f"  overlap of top-{args.cap} with SI long leg : "
              f"{100*st.mean(ov):>5.1f}%   (random would give "
              f"{100*st.mean(rnd_ov):.1f}%)")
        print(f"  Spearman(model prob, days_to_cover)        : "
              f"{st.mean(rho) if rho else float('nan'):>+6.3f}")
        print(f"  excess, top-{args.cap} as-is                       : "
              f"{100*st.mean(ex_all):>+6.3f}pp")
        print(f"  excess, top-{args.cap} EXCLUDING SI long names     : "
              f"{100*st.mean(ex_resid) if ex_resid else float('nan'):>+6.3f}pp")
        print(f"  excess, model retrained WITHOUT SI features : "
              f"{100*st.mean(ex_nosi):>+6.3f}pp\n")
        agg["ov"].append(st.mean(ov))
        agg["rnd"].append(st.mean(rnd_ov))
        agg["rho"].append(st.mean(rho) if rho else 0.0)
        agg["all"].append(st.mean(ex_all))
        agg["resid"].append(st.mean(ex_resid) if ex_resid else 0.0)
        agg["nosi"].append(st.mean(ex_nosi))

    if not agg["ov"]:
        print("no seeds produced results")
        return
    print("=" * 66)
    print("ACROSS SEEDS")
    print("=" * 66)
    print(f"  overlap with SI long leg      {100*st.mean(agg['ov']):>6.1f}%"
          f"   vs {100*st.mean(agg['rnd']):.1f}% random")
    print(f"  Spearman(prob, days_to_cover) {st.mean(agg['rho']):>+7.3f}")
    print(f"  excess as-is                  {100*st.mean(agg['all']):>+6.3f}pp"
          f"   ({sum(1 for x in agg['all'] if x>0)}/{len(agg['all'])} seeds +)")
    print(f"  excess excluding SI longs     {100*st.mean(agg['resid']):>+6.3f}pp"
          f"   ({sum(1 for x in agg['resid'] if x>0)}/{len(agg['resid'])} +)")
    print(f"  excess without SI features    {100*st.mean(agg['nosi']):>+6.3f}pp"
          f"   ({sum(1 for x in agg['nosi'] if x>0)}/{len(agg['nosi'])} +)")

    print("\n  Overlap must be read against the RANDOM column: with a quintile")
    print("  long leg, ~20% is the null, not zero.\n")
    print("  If the residual and no-SI-feature rows hold up, the h=40 model is")
    print("  a genuinely separate signal. If they collapse, it is the SI brick")
    print("  seen through a different lens, and the fund has one edge measured")
    print("  twice rather than two to size against each other.")


if __name__ == "__main__":
    main()
