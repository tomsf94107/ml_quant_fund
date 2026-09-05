#!/usr/bin/env python3
"""
target_comparison_test.py — is the direction model's problem its TARGET?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

THE THESIS UNDER TEST
    The direction model was killed on 2026-05-31 as "near-coin-flip AND
    inverted at the extremes". Six attempts on 2026-09-04/05 to improve it via
    features -- insider constructions, rate beta, leverage, IV skew, a selective
    classifier -- all failed.

    docs/A8_implementation_plan.md offers a different diagnosis: "A8
    (top-decile cross-sectional target) achieved OOS AUC 0.677 ... This is a NEW
    alpha source distinct from current production (any-positive target, mostly
    macro-driven)." That is a claim about the LABEL, not the features -- the
    same inputs, a different question asked of them.

THE TRAP, AND WHY THIS SCRIPT DOES NOT USE AUC
    AUC is NOT comparable across targets. A rarer label is mechanically easier
    to separate: predicting "top decile" at a 10% base rate can post a higher
    AUC than "any positive" at 51% without being more useful. So A8's 0.677
    against production's 0.535 may be an artifact of label rarity rather than
    evidence of better alpha, and the plan has carried that comparison since May
    on the strength of a non-comparable number.

    PCT7 is the proof. Its target is rare, its AUC is high, its within-date
    selection edge is real (48/65 sessions above their own base rate) -- and its
    tradeable return is NEGATIVE: -1.97% per cohort, 41 of 64 days losing,
    identical at every capital level. Higher AUC, worse outcome.

    So every target here is scored the same way: the FORWARD RETURN of the top
    decile of its own predictions, day-weighted. That is comparable across
    targets because it is measured in the same units -- money -- regardless of
    what the label was.

TARGETS
    any_positive     fwd 5d return > 0            current production
    top_decile       top 10% of that date's cross-section    A8's claim
    pct7             fwd 5d return >= +7%         PCT7's
    triple_barrier   +1 if a +2 sigma profit barrier is hit before a -1 sigma
                     stop or the 5-day limit; 0 otherwise. Lopez de Prado's
                     method, which studies find "outperforms traditional
                     labeling techniques like fixed time horizon", and which
                     unlike the others encodes the PATH rather than only the
                     endpoint. Barriers scale with each name's own 20-day
                     volatility, so a fixed threshold does not misfire on
                     heteroskedastic returns.

METHOD
    Same features, same tickers, same temporal split, one model per target.
    Fitted on the first 70% of dates, scored on the last 30%. Never a random
    split -- adjacent dates share market conditions and would leak.

    Reported per target: base rate, test AUC (for reference only, NOT for
    comparison across targets), and the day-weighted forward return of its top
    decile, against the universe over the same days.

    python analysis/target_comparison_test.py --tickers 60
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")


def auc(scores, labels):
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
    print(f"building {len(universe)} tickers from {args.start} -- slow\n")

    # panel: (ticker, date) -> features, and forward path
    X = {}
    fwd = {}
    path = {}
    vol = {}
    feat_names = None
    built = 0
    for i, t in enumerate(universe, 1):
        try:
            df = build_feature_dataframe(t, start_date=args.start,
                                         training_mode=True)
            if df is None or len(df) < 300:
                continue
            num = df.select_dtypes("number")
            drop = [c for c in num.columns if c.startswith("target_")]
            num = num.drop(columns=drop, errors="ignore")
            if feat_names is None:
                feat_names = list(num.columns)
            ds = [str(d)[:10] for d in df["date"]]
            cl = list(df["close"]) if "close" in df.columns else None
            if cl is None:
                continue
            rets = [0.0] + [(cl[j] - cl[j-1]) / cl[j-1] if cl[j-1] else 0.0
                            for j in range(1, len(cl))]
            for j in range(20, len(ds) - H):
                a = cl[j]
                if not a:
                    continue
                w = [rets[k] for k in range(j - 20, j)]
                sd = st.pstdev(w) if len(w) > 5 else 0.0
                nxt = [cl[j + k] for k in range(1, H + 1)]
                if any(x is None for x in nxt):
                    continue
                r = (cl[j + H] - a) / a
                if abs(r) > 0.8:
                    continue
                X[(t, ds[j])] = [float(v) if v == v else float("nan")
                                 for v in num.iloc[j].tolist()]
                fwd[(t, ds[j])] = r
                path[(t, ds[j])] = [(p - a) / a for p in nxt]
                vol[(t, ds[j])] = sd
            built += 1
            if i % 20 == 0:
                print(f"  ...{i} tickers, {built} usable, {len(X):,} rows")
        except Exception:
            continue
    print(f"\n{built} tickers, {len(X):,} panel rows, "
          f"{len(feat_names or [])} features\n")
    if len(X) < 5000:
        print("too few rows")
        return

    dates = sorted({k[1] for k in X})
    cut = dates[int(len(dates) * 0.70)]
    print(f"train < {cut}, test >= {cut}\n")

    # ---- build the four label sets ----
    bydate = defaultdict(list)
    for k in X:
        bydate[k[1]].append(k)

    labels = {n: {} for n in ("any_positive", "top_decile", "pct7",
                              "triple_barrier")}
    for d, keys in bydate.items():
        rs = sorted((fwd[k] for k in keys), reverse=True)
        cutoff = rs[max(0, len(rs) // 10 - 1)] if len(rs) >= 10 else None
        for k in keys:
            r = fwd[k]
            labels["any_positive"][k] = 1 if r > 0 else 0
            labels["top_decile"][k] = (1 if cutoff is not None and r >= cutoff
                                       else 0)
            labels["pct7"][k] = 1 if r >= 0.07 else 0
            s = vol[k]
            up, dn = (2.0 * s, -1.0 * s) if s > 0 else (0.04, -0.02)
            lab = 0
            for p in path[k]:
                if p >= up:
                    lab = 1
                    break
                if p <= dn:
                    lab = 0
                    break
            labels["triple_barrier"][k] = lab

    keys_tr = [k for k in X if k[1] < cut]
    keys_te = [k for k in X if k[1] >= cut]
    Xtr = [X[k] for k in keys_tr]
    Xte = [X[k] for k in keys_te]

    uni_by_date = defaultdict(list)
    for k in keys_te:
        uni_by_date[k[1]].append(fwd[k])
    uni_day = st.mean(st.mean(v) for v in uni_by_date.values())

    print(f"  {'target':<16}{'base':>7}{'AUC*':>8}{'top-decile ret':>16}"
          f"{'vs universe':>13}{'win days':>10}")
    for name in ("any_positive", "top_decile", "pct7", "triple_barrier"):
        ytr = [labels[name][k] for k in keys_tr]
        yte = [labels[name][k] for k in keys_te]
        if sum(ytr) < 100 or sum(yte) < 50:
            print(f"  {name:<16}   too few positives")
            continue
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr)
        p = [float(v) for v in m.predict_proba(Xte)[:, 1]]
        a = auc(p, yte)

        # day-weighted forward return of each date's top decile by prediction
        byd = defaultdict(list)
        for i, k in enumerate(keys_te):
            byd[k[1]].append((p[i], fwd[k]))
        day_rets = []
        for d, v in byd.items():
            if len(v) < 10:
                continue
            v.sort(reverse=True)
            top = v[:max(1, len(v) // 10)]
            day_rets.append(st.mean(r for _, r in top))
        if not day_rets:
            print(f"  {name:<16}   no scoreable days")
            continue
        mr = st.mean(day_rets)
        wins = sum(1 for d, v in byd.items()
                   if len(v) >= 10
                   and st.mean(r for _, r in sorted(v, reverse=True)
                               [:max(1, len(v)//10)])
                   > st.mean(r for _, r in v))
        print(f"  {name:<16}{100*sum(yte)/len(yte):>6.1f}%{a:>8.4f}"
              f"{100*mr:>15.2f}%{100*(mr-uni_day):>+12.2f}pp"
              f"{wins:>6}/{len(day_rets)}")

    print(f"\n  universe day-weighted mean over the test period: "
          f"{100*uni_day:+.2f}%")
    print("\n  * AUC is shown for reference and is NOT comparable across")
    print("    targets: a rarer label is mechanically easier to separate.")
    print("    PCT7 is the proof -- rare target, high AUC, real within-date")
    print("    edge, and a NEGATIVE tradeable return.")
    print("\n  The comparable column is 'vs universe': the day-weighted forward")
    print("  return of each target's own top decile, in money, over the same")
    print("  days. If any_positive is not the worst, the relabeling thesis is")
    print("  wrong and the rebuild needs a different idea.")


if __name__ == "__main__":
    main()
