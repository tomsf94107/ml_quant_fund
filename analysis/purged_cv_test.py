#!/usr/bin/env python3
"""
purged_cv_test.py — is the measured AUC inflated by label overlap?

READ-ONLY on the model. Trains throwaway copies; writes nothing to any database.

THE PROBLEM
    train_model uses a 60/20/20 temporal split. That prevents the future from
    leaking into the past in the obvious way, but it does NOT handle LABEL
    OVERLAP.

    An h=5 prediction made on Monday is labelled by the return through Friday.
    A prediction made on Tuesday is labelled by the return through the
    following Monday. Those two labels share FOUR of five days. At the boundary
    between train and validation, the last training samples and the first
    validation samples are labelled by overlapping windows -- so information
    about the validation outcome is already inside the training set.

    The effect inflates measured AUC by an unknown amount. Every downstream
    conclusion rests on AUC 0.5362: the 2-4pp lift ceiling, the judgement that
    May was anomalous, the decision about whether the direction model is worth
    running. If the true figure is 0.52, those change.

THE FIX BEING TESTED
    Lopez de Prado's PURGING and EMBARGO:

      PURGE   drop training samples whose label window overlaps the test set.
              For horizon h, that is the h-1 samples immediately before the
              test block, and the h-1 after it.
      EMBARGO an additional gap after the test block, to handle serial
              correlation in features that outlives the label window. A common
              default is 1% of the sample; here it is expressed in trading
              days and defaults to h.

    The script fits the same model three ways on the same tickers:

      plain     the current 60/20/20 split, as the system does it now
      purged    same split, with overlapping boundary samples removed
      pe        purged plus embargo

    The difference between "plain" and "pe" IS the leakage.

WHY THE TOP DECILE IS TESTED SEPARATELY
    Pooled AUC averages over every prediction, so leakage concentrated in the
    model's most CONFIDENT calls could be invisible there while inflating
    exactly the subset that matters. The high-confidence cohort is what gets
    traded and what the operator watches; a boundary-leaked feature driving the
    extreme probabilities would show up in top-decile accuracy and nowhere else.

    So the script reports both: pooled AUC, and accuracy in the top decile of
    predicted probability, under each variant.

WHAT WOULD BE A REAL FINDING
    A drop of 0.005 or less is noise at this sample size. A drop of 0.01-0.02
    materially changes the ceiling. A drop that takes AUC to 0.51 or below means
    the measured edge was substantially an artifact, and the honest response is
    to stop adding features and reconsider the model.

    Reported per horizon, pooled across tickers, with the count of samples
    purged so the cost of the correction is visible.

    python analysis/purged_cv_test.py --tickers 40 --horizon 5
"""
import argparse
import math
import random
import sys
import warnings

warnings.filterwarnings("ignore")


def auc(scores, labels):
    n = len(scores)
    pos = sum(labels)
    neg = n - pos
    if not pos or not neg:
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
    return (rs - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=40,
                    help="how many tickers to test; more is slower")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=None,
                    help="embargo in trading days; defaults to the horizon")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    h = args.horizon
    embargo = args.embargo if args.embargo is not None else h

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe, add_forecast_targets
    from xgboost import XGBClassifier

    universe = [l.strip() for l in open("tickers.txt") if l.strip()]
    rnd = random.Random(args.seed)
    rnd.shuffle(universe)
    picked = universe[:args.tickers]

    print(f"purged CV test — h={h}, embargo={embargo} trading days, "
          f"{len(picked)} tickers\n")
    print("  label overlap for h={0}: predictions {0} days apart share "
          "{1} of {0} outcome days".format(h, h - 1))
    print(f"  purge removes the {h-1} samples either side of the test block; "
          f"embargo adds {embargo} more after\n")

    res = {"plain": [], "purged": [], "pe": []}
    top = {k: [0, 0] for k in res}        # [n, hits] in the top decile
    base_all = {k: [0, 0] for k in res}   # [n, hits] over all test rows
    purged_counts = []
    done = 0

    for tk in picked:
        try:
            df = build_feature_dataframe(tk, start_date="2018-01-01",
                                         training_mode=True)
            df = add_forecast_targets(df, horizons=(1, 3, 5))
            col = f"target_{h}d"
            if col not in df.columns:
                continue
            df = df.dropna(subset=[col])
            drop = [c for c in df.columns
                    if c.startswith("target_") or c in ("date", "ticker")]
            X = df.drop(columns=drop, errors="ignore").select_dtypes("number")
            y = df[col].astype(int)
            if len(X) < 400 or y.nunique() < 2:
                continue

            n = len(X)
            tr_end = int(n * 0.60)
            te_start = int(n * 0.80)
            Xte, yte = X.iloc[te_start:], y.iloc[te_start:]
            if yte.nunique() < 2:
                continue

            variants = {
                # current behaviour: train on everything up to 60%
                "plain": list(range(0, tr_end)),
                # purge the h-1 samples whose label window reaches into val/test
                "purged": list(range(0, max(tr_end - (h - 1), 1))),
                # purge plus embargo
                "pe": list(range(0, max(tr_end - (h - 1) - embargo, 1))),
            }
            purged_counts.append(len(variants["plain"]) - len(variants["pe"]))

            for name, idx in variants.items():
                if len(idx) < 200:
                    continue
                m = XGBClassifier(n_estimators=120, max_depth=3,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss",
                                  verbosity=0)
                m.fit(X.iloc[idx], y.iloc[idx])
                p = [float(v) for v in m.predict_proba(Xte)[:, 1]]
                a = auc(p, list(yte))
                if a is not None:
                    res[name].append(a)
                # Top-decile accuracy and its lift over this ticker's own base
                # rate. Pooled per ticker as raw counts, not as an average of
                # per-ticker percentages: a ticker with 4 test rows would
                # otherwise carry the same weight as one with 200.
                pairs = sorted(zip(p, list(yte)), reverse=True)
                k = max(1, len(pairs) // 10)
                top[name][0] += k
                top[name][1] += sum(yy for _, yy in pairs[:k])
                base_all[name][0] += len(pairs)
                base_all[name][1] += sum(yy for _, yy in pairs)
            done += 1
            if done % 10 == 0:
                print(f"  ...{done} tickers")
        except Exception as e:
            continue

    if not res["plain"]:
        print("no tickers produced a usable fit")
        return

    print(f"\n  fitted {len(res['plain'])} tickers\n")
    print(f"  {'variant':<10}{'mean AUC':>11}{'median':>10}{'n':>6}")
    means = {}
    for k in ("plain", "purged", "pe"):
        v = res[k]
        if not v:
            continue
        means[k] = sum(v) / len(v)
        sv = sorted(v)
        print(f"  {k:<10}{means[k]:>11.4f}{sv[len(sv)//2]:>10.4f}{len(v):>6}")

    print(f"\n  {'variant':<10}{'top-dec n':>11}{'top-dec acc':>13}"
          f"{'base':>8}{'lift':>9}")
    lifts = {}
    for k in ("plain", "purged", "pe"):
        tn, th = top[k]
        bn, bh = base_all[k]
        if not tn or not bn:
            continue
        tacc = 100.0 * th / tn
        bacc = 100.0 * bh / bn
        lifts[k] = tacc - bacc
        print(f"  {k:<10}{tn:>11}{tacc:>12.1f}%{bacc:>7.1f}%"
              f"{lifts[k]:>+8.1f}pp")
    if "plain" in lifts and "pe" in lifts:
        dl = lifts["plain"] - lifts["pe"]
        print(f"\n  top-decile lift leakage (plain - purged+embargo): "
              f"{dl:+.1f}pp")
        print("  This is the number that matters more than pooled AUC: the "
              "high-confidence\n  cohort is what gets traded. A drop here with "
              "flat AUC would mean leakage\n  concentrated in exactly the "
              "predictions that are acted on.")

    if "plain" in means and "pe" in means:
        d = means["plain"] - means["pe"]
        print(f"\n  leakage estimate (plain - purged+embargo): {d:+.4f}")
        if abs(d) < 0.005:
            v = "NOISE at this sample size -- the split was already adequate"
        elif d < 0.02:
            v = "MATERIAL -- the ceiling moves, conclusions need restating"
        else:
            v = "LARGE -- the measured edge was substantially an artifact"
        print(f"  -> {v}")
        print(f"\n  training samples removed per ticker: "
              f"{sum(purged_counts)/len(purged_counts):.0f} of "
              f"~{int(0.6*1300)} ({100*sum(purged_counts)/len(purged_counts)/780:.1f}%)")

    print("\n  A negative difference means purging IMPROVED measured AUC, which "
          "would be\n  a sign the test itself is noisy rather than that leakage "
          "helps. Read the\n  median alongside the mean: a few tickers with "
          "extreme AUC can move the mean.")


if __name__ == "__main__":
    main()
