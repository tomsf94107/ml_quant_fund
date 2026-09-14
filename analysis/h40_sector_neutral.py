#!/usr/bin/env python3
"""
h40_sector_neutral.py — is the h=40 edge stock-level, or a sector bet?

READ-ONLY. Trains throwaway copies. Writes nothing.

THE QUESTION
    Every number measured for h=40 so far is a RAW cross-sectional result: rank
    every name by predicted probability, buy the top ones, measure excess return.
    That answers "does the ranking work". It does not answer "does the ranking
    work WITHIN a sector, or is it mostly picking whichever sector is running".

    Those are different books. If the edge is a sector tilt, sector exposure is
    available through eleven SPDR ETFs at a few basis points, and a
    stock-selection model that reproduces it is an expensive way to buy XLK.

WHY THIS TEST, IN THIS FORM
    validate_borrow_battery.py test 2 already established the method and the bar
    for this fund: demean the feature AND the return within `sector` per date,
    recompute the IC, and require that at least 40% of the raw IC survives.

    The borrow-fee residual retained 34% -- its raw per-date IC of +0.0273 at
    NW-t +2.30 fell to +0.0092 at t +0.92 -- and the axis was closed for it on
    2026-09-10. Two thirds of that signal was "this sector is hard to borrow".
    The same test, the same bar, applied to the thing the fund actually runs.

WHAT THIS IS NOT
    analysis/sector_ic_breakdown.py fits a SEPARATE model per sector and reports
    each sector's own IC. That asks whether a sector-specific model works, on
    15-45 names per sector -- a different and much thinner question. It also
    defaults horizon=5, where this model is BELOW random (AUC 0.478 against
    0.577 at h=40), and tags unmapped names with SECTOR_ETF_MAP.get(tk, "XLK"),
    which silently labels every miss as tech and would manufacture the "tech is
    better" answer it exists to test. Not used here.

SECTOR LABELS
    features.builder.resolve_sector_etf -- the same three-tier resolution the
    model's own sector_rel_ret feature uses: SECTOR_ETF_MAP (429 hand-curated)
    -> tickers_metadata.csv bucket -> data/sector_etf_sic.csv (1,885 names from
    SEC SIC codes) -> market. Using a DIFFERENT sector map than the features
    were built with would decompose by one taxonomy a signal trained under
    another.

PANEL CONSTRUCTION
    Mirrors h40_book_test.py exactly -- same universe file, same seeded shuffle,
    same 20-row warmup, same |r| > 1.5 filter, same quarterly anchors from the
    55th percentile of months, same XGBClassifier parameters. The point is that
    the raw IC reported here is comparable to the excess returns reported there.

READ IT PER SEED
    Three days of arms established that one number from this construction is
    unreliable: cap-3 on a FIXED scored set moved 4.17pp across seeds when only
    the training companions changed. A retained fraction from a single seed
    means nothing. Three seeds agreeing is the minimum, and disagreement across
    seeds is itself the finding.

USAGE
    python analysis/h40_sector_neutral.py
    python analysis/h40_sector_neutral.py --tickers 400 --seeds 3
"""
import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np


def _nw_t(x, lag):
    """Newey-West t-stat on the mean of x, `lag` overlapping periods."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if s <= 0:
        return float("nan")
    return x.mean() / math.sqrt(s / n)


def _rank(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def _spearman(p, r):
    if len(p) < 3:
        return None
    rp, rr = _rank(p), _rank(r)
    if rp.std() == 0 or rr.std() == 0:
        return None
    c = np.corrcoef(rp, rr)[0, 1]
    return None if c != c else float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tickers", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--universe", default="tickers.txt")
    ap.add_argument("--min-names", type=int, default=20,
                    help="minimum names on a date for the RAW leg")
    # SEPARATE FLOOR FOR THE SECTOR LEG. A first run used --min-names for both
    # and produced zero neutral dates: 80 tickers across ~11 sectors is ~7 per
    # sector, so every sector-date was dropped. Demeaning against 8 names is
    # already thin; against 3 it is noise. 8 is the floor, and sectors below it
    # are dropped rather than demeaned badly.
    ap.add_argument("--min-sector-names", type=int, default=8,
                    help="minimum names in ONE sector on a date to demean it")
    ap.add_argument("--retain-bar", type=float, default=0.40,
                    help="validate_borrow_battery.py test 2's bar; borrow "
                         "retained 0.34 and was closed")
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe, resolve_sector_etf
    from xgboost import XGBClassifier

    uni_all = [l.strip().upper() for l in open(args.universe) if l.strip()]
    print(f"h={H} sector-neutral test — {args.seeds} seeds x {args.tickers} "
          f"tickers from {args.universe}")
    print(f"bar: sector-neutral must retain >= {args.retain_bar:.0%} of raw IC "
          f"(borrow retained 34% and was closed)\n")

    agg = defaultdict(list)

    for seed in range(1, args.seeds + 1):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        X, fwd, sec = {}, {}, {}
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
                _s = resolve_sector_etf(t)
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
                    sec[(t, ds[j])] = _s
            except Exception:
                continue
        if len(X) < 15000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        raw_ic, neu_ic = [], []
        sec_counts = defaultdict(int)

        for i in range(len(anchors) - 1):
            tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr_end]
            kte = [k for k in X if tr_end <= k[1] < te_end]
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0)
            m.fit([X[k] for k in ktr], [1 if fwd[k] > 0 else 0 for k in ktr])
            p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]

            byd = defaultdict(list)
            for z, k in enumerate(kte):
                byd[k[1]].append((p[z], fwd[k], sec[k]))

            for d, rows in byd.items():
                if len(rows) < args.min_names:
                    continue
                pv = [x[0] for x in rows]
                rv = [x[1] for x in rows]
                ic = _spearman(pv, rv)
                if ic is not None:
                    raw_ic.append(ic)

                # SECTOR-NEUTRAL LEG. Demean prediction AND return within each
                # sector on this date, then pool the residuals and IC them.
                # A sector with fewer than min_names on a date is dropped rather
                # than demeaned against two or three observations.
                bys = defaultdict(list)
                for pp, rr, ss in rows:
                    bys[ss].append((pp, rr))
                np_, nr_ = [], []
                for ss, vals in bys.items():
                    if len(vals) < args.min_sector_names:
                        continue
                    sec_counts[ss] += 1
                    mp = sum(v[0] for v in vals) / len(vals)
                    mr = sum(v[1] for v in vals) / len(vals)
                    for pp, rr in vals:
                        np_.append(pp - mp)
                        nr_.append(rr - mr)
                if len(np_) >= args.min_names:
                    ic2 = _spearman(np_, nr_)
                    if ic2 is not None:
                        neu_ic.append(ic2)

        if len(raw_ic) < 10 or len(neu_ic) < 10:
            print(f"seed {seed}: too few dates "
                  f"(raw {len(raw_ic)}, neutral {len(neu_ic)}), skipped\n")
            continue

        rm, nm = float(np.mean(raw_ic)), float(np.mean(neu_ic))
        rt = _nw_t(raw_ic, H)
        nt = _nw_t(neu_ic, H)
        ret = (nm / rm) if rm else float("nan")
        agg["raw"].append(rm)
        agg["neu"].append(nm)
        agg["ret"].append(ret)
        agg["rt"].append(rt)
        agg["nt"].append(nt)

        print(f"SEED {seed} — {len(X):,} rows, {len(raw_ic)} dates, "
              f"{len(bys) if 'bys' in dir() else '?'} sectors on the last date")
        print(f"  raw per-date IC          {rm:+.4f}   NW t {rt:+.2f}   "
              f"{100*float(np.mean([1 if x>0 else 0 for x in raw_ic])):.0f}% of dates > 0")
        print(f"  sector-neutral IC        {nm:+.4f}   NW t {nt:+.2f}   "
              f"{100*float(np.mean([1 if x>0 else 0 for x in neu_ic])):.0f}% of dates > 0")
        print(f"  retained                 {ret:.0%}"
              f"{'  <-- below the 40% bar' if ret < args.retain_bar else ''}\n")

    if not agg["ret"]:
        print("no seed produced enough dates.")
        return

    print("=" * 70)
    print("ACROSS SEEDS")
    print("=" * 70)
    print(f"  raw IC             {np.mean(agg['raw']):+.4f}   "
          f"(seeds: {', '.join(f'{x:+.4f}' for x in agg['raw'])})")
    print(f"  sector-neutral IC  {np.mean(agg['neu']):+.4f}   "
          f"(seeds: {', '.join(f'{x:+.4f}' for x in agg['neu'])})")
    print(f"  retained           {np.mean(agg['ret']):.0%}   "
          f"(seeds: {', '.join(f'{x:.0%}' for x in agg['ret'])})")
    print()
    _mean_ret = float(np.mean(agg["ret"]))
    _all_pass = all(x >= args.retain_bar for x in agg["ret"])
    _all_fail = all(x < args.retain_bar for x in agg["ret"])
    if _all_pass:
        print(f"  READ: every seed retains >= {args.retain_bar:.0%}. The edge")
        print("  survives sector-neutralisation, so it is stock-level and a")
        print("  per-sector breakdown would be decoration rather than a finding.")
    elif _all_fail:
        print(f"  READ: every seed retains < {args.retain_bar:.0%}. Most of the")
        print("  h=40 edge is a SECTOR BET, not stock selection -- the same")
        print("  verdict that closed the borrow-fee axis at 34% on 2026-09-10.")
        print("  Sector exposure costs a few bp through eleven SPDR ETFs.")
    else:
        print("  READ: seeds DISAGREE about whether the edge survives.")
        print("  That is the finding. It is consistent with the training-panel")
        print("  instability measured 2026-09-13 -- a single retained fraction")
        print("  from this construction is not a measurement.")
    print()
    print("  Neutralisation demeans the PREDICTION and the RETURN within each")
    print("  sector per date, pools the residuals, and ranks those. Sectors")
    print(f"  with fewer than {args.min_sector_names} names on a date are dropped")
    print("  rather than demeaned against a handful of observations.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
