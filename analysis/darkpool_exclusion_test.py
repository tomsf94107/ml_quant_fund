#!/usr/bin/env python3
"""
darkpool_exclusion_test.py — does excluding the Boulton cell improve a book?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHAT IS ALREADY ESTABLISHED (2026-09-05)
    analysis/darkpool_volume_si_test.py replicated Boulton et al., "Short
    selling and dark pool volume", on this fund's data. Stocks in the HIGH
    days-to-cover AND HIGH dark-pool-volume-share cell underperform:

        h=20   -2.884pp excess, 0 of 3 seeds positive
        h=40   -7.145pp excess, 0 of 3 seeds positive, one seed at NW t -3.33

    The direction was PREDICTED BY THE PAPER BEFORE THE TEST RAN, not found by
    searching. That provenance matters: the h=40 book was discovered by sweeping
    horizons, which invites data-mining doubt; this was a stated hypothesis that
    replicated. Seed 1 alone showed +2.639pp -- the opposite sign -- and a
    plausible-sounding "short covering" story was proposed for it before the
    other two seeds reversed it. That is exactly the retrospective
    rationalisation the multi-seed rule exists to catch.

WHY EXCLUSION RATHER THAN SHORTING
    Both this fund's own evidence and the literature say the short side is not
    usable. si_leg_decomp.py measured the SI brick at 79% long leg / 21% short,
    with the short leg NOT individually significant. And a 2026 cross-sectional
    ML study found the absence of negative alphas at the bottom of the ranking
    confirms models cannot reliably identify underperformers, so shorting
    bottom-ranked names introduces estimation error from positions the model
    cannot distinguish from average performers.

    So the tradeable form of a reliable negative signal is a FILTER: do not hold
    names in the cell. That is what this measures.

WHAT IS MEASURED
    The h=40 cap-3 and prob>=0.70 books, run twice per seed on identical folds:
      baseline   as today
      filtered   names in the high-dtc/high-dp_share cell removed from the
                 candidate set BEFORE ranking

    Reported as the difference. A filter earns its place only if it improves the
    book -- a signal that predicts underperformance in a cell the model never
    picks from is true and useless.

    Also reported: how often the filter actually BINDS, i.e. how many of the
    book's picks it removes. A filter that never fires cannot help, and one that
    removes most picks is a different strategy rather than a filter.

LIMITS, STATED FIRST
    - Dark-pool data covers 416 tickers, not the expanded 1,920. This cannot
      extend to the wider universe without more UW fetching.
    - The dark-pool history is shorter than the price history, so the usable
      window is smaller than the h=40 book's own backtest.
    - Days-to-cover is taken at settlement + 12 calendar days. FINRA
      disseminates ~8 business days after settlement, and entry at settlement is
      a look-ahead worth roughly 10% of the SI brick's measured edge.
    - Bar is NW t > 3.0 per Harvey, Liu & Zhu, not the conventional 2.0.

    python analysis/darkpool_exclusion_test.py --seeds 3 --tickers 80
"""
import argparse
import datetime
import math
import os
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
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--dtc-q", type=float, default=0.67)
    ap.add_argument("--dp-q", type=float, default=0.67)
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    ic = sqlite3.connect("file:institutional_trades.db?mode=ro", uri=True)
    dp = defaultdict(lambda: defaultdict(float))
    for t, d, v in ic.execute(
            "SELECT ticker, trade_date, notional_usd FROM institutional_trades "
            "WHERE notional_usd IS NOT NULL AND is_dark_pool = 1 "
            "AND is_canceled = 0"):
        if t and d:
            dp[str(t).upper()][str(d)[:10]] += float(v or 0)
    ic.close()

    sc = sqlite3.connect("file:short_interest.db?mode=ro", uri=True)
    si = defaultdict(list)
    for t, d, v in sc.execute(
            "SELECT ticker, settlement_date, days_to_cover FROM "
            "short_interest WHERE days_to_cover IS NOT NULL "
            "AND days_to_cover <= 50 ORDER BY settlement_date"):
        si[str(t).upper()].append((str(d)[:10], float(v)))
    sc.close()

    def dtc_asof(t, d):
        h = si.get(t)
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

    uni = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    have = [t for t in uni if t in dp and t in si]
    print(f"dark-pool + short-interest coverage: {len(have)} of {len(uni)} "
          f"names\n")
    if len(have) < 50:
        raise SystemExit("too few names carry both sources")

    agg = defaultdict(list)
    for seed in range(1, args.seeds + 1):
        u = have[:]
        random.Random(seed).shuffle(u)
        sample = u[:args.tickers]

        X, fwd, dpshare, dtcv = {}, {}, {}, {}
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
                vol = list(df["volume"]) if "volume" in df.columns else None
                for j in range(args.window, len(ds) - H):
                    a, b = cl[j], cl[j + H]
                    if not a or not b:
                        continue
                    r = (b - a) / a
                    if abs(r) > 1.5:
                        continue
                    k = (t, ds[j])
                    X[k] = [float(v) if v == v else float("nan")
                            for v in num.iloc[j].tolist()]
                    fwd[k] = r
                    if vol:
                        dollar = sum((cl[m] or 0) * (vol[m] or 0)
                                     for m in range(j - args.window, j))
                        dpv = sum(dp[t].get(ds[m], 0.0)
                                  for m in range(j - args.window, j))
                        dpshare[k] = (dpv / dollar) if dollar > 0 else None
                    dtcv[k] = dtc_asof(t, ds[j])
            except Exception:
                continue
        if len(X) < 15000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        base = defaultdict(list)
        filt = defaultdict(list)
        n_removed, n_days, n_bind = 0, 0, 0

        for i in range(len(anchors) - 1):
            tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
            ktr = [k for k in X if k[1] < tr_end]
            kte = [k for k in X if tr_end <= k[1] < te_end]
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss",
                              verbosity=0)
            m.fit([X[k] for k in ktr], [1 if fwd[k] > 0 else 0 for k in ktr])
            p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]
            byd = defaultdict(list)
            for z, k in enumerate(kte):
                byd[k[1]].append((p[z], fwd[k], k))

            for d, rows in byd.items():
                if len(rows) < args.min_names:
                    continue
                n_days += 1
                mkt = st.mean(x[1] for x in rows)
                # the Boulton cell, computed cross-sectionally on THIS date
                dd = [(x, dtcv.get(x[2]), dpshare.get(x[2])) for x in rows]
                usable = [(x, a, b) for x, a, b in dd
                          if a is not None and b is not None]
                excluded = set()
                if len(usable) >= 12:
                    dq = sorted(a for _, a, _ in usable)
                    sq = sorted(b for _, _, b in usable)
                    dhi = dq[int(len(dq) * args.dtc_q)]
                    shi = sq[int(len(sq) * args.dp_q)]
                    excluded = {x[2] for x, a, b in usable
                                if a >= dhi and b >= shi}
                    n_removed += len(excluded)
                ranked = sorted(rows, reverse=True)
                keep = [x for x in ranked if x[2] not in excluded]
                if not keep:
                    continue
                if {x[2] for x in ranked[:max(CAPS)]} & excluded:
                    n_bind += 1
                for c in CAPS:
                    base[c].append(st.mean(x[1] for x in ranked[:c]) - mkt)
                    filt[c].append(st.mean(x[1] for x in keep[:c]) - mkt)
                hb = [x for x in ranked if x[0] >= THRESH]
                hf = [x for x in keep if x[0] >= THRESH]
                if hb and hf:
                    base["thr"].append(st.mean(x[1] for x in hb) - mkt)
                    filt["thr"].append(st.mean(x[1] for x in hf) - mkt)

        if not base[CAPS[0]]:
            print(f"seed {seed}: no scoreable rebalances\n")
            continue
        print(f"SEED {seed} — {len(X):,} rows, {len(base[CAPS[0]])} rebalances")
        print(f"  filter removed {n_removed/max(n_days,1):.1f} names/date and "
              f"touched a top-{max(CAPS)} pick on {n_bind}/{n_days} dates")
        print(f"  {'book':<10}{'baseline':>11}{'filtered':>11}{'delta':>11}"
              f"{'NW t':>8}")
        for key, lab in [(c, f"cap {c}") for c in CAPS] + [("thr",
                                                            "prob>=0.70")]:
            b, f = base.get(key, []), filt.get(key, [])
            if len(b) < 10:
                continue
            d = [f[i] - b[i] for i in range(len(b))]
            t_ = nw_t(d, 3) or 0.0
            print(f"  {lab:<10}{100*st.mean(b):>+10.3f}pp"
                  f"{100*st.mean(f):>+10.3f}pp{100*st.mean(d):>+10.3f}pp"
                  f"{t_:>+8.2f}"
                  + ("  PASSES t>3" if abs(t_) > T_HURDLE else ""))
            agg[key].append(st.mean(d))
        print()

    if not agg:
        print("no results")
        return
    print("=" * 60)
    print("ACROSS SEEDS — improvement from excluding the Boulton cell")
    print("=" * 60)
    print(f"  {'book':<12}{'mean delta':>13}{'seeds +':>10}")
    for key, lab in [(c, f"cap {c}") for c in CAPS] + [("thr", "prob>=0.70")]:
        v = agg.get(key, [])
        if not v:
            continue
        print(f"  {lab:<12}{100*st.mean(v):>+12.3f}pp"
              f"{sum(1 for x in v if x > 0):>6}/{len(v)}")

    print("\n  A filter earns its place only if it IMPROVES the book. A signal")
    print("  that predicts underperformance in a cell the model never picks")
    print("  from is true and useless -- which is why the bind rate is")
    print("  reported alongside.\n")
    print(f"  Bar is NW t > {T_HURDLE}: Harvey, Liu & Zhu argue a new factor")
    print("  needs t > 3.0 given hundreds of published factors and extensive")
    print("  data mining. Between 2 and 3 is NOT ESTABLISHED.")


if __name__ == "__main__":
    main()
