#!/usr/bin/env python3
"""
h40_book_test.py — threshold vs cap, and does the h=40 edge survive cost?

READ-ONLY. Trains throwaway copies. Writes nothing.

WHAT IS ALREADY ESTABLISHED (2026-09-05)
    XGBoost on the existing feature set, h=40, quarterly refits, 3 seeds:
        cap-3 day-weighted excess +5.194pp, positive 3/3 seeds
        cap-5 +4.476pp 3/3, cap-10 +3.724pp 3/3
        AUC 0.577 -- the model's only above-random AUC in the sweep
        turnover 27%, against 61% at h=3
    And it is NOT the short-interest brick reappearing: overlap with the
    lowest-DTC quintile is 32.2% against 19.5% random, Spearman(prob, DTC) is
    -0.060, and retraining with short_ratio and all its derivatives REMOVED
    gives +5.376pp -- slightly stronger, 3/3 seeds.

WHAT THIS SCRIPT ADDS, AND WHY IN THIS ORDER

  1. THRESHOLD vs CAP. Everything so far took a fixed count -- top 3, top 5.
     Production does not work that way: the gate is prob_eff >= 0.70. A fixed
     threshold takes however many clear the bar, which varies by day. That
     matters at h=40 precisely because AUC is 0.577, so probabilities should
     actually spread here, unlike h=5 where the model was near-random and almost
     nothing cleared 0.70.
     Reported with NAMES PER DAY, because a threshold that fires on 40 names
     some days and zero on others is a different instrument from a top-3 book
     even at the same average excess.

  2. COST. +5.134pp gross at 27% turnover over 40 days. A ladder rather than a
     single assumed figure, because the right bps depends on the names and the
     broker, and because every h=5 result this session died on friction rather
     than on sign.

  3. DRAWDOWN AND CONCENTRATION. PCT7 passed significance, nulls, regime
     stability, month-consistency and a cost ladder on 2026-09-05 and was still
     retracted -- because nobody had asked whether the measured quantity was
     capturable. Its pooled +1.46% was -1.97% day-weighted, and 38 of 64 days
     lost money. So: worst stretch, share of periods negative, and how
     concentrated the picks are in a few tickers.

METHOD
    Same construction as the horizon sweep: one feature panel per seed,
    quarterly refits, day-weighted excess over the SAME DAY's universe so market
    direction cannot flatter it, multi-seed because three single-seed results
    reversed on replication today.

    python analysis/h40_book_test.py --seeds 3 --tickers 80
"""
import argparse
import os
import math
import statistics as st
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# Widened 2026-09-14. Eight seeds showed reliability rising monotonically with
# name count -- cap-3 cleared t>3 in 4 of 8 seeds, cap-5 in 5, cap-10 in 6, and
# prob>=0.6 at 159 names/day in 8 of 8. The transition from unreliable to
# dependable happens somewhere between 10 and 159 names and had never been
# looked at. These intermediate caps locate it, and the smallest cap that holds
# across seeds is the smallest tradeable book whose edge is measured rather
# than hoped for.
CAPS = (1, 3, 5, 10, 15, 25, 40, 60, 100)
THRESHOLDS = (0.50, 0.55, 0.60, 0.70)


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
    ap.add_argument("--seed-start", type=int, default=1,
                    help="first seed. Default 1 reproduces every earlier arm; "
                         "9 gives a fresh, non-overlapping draw set.")
    ap.add_argument("--tickers", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--min-names", type=int, default=25)
    # ROLLING vs EXPANDING vs WEIGHTED. These three are the only difference
    # between the arms; folds, seeds, universe and horizon are identical, so a
    # difference in the result is attributable to the training slice alone.
    ap.add_argument("--window", type=int, default=0,
                    help="rolling window in CALENDAR days. 0 = expanding, "
                         "which is what production does (TRAIN_START=2018, no "
                         "decay). 441 is the ~7-quarter figure from the "
                         "volatility literature; it is a starting point to "
                         "measure, not a recommendation to adopt.")
    ap.add_argument("--half-life", type=int, default=0,
                    help="recency half-life in calendar days. 0 = uniform. "
                         "252 means a row one year older counts half as much. "
                         "Keeps every row, unlike --window, so it has no "
                         "sample-size cost.")
    ap.add_argument("--universe", default="tickers.txt")
    # TRAIN WIDE, SCORE NARROW (2026-09-13). Defaults to --universe, so every
    # existing invocation is unchanged.
    #
    # The two arms run on 2026-09-13 could not separate the estimation claim
    # from the slice. Arm A sampled 400 from tickers.txt and scored +1.543pp;
    # arm B sampled 400 from tickers_expanded.txt and scored +4.437pp -- but
    # 1,512 of those 1,924 names are mid and small, and the survivorship bound
    # had already measured this edge monotone in illiquidity, mega +1.721pp to
    # small +4.932pp. Arm B landed in the small band. Different names, not a
    # better estimate.
    #
    # With --score-universe the MODEL fits on the wide draw and the RANKING is
    # restricted to the narrow set, so both arms score the same names and any
    # difference is the training panel. That is what "train wide, trade narrow"
    # means in production, and it is what the expansion case actually asserts.
    #
    # This touches the TEST keys only. ktr is unchanged, so nothing the model
    # learned changes -- only which names compete for the cap slots.
    ap.add_argument("--score-universe", default=None,
                    help="rank only names in this file; model still trains on "
                         "the full --universe draw. Default: same as --universe")
    # WITHOUT THIS, --score-universe IS CONFOUNDED BY SELECTIVITY.
    # 422 traded names in a 1,924 pool is 22%, so a 400-name draw contains only
    # ~87 of them (measured: 86, 94, 82 for seeds 1-3). Arm A picks cap-3 from
    # 400 candidates, 0.75% selectivity; that arm would pick 3 from ~87, 3.4%.
    # A lower score is then selectivity, not the training panel.
    # --pin-score puts every scored name in the draw FIRST and fills the rest
    # at random, so the scored set is held fixed across arms and the only
    # difference is the extra training names. That is the actual
    # train-wide/trade-narrow comparison.
    ap.add_argument("--pin-score", action="store_true",
                    help="always include every --score-universe name in the "
                         "training draw; fill the remainder at random")
    args = ap.parse_args()
    H = args.horizon

    import datetime as _dt
    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    from xgboost import XGBClassifier
    import random

    # --universe lets the expanded set be tested WITHOUT swapping tickers.txt,
    # which every cron job reads. The h=40 shadow book in particular is frozen
    # on the current 415 names and changing its universe would void it.
    uni_all = [l.strip().upper() for l in open(args.universe) if l.strip()]
    score_set = None
    if args.score_universe:
        score_set = {l.strip().upper() for l in open(args.score_universe)
                     if l.strip()}
        print(f"  scoring restricted to {len(score_set)} names from "
              f"{args.score_universe}; model trains on the full "
              f"{args.universe} draw")
    _arm = ("expanding (production)" if args.window == 0 and args.half_life == 0
            else f"rolling {args.window}d" if args.window > 0
            else f"recency half-life {args.half_life}d")
    print(f"h={H} book test — {args.seeds} seeds x {args.tickers} tickers")
    print(f"training slice: {_arm}\n")

    agg_cap = defaultdict(list)
    agg_ls = defaultdict(list)
    pooled_ls = defaultdict(list)
    agg_thr = defaultdict(list)
    agg_n = defaultdict(list)
    agg_turn = defaultdict(list)
    dd_all, neg_all, conc_all = [], [], []

    # SEED OFFSET. Seeds are the universe-draw randomisation, and seeds 1-8
    # have been used for every arm so far. A result that holds on 1-8 and not
    # on 9-16 was a property of those eight draws, not of the construction.
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        if args.pin_score and score_set:
            _pin = [t for t in uni_all if t in score_set]
            _rest = [t for t in u if t not in score_set]
            u = _pin + _rest
            if seed == 1:
                print(f"  pinned {len(_pin)} scored names into every draw; "
                      f"{max(0, args.tickers - len(_pin))} random fill")
        X, fwd = {}, {}
        for t in u[:args.tickers]:
            try:
                df = build_feature_dataframe(t, start_date=args.start,
                                             training_mode=True)
                if df is None or len(df) < 400 or "close" not in df.columns:
                    continue
                num = df.select_dtypes("number")
                num = num.drop(columns=[c for c in num.columns
                                        if c.startswith("target_")
                                        # ML_QUANT_NO_DARKPOOL=1 drops these so
                                        # their contribution can be isolated.
                                        # This script builds its matrix from
                                        # select_dtypes("number"), NOT from
                                        # FEATURE_COLUMNS, so the flag in
                                        # models/classifier.py has no effect
                                        # here -- a first A/B returned figures
                                        # identical to three decimals because
                                        # both runs used the same columns.
                                        or (os.environ.get(
                                            "ML_QUANT_NO_DARKPOOL") == "1"
                                            and c in ("dp_volume_share",
                                                      "boulton_cell"))],
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
        if len(X) < 15000:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue

        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]

        # LONG/SHORT LEG (2026-09-15). The h=5 book gives +0.436% per period
        # gross and dies at 10bps/leg, because four crossings -- buy and sell
        # the long, sell short and cover the short -- recur every five days.
        # At h=40 the same four crossings amortise over eight times the holding
        # period, so the cost per period is unchanged while the edge has eight
        # times as long to accumulate. That is the only untested route left
        # after tighter gates were tried at h=5 and made it worse: 0.75/0.30
        # gave the same gross with HALF the t-statistic, so the edge is diffuse
        # rather than concentrated in the tail.
        #
        # Built on the SAME fold, seed and walk-forward as the cap legs, so the
        # numbers are directly comparable rather than a separate experiment.
        ls_ex = defaultdict(list)
        cap_ex = defaultdict(list)
        cap_turn = defaultdict(list)
        thr_ex = defaultdict(list)
        thr_n = defaultdict(list)
        thr_turn = defaultdict(list)
        picks = []
        prev_cap = {c: set() for c in CAPS}
        prev_thr = {t: set() for t in THRESHOLDS}

        for i in range(len(anchors) - 1):
            tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
            # THE ONE LINE THE ARMS DIFFER AT.
            # Expanding (production): every row before the anchor.
            # Rolling: only the trailing --window calendar days.
            if args.window > 0:
                _lo = (_dt.date.fromisoformat(tr_end)
                       - _dt.timedelta(days=args.window)).isoformat()
                ktr = [k for k in X if _lo <= k[1] < tr_end]
            else:
                ktr = [k for k in X if k[1] < tr_end]
            kte = [k for k in X if tr_end <= k[1] < te_end
                   and (score_set is None or k[0] in score_set)]
            # NOTE the kte floor. Narrowing the scoring set cuts this count, and
            # a fold silently skipped here means fewer rebalances and an arm
            # that is not comparable to the others. The run prints the fold
            # count so a shortfall is visible rather than inferred.
            if len(ktr) < 4000 or len(kte) < 800:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0)
            # Recency weights, if asked. 0.5 ** (age / half_life) measured
            # back from the anchor, so the newest row weighs 1.0 and older rows
            # decay geometrically. XGBoost takes these directly as
            # sample_weight -- no resampling, no rows discarded.
            _w = None
            if args.half_life > 0:
                _end = _dt.date.fromisoformat(tr_end)
                _w = [0.5 ** ((_end - _dt.date.fromisoformat(k[1])).days
                              / args.half_life) for k in ktr]
            m.fit([X[k] for k in ktr], [1 if fwd[k] > 0 else 0 for k in ktr],
                  sample_weight=_w)
            p = [float(v) for v in m.predict_proba([X[k] for k in kte])[:, 1]]
            byd = defaultdict(list)
            for z, k in enumerate(kte):
                byd[k[1]].append((p[z], fwd[k], k[0]))

            for d in sorted(byd):
                v = sorted(byd[d], reverse=True)
                if len(v) < args.min_names:
                    continue
                mkt = st.mean(x[1] for x in v)
                for c in CAPS:
                    sel = v[:c]
                    cap_ex[c].append(st.mean(x[1] for x in sel) - mkt)
                    # Long the top c, short the bottom c, equal dollars. The
                    # market move cancels, so no `- mkt` here: the benchmark IS
                    # the short leg. Half the sum because the book is two legs
                    # of equal size, not one leg of double.
                    if len(v) >= 2 * c:
                        _bot = v[-c:]
                        ls_ex[c].append(0.5 * (st.mean(x[1] for x in sel)
                                               - st.mean(x[1] for x in _bot)))
                    cur = {x[2] for x in sel}
                    if prev_cap[c]:
                        cap_turn[c].append(
                            100.0 * len(cur - prev_cap[c]) / max(len(cur), 1))
                    prev_cap[c] = cur
                    if c == 3:
                        picks += list(cur)
                for th in THRESHOLDS:
                    sel = [x for x in v if x[0] >= th]
                    thr_n[th].append(len(sel))
                    if not sel:
                        continue
                    thr_ex[th].append(st.mean(x[1] for x in sel) - mkt)
                    cur = {x[2] for x in sel}
                    if prev_thr[th]:
                        thr_turn[th].append(
                            100.0 * len(cur - prev_thr[th]) / max(len(cur), 1))
                    prev_thr[th] = cur

        if not cap_ex[3]:
            print(f"seed {seed}: no scoreable rebalances\n")
            continue

        print(f"SEED {seed} — {len(X):,} rows, {len(cap_ex[3])} rebalances")
        print(f"  {'selection':<14}{'excess':>10}{'NW t':>8}{'names/day':>11}"
              f"{'turnover':>10}")
        for c in CAPS:
            print(f"  {'cap '+str(c):<14}{100*st.mean(cap_ex[c]):>+9.3f}pp"
                  f"{(nw_t(cap_ex[c], 3) or 0):>+8.2f}{c:>11}"
                  f"{(st.mean(cap_turn[c]) if cap_turn[c] else 0):>9.0f}%")
            agg_cap[c].append(st.mean(cap_ex[c]))
        for th in THRESHOLDS:
            if not thr_ex[th]:
                print(f"  {'prob>='+str(th):<14}   never fires")
                continue
            print(f"  {'prob>='+str(th):<14}{100*st.mean(thr_ex[th]):>+9.3f}pp"
                  f"{(nw_t(thr_ex[th], 3) or 0):>+8.2f}"
                  f"{st.mean(thr_n[th]):>11.1f}"
                  f"{(st.mean(thr_turn[th]) if thr_turn[th] else 0):>9.0f}%")
            agg_thr[th].append(st.mean(thr_ex[th]))
            agg_n[th].append(st.mean(thr_n[th]))
            agg_turn[th].append(st.mean(thr_turn[th]) if thr_turn[th] else 0)

        # LONG/SHORT rows. No `- mkt`: the short leg IS the benchmark, so
        # subtracting the market as well would double-count the hedge. That is
        # why this cannot reuse cap_ex with a negative index.
        if ls_ex.get(3):
            print(f"  {'long/short':<14}{'excess':>9}{'NW t':>8}{'names':>11}")
            for c in CAPS:
                if not ls_ex.get(c) or len(ls_ex[c]) < 5:
                    continue
                print(f"  {'L/S cap ' + str(c):<14}"
                      f"{100*st.mean(ls_ex[c]):>+9.3f}pp"
                      f"{(nw_t(ls_ex[c], H) or 0):>+8.2f}"
                      f"{2*c:>11}")
                agg_ls[c].append(st.mean(ls_ex[c]))
                # Keep the RAW per-rebalance series, not just its mean. The
                # standing bar is NW t > 3.0 on the NET figure, and a t cannot
                # be recovered from eight seed means -- it needs the rebalances
                # underneath them. Pooling across seeds treats each seed's
                # rebalances as additional observations of the same book, which
                # is what they are: same period, same construction, different
                # 400-name draw from the same 422.
                pooled_ls[c].extend(ls_ex[c])
            print()

        e = cap_ex[3]
        eq, peak, mdd = 1.0, 1.0, 0.0
        for r in e:
            eq *= (1 + r / H)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
        neg = 100.0 * sum(1 for x in e if x < 0) / len(e)
        c5 = 100.0 * sum(n for _, n in Counter(picks).most_common(5)) / max(len(picks), 1)
        print(f"  cap-3 path: max drawdown {100*mdd:.1f}%, "
              f"{neg:.0f}% of rebalances negative, top-5 tickers "
              f"{c5:.0f}% of picks, {len(set(picks))} distinct\n")
        dd_all.append(mdd); neg_all.append(neg); conc_all.append(c5)

    if not agg_cap[3]:
        print("no seeds produced results")
        return

    print("=" * 70)
    print("ACROSS SEEDS")
    print("=" * 70)
    print(f"  {'selection':<14}{'excess':>10}{'seeds +':>9}{'names/day':>11}"
          f"{'turnover':>10}")
    for c in CAPS:
        v = agg_cap[c]
        print(f"  {'cap '+str(c):<14}{100*st.mean(v):>+9.3f}pp"
              f"{sum(1 for x in v if x>0):>5}/{len(v)}{c:>11}")
    for th in THRESHOLDS:
        v = agg_thr.get(th, [])
        if not v:
            continue
        print(f"  {'prob>='+str(th):<14}{100*st.mean(v):>+9.3f}pp"
              f"{sum(1 for x in v if x>0):>5}/{len(v)}"
              f"{st.mean(agg_n[th]):>11.1f}{st.mean(agg_turn[th]):>9.0f}%")

    if agg_ls.get(3):
        print()
        print(f"  {'long/short':<14}{'excess':>10}{'seeds +':>9}{'names':>11}")
        for c in CAPS:
            v = agg_ls.get(c, [])
            if not v:
                continue
            print(f"  {'L/S cap '+str(c):<14}{100*st.mean(v):>+9.3f}pp"
                  f"{sum(1 for x in v if x>0):>5}/{len(v)}{2*c:>11}")

        # LADDER ON THE L/S BOOK. Four crossings per rebalance -- buy and sell
        # the long leg, sell short and cover the short -- the same count as the
        # h=5 book, but here they amortise over 40 days instead of 5. That is
        # the whole reason this was worth testing: at h=5 the identical
        # construction gave +0.436pp at NW t +1.45 and went to zero at 10bps
        # per leg.
        _lsc = 10 if agg_ls.get(10) else 3
        _g = st.mean(agg_ls[_lsc])
        _P = pooled_ls.get(_lsc, [])
        # COST CONVENTION, corrected 2026-09-15. The L/S return is HALF-SCALED
        # -- 0.5 * (mean(long) - mean(short)) -- so the book is $0.5 long and
        # $0.5 short, gross exposure 1.0. Four crossings at $0.5 notional each
        # is 2 bps-units, not 4. The first version charged 4 and so overstated
        # cost by exactly 2x. That direction can only manufacture a FALSE FAIL,
        # never a false pass, and given the h=5 book died at 10bps/leg a false
        # fail is precisely the outcome that would wrongly close this axis.
        print(f"\n  COST LADDER on L/S cap-{_lsc} ({100*_g:+.3f}pp gross, "
              f"{len(_P)} pooled rebalances across {len(agg_ls[_lsc])} seeds)")
        print(f"      {'bps/leg':>9}{'net':>12}{'NW t':>9}")
        for _bps in (0, 5, 10, 20, 40, 60, 100):
            _c = 2.0 * _bps / 1e4
            _net = [x - _c for x in _P]
            _t = nw_t(_net, H) if _net else 0
            print(f"      {_bps:>9}{100*(st.mean(_net) if _net else 0):>+11.3f}pp"
                  f"{(_t or 0):>+9.2f}")
        print(f"\n  The t is on the POOLED per-rebalance series, which is what")
        print(f"  the standing bar (NW t > 3.0 on the NET figure) is written")
        print(f"  against. Per-seed gross t ran lower; pooling adds observations")
        print(f"  but the rebalances overlap across seeds, so treat this as the")
        print(f"  optimistic end of the range rather than the number.")
        print(f"\n  Borrow is charged separately and is small at this horizon:")
        print(f"  the bottom decile runs 127-173bps annualised against a 196bps")
        print(f"  universe average, so 40 days costs roughly 20-27bps -- about")
        print(f"  1% of a {100*_g:.1f}pp gross edge. Execution is what decides it.")

    print(f"\n  COST LADDER on cap-3 ({100*st.mean(agg_cap[3]):+.3f}pp gross, "
          f"~27% turnover, one round trip per rebalance)")
    print(f"    {'bps/leg':>9}{'net':>11}")
    g = st.mean(agg_cap[3])
    for bps in (0, 5, 10, 20, 40, 100):
        print(f"    {bps:>9}{100*(g - 2*bps/10000.0):>+10.3f}pp")

    if dd_all:
        print(f"\n  cap-3 path across seeds: max drawdown "
              f"{100*st.mean(dd_all):.1f}%, "
              f"{st.mean(neg_all):.0f}% of rebalances negative, "
              f"top-5 tickers {st.mean(conc_all):.0f}% of picks")

    print("\n  A threshold that matches cap-3's excess with MORE names is a")
    print("  better book -- same edge, less idiosyncratic risk. One that fires")
    print("  on almost nothing is a top-1 book wearing a different label, and")
    print("  the names/day column is what tells them apart.\n")
    print("  Drawdown here scales cohort returns by 1/H and assumes equal")
    print("  weights and free fills. It is a sanity check on the path, not a")
    print("  backtest.")


if __name__ == "__main__":
    main()
