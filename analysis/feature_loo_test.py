#!/usr/bin/env python3
"""
feature_loo_test.py — leave-one-out: which features are load-bearing?

READ-ONLY on the model. Trains throwaway copies. Writes nothing.

WHY
    The system carries 119 feature columns and nobody knows which matter.
    Importance rankings do not answer it: importance measures how often a tree
    SPLITS on a feature, not what the book earns without it. Two features can
    both score 5.0 while one is load-bearing and the other is a proxy for
    something already present.

    2026-09-05 produced several cases where that distinction mattered:
      - vix_term_structure scored 4.764 then 0.000 for ten weeks while pinned to
        a literal. Importance caught it late; a drop test would have shown it
        contributed nothing.
      - rate_beta had raw IC t=+3.80 at h=20 and retained only 37-41% after
        orthogonalisation against the existing columns -- it was a proxy.
      - The XGB+LightGBM ensemble adds +0.002 AUC (t=+0.63) despite a header
        claiming +1-3%.

    Method borrowed from a market-positioning study that rebuilt its composite
    21 times, once per ingredient removed, and published the range of the
    verdict statistic. Exactly one variant changed the verdict class, and the
    study named that ingredient as the hinge. That is a stronger statement than
    any importance table.

WHAT IS MEASURED
    Baseline: the h=40 cap-3 book -- the one candidate to survive every check on
    2026-09-05 (+1.70pp excess per period 2021-2025, four of five years
    positive, independent of the SI brick).

    Then, once per feature: drop it, retrain, re-measure. The DELTA against
    baseline is what the feature contributes.

      delta strongly NEGATIVE  load-bearing; removing it costs money
      delta near ZERO          redundant -- either uninformative or a proxy for
                               something else still present
      delta POSITIVE           the feature HURTS; the model is better without it

    Features are dropped in GROUPS as well as singly. A feature with five
    correlated siblings will show zero delta alone while the group matters --
    dropping ma_5 is nothing when ma_10 and ma_20 remain. Single-drop results
    without a group test systematically understate correlated families.

HONEST LIMITS, STATED FIRST
    - 119 retrains per seed. This is the most expensive test in the repo.
      Default is a coarse pass: groups first, then singles only for features
      whose group mattered.
    - At ~14-30 independent 40-day periods, a single feature's delta is inside
      the noise. Read the RANKING and the group results, not any one number.
    - Multi-seed is mandatory. Three single-seed results reversed on replication
      on 2026-09-05: a top-N book at +1.17pp went to -0.81pp, a linear economic
      edge at +0.645pp went to -0.021pp, and 3-of-3 consensus came in at
      -0.173pp against 1-of-3.

    python analysis/feature_loo_test.py --groups          # fast, families only
    python analysis/feature_loo_test.py --singles         # slow, every feature
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# Families that move together. A single drop inside one of these tells you
# almost nothing, because the siblings carry the same information.
GROUPS = {
    "moving_avg":   ("ma_5", "ma_10", "ma_20", "ma_50", "ma5_above_ma20",
                     "ma20_above_ma50"),
    "bollinger":    ("bb_upper", "bb_lower", "bb_width", "bb_pct"),
    "returns":      ("return_1d", "return_3d", "return_5d", "return_20d",
                     "return_60d"),
    "volatility":   ("volatility_5d", "volatility_10d", "atr", "vol_zscore_60d",
                     "vol_10d_self_rank"),
    "volume":       ("volume", "volume_zscore", "volume_spike", "obv",
                     "obv_trend", "vol_surge_eod", "vwap", "vwap_dev_eod"),
    "momentum":     ("rsi_14", "macd", "rsi_below_30", "intraday_momentum",
                     "premarket_gap"),
    "macro":        ("oil_ret", "oil_spy_corr", "spy_ret", "xlk_ret", "dxy_ret",
                     "yield_10y", "vix_close", "vix_ret", "vix_term_structure",
                     "lqd_hyg_spread"),
    "sector":       ("xle_ret_5d", "xlv_ret_5d", "xlf_ret_5d", "xlk_ret_5d",
                     "xlu_ret_5d", "xli_ret_5d", "xlp_ret_5d", "xly_ret_5d",
                     "xlc_ret_5d", "xlre_ret_5d", "xlb_ret_5d",
                     "sector_rel_ret", "semi_etf_momentum_60d"),
    "short_int":    ("short_ratio", "short_pct_float", "vol_x_short",
                     "is_squeeze_setup"),
    "insider":      ("insider_net_shares", "insider_7d", "insider_21d",
                     "insider_60d", "insider_90d"),
    "earnings":     ("eps_surprise", "rev_surprise", "days_to_earnings",
                     "post_earnings_1d", "post_earnings_3d", "post_earnings_5d",
                     "expected_move_perc", "pre_earnings_drift",
                     "post_earnings_drift", "is_earnings_week"),
    "eightk":       ("eightk_filings_30d", "eightk_days_since_last",
                     "eightk_exec_change_30d", "eightk_material_agreement_30d",
                     "eightk_reg_fd_30d", "eightk_other_events_30d"),
    "sentiment":    ("sentiment_score", "finbert_sentiment", "finbert_mult",
                     "finbert_sentiment_earnings", "monday_sentiment",
                     "fear_greed"),
    "fundamental":  ("fund_gp_assets", "fund_op_equity", "fund_ni_margin",
                     "fund_bm", "fund_ep", "rev_growth_yoy", "rev_growth_qoq"),
    "institutional": ("inst_block_buy_sell_7d", "inst_signed_flow_30d",
                      "inst_auction_imbal_5d", "inst_signed_flow_5d"),
    "risk":         ("risk_today", "risk_next_1d", "risk_next_3d",
                     "risk_prev_1d"),
    "52week":       ("high_52w_ratio", "low_52w_ratio", "rev_x_low52w"),
    "calendar":     ("day_of_week", "is_month_end", "is_pandemic"),
    "beta":         ("beta_60d",),
}


def build(universe, start, H):
    from features.builder import build_feature_dataframe
    X, fwd, cols = {}, {}, None
    for t in universe:
        try:
            df = build_feature_dataframe(t, start_date=start,
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
    return X, fwd, cols


def measure(X, fwd, keep_idx, anchors, cap, min_names):
    """Day-weighted excess of the top-cap book, over quarterly refits."""
    from xgboost import XGBClassifier
    from collections import defaultdict as dd
    ex = []
    for i in range(len(anchors) - 1):
        tr_end, te_end = anchors[i] + "-01", anchors[i + 1] + "-01"
        ktr = [k for k in X if k[1] < tr_end]
        kte = [k for k in X if tr_end <= k[1] < te_end]
        if len(ktr) < 4000 or len(kte) < 800:
            continue
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
        m.fit([[X[k][j] for j in keep_idx] for k in ktr],
              [1 if fwd[k] > 0 else 0 for k in ktr])
        p = m.predict_proba([[X[k][j] for j in keep_idx] for k in kte])[:, 1]
        byd = dd(list)
        for z, k in enumerate(kte):
            byd[k[1]].append((float(p[z]), fwd[k]))
        for d, v in byd.items():
            if len(v) < min_names:
                continue
            v.sort(reverse=True)
            ex.append(st.mean(r for _, r in v[:cap])
                      - st.mean(r for _, r in v))
    return st.mean(ex) if ex else None, len(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--min-names", type=int, default=25)
    ap.add_argument("--groups", action="store_true", default=True)
    ap.add_argument("--singles", action="store_true",
                    help="also drop every feature individually -- 119 retrains "
                         "per seed, hours")
    args = ap.parse_args()
    H = args.horizon

    sys.path.insert(0, ".")
    import random
    uni_all = [l.strip().upper() for l in open("tickers.txt") if l.strip()]

    print(f"leave-one-out — h={H}, cap-{args.cap}, {args.seeds} seeds x "
          f"{args.tickers} tickers\n")
    agg = defaultdict(list)
    base_all = []

    for seed in range(1, args.seeds + 1):
        u = uni_all[:]
        random.Random(seed).shuffle(u)
        X, fwd, cols = build(u[:args.tickers], args.start, H)
        if len(X) < 15000 or cols is None:
            print(f"seed {seed}: only {len(X)} rows, skipped\n")
            continue
        idx = {c: i for i, c in enumerate(cols)}
        dates = sorted({k[1] for k in X})
        months = sorted({d[:7] for d in dates})
        anchors = months[int(len(months) * 0.55)::3]
        allidx = list(range(len(cols)))

        base, nreb = measure(X, fwd, allidx, anchors, args.cap, args.min_names)
        if base is None:
            print(f"seed {seed}: baseline unmeasurable\n")
            continue
        print(f"SEED {seed} — {len(X):,} rows, {len(cols)} features, "
              f"{nreb} rebalances")
        print(f"  baseline: {100*base:+.3f}pp\n")
        base_all.append(base)

        targets = []
        for g, feats in GROUPS.items():
            drop = {idx[f] for f in feats if f in idx}
            if drop:
                targets.append((f"[{g}]", drop, len(drop)))
        if args.singles:
            for c in cols:
                targets.append((c, {idx[c]}, 1))

        print(f"  {'dropped':<22}{'n':>4}{'excess':>10}{'delta':>10}")
        for label, drop, nf in targets:
            keep = [i for i in allidx if i not in drop]
            if len(keep) < 10:
                continue
            v, _ = measure(X, fwd, keep, anchors, args.cap, args.min_names)
            if v is None:
                continue
            d = v - base
            flag = ("  <-- load-bearing" if d <= -0.005
                    else "  <-- HURTS" if d >= 0.005 else "")
            print(f"  {label:<22}{nf:>4}{100*v:>+9.3f}pp{100*d:>+9.3f}pp{flag}")
            agg[label].append(d)
        print()

    if not agg:
        print("no results")
        return
    print("=" * 62)
    print(f"ACROSS SEEDS — baseline {100*st.mean(base_all):+.3f}pp")
    print("=" * 62)
    rows = sorted(((st.mean(v), k, v) for k, v in agg.items() if len(v) >= 1))
    print(f"  {'dropped':<22}{'mean delta':>12}{'seeds':>7}{'agree':>7}")
    for m, k, v in rows:
        agree = sum(1 for x in v if (x < 0) == (m < 0))
        print(f"  {k:<22}{100*m:>+11.3f}pp{len(v):>7}{agree:>4}/{len(v)}")

    print("\n  NEGATIVE delta = removing it COST money = load-bearing.")
    print("  POSITIVE delta = the model is BETTER without it.")
    print("  Near zero = redundant, or a proxy for something still present.\n")
    print("  Read the RANKING, not any single number: at ~14-30 independent")
    print("  40-day periods one feature's delta sits inside the noise. The")
    print("  'agree' column shows whether the seeds point the same way, which")
    print("  matters more than the magnitude.")


if __name__ == "__main__":
    main()
