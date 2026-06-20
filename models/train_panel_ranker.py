"""
train_panel_ranker.py — LightGBM lambdarank trained on the ALPHA PANEL.

WHY THIS EXISTS (2026-06-20):
The validated cross-sectional signals (pc_ratio_snap, short_pct_float, iv_skew_snap)
have real history ONLY in the alpha panel parquets — NOT in prediction_features
(62 days) and NOT reconstructable via fresh build_feature_dataframe (today-snapshot
broadcast). All production models (per-ticker classifier, cross-sectional ensemble,
global ranker) exclude them. This model trains DIRECTLY on the panel, where those
signals live as genuine daily cross-sections, so it can finally use them.

It is also the learning-to-rank model the research flagged as the #1 untested
high-value model: optimizes ranking (rank-IC / NDCG) not per-name direction.

CORRECTNESS (leakage control):
- Strict TEMPORAL walk-forward: train on past dates, test on future dates only.
- EMBARGO: drop the last `embargo` train dates before each test fold. With h=5
  the label uses 5 forward trading days, so the final train days' labels are built
  from prices that overlap the test window -> purge them.
- No shuffling, no random k-fold, no cross-date leakage in normalization (the panel
  cs_* transforms are already per-date cross-sectional, computed within each day).

EVAL:
- OOS rank-IC per TEST DATE (Spearman of predicted score vs actual_return across
  tickers that day), then summarized per fold and overall. Report the DISTRIBUTION,
  not just the mean (tight@0=no signal; wide straddling 0=regime-dependent).
"""
import sys, argparse, json
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd

PANEL_DIR = Path("data/alpha_panel")
DB = Path("accuracy.db")
STRANDED = ["pc_ratio_snap__cs_rank", "short_pct_float__cs_rank", "iv_skew_snap__cs_rank"]

NONFEAT = {"ticker","date","actual_return","actual_up","horizon",
           "prediction_date","fwd_ret_5d","fwd_ret_3d","fwd_ret_1d"}


def load_data(horizon):
    from analysis.alpha_fitness import _load_panel, _merge_outcomes
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, horizon)
    m = m.dropna(subset=["actual_return"]).sort_values(["date","ticker"]).reset_index(drop=True)
    feats = [c for c in m.columns if c not in NONFEAT]
    return m, feats


def rank_ic_by_date(df, score_col="__score__", ret_col="actual_return"):
    """Spearman rank-IC per date; returns a Series indexed by date."""
    def _ic(g):
        if len(g) < 5 or g[score_col].nunique() < 3:
            return np.nan
        return g[score_col].rank().corr(g[ret_col].rank())
    return df.groupby("date", group_keys=False).apply(_ic, include_groups=False).dropna()


def walk_forward(m, feats, horizon, n_folds=5, embargo=None, num_rounds=300, verbose=True):
    import lightgbm as lgb
    if embargo is None:
        embargo = horizon  # purge >= label horizon

    dates = np.array(sorted(m["date"].unique()))
    nd = len(dates)
    # sequential expanding-window folds over the date axis
    fold_edges = np.linspace(int(nd*0.4), nd, n_folds+1).astype(int)  # first 40% = initial train
    fold_edges = sorted(set(fold_edges))

    all_test_ic = []
    fold_summaries = []
    stranded_imp_acc = {s: [] for s in STRANDED if s in feats}

    for k in range(len(fold_edges)-1):
        test_start_i = fold_edges[k]
        test_end_i   = fold_edges[k+1]
        if test_start_i >= test_end_i:
            continue
        test_dates  = dates[test_start_i:test_end_i]
        # train = all dates strictly before test_start, minus embargo gap
        train_cut_i = max(0, test_start_i - embargo)
        train_dates = dates[:train_cut_i]
        if len(train_dates) < 30:
            continue

        tr = m[m["date"].isin(train_dates)]
        te = m[m["date"].isin(test_dates)]

        Xtr, ytr = tr[feats].values, tr["actual_return"].values
        Xte = te[feats].values
        grp_tr = tr.groupby("date").size().values  # lambdarank groups = per-date

        # lambdarank needs integer relevance labels: bucket forward returns into
        # per-date quantile ranks (0..K). Higher return -> higher relevance.
        def relevance(frame):
            r = frame.groupby("date")["actual_return"].rank(pct=True)
            return (r * 9.99).astype(int).values  # 0..9 relevance grades
        ytr_rel = relevance(tr)

        dtrain = lgb.Dataset(Xtr, label=ytr_rel, group=grp_tr,
                             feature_name=feats, free_raw_data=False)
        params = dict(objective="lambdarank", metric="ndcg",
                      ndcg_eval_at=[10,20], learning_rate=0.05,
                      num_leaves=31, min_data_in_leaf=50,
                      lambda_l1=1.0, lambda_l2=10.0,
                      max_depth=4, verbose=-1, force_col_wise=True)
        model = lgb.train(params, dtrain, num_boost_round=num_rounds)

        te = te.copy()
        te["__score__"] = model.predict(Xte)
        ic = rank_ic_by_date(te)
        all_test_ic.append(ic)

        # accumulate stranded-signal importance (gain)
        imp = dict(zip(feats, model.feature_importance(importance_type="gain")))
        for s in stranded_imp_acc:
            stranded_imp_acc[s].append(imp.get(s, 0.0))

        fold_summaries.append({
            "fold": k+1,
            "train_dates": len(train_dates),
            "test_dates": len(test_dates),
            "test_window": f"{pd.Timestamp(test_dates[0]).date()}..{pd.Timestamp(test_dates[-1]).date()}",
            "mean_ic": round(float(ic.mean()),5),
            "ic_t": round(float(ic.mean()/ic.std()*np.sqrt(len(ic))),2) if ic.std()>0 else 0.0,
            "n_ic_days": len(ic),
        })
        if verbose:
            fs = fold_summaries[-1]
            print(f"  fold {fs['fold']}: test {fs['test_window']} "
                  f"({fs['test_dates']}d) mean_ic={fs['mean_ic']:+.5f} t={fs['ic_t']:+.2f}")

    full_ic = pd.concat(all_test_ic) if all_test_ic else pd.Series(dtype=float)
    return full_ic, fold_summaries, stranded_imp_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--embargo", type=int, default=None)
    args = ap.parse_args()

    print(f"=== PANEL RANKER — h={args.horizon}, {args.folds} folds, embargo={args.embargo or args.horizon} ===")
    m, feats = load_data(args.horizon)
    print(f"loaded: {len(m)} rows, {len(feats)} features, {m['date'].nunique()} dates, "
          f"{m['ticker'].nunique()} tickers")
    present_stranded = [s for s in STRANDED if s in feats]
    print(f"stranded signals present: {present_stranded}")

    full_ic, folds, stranded_imp = walk_forward(
        m, feats, args.horizon, n_folds=args.folds,
        embargo=args.embargo, num_rounds=args.rounds)

    print("\n=== OOS RANK-IC (pooled across all test dates) ===")
    if len(full_ic):
        mean, sd, n = full_ic.mean(), full_ic.std(), len(full_ic)
        t = mean/sd*np.sqrt(n) if sd>0 else 0
        print(f"  mean rank-IC: {mean:+.5f}")
        print(f"  IC t-stat:    {t:+.2f}  (n={n} test-date ICs)")
        print(f"  IC>0 days:    {int((full_ic>0).sum())}/{n} ({100*(full_ic>0).mean():.0f}%)")
        print(f"  per-fold IC:  {[f['mean_ic'] for f in folds]}")
    print("\n=== STRANDED SIGNAL IMPORTANCE (gain, avg across folds) ===")
    for s, vals in stranded_imp.items():
        print(f"  {s}: {np.mean(vals):.1f}  (the whole point — usable here)")

    out = {"horizon": args.horizon, "folds": folds,
           "pooled_mean_ic": round(float(full_ic.mean()),5) if len(full_ic) else None,
           "stranded_importance": {s: round(float(np.mean(v)),2) for s,v in stranded_imp.items()}}
    Path("reports").mkdir(exist_ok=True)
    rp = Path(f"reports/panel_ranker_h{args.horizon}.json")
    rp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {rp}")


if __name__ == "__main__":
    main()
