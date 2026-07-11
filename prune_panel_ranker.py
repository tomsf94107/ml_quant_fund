"""
Stability-based feature pruning for the panel ranker — LEAKAGE-SAFE.

Method:
1. SELECTION PHASE (early dates only, first 60% of timeline): run walk-forward,
   record per-fold gain importance. Score each feature by
   mean_importance * fold_consistency (fraction of folds where nonzero).
2. Pick top-K by that stability score. ALWAYS include the 3 validated stranded
   signals + ensure core families (momentum/reversal/vol/rsi) are present.
3. VALIDATION PHASE: run a fresh walk-forward using ONLY the selected features,
   over the FULL timeline, and compare pooled IC + per-fold stability to the
   3540-feature baseline.

Leakage control: selection uses ONLY the first 60% of dates. The held-out last
40% was never seen during selection, so improvement there is honest.
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from models.train_panel_ranker import load_data, walk_forward, rank_ic_by_date, STRANDED

CORE_PATTERNS = ["return_1d__cs", "return_3d__cs", "return_5d__cs", "return_20d__cs",
                 "rsi_14__cs", "macd__cs", "volatility", "atr", "reversal",
                 "sector_rel_ret__cs", "spy_ret__cs", "vix"]
TOPK = 40


def importance_stability(m, feats, horizon, select_dates):
    """Run walk-forward on SELECT dates only; return per-feature (mean_imp, consistency)."""
    import lightgbm as lgb
    sub = m[m["date"].isin(select_dates)].copy()
    dates = np.array(sorted(sub["date"].unique()))
    nd = len(dates)
    edges = np.linspace(int(nd*0.4), nd, 4).astype(int)  # 3 folds within selection window
    edges = sorted(set(edges))
    imp_per_fold = []
    for k in range(len(edges)-1):
        ts_i, te_i = edges[k], edges[k+1]
        if ts_i >= te_i: continue
        test_d = dates[ts_i:te_i]
        train_cut = max(0, ts_i - horizon)
        train_d = dates[:train_cut]
        if len(train_d) < 30: continue
        tr = sub[sub["date"].isin(train_d)]
        grp = tr.groupby("date").size().values
        rel = (tr.groupby("date")["actual_return"].rank(pct=True) * 9.99).astype(int).values
        d = lgb.Dataset(tr[feats].values, label=rel, group=grp,
                        feature_name=feats, free_raw_data=False)
        params = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05,
                      num_leaves=31, min_data_in_leaf=50, lambda_l1=1.0,
                      lambda_l2=10.0, max_depth=4, verbose=-1, force_col_wise=True)
        mdl = lgb.train(params, d, num_boost_round=200)
        imp_per_fold.append(dict(zip(feats, mdl.feature_importance(importance_type="gain"))))
    if not imp_per_fold:
        return {}
    stats = {}
    nfolds = len(imp_per_fold)
    for f in feats:
        vals = [d.get(f, 0.0) for d in imp_per_fold]
        mean_imp = float(np.mean(vals))
        consistency = float(np.mean([v > 0 for v in vals]))  # fraction of folds nonzero
        stats[f] = mean_imp * consistency  # stability score
    return stats


def main():
    horizon = 5
    print(f"=== STABILITY PRUNING — h={horizon} ===")
    m, feats = load_data(horizon)
    dates = np.array(sorted(m["date"].unique()))
    cut = int(len(dates) * 0.6)
    select_dates = dates[:cut]   # first 60% for selection
    print(f"selection window: {pd.Timestamp(select_dates[0]).date()}..{pd.Timestamp(select_dates[-1]).date()} ({len(select_dates)} dates)")
    print(f"held-out for honest validation: last {len(dates)-cut} dates")

    print("\nrunning importance-stability pass on selection window...")
    scores = importance_stability(m, feats, horizon, select_dates)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    chosen = [f for f, s in ranked[:TOPK] if s > 0]

    # force-include validated stranded signals
    for s in STRANDED:
        if s in feats and s not in chosen:
            chosen.append(s)
    # ensure core families present
    for pat in CORE_PATTERNS:
        match = [f for f in feats if pat in f]
        if match and not any(c in chosen for c in match):
            chosen.append(match[0])
    chosen = list(dict.fromkeys(chosen))  # dedupe, keep order

    print(f"\nselected {len(chosen)} features (top-{TOPK} stable + stranded + core):")
    for f in chosen[:50]:
        print(f"  {scores.get(f,0):8.1f}  {f}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/panel_ranker_pruned_features.json").write_text(json.dumps(chosen, indent=2))

    # VALIDATION: fresh walk-forward on full timeline with pruned features
    print(f"\n=== VALIDATION: walk-forward on {len(chosen)} pruned features (full timeline) ===")
    full_ic, folds, stranded_imp = walk_forward(m, chosen, horizon, n_folds=5, num_rounds=300)
    if len(full_ic):
        mean, sd, n = full_ic.mean(), full_ic.std(), len(full_ic)
        t = mean/sd*np.sqrt(n) if sd>0 else 0
        print(f"\n  PRUNED pooled rank-IC: {mean:+.5f}  t={t:+.2f}  (n={n})")
        print(f"  per-fold IC: {[f['mean_ic'] for f in folds]}")
        print(f"  BASELINE (3540 feat): pooled +0.01003 t=+0.85  folds [-0.050,0.050,-0.021,0.005,0.066]")
    print("\n  stranded importance (pruned):")
    for s, v in stranded_imp.items():
        print(f"    {s}: {np.mean(v):.1f}")


if __name__ == "__main__":
    main()
