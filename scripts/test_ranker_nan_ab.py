"""
scripts/test_ranker_nan_ab.py
─────────────────────────────
A/B test ML_QUANT_NATIVE_NAN and ML_QUANT_MISSING_INDICATORS on the
GLOBAL ranker. Build panel ONCE with flag=ON (so includes indicators),
then run 4 configs by selecting subsets / changing NaN handling.

Metric: OOS Q5-Q1 spread on horizon-day forward returns (2025+ holdout).
Baseline (commit e3790ad): +1.56pp for h=5.

Usage:
    python scripts/test_ranker_nan_ab.py --horizon 5
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_panel_once(horizon):
    """Build panel ONCE with all features (including indicators)."""
    # Force both flags so the panel has all possible columns
    os.environ["ML_QUANT_INST_FEATURES"] = "1"
    os.environ["ML_QUANT_MISSING_INDICATORS"] = "1"
    
    # Fresh module imports
    for mod in list(sys.modules):
        if mod.startswith(("features.", "models.")):
            del sys.modules[mod]
    
    from features.builder import build_feature_dataframe
    from models.classifier import FEATURE_COLUMNS as FC_ALL
    
    tickers = [t.strip().upper() for t in (ROOT / "tickers.txt").read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    
    all_dfs = []
    print(f"Building panel for {len(tickers)} tickers...")
    for i, t in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {t}")
        try:
            df = build_feature_dataframe(t, start_date="2020-01-01")
            if df.empty or len(df) < 200:
                continue
            df = df.copy()
            df["fwd_ret"] = (df["close"].shift(-horizon) / df["close"]) - 1.0
            df["_ticker"] = t
            all_dfs.append(df)
        except Exception:
            continue
    
    pooled = pd.concat(all_dfs, ignore_index=True).dropna(subset=["fwd_ret"])
    print(f"Pooled panel: {len(pooled)} rows × {len(FC_ALL)} features")
    
    # Ensure all FC_ALL cols exist
    for c in FC_ALL:
        if c not in pooled.columns:
            pooled[c] = 0.0
    
    return pooled, list(FC_ALL)


def run_one_config(pooled, fc_all, cfg_name, flag_native_nan, flag_indicators, horizon=5):
    """Train + evaluate one ranker config from pre-built panel."""
    import lightgbm as lgb
    
    # Pick feature columns based on indicator flag
    if flag_indicators:
        feat_cols = list(fc_all)
    else:
        feat_cols = [c for c in fc_all if not c.endswith("_has_value")]
    
    df = pooled.copy()
    
    # Apply NaN handling based on native_nan flag
    if flag_native_nan:
        pass  # leave NaN in place — LGBMRanker handles natively
    else:
        # Default: median-fill non-inst features (inst stays NaN — ranker default)
        non_inst = [c for c in feat_cols if not c.startswith("inst_")]
        df[non_inst] = df[non_inst].fillna(df[non_inst].median())
    
    # Chronological split
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < "2025-01-01"].copy()
    test_df = df[df["date"] >= "2025-01-01"].copy()
    
    train_df = train_df.sort_values("date").reset_index(drop=True)
    train_df["rank_pct"] = train_df.groupby("date")["fwd_ret"].rank(pct=True)
    train_df["relevance"] = pd.cut(train_df["rank_pct"], bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=[0,1,2,3,4]).astype(int)
    train_groups = train_df.groupby("date").size().values.tolist()
    
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05,
                            num_leaves=31, reg_alpha=0.1, reg_lambda=0.1, verbose=-1, random_state=42)
    ranker.fit(train_df[feat_cols].values, train_df["relevance"].values, group=train_groups)
    
    # Predict on test
    test_df = test_df.sort_values("date").reset_index(drop=True)
    test_df["raw_score"] = ranker.predict(test_df[feat_cols].values)
    test_df["pred_rank_pct"] = test_df.groupby("date")["raw_score"].rank(pct=True)
    test_df["pred_quintile"] = pd.cut(test_df["pred_rank_pct"], bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=["Q1","Q2","Q3","Q4","Q5"]).astype(str)
    
    q5_ret = test_df[test_df["pred_quintile"] == "Q5"]["fwd_ret"].mean() * 100
    q1_ret = test_df[test_df["pred_quintile"] == "Q1"]["fwd_ret"].mean() * 100
    spread = q5_ret - q1_ret
    
    importances = ranker.feature_importances_
    inst_imp = sum(float(imp) for n, imp in zip(feat_cols, importances) if n.startswith("inst_") and not n.endswith("_has_value"))
    ind_imp = sum(float(imp) for n, imp in zip(feat_cols, importances) if n.endswith("_has_value"))
    
    return {
        "config": cfg_name,
        "q5": q5_ret, "q1": q1_ret, "spread": spread,
        "n_feat": len(feat_cols), "inst_imp": inst_imp, "ind_imp": ind_imp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    
    # Build panel ONCE
    pooled, fc_all = build_panel_once(args.horizon)
    print(f"\n{'Config':<14} {'Q5%':>8} {'Q1%':>8} {'Spread':>10} {'InstImp':>9} {'IndImp':>8} {'NFeat':>6}")
    print("-" * 72)
    
    configs = [
        ("default",       False, False),
        ("native_nan",    True,  False),
        ("indicators",    False, True),
        ("both",          True,  True),
    ]
    
    for cfg, fnan, find in configs:
        r = run_one_config(pooled, fc_all, cfg, fnan, find, horizon=args.horizon)
        print(f"{r['config']:<14} {r['q5']:>8.3f} {r['q1']:>8.3f} {r['spread']:>+9.3f}pp "
              f"{r['inst_imp']:>9.0f} {r['ind_imp']:>8.0f} {r['n_feat']:>6}")


if __name__ == "__main__":
    main()
