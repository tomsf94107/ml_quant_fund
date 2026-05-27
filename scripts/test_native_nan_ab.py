"""
scripts/test_native_nan_ab.py
─────────────────────────────
A/B test ML_QUANT_NATIVE_NAN and ML_QUANT_MISSING_INDICATORS flags.

For each ticker, trains 4 model configurations:
  - default (no flags)
  - native_nan only
  - missing_indicators only
  - native_nan + missing_indicators (combined)

Reports OOS AUC (from train_ensemble's 3-way split) and inst feature importance.

Usage:
    python scripts/test_native_nan_ab.py --tickers NVDA AAPL TSLA MSFT EME
"""
import argparse
import os
import sys
import importlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def force_reimport():
    """Clear cached modules so flag changes take effect."""
    for mod in list(sys.modules):
        if mod.startswith(("features.", "models.")):
            del sys.modules[mod]


def run_config(ticker, flag_native_nan, flag_indicators, horizon=5):
    """Train one config, return (test_auc, inst_importance_sum)."""
    # Set flags
    if flag_native_nan:
        os.environ["ML_QUANT_NATIVE_NAN"] = "1"
    else:
        os.environ.pop("ML_QUANT_NATIVE_NAN", None)
    if flag_indicators:
        os.environ["ML_QUANT_MISSING_INDICATORS"] = "1"
    else:
        os.environ.pop("ML_QUANT_MISSING_INDICATORS", None)
    
    # Always-on inst features
    os.environ["ML_QUANT_INST_FEATURES"] = "1"
    
    force_reimport()
    
    from features.builder import build_feature_dataframe, add_forecast_targets
    from models.ensemble import train_ensemble
    from models.classifier import TARGET_HORIZONS, FEATURE_COLUMNS
    
    df = build_feature_dataframe(ticker, start_date="2024-01-01")
    df = add_forecast_targets(df, horizons=TARGET_HORIZONS)
    
    try:
        result = train_ensemble(ticker, df, horizon=horizon, save=False)
    except Exception as e:
        import traceback
        print(f"[{ticker} flag_nan={flag_native_nan} flag_ind={flag_indicators}] EXCEPTION:")
        traceback.print_exc()
        return (None, None, None)
    
    test_auc = result.metrics.get("roc_auc")
    
    # LGB feature importance — focus on inst_* features
    lgb_inner = result.lgb_model.calibrated_classifiers_[0].estimator
    importances = lgb_inner.feature_importances_
    feat_names = result.feature_cols
    
    inst_imp = 0.0
    indicator_imp = 0.0
    for name, imp in zip(feat_names, importances):
        if name.startswith("inst_") and not name.endswith("_has_value"):
            inst_imp += float(imp)
        elif name.endswith("_has_value"):
            indicator_imp += float(imp)
    
    return (test_auc, inst_imp, indicator_imp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["NVDA", "AAPL", "TSLA", "MSFT", "EME"])
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    
    configs = [
        ("default",       False, False),
        ("native_nan",    True,  False),
        ("indicators",    False, True),
        ("both",          True,  True),
    ]
    
    print(f"\n{'Ticker':<8} {'Config':<14} {'TestAUC':>8} {'InstImp':>9} {'IndImp':>8}")
    print("-" * 60)
    
    results = []
    for ticker in args.tickers:
        for cfg_name, fnan, find in configs:
            auc, inst_imp, ind_imp = run_config(ticker, fnan, find, horizon=args.horizon)
            if auc is None:
                print(f"{ticker:<8} {cfg_name:<14}  ERROR")
                continue
            ind_str = f"{ind_imp:.1f}" if ind_imp is not None else "—"
            print(f"{ticker:<8} {cfg_name:<14} {auc:>8.4f} {inst_imp:>9.1f} {ind_str:>8}")
            results.append((ticker, cfg_name, auc, inst_imp, ind_imp))
        print()
    
    # Aggregate by config
    print("\n=== AGGREGATE (mean AUC across tickers) ===")
    for cfg_name, _, _ in configs:
        cfg_aucs = [r[2] for r in results if r[1] == cfg_name and r[2] is not None]
        if cfg_aucs:
            mean_auc = np.mean(cfg_aucs)
            print(f"  {cfg_name:<14}  mean_auc={mean_auc:.4f}  n={len(cfg_aucs)}")


if __name__ == "__main__":
    main()
