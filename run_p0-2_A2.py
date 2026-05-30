import sys, time, os
import pandas as pd
from analysis.walk_forward import (
    load_panel_pit, detect_feature_cols, walk_forward_backtest, print_backtest_report,
)
from models.classifier import XGB_PARAMS

DB="p0-2_A2.db"; PARQUET="p0-2_A2_panel.parquet"; PREFIX="p0-2_A2"
H=5; FOLDS=5; EMBARGO=5

print(f"=== P0-2 A2 START {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

# 1. BUILD (or resume from parquet if a prior run completed the build)
if os.path.exists(PARQUET):
    print(f"resuming: loading cached panel {PARQUET}", flush=True)
    df = pd.read_parquet(PARQUET)
else:
    print(f"building PIT panel from {DB} (~59h, training_mode, UW-free)...", flush=True)
    t0=time.time()
    df = load_panel_pit(DB, horizon=H, since=None, limit=None)
    print(f"panel built: {len(df):,} rows in {(time.time()-t0)/3600:.2f}h", flush=True)
    if df.empty:
        sys.exit("FAIL: empty panel")
    df.to_parquet(PARQUET, index=False)
    print(f"checkpoint saved -> {PARQUET}", flush=True)

# 2. SCORE (fast; re-runnable from parquet)
feat = detect_feature_cols(df)
print(f"scoring: {len(df):,} rows x {len(feat)} features, "
      f"{df['prediction_date'].min().date()} -> {df['prediction_date'].max().date()}", flush=True)
print(f"target: {df['actual_up'].mean()*100:.1f}% positive", flush=True)
params=dict(XGB_PARAMS)
print(f"[config=production] depth={params.get('max_depth')} "
      f"n_est={params.get('n_estimators')} lambda={params.get('reg_lambda')}", flush=True)
folds, overall = walk_forward_backtest(df, feat, n_folds=FOLDS, embargo=EMBARGO, model_params=params)
print_backtest_report(folds, overall)
folds.to_csv(f"{PREFIX}_folds.csv", index=False)
print(f"wrote {PREFIX}_folds.csv", flush=True)
print(f"=== P0-2 A2 DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
