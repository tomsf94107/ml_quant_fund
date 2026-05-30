"""
analysis/eval_global_pit.py — honest 5yr purged-WF eval of the GLOBAL
cross-sectional model, comparable apples-to-apples with P0-2 per-ticker.

The plan-doc 0.58 came from validate_oos(): a SINGLE chronological split at
2026-04-01 (test window ~2 months, one regime, no embargo). This re-runs the
SAME purged k-fold + embargo harness that gave per-ticker 0.49 over 5 years.

RULE #1: forces training_mode=True in the panel build (UW-free, weekend-safe).
"""
import argparse, time, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.builder import build_feature_dataframe
from models.classifier import FEATURE_COLUMNS
from analysis.walk_forward import walk_forward_backtest, print_backtest_report
from models.classifier import XGB_PARAMS


def build_pooled_pit(tickers, start_date, horizon):
    """Pooled cross-sectional panel, PIT-honest (training_mode=True)."""
    dfs = []
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            df = build_feature_dataframe(tk, start_date=start_date, training_mode=True)
            if df.empty or len(df) < 200:
                continue
            df = df.copy()
            df["actual_return"] = df["close"].shift(-horizon) / df["close"] - 1.0
            df["actual_up"] = (df["actual_return"] > 0).astype(int)
            df["prediction_date"] = pd.to_datetime(df["date"])
            df["ticker"] = tk
            dfs.append(df)
        except Exception as e:
            print(f"  {tk} failed: {type(e).__name__}: {e}", flush=True)
    pooled = pd.concat(dfs, ignore_index=True).dropna(subset=["actual_return"])
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start-date", default="2020-01-01")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in
               (ROOT / args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== GLOBAL PIT eval: {len(tickers)} tickers, h={args.horizon}, "
          f"{args.folds} folds, embargo={args.embargo}d ===", flush=True)

    t0 = time.time()
    pooled = build_pooled_pit(tickers, args.start_date, args.horizon)
    print(f"pooled panel: {len(pooled):,} rows, "
          f"{pooled['prediction_date'].min().date()} -> {pooled['prediction_date'].max().date()}, "
          f"built in {(time.time()-t0)/60:.1f}m", flush=True)

    feat = [c for c in FEATURE_COLUMNS if c in pooled.columns]
    print(f"features: {len(feat)}/{len(FEATURE_COLUMNS)} present", flush=True)
    print(f"positive class: {pooled['actual_up'].mean()*100:.1f}%", flush=True)

    params = dict(XGB_PARAMS)
    print(f"[config=production] depth={params.get('max_depth')} "
          f"n_est={params.get('n_estimators')} lambda={params.get('reg_lambda')}", flush=True)

    folds, overall = walk_forward_backtest(
        pooled, feat, n_folds=args.folds, embargo=args.embargo, model_params=params)
    print_backtest_report(folds, overall)

    out = args.out or f"analysis/global_pit_eval_h{args.horizon}.csv"
    folds.to_csv(ROOT / out, index=False)
    print(f"\nwrote {out}", flush=True)
    print(f"\n=== COMPARISON ===", flush=True)
    print(f"GLOBAL pooled OOS AUC (this, 5yr purged): {overall.get('pooled_oos_auc'):.4f}", flush=True)
    print(f"per-ticker OOS AUC (P0-2, 5yr purged):    0.487 / 0.493", flush=True)
    print(f"GLOBAL validate_oos (single 2mo split):   0.580  <- the number being tested", flush=True)


if __name__ == "__main__":
    main()
