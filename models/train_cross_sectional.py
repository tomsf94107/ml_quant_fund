"""
models/train_cross_sectional.py
────────────────────────────────
Path A: pooled cross-sectional global model. One model per horizon trained
on all 125 tickers' data combined.

Hypothesis: per-ticker training has too few cross-sectional events (e.g.
~6 exec_change events per ticker) for trees to learn from. Pooling across
tickers gives ~750+ events of each kind, enabling cross-sectional alpha
extraction.

Saves as models/saved/GLOBAL_ensemble_{horizon}d.joblib using EnsembleResult.
Reuses EnsembleResult so signal pipeline can load identically.

Run:
    python -m models.train_cross_sectional
    python -m models.train_cross_sectional --horizons 5  # specific horizon
    python -m models.train_cross_sectional --validate    # OOS chronological split
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _read_ticker_file(path: Path) -> list[str]:
    return [
        t.strip().upper() for t in path.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]


def build_pooled_panel(
    tickers: list[str],
    start_date: str = "2020-01-01",
    verbose: bool = True,
) -> pd.DataFrame:
    """Build features for all tickers and concat into a single DataFrame."""
    from features.builder import build_feature_dataframe, add_forecast_targets
    from models.classifier import TARGET_HORIZONS

    all_dfs = []
    log.info(f"Building features for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            log.info(f"  [{i}/{len(tickers)}] {ticker}")
        try:
            df = build_feature_dataframe(ticker, start_date=start_date)
            if df.empty or len(df) < 200:
                continue
            df = add_forecast_targets(df, horizons=TARGET_HORIZONS)
            df['_ticker'] = ticker
            all_dfs.append(df)
        except Exception as e:
            log.warning(f"  {ticker} failed: {type(e).__name__}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No tickers produced valid data")

    pooled = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Pooled panel: {len(pooled)} rows across {len(all_dfs)} tickers")
    return pooled


def train_one_horizon_pooled(
    df_pooled: pd.DataFrame,
    horizon: int,
    save: bool = True,
    verbose: bool = True,
) -> object:
    """Train an EnsembleResult on the pooled panel for one horizon."""
    from models.classifier import FEATURE_COLUMNS
    from models.ensemble import train_ensemble, EnsembleResult, MODEL_DIR

    target_col = f"target_{horizon}d"
    if target_col not in df_pooled.columns:
        raise ValueError(f"Target column {target_col} not in panel")

    # Drop rows with NaN target
    n_before = len(df_pooled)
    df_train = df_pooled.dropna(subset=[target_col]).copy()
    log.info(f"  h={horizon}d: dropped {n_before - len(df_train)} NaN rows → {len(df_train)} usable")

    # Ensure all FEATURE_COLUMNS exist (fill missing with 0)
    for c in FEATURE_COLUMNS:
        if c not in df_train.columns:
            df_train[c] = 0.0

    # Fill NaN feature values with median per column (cross-sectional median)
    df_train[FEATURE_COLUMNS] = df_train[FEATURE_COLUMNS].fillna(df_train[FEATURE_COLUMNS].median())

    # Hijack the train_ensemble signature with a synthetic "ticker" name
    # The function takes a ticker name for saving; we use "GLOBAL"
    log.info(f"  h={horizon}d: training pooled ensemble on {len(df_train)} rows × {len(FEATURE_COLUMNS)} features")
    t0 = time.time()
    result = train_ensemble(
        ticker="GLOBAL",
        df=df_train,
        horizon=horizon,
        verbose=verbose,
        save=save,
    )
    log.info(f"  h={horizon}d: training done in {time.time()-t0:.1f}s")
    log.info(f"  h={horizon}d: train_auc={result.metrics.get('train_auc', 'N/A')}  test_auc={result.metrics.get('test_auc', 'N/A')}")
    return result


def validate_oos(df_pooled: pd.DataFrame, horizon: int, split_date: str = "2026-04-01"):
    """Chronological train/test split. Train on data before split_date, test after."""
    from models.classifier import FEATURE_COLUMNS
    from models.ensemble import train_ensemble
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    target_col = f"target_{horizon}d"
    df = df_pooled.dropna(subset=[target_col]).copy()
    df['date'] = pd.to_datetime(df['date'])
    
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    
    log.info(f"  OOS split: train n={len(train_df)} (before {split_date}), test n={len(test_df)} (after)")
    
    if len(test_df) < 100:
        log.warning(f"  Test set too small ({len(test_df)})")
        return None
    
    # Fill features
    for c in FEATURE_COLUMNS:
        if c not in train_df.columns:
            train_df[c] = 0.0
        if c not in test_df.columns:
            test_df[c] = 0.0
    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].fillna(train_df[FEATURE_COLUMNS].median())
    test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(train_df[FEATURE_COLUMNS].median())  # use TRAIN median
    
    # Train (don't save validation runs)
    result = train_ensemble(
        ticker=f"GLOBAL_VAL",
        df=train_df,
        horizon=horizon,
        verbose=False,
        save=False,
    )
    
    # Predict on test
    pred = result.predict_proba(test_df)
    auc = roc_auc_score(test_df[target_col], pred)
    acc = accuracy_score(test_df[target_col], pred > 0.5)
    
    log.info(f"  h={horizon}d OOS: AUC={auc:.4f}  acc={acc*100:.1f}%  n_test={len(test_df)}")
    return {"horizon": horizon, "oos_auc": auc, "oos_acc": acc, "n_test": len(test_df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--validate", action="store_true", help="Run OOS validation, don't save")
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N tickers (debug)")
    args = ap.parse_args()

    tickers = _read_ticker_file(Path("tickers.txt"))
    if args.limit:
        tickers = tickers[:args.limit]
    
    log.info(f"=== Cross-sectional pooled training ===")
    log.info(f"Tickers: {len(tickers)} | Horizons: {args.horizons} | Start: {args.start}")
    log.info(f"Mode: {'VALIDATION (no save)' if args.validate else 'PRODUCTION (save GLOBAL models)'}")

    panel = build_pooled_panel(tickers, start_date=args.start)
    
    results = []
    for h in args.horizons:
        log.info(f"\n──── Horizon {h}d ────")
        if args.validate:
            r = validate_oos(panel, h)
            if r:
                results.append(r)
        else:
            result = train_one_horizon_pooled(panel, h, save=True, verbose=False)
            results.append({"horizon": h, "train_auc": result.metrics.get("train_auc"), "test_auc": result.metrics.get("test_auc")})

    log.info("\n=== SUMMARY ===")
    for r in results:
        log.info(f"  {r}")


if __name__ == "__main__":
    main()
