"""
models/train_top_decile.py
──────────────────────────
A8: cross-sectional top-decile target model.

Trains a pooled cross-sectional EnsembleResult where target = 1 if a ticker's
forward h-day return is in the top 10% across the universe on that date.

Per A8 finding (May 25 2026): cross-sectional ranking removes macro confounders
by construction. If AUC > 0.50, the model found ticker-specific signal.
Original A8 v1 achieved OOS AUC 0.677 with this approach.

Reverse-engineered May 27 2026 from existing .joblib + meta.json artifacts
(training script was lost in a one-off notebook session).

Run:
    python -m models.train_top_decile --horizon 5
    python -m models.train_top_decile --horizon 5 --validate
    python -m models.train_top_decile --horizons 1 3 5  (for Phase 3F multi-horizon)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _read_ticker_file(path: Path) -> list[str]:
    return [
        t.strip().upper() for t in path.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]


def compute_top_decile_target(df: pd.DataFrame, horizon: int, threshold: float = 0.90) -> pd.Series:
    """For each date, mark tickers whose fwd_{h}d_ret is in top 10% (or threshold).
    
    Args:
        df: panel with columns ['date', '_ticker', 'close']
        horizon: forward days
        threshold: rank percentile cutoff (0.90 = top 10%)
    
    Returns:
        Series of 0/1 aligned with df.index. NaN where fwd return is NaN.
    """
    df = df.copy()
    # Compute forward return per ticker
    df_sorted = df.sort_values(['_ticker', 'date']).copy()
    df_sorted['fwd_ret'] = df_sorted.groupby('_ticker')['close'].transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    # Rank within date (cross-sectional)
    df_sorted['rank_pct'] = df_sorted.groupby('date')['fwd_ret'].rank(pct=True)
    # Top decile = rank >= 0.90
    df_sorted['target_top_decile'] = (df_sorted['rank_pct'] >= threshold).astype(float)
    # Mark NaN where forward return is unknown
    df_sorted.loc[df_sorted['fwd_ret'].isna(), 'target_top_decile'] = np.nan
    # Restore original index order
    return df_sorted.sort_index()['target_top_decile']


def build_pooled_panel(tickers, start_date="2020-01-01"):
    """Build features for all tickers and concat into single DataFrame.
    
    Mirrors models.train_cross_sectional.build_pooled_panel.
    """
    from features.builder import build_feature_dataframe

    all_dfs = []
    log.info(f"Building features for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            log.info(f"  [{i}/{len(tickers)}] {ticker}")
        try:
            df = build_feature_dataframe(ticker, start_date=start_date)
            if df.empty or len(df) < 200:
                continue
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


def train_a8(df_pooled, horizon=5, save=False, name_suffix=""):
    """Train A8 (top-decile target) on pooled panel.
    
    Args:
        df_pooled: panel with features + date + _ticker + close
        horizon: 5 for original A8
        save: if True, save to models/research/A8_top_decile_{h}d_{date}.joblib
        name_suffix: optional suffix for filename
    
    Returns:
        EnsembleResult
    """
    from models.classifier import FEATURE_COLUMNS
    from models.ensemble import train_ensemble
    from datetime import datetime

    log.info(f"Computing top-decile target for h={horizon}d...")
    df_pooled['target_top_decile'] = compute_top_decile_target(df_pooled, horizon)
    n_before = len(df_pooled)
    df_train = df_pooled.dropna(subset=['target_top_decile']).copy()
    log.info(f"  Dropped {n_before - len(df_train)} NaN-target rows → {len(df_train)} usable")

    # train_ensemble expects target_{h}d column
    # Hijack: rename top_decile target to standard target_{h}d
    df_train[f'target_{horizon}d'] = df_train['target_top_decile'].astype(int)
    
    # Diagnostic: confirm base rate ~10%
    base_rate = df_train[f'target_{horizon}d'].mean()
    log.info(f"  Top-decile base rate: {base_rate:.3f} (expect ~0.10)")
    if abs(base_rate - 0.10) > 0.03:
        log.warning(f"  Base rate {base_rate:.3f} is far from 0.10 — check target")

    # Ensure all FEATURE_COLUMNS exist
    for c in FEATURE_COLUMNS:
        if c not in df_train.columns:
            df_train[c] = 0.0
    
    # Median-fill non-inst features (same as train_cross_sectional.py)
    non_inst = [c for c in FEATURE_COLUMNS if not c.startswith("inst_")]
    df_train[non_inst] = df_train[non_inst].fillna(df_train[non_inst].median())

    log.info(f"  Training A8 on {len(df_train)} rows × {len(FEATURE_COLUMNS)} features...")
    t0 = time.time()
    ticker_name = "A8_top_decile" + (f"_{name_suffix}" if name_suffix else "")
    result = train_ensemble(
        ticker=ticker_name,
        df=df_train,
        horizon=horizon,
        verbose=True,
        save=False,  # We save manually to research/
    )
    log.info(f"  Training done in {time.time()-t0:.1f}s")
    log.info(f"  AUC: train={result.metrics.get('train_auc', 'N/A')}  test={result.metrics.get('roc_auc', 'N/A')}")

    if save:
        from scripts.save_experiment_artifact import save_artifact
        today = datetime.now().strftime("%Y%m%d")
        exp_id = f"A8_top_decile_{horizon}d{('_'+name_suffix) if name_suffix else ''}"
        
        # Get feature importances
        try:
            lgb_inner = result.lgb_model.calibrated_classifiers_[0].estimator
            importances = list(zip(result.feature_cols, lgb_inner.feature_importances_))
            top_15 = sorted(importances, key=lambda x: -x[1])[:15]
            feature_imp = {"top_15": [[n, float(v)] for n, v in top_15]}
        except Exception:
            feature_imp = {}
        
        model_path, meta_path = save_artifact(
            experiment_id=exp_id,
            result=result,
            horizon=horizon,
            target_definition=f"cross-sectional rank(fwd_{horizon}d_ret) per date >= 0.90",
            train_size=result.metrics.get('n_train', 0),
            test_size=result.metrics.get('n_test', 0),
            oos_auc=result.metrics.get('roc_auc', 0.0),
            feature_importances=feature_imp,
            notes=f"A8 retrained from train_top_decile.py on {today}",
        )
        log.info(f"  Saved: {model_path}")
        log.info(f"          {meta_path}")
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=[5])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--save", action="store_true", help="Save to models/research/")
    ap.add_argument("--limit", type=int, default=None, help="Limit tickers for debug")
    args = ap.parse_args()
    
    tickers = _read_ticker_file(Path("tickers.txt"))
    if args.limit:
        tickers = tickers[:args.limit]
        log.info(f"DEBUG MODE: limited to {len(tickers)} tickers")
    
    log.info(f"=== A8 Top-Decile Training ===")
    log.info(f"Tickers: {len(tickers)} | Horizons: {args.horizons} | Start: {args.start}")
    log.info(f"Save: {args.save}")
    
    panel = build_pooled_panel(tickers, start_date=args.start)
    
    for h in args.horizons:
        log.info(f"\n──── A8 h={h}d ────")
        result = train_a8(panel.copy(), horizon=h, save=args.save)
        log.info(f"  Final: AUC={result.metrics.get('roc_auc'):.4f}")


if __name__ == "__main__":
    main()
