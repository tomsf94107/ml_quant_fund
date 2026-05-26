"""
models/train_global_ranker.py
─────────────────────────────
Path A v2: pooled cross-sectional RANKING model using LightGBMRanker.

Why ranking, not classification (May 26 2026 finding):
  Pooled classification gives nearly identical predictions for large-caps
  (NVDA/AAPL/MSFT/TSLA all = 0.5058 with 97 features). Trees can't
  discriminate between similar stocks on absolute up/down direction.

Ranking approach (industry standard for cross-sectional quant):
  Within each trading day, rank 125 tickers by forward N-day return.
  Output: relative score that varies cross-sectionally.
  Top quintile = expected to outperform peers.

Saves as models/saved/GLOBAL_ranker_{horizon}d.joblib using a custom wrapper.

Usage:
    python -m models.train_global_ranker
    python -m models.train_global_ranker --horizons 5  # specific
    python -m models.train_global_ranker --validate    # OOS test
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

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "saved"


from models.wrappers import GlobalRankerResult


def build_pooled_panel_with_returns(tickers, start_date="2020-01-01", horizon=5, verbose=True):
    """Build features pooled across tickers + compute forward returns for ranking."""
    from features.builder import build_feature_dataframe

    all_dfs = []
    log.info(f"Building panel for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers, 1):
        if i % 25 == 0:
            log.info(f"  [{i}/{len(tickers)}] {ticker}")
        try:
            df = build_feature_dataframe(ticker, start_date=start_date)
            if df.empty or len(df) < 200:
                continue
            # Compute continuous forward returns
            df = df.copy()
            df[f'fwd_ret_{horizon}d'] = (df['close'].shift(-horizon) / df['close']) - 1.0
            df['_ticker'] = ticker
            all_dfs.append(df)
        except Exception as e:
            log.warning(f"  {ticker} failed: {type(e).__name__}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No valid tickers")

    pooled = pd.concat(all_dfs, ignore_index=True)
    pooled = pooled.dropna(subset=[f'fwd_ret_{horizon}d'])
    log.info(f"Pooled: {len(pooled)} rows across {len(all_dfs)} tickers (with fwd_ret_{horizon}d)")
    return pooled


def train_ranker(df_pooled, horizon, save=True, verbose=True):
    """Train LGBMRanker on pooled panel grouped by date."""
    import lightgbm as lgb
    from models.classifier import FEATURE_COLUMNS

    # Ensure FEATURE_COLUMNS exist
    for c in FEATURE_COLUMNS:
        if c not in df_pooled.columns:
            df_pooled[c] = 0.0

    # Fill NaN features (non-inst) with median
    non_inst = [c for c in FEATURE_COLUMNS if not c.startswith("inst_")]
    df_pooled[non_inst] = df_pooled[non_inst].fillna(df_pooled[non_inst].median())

    # Convert fwd_ret to ranking score per date
    fwd_col = f'fwd_ret_{horizon}d'
    
    # Sort by date for proper grouping
    df_pooled = df_pooled.sort_values('date').reset_index(drop=True)
    
    # Per-date ranking: 0-4 (5 quintiles, integer relevance scores)
    df_pooled['rank_within_day'] = df_pooled.groupby('date')[fwd_col].rank(method='dense', pct=True)
    df_pooled['relevance'] = pd.cut(df_pooled['rank_within_day'], 
                                     bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
                                     labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Filter dates with at least 2 tickers (need group of size >=2)
    date_counts = df_pooled.groupby('date').size()
    valid_dates = date_counts[date_counts >= 2].index
    df_pooled = df_pooled[df_pooled['date'].isin(valid_dates)].copy()
    df_pooled = df_pooled.sort_values('date').reset_index(drop=True)
    
    # Group sizes for ranker
    group_sizes = df_pooled.groupby('date').size().values.tolist()
    
    X = df_pooled[FEATURE_COLUMNS].values
    y = df_pooled['relevance'].values
    
    log.info(f"  h={horizon}d: training ranker on {len(df_pooled)} rows × {len(FEATURE_COLUMNS)} features")
    log.info(f"  h={horizon}d: {len(group_sizes)} groups (trading days), avg size {np.mean(group_sizes):.1f}")
    
    # Train LGBMRanker with lambdarank
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42,
    )
    
    t0 = time.time()
    ranker.fit(X, y, group=group_sizes)
    log.info(f"  h={horizon}d: training done in {time.time()-t0:.1f}s")

    # Quick prediction range check
    preds = ranker.predict(X[:1000])
    log.info(f"  h={horizon}d: sample predictions: min={preds.min():.4f}, max={preds.max():.4f}, std={preds.std():.4f}")
    
    metrics = {
        "n_train": len(df_pooled),
        "n_features": len(FEATURE_COLUMNS),
        "n_groups": len(group_sizes),
        "avg_group_size": float(np.mean(group_sizes)),
        "pred_std": float(preds.std()),
    }
    
    result = GlobalRankerResult(
        ranker=ranker,
        feature_cols=FEATURE_COLUMNS,
        horizon=horizon,
        ticker="GLOBAL",
        metrics=metrics,
    )
    
    if save:
        import joblib
        out_path = MODEL_DIR / f"GLOBAL_ranker_{horizon}d.joblib"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, out_path)
        log.info(f"  h={horizon}d: saved to {out_path}")
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument("--start-date", default="2020-01-01")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in (ROOT / "tickers.txt").read_text().splitlines() 
               if t.strip() and not t.startswith("#")]
    log.info(f"Tickers: {len(tickers)}")

    for h in args.horizons:
        log.info(f"\n──── Horizon {h}d ────")
        try:
            df_pooled = build_pooled_panel_with_returns(tickers, start_date=args.start_date, horizon=h)
            result = train_ranker(df_pooled, horizon=h, save=not args.no_save)
            log.info(f"  h={h}d: metrics={result.metrics}")
        except Exception as e:
            log.error(f"  h={h}d FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
