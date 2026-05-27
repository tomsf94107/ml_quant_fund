"""
scripts/generate_a8_oos_panel.py
─────────────────────────────────
Generate walk-forward OOS predictions from A8 (top-decile cross-sectional model).

For each prediction date D, the A8 score uses ONLY data BEFORE D - PURGE_DAYS.
This prevents look-ahead leakage when A8 prob is used as a feature in the
main per-ticker model.

ARCHITECTURE:
  1. Build pooled panel ONCE for full date range
  2. Cache panel to /tmp/a8_pooled_panel.pkl
  3. For each weekly cutoff:
     a. Slice panel to (date < cutoff - PURGE_DAYS)
     b. Compute top-decile target on slice
     c. Train A8 ensemble
     d. Score on dates [cutoff, cutoff + 7) for all tickers
     e. Append to OOS panel
  4. Save OOS panel to data/a8_oos_panel.parquet

OUTPUT:
  data/a8_oos_panel.parquet with columns:
    date           — score date
    ticker         — ticker symbol
    a8_prob        — A8 prob(top-decile)
    cutoff         — training data cutoff (for audit)

USAGE:
    python scripts/generate_a8_oos_panel.py [--start 2020-07-01] [--end 2026-05-27]
    python scripts/generate_a8_oos_panel.py --resume   # continue from existing parquet
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PURGE_DAYS = 5  # h=5 forward returns settle by then
MIN_TRAIN_DAYS = 180  # 6 months minimum training window
CACHE_PATH = Path("/tmp/a8_pooled_panel.pkl")
OUTPUT_PATH = ROOT / "data" / "a8_oos_panel.parquet"


def build_or_load_pooled_panel():
    """Build pooled panel ONCE, cache to /tmp."""
    if CACHE_PATH.exists():
        log.info(f"Loading cached panel from {CACHE_PATH}")
        panel = pd.read_pickle(CACHE_PATH)
        log.info(f"  {len(panel)} rows × {len(panel.columns)} cols")
        return panel
    
    log.info("Building pooled panel from scratch (5 min)...")
    from models.train_top_decile import build_pooled_panel, _read_ticker_file
    
    tickers = _read_ticker_file(ROOT / "tickers.txt")
    panel = build_pooled_panel(tickers, start_date="2020-01-01")
    
    log.info(f"Caching panel to {CACHE_PATH}")
    panel.to_pickle(CACHE_PATH)
    return panel


def train_a8_slice(panel_slice, horizon=5):
    """Train A8 on a panel slice. Returns EnsembleResult."""
    from models.classifier import FEATURE_COLUMNS
    from models.ensemble import train_ensemble
    from models.train_top_decile import compute_top_decile_target
    
    # Compute target
    panel_slice = panel_slice.copy()
    panel_slice['target_top_decile'] = compute_top_decile_target(panel_slice, horizon)
    
    df_train = panel_slice.dropna(subset=['target_top_decile']).copy()
    df_train[f'target_{horizon}d'] = df_train['target_top_decile'].astype(int)
    
    # Ensure feature columns exist + fillna
    for c in FEATURE_COLUMNS:
        if c not in df_train.columns:
            df_train[c] = 0.0
    non_inst = [c for c in FEATURE_COLUMNS if not c.startswith("inst_")]
    df_train[non_inst] = df_train[non_inst].fillna(df_train[non_inst].median())
    
    if len(df_train) < 1000:
        return None  # Not enough data
    
    result = train_ensemble(
        ticker="A8_WF",  # walk-forward
        df=df_train,
        horizon=horizon,
        verbose=False,
        save=False,
    )
    return result


def score_on_dates(panel, result, score_dates, horizon=5):
    """Score A8 on given dates for all tickers. Returns DataFrame."""
    from models.classifier import FEATURE_COLUMNS
    
    # Filter panel to score_dates
    score_panel = panel[panel['date'].isin(score_dates)].copy()
    if score_panel.empty:
        return pd.DataFrame()
    
    # Ensure features + fillna
    for c in FEATURE_COLUMNS:
        if c not in score_panel.columns:
            score_panel[c] = 0.0
    non_inst = [c for c in FEATURE_COLUMNS if not c.startswith("inst_")]
    score_panel[non_inst] = score_panel[non_inst].fillna(score_panel[non_inst].median())
    
    # Predict
    score_panel['a8_prob'] = result.predict_proba(score_panel)
    return score_panel[['date', '_ticker', 'a8_prob']].rename(columns={'_ticker': 'ticker'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-07-01", help="First date to score (after MIN_TRAIN_DAYS)")
    ap.add_argument("--end", default=None, help="Last date to score (default: today)")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Resume from existing OOS panel")
    args = ap.parse_args()
    
    end_date = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start_date = pd.Timestamp(args.start)
    
    log.info("=== A8 Walk-Forward OOS Panel Generator ===")
    log.info(f"Score window: {start_date.date()} → {end_date.date()}")
    log.info(f"Purge buffer: {PURGE_DAYS} days")
    log.info(f"Min training: {MIN_TRAIN_DAYS} days")
    
    # Build/load panel
    panel = build_or_load_pooled_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    
    # Resume support
    existing_oos = None
    if args.resume and OUTPUT_PATH.exists():
        existing_oos = pd.read_parquet(OUTPUT_PATH)
        existing_oos['date'] = pd.to_datetime(existing_oos['date'])
        last_scored = existing_oos['date'].max()
        log.info(f"Resuming from {last_scored.date()} (existing {len(existing_oos)} rows)")
        start_date = last_scored + timedelta(days=1)
    
    # Generate weekly cutoffs (every Saturday)
    cutoffs = pd.date_range(start_date, end_date, freq='W-SAT')
    log.info(f"Weekly cutoffs: {len(cutoffs)} (every Saturday)")
    
    all_oos = []
    if existing_oos is not None:
        all_oos.append(existing_oos)
    
    t0 = time.time()
    for i, cutoff in enumerate(cutoffs, 1):
        train_cutoff = cutoff - timedelta(days=PURGE_DAYS)
        train_slice = panel[panel['date'] < train_cutoff].copy()
        
        days_in_train = (train_slice['date'].max() - train_slice['date'].min()).days if len(train_slice) else 0
        if days_in_train < MIN_TRAIN_DAYS:
            log.warning(f"  [{i}/{len(cutoffs)}] {cutoff.date()}: only {days_in_train}d train data, skip")
            continue
        
        # Train A8 on slice
        result = train_a8_slice(train_slice, horizon=args.horizon)
        if result is None:
            log.warning(f"  [{i}/{len(cutoffs)}] {cutoff.date()}: train_a8_slice returned None, skip")
            continue
        
        # Score on next week's dates [cutoff, cutoff+7)
        next_week_end = cutoff + timedelta(days=7)
        score_dates_range = pd.date_range(cutoff, next_week_end - timedelta(days=1))
        score_dates = [d for d in score_dates_range if d.weekday() < 5]  # business days only
        
        oos_chunk = score_on_dates(panel, result, score_dates, horizon=args.horizon)
        oos_chunk['cutoff'] = cutoff
        all_oos.append(oos_chunk)
        
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(cutoffs) - i) / rate
            log.info(f"  [{i}/{len(cutoffs)}] {cutoff.date()}  train_rows={len(train_slice)}  "
                     f"scored={len(oos_chunk)}  rate={rate:.1f}/s  eta={remaining/60:.1f}min")
    
    if not all_oos:
        log.error("No OOS data generated")
        return
    
    final = pd.concat(all_oos, ignore_index=True)
    final = final.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"\nSaved {len(final)} rows to {OUTPUT_PATH}")
    log.info(f"Date range: {final['date'].min().date()} → {final['date'].max().date()}")
    log.info(f"Unique tickers: {final['ticker'].nunique()}")
    log.info(f"a8_prob stats: mean={final['a8_prob'].mean():.4f}  std={final['a8_prob'].std():.4f}  "
             f"min={final['a8_prob'].min():.4f}  max={final['a8_prob'].max():.4f}")


if __name__ == "__main__":
    main()
