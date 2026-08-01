"""
scripts/backtest_8k_inst_gating.py
─────────────────────────────────────
Path B backtest: post-prediction gating overlay using 8-K + inst flow + earnings signals.

For each closed BUY prediction in accuracy.db, evaluate whether suppression/boosting
rules improve hit rate.

Rules tested:
  R1. SUPPRESS if eightk_exec_change_30d == 1  (pooled cross-sectional edge: -5.4pp)
  R2. SUPPRESS if days_since_earnings ≤ 5 AND inst_signed_flow_5d < threshold
  R3. SUPPRESS if eightk_exec_change_30d == 1 AND inst_signed_flow_5d < threshold (combo)
  R4. BOOST score if eightk_other_events_30d == 1 (pooled edge: +6.0pp)
"""

import os
os.environ['ML_QUANT_INST_FEATURES'] = '1'

import sqlite3
import sys
from datetime import datetime
import pandas as pd
import numpy as np

DB_PATH = "accuracy.db"
EARN_DB = "earnings.db"


def load_buys() -> pd.DataFrame:
    """Load all closed BUY predictions with their inst features and outcomes."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql("""
        SELECT 
            p.ticker, p.prediction_date, p.horizon,
            p.prob_up, p.prob_raw,
            o.actual_return, o.actual_up,
            pf.inst_signed_flow_5d
        FROM predictions p
        JOIN outcomes o USING (ticker, prediction_date, horizon)
        LEFT JOIN prediction_features pf USING (ticker, prediction_date, horizon)
        WHERE p.signal = 'BUY'
        ORDER BY p.ticker, p.prediction_date, p.horizon
    """, conn)
    conn.close()
    return df


def attach_eightk(df: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, prediction_date), compute the 8-K features as of that date."""
    from data.alpha_sources import load_eightk_pit
    
    cols_to_add = ['eightk_exec_change_30d', 'eightk_material_agreement_30d',
                   'eightk_reg_fd_30d', 'eightk_other_events_30d',
                   'eightk_filings_30d', 'eightk_days_since_last']
    for c in cols_to_add:
        df[c] = np.nan
    
    # Group by ticker for efficient lookup
    unique_tickers = df['ticker'].unique()
    print(f"Attaching 8-K features for {len(unique_tickers)} tickers...")
    
    for i, ticker in enumerate(unique_tickers, 1):
        if i % 20 == 0:
            print(f"  {i}/{len(unique_tickers)}")
        sub_idx = df.index[df['ticker'] == ticker]
        if len(sub_idx) == 0:
            continue
        dates = pd.to_datetime(df.loc[sub_idx, 'prediction_date'])
        try:
            eightk_df = load_eightk_pit(ticker, dates)
            for c in cols_to_add:
                if c in eightk_df.columns:
                    # Map back by date
                    for idx, d in zip(sub_idx, dates):
                        if d in eightk_df.index:
                            df.at[idx, c] = eightk_df.loc[d, c]
        except Exception as e:
            pass
    return df


def attach_earnings(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, compute days_since_earnings."""
    conn = sqlite3.connect(EARN_DB, timeout=30)
    earn = pd.read_sql("""
        SELECT ticker, date(report_date) AS report_date
        FROM earnings_surprises
        WHERE eps_actual IS NOT NULL
        ORDER BY ticker, report_date
    """, conn)
    conn.close()
    earn['report_date'] = pd.to_datetime(earn['report_date'])
    
    df['days_since_earnings'] = np.nan
    df['_pred_dt'] = pd.to_datetime(df['prediction_date'])
    
    for ticker in df['ticker'].unique():
        e_sub = earn[earn['ticker'] == ticker].sort_values('report_date')
        if len(e_sub) == 0:
            continue
        d_sub_idx = df.index[df['ticker'] == ticker]
        for idx in d_sub_idx:
            pd_dt = df.at[idx, '_pred_dt']
            past = e_sub[e_sub['report_date'] <= pd_dt]
            if len(past) > 0:
                last_earn = past['report_date'].max()
                df.at[idx, 'days_since_earnings'] = (pd_dt - last_earn).days
    
    df = df.drop(columns='_pred_dt')
    return df


def measure(df: pd.DataFrame, mask, label: str, baseline_hit: float) -> dict:
    """Return dict of metrics for given suppression mask."""
    kept = df[~mask]
    suppressed = df[mask]
    n_supp = len(suppressed)
    n_kept = len(kept)
    
    if n_kept == 0:
        return {'rule': label, 'n_supp': n_supp, 'n_kept': 0, 'kept_hit': None}
    
    kept_hit = kept['actual_up'].mean()
    supp_hit = suppressed['actual_up'].mean() if n_supp > 0 else None
    
    return {
        'rule': label,
        'n_supp': n_supp,
        'n_kept': n_kept,
        'kept_hit_rate': kept_hit,
        'kept_return_avg': kept['actual_return'].mean(),
        'supp_hit_rate': supp_hit,
        'supp_return_avg': suppressed['actual_return'].mean() if n_supp > 0 else None,
        'lift_pp': (kept_hit - baseline_hit) * 100,
    }


def main():
    print("Loading closed BUY predictions...")
    df = load_buys()
    print(f"  Loaded {len(df)} BUYs")
    
    print("\nAttaching 8-K features...")
    df = attach_eightk(df)
    
    print("\nAttaching earnings dates...")
    df = attach_earnings(df)
    
    # Drop rows with missing critical fields
    n_before = len(df)
    df_full = df.dropna(subset=['actual_up', 'inst_signed_flow_5d'])
    print(f"\nAfter dropping NaN: {len(df_full)} / {n_before}")
    
    # ─── BASELINE ──────────────────────────────────────────────────────
    print("\n=== BASELINE ===")
    for h in sorted(df_full['horizon'].unique()):
        sub = df_full[df_full['horizon'] == h]
        if len(sub) == 0:
            continue
        baseline = sub['actual_up'].mean()
        avg_ret = sub['actual_return'].mean()
        print(f"  h={h}: n={len(sub)} baseline_hit={baseline*100:.1f}% avg_return={avg_ret*100:+.2f}%")
    
    # ─── RULES PER HORIZON ─────────────────────────────────────────────
    print("\n=== RULES PER HORIZON ===")
    for h in sorted(df_full['horizon'].unique()):
        sub = df_full[df_full['horizon'] == h].copy()
        baseline_hit = sub['actual_up'].mean()
        print(f"\n--- h={h} (baseline hit_rate={baseline_hit*100:.1f}%, n={len(sub)}) ---")
        
        results = []
        # R1: eightk_exec_change
        mask = sub['eightk_exec_change_30d'].fillna(0) > 0
        results.append(measure(sub, mask, "R1: suppress if eightk_exec_change_30d", baseline_hit))
        
        # R2: post-earnings + neg inst flow
        for inst_thr in [-0.05, -0.03, -0.01, 0.0]:
            for n_days in [3, 5, 7]:
                mask = (sub['days_since_earnings'].fillna(99) <= n_days) & (sub['inst_signed_flow_5d'].fillna(0) < inst_thr)
                if mask.sum() == 0:
                    continue
                results.append(measure(sub, mask, f"R2: days<={n_days} & inst<{inst_thr}", baseline_hit))
        
        # R3: combo exec_change + neg inst
        for inst_thr in [-0.03, 0.0]:
            mask = (sub['eightk_exec_change_30d'].fillna(0) > 0) & (sub['inst_signed_flow_5d'].fillna(0) < inst_thr)
            if mask.sum() == 0:
                continue
            results.append(measure(sub, mask, f"R3: exec_change & inst<{inst_thr}", baseline_hit))
        
        # Show all rules sorted by lift
        results = [r for r in results if r.get('kept_hit_rate') is not None]
        results.sort(key=lambda x: -x.get('lift_pp', 0))
        
        print(f"  {'Rule':<55} {'n_supp':<8} {'kept_hit':<10} {'lift_pp':<10} {'supp_hit':<10}")
        for r in results[:10]:
            supp_str = f"{r['supp_hit_rate']*100:.1f}%" if r['supp_hit_rate'] is not None else "N/A"
            print(f"  {r['rule']:<55} {r['n_supp']:<8} {r['kept_hit_rate']*100:.1f}%      {r['lift_pp']:+.2f}pp     {supp_str}")


if __name__ == "__main__":
    main()
