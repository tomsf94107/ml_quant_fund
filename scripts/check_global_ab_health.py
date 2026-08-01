"""
scripts/check_global_ab_health.py
──────────────────────────────────
Layer 3 of Path A A/B observability: standalone weekly health check.

Run anytime to see comparison metrics between per-ticker and GLOBAL
cross-sectional model. After ~4 weeks of data, use this to decide
whether to promote GLOBAL.

Usage:
    python scripts/check_global_ab_health.py [--since YYYY-MM-DD]

Reports:
  - Coverage (% of predictions with both probs)
  - Correlation per-ticker vs GLOBAL
  - BUY agreement matrix
  - Hit rate by source (where outcomes are available)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="Only consider predictions on/after this date (YYYY-MM-DD)")
    ap.add_argument("--db", default="accuracy.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    
    # Base query
    where_clauses = ["prob_up IS NOT NULL"]
    params = []
    if args.since:
        where_clauses.append("prediction_date >= ?")
        params.append(args.since)
    where = " AND ".join(where_clauses)
    
    df = pd.read_sql(
        f"""
        SELECT 
            p.ticker, p.prediction_date, p.horizon, p.signal,
            p.prob_up, p.prob_up_global,
            o.actual_up, o.actual_return
        FROM predictions p
        LEFT JOIN outcomes o USING (ticker, prediction_date, horizon)
        WHERE {where}
        ORDER BY p.prediction_date DESC, p.ticker, p.horizon
        """,
        conn, params=params
    )
    conn.close()
    
    if len(df) == 0:
        print("No predictions found.")
        sys.exit(0)
    
    print("=" * 70)
    print("Path A A/B Health Check")
    print("=" * 70)
    print(f"Date range:  {df['prediction_date'].min()} → {df['prediction_date'].max()}")
    print(f"Total predictions:           {len(df)}")
    print(f"With GLOBAL prediction:      {df['prob_up_global'].notna().sum()}  ({df['prob_up_global'].notna().mean()*100:.1f}%)")
    print(f"With actual outcome:         {df['actual_up'].notna().sum()}  ({df['actual_up'].notna().mean()*100:.1f}%)")
    print(f"With BOTH GLOBAL + outcome:  {((df['prob_up_global'].notna()) & (df['actual_up'].notna())).sum()}")
    print()
    
    # Per-horizon analysis
    for h in sorted(df['horizon'].unique()):
        sub = df[df['horizon'] == h]
        sub_both = sub.dropna(subset=['prob_up_global'])
        
        print(f"─── Horizon {h}d ──────────────────────────────")
        print(f"  Total: {len(sub)}, with GLOBAL: {len(sub_both)}")
        if len(sub_both) < 5:
            print(f"  Skipping — too few rows with GLOBAL")
            print()
            continue
        
        # Distribution comparison
        print(f"  per-ticker:  mean={sub_both['prob_up'].mean():.3f}  std={sub_both['prob_up'].std():.3f}")
        print(f"  GLOBAL:      mean={sub_both['prob_up_global'].mean():.3f}  std={sub_both['prob_up_global'].std():.3f}")
        
        # Correlation
        corr = sub_both[['prob_up', 'prob_up_global']].corr().iloc[0,1]
        print(f"  Correlation: {corr:.3f}")
        
        # BUY agreement
        sub_both = sub_both.copy()
        sub_both['pt_buy'] = sub_both['prob_up'] > 0.55
        sub_both['gl_buy'] = sub_both['prob_up_global'] > 0.55
        n_both = ((sub_both['pt_buy']) & (sub_both['gl_buy'])).sum()
        n_only_pt = ((sub_both['pt_buy']) & (~sub_both['gl_buy'])).sum()
        n_only_gl = ((~sub_both['pt_buy']) & (sub_both['gl_buy'])).sum()
        n_neither = ((~sub_both['pt_buy']) & (~sub_both['gl_buy'])).sum()
        total_pt_buys = sub_both['pt_buy'].sum()
        agree_pct = (n_both / total_pt_buys * 100) if total_pt_buys > 0 else 0
        print(f"  BUY agreement (>0.55):")
        print(f"    Both BUY: {n_both}   Only per-ticker: {n_only_pt}   Only GLOBAL: {n_only_gl}   Both HOLD: {n_neither}")
        print(f"    Per-ticker BUYs that GLOBAL agrees with: {agree_pct:.1f}%")
        
        # Hit rate where outcomes available
        sub_outcome = sub_both.dropna(subset=['actual_up'])
        if len(sub_outcome) >= 10:
            print(f"  Hit rate (n={len(sub_outcome)} with outcomes):")
            for src_label, mask in [
                ("per-ticker BUYs", sub_outcome['pt_buy']),
                ("GLOBAL BUYs",     sub_outcome['gl_buy']),
                ("BOTH BUY",        sub_outcome['pt_buy'] & sub_outcome['gl_buy']),
                ("only per-ticker", sub_outcome['pt_buy'] & ~sub_outcome['gl_buy']),
                ("only GLOBAL",     ~sub_outcome['pt_buy'] & sub_outcome['gl_buy']),
            ]:
                msub = sub_outcome[mask]
                if len(msub) >= 3:
                    hit = msub['actual_up'].mean()
                    ret = msub['actual_return'].mean()
                    print(f"    {src_label:20s} n={len(msub):4d} hit={hit*100:.1f}% avg_ret={ret*100:+.2f}%")
        else:
            print(f"  Outcomes pending: n={len(sub_outcome)} (need {5+h} business days after prediction)")
        print()
    
    print("Decision criteria (after ~4 weeks / 1000+ predictions with outcomes):")
    print("  - If GLOBAL hit rate consistently >= per-ticker + 2pp: promote GLOBAL")
    print("  - If GLOBAL hit rate consistently below per-ticker:    keep per-ticker")
    print("  - If marginal/inconclusive: extend A/B period 4 more weeks")


if __name__ == "__main__":
    main()
