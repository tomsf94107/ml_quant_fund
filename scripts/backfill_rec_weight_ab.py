"""
scripts/backfill_rec_weight_ab.py
─────────────────────────────────
Backfill portfolio_returns_ab table with realized returns for both:
  - Equal-weight: 1/N per BUY ticker
  - Conviction-weight (REC %): max(prob_raw - 0.5, 0) / sum

For every (prediction_date, horizon) where actual_return is available.

Usage:
    python scripts/backfill_rec_weight_ab.py [--dry-run] [--horizon H]

Idempotent: uses INSERT OR REPLACE.
"""
import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "accuracy.db"


def compute_ab_row(group: pd.DataFrame) -> dict | None:
    """For one (date, horizon) group of BUYs, return A/B metrics.
    
    Returns None if no BUYs OR no actual_return data.
    """
    buys = group[group['signal'] == 'BUY'].copy()
    if buys.empty:
        return None
    
    # Use COALESCE(prob_raw, prob_up) — prob_raw missing for pre-May-25 rows
    buys['prob_signal'] = buys['prob_raw'].fillna(buys['prob_up'])
    
    # Need actual_return for all BUYs
    if buys['actual_return'].isnull().any():
        return None  # outcomes incomplete
    
    n = len(buys)
    
    # Equal weight
    w_equal = np.full(n, 1.0 / n)
    ret_equal = float((w_equal * buys['actual_return'].values).sum())
    
    # Conviction weight (long-only neutralizer math)
    sig_val = np.maximum(buys['prob_signal'].values - 0.5, 0)
    sig_sum = sig_val.sum()
    if sig_sum <= 0:
        # All BUYs have prob_signal <= 0.5 — degenerate, skip
        return None
    w_rec = sig_val / sig_sum
    ret_rec = float((w_rec * buys['actual_return'].values).sum())
    
    return {
        "prediction_date": buys['prediction_date'].iloc[0],
        "horizon":         int(buys['horizon'].iloc[0]),
        "n_buys":          n,
        "ret_equal":       round(ret_equal, 6),
        "ret_rec":         round(ret_rec, 6),
        "diff":            round(ret_rec - ret_equal, 6),
        "avg_prob_raw":    round(float(buys['prob_signal'].mean()), 4),
        "avg_signal_val":  round(float(sig_val.mean()), 4),
        "weight_max":      round(float(w_rec.max()), 4),
        "weight_min":      round(float(w_rec.min()), 4),
        "computed_at":     datetime.now().isoformat(timespec='seconds'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute but don't write")
    ap.add_argument("--horizon", type=int, choices=[1, 3, 5], help="Only this horizon")
    args = ap.parse_args()
    
    conn = sqlite3.connect(DB, timeout=30)
    q = """
        SELECT p.prediction_date, p.ticker, p.horizon, p.prob_up, p.prob_raw,
               p.signal, o.actual_return
        FROM predictions p
        JOIN outcomes o
          ON p.ticker = o.ticker
         AND p.prediction_date = o.prediction_date
         AND p.horizon = o.horizon
        WHERE p.signal = 'BUY'
          AND o.actual_return IS NOT NULL
    """
    params = []
    if args.horizon:
        q += " AND p.horizon = ?"
        params.append(args.horizon)
    
    df = pd.read_sql(q, conn, params=params)
    print(f"Loaded {len(df)} BUY rows with outcomes")
    
    rows = []
    skipped = 0
    for (date, h), group in df.groupby(['prediction_date', 'horizon']):
        result = compute_ab_row(group)
        if result is None:
            skipped += 1
            continue
        rows.append(result)
    
    print(f"Computed: {len(rows)} (date, horizon) buckets")
    print(f"Skipped:  {skipped} (no BUYs or incomplete outcomes)")
    
    if args.dry_run:
        print("\n=== DRY RUN — showing first 10 ===")
        for r in rows[:10]:
            print(f"  {r['prediction_date']} h={r['horizon']} n={r['n_buys']:2} "
                  f"equal={r['ret_equal']:+.4f} rec={r['ret_rec']:+.4f} "
                  f"diff={r['diff']:+.4f}")
        conn.close()
        return
    
    # Write to DB
    cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT OR REPLACE INTO portfolio_returns_ab
            (prediction_date, horizon, n_buys, ret_equal, ret_rec, diff,
             avg_prob_raw, avg_signal_val, weight_max, weight_min, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (r['prediction_date'], r['horizon'], r['n_buys'],
              r['ret_equal'], r['ret_rec'], r['diff'],
              r['avg_prob_raw'], r['avg_signal_val'],
              r['weight_max'], r['weight_min'], r['computed_at']))
    conn.commit()
    print(f"\nWrote {len(rows)} rows to portfolio_returns_ab")
    
    # Quick aggregate
    print("\n=== AGGREGATE SUMMARY ===")
    for h in [1, 3, 5]:
        h_rows = [r for r in rows if r['horizon'] == h]
        if not h_rows:
            continue
        mean_diff = np.mean([r['diff'] for r in h_rows])
        cum_equal = np.prod([1 + r['ret_equal'] for r in h_rows]) - 1
        cum_rec = np.prod([1 + r['ret_rec'] for r in h_rows]) - 1
        print(f"  h={h}d: {len(h_rows)} days, mean_diff={mean_diff:+.4f}, "
              f"cum_equal={cum_equal:+.2%}, cum_rec={cum_rec:+.2%}, "
              f"cum_diff={cum_rec - cum_equal:+.2%}")
    
    conn.close()


if __name__ == "__main__":
    main()
