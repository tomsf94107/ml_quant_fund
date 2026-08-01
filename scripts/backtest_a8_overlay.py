"""
scripts/backtest_a8_overlay.py
─────────────────────────────
Phase 2H backtest: does A8 prob_top_decile improve BUY portfolios
when used as an overlay filter?

For each historical (date, horizon) with mature outcomes:
  1. Get all BUY signals
  2. Look up a8_prob for each
  3. Compare portfolio returns under different filter strategies:
     - production: no filter
     - filter_010: a8_prob > 0.10
     - filter_015: a8_prob > 0.15
     - filter_020: a8_prob > 0.20
     - top_half:   a8 > median of BUYs that day
     - top_decile: a8 > 0.25
  4. Aggregate: cumulative return, win rate, n_filtered

Usage:
    python scripts/backtest_a8_overlay.py [--horizon 5]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_data(horizon=5):
    """Load BUYs with outcomes + a8 panel, join."""
    # 1. Historical BUYs with mature outcomes
    conn = sqlite3.connect(ROOT / "accuracy.db", timeout=30)
    buys = pd.read_sql("""
        SELECT p.prediction_date, p.ticker, p.horizon, p.prob_up, p.prob_raw,
               o.actual_return
        FROM predictions p
        JOIN outcomes o USING (ticker, prediction_date, horizon)
        WHERE p.signal='BUY' 
          AND p.horizon=?
          AND o.actual_return IS NOT NULL
    """, conn, params=(horizon,))
    conn.close()
    buys["prediction_date"] = pd.to_datetime(buys["prediction_date"])
    
    # 2. A8 OOS panel
    panel = pd.read_parquet(ROOT / "data" / "a8_oos_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    
    # 3. Join
    merged = buys.merge(
        panel.rename(columns={"date": "prediction_date"}),
        on=["ticker", "prediction_date"], how="left"
    )
    
    print(f"BUYs: {len(buys)}, joined with a8_prob: {merged['a8_prob'].notna().sum()}")
    print(f"BUYs without a8_prob: {merged['a8_prob'].isna().sum()}")
    
    return merged


def compute_strategy(df, strategy_name, filter_fn):
    """For each date, apply filter, compute portfolio return.
    
    Returns: DataFrame with date, n_buys, n_filtered, ret_equal, cum_return
    """
    rows = []
    for date, group in df.groupby("prediction_date"):
        all_buys = group
        kept = filter_fn(group)
        n_buys = len(all_buys)
        n_kept = len(kept)
        
        if n_kept == 0:
            ret = 0.0  # no positions
        else:
            ret = float(kept["actual_return"].mean())  # equal-weight
        
        rows.append({
            "date": date,
            "n_buys": n_buys,
            "n_kept": n_kept,
            "n_filtered": n_buys - n_kept,
            "ret": ret,
        })
    
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["cum_return"] = (1 + daily["ret"]).cumprod() - 1
    
    win_days = (daily["ret"] > 0).sum()
    return {
        "strategy": strategy_name,
        "n_days": len(daily),
        "mean_ret": daily["ret"].mean(),
        "cum_return": daily["cum_return"].iloc[-1] if len(daily) else 0,
        "win_rate": win_days / len(daily) if len(daily) else 0,
        "sharpe": daily["ret"].mean() / daily["ret"].std() * np.sqrt(252) if daily["ret"].std() > 0 else 0,
        "max_dd": (daily["cum_return"] - daily["cum_return"].cummax()).min(),
        "avg_n_buys": daily["n_buys"].mean(),
        "avg_n_kept": daily["n_kept"].mean(),
        "avg_n_filtered": daily["n_filtered"].mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    
    df = load_data(horizon=args.horizon)
    
    # Strategy definitions
    strategies = [
        ("production",   lambda g: g),  # no filter
        ("filter_005",   lambda g: g[g["a8_prob"].fillna(0.10) > 0.05]),
        ("filter_010",   lambda g: g[g["a8_prob"].fillna(0.10) > 0.10]),
        ("filter_015",   lambda g: g[g["a8_prob"].fillna(0.10) > 0.15]),
        ("filter_020",   lambda g: g[g["a8_prob"].fillna(0.10) > 0.20]),
        ("filter_025",   lambda g: g[g["a8_prob"].fillna(0.10) > 0.25]),
        ("top_half",     lambda g: g[g["a8_prob"].fillna(0.10) > g["a8_prob"].fillna(0.10).median()]),
        ("top_quartile", lambda g: g[g["a8_prob"].fillna(0.10) > g["a8_prob"].fillna(0.10).quantile(0.75)]),
    ]
    
    print(f"\n{'='*90}")
    print(f"PHASE 2H BACKTEST — h={args.horizon}d, {len(df)} BUYs across "
          f"{df['prediction_date'].dt.date.min()} → {df['prediction_date'].dt.date.max()}")
    print(f"{'='*90}")
    
    results = []
    for name, fn in strategies:
        r = compute_strategy(df, name, fn)
        results.append(r)
    
    rdf = pd.DataFrame(results)
    rdf["cum_pct"] = rdf["cum_return"] * 100
    rdf["mean_pct"] = rdf["mean_ret"] * 100
    rdf["dd_pct"] = rdf["max_dd"] * 100
    
    # Display
    display = rdf[["strategy", "n_days", "avg_n_buys", "avg_n_kept", 
                   "mean_pct", "cum_pct", "win_rate", "sharpe", "dd_pct"]].copy()
    display.columns = ["strategy", "days", "avg_buys", "avg_kept", 
                       "mean_ret%", "cum_ret%", "win_rate", "sharpe", "max_dd%"]
    print()
    print(display.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 100 else f"{x:.1f}"))
    
    # Comparison to production
    prod = next(r for r in results if r["strategy"] == "production")
    print(f"\n{'='*90}")
    print(f"vs production (cum {prod['cum_return']*100:+.2f}%):")
    print(f"{'='*90}")
    for r in results:
        if r["strategy"] == "production":
            continue
        diff = (r["cum_return"] - prod["cum_return"]) * 100
        retain_pct = r["avg_n_kept"] / r["avg_n_buys"] * 100 if r["avg_n_buys"] else 0
        flag = "✓" if diff > 0 else "✗"
        print(f"  {flag} {r['strategy']:<15} diff={diff:+.2f}pp  retains={retain_pct:.0f}% of BUYs")


if __name__ == "__main__":
    main()
