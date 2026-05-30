"""
analysis/residual_reversal.py — STEP 2 (proper fix): mean-reversion signal.

Builds the reversion signal in its SIMPLEST CORRECT form and backtests it net of
cost. This is the proper replacement for the broken direction model: instead of a
classifier whose DOWN calls are inverted (verified: DOWN-calls-right 40% at h=5),
we rank names DIRECTLY by overextension and expect reversion.

SIGNAL (simplest correct version):
  resid = cs_demean(trailing 5d return)   # relative over/under-performance vs peers
  reversion score = -resid                # most oversold-relative -> highest score (BUY)
Then form deciles, long the most-oversold decile, short the most-overextended, and
measure forward actual_return spread NET of 10bps/turnover cost with a no-trade band.

Reuses tested machinery: cs_demean (features.alpha_transformations), load_bucket_map.
No new data — return_5d + actual_return already in accuracy.db (4600 joinable h=5 rows).

GATE: net-of-cost Sharpe > 0.3 -> proceed. Below -> try sector-residual / PCA versions.
"""
import argparse, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.alpha_transformations import cs_demean, group_neutralize

COST_BPS = 10.0
DECILE = 0.10


def load(db, horizon, days):
    q = """
      SELECT pf.ticker, pf.prediction_date, pf.return_5d, pf.return_1d, pf.return_20d,
             o.actual_return
      FROM prediction_features pf
      JOIN outcomes o ON pf.ticker=o.ticker AND pf.prediction_date=o.prediction_date
                     AND pf.horizon=o.horizon
      WHERE pf.horizon=? AND pf.prediction_date >= date('now', ?)
        AND pf.return_5d IS NOT NULL AND o.actual_return IS NOT NULL
    """
    con = sqlite3.connect(db)
    df = pd.read_sql(q, con, params=[horizon, f"-{days} day"])
    con.close()
    return df


def backtest(df, lookback_col, sector_map=None, exit_band=0.30):
    """
    Reversion signal = -cs_demean(lookback return). Optionally sector-residualize.
    Long bottom decile of the RESIDUAL (most oversold), short top, hold via no-trade band.
    Returns (gross_sharpe, net_sharpe, turnover, n_dates).
    """
    # pivot to date x ticker panel of the lookback return
    panel = df.pivot_table(index="prediction_date", columns="ticker", values=lookback_col)
    ret_panel = df.pivot_table(index="prediction_date", columns="ticker", values="actual_return")
    # residual: cross-sectional demean (relative perf). sector version if map given.
    if sector_map is not None:
        resid = group_neutralize(panel, sector_map)
    else:
        resid = cs_demean(panel)
    score = -resid  # most-oversold-relative -> highest score -> long

    long_held, short_held = set(), set()
    spreads, turns = [], []
    for d in score.index:
        row = score.loc[d].dropna()
        if len(row) < 10:
            continue
        rr = ret_panel.loc[d]
        order = row.sort_values()           # ascending: low score = overextended (short)
        ranks = {t: i/len(order) for i, t in enumerate(order.index)}
        k = max(1, len(order)//10)
        new_long  = {t for t in long_held  if t in ranks and ranks[t] >= 1-exit_band}
        new_short = {t for t in short_held if t in ranks and ranks[t] <= exit_band}
        new_long  |= set(order.tail(k).index)   # highest score = most oversold = LONG
        new_short |= set(order.head(k).index)   # lowest score = overextended = SHORT
        if long_held or short_held:
            lt = 1.0 - len(new_long & long_held)/max(1,len(new_long))
            st = 1.0 - len(new_short & short_held)/max(1,len(new_short))
            turns.append((lt+st)/2)
        lr = np.nanmean([rr.get(t, np.nan) for t in new_long])  if new_long  else 0.0
        sr = np.nanmean([rr.get(t, np.nan) for t in new_short]) if new_short else 0.0
        spreads.append((lr if not np.isnan(lr) else 0) - (sr if not np.isnan(sr) else 0))
        long_held, short_held = new_long, new_short
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    turn = float(np.mean(turns)) if turns else 1.0
    cost = turn * (COST_BPS/1e4) * 2.0
    gross = (sa.mean()/sd*np.sqrt(50)) if sd>0 else float("nan")
    net   = ((sa.mean()-cost)/sd*np.sqrt(50)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), round(turn,3), len(sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--days", type=int, default=400)
    args = ap.parse_args()

    df = load(args.db, args.horizon, args.days)
    print(f"=== RESIDUAL REVERSAL (h={args.horizon}d, {len(df)} rows) ===\n")
    from analysis.build_alpha_panel import load_bucket_map
    try:
        smap = load_bucket_map()
    except Exception:
        smap = None

    print(f"  {'signal':<34}{'turnover':>10}{'gross Sh':>10}{'net Sh':>10}{'n':>7}")
    print("  " + "-"*70)
    configs = [
        ("cs-demean 5d return (simplest)", "return_5d", None),
        ("cs-demean 1d return",            "return_1d", None),
        ("cs-demean 20d return",           "return_20d", None),
    ]
    if smap:
        configs.append(("sector-residual 5d return", "return_5d", smap))
    for label, col, sm in configs:
        if col not in df.columns or df[col].isna().all():
            print(f"  {label:<34}  (no data for {col})"); continue
        r = backtest(df, col, sector_map=sm)
        if r:
            g, nt, tu, nd = r
            print(f"  {label:<34}{tu*100:>9.1f}%{g:>+10.3f}{nt:>+10.3f}{nd:>7d}")
    print("\n  GATE: net Sh > 0.3 -> proper reversion signal works -> proceed to wiring.")
    print("  Note: -return means oversold->LONG. Compare cs-demean vs sector-residual")
    print("  (your Stage-4 finding: sector-neutralizing may HURT — verify here).")


if __name__ == "__main__":
    main()
