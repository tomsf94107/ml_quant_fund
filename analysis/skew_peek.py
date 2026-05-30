"""
analysis/skew_peek.py — STEP 2c (option A): THIN-DATA PEEK at the options skew axis.

NOT a validation — only ~51 dates / 10 weeks of IV-leg data exist (Mar 21-May 30).
At a weekly rebalance that's ~10 independent windows. This tells us DIRECTIONALLY
whether option skew / iv_rank carries any signal on these names — i.e. whether the
options axis is worth the wait for VRP (which needs ~Aug data). Treat every number
as a hint with huge error bars, NOT a tradeable result.

skew_25d > 0 = put IV > call IV = "BEARISH" label (downside hedging demand).
Tests: does high skew predict DOWN (label correct) or UP (label inverted / fear=bounce)?
Also tests iv_rank. Decile spread, net of 10bps/turnover, joined to h=5 outcomes.
"""
import argparse, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0


def load(db, horizon, col):
    q = f"""
      SELECT s.date, s.ticker, s.{col} as sig, o.actual_return
      FROM options_skew_history s
      JOIN outcomes o ON s.ticker=o.ticker AND s.date=o.prediction_date
      WHERE o.horizon=? AND s.{col} IS NOT NULL AND o.actual_return IS NOT NULL
    """
    con = sqlite3.connect(db); df = pd.read_sql(q, con, params=[horizon]); con.close()
    return df


def decile_spread(df, sign, winsor=0.25, min_names=8):
    """sign=+1: long HIGH signal / short low. sign=-1: reverse. Net of cost."""
    spreads = []
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["sig", "actual_return"])
        if len(g) < min_names:
            continue
        g = g.copy()
        g["ar"] = g["actual_return"].clip(-winsor, winsor)
        g = g.sort_values("sig")
        k = max(1, len(g)//5)  # quintiles (thin data -> wider buckets)
        lo = g.head(k)["ar"].mean()
        hi = g.tail(k)["ar"].mean()
        spreads.append(sign * (hi - lo))
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    cost = 1.0*(COST_BPS/1e4)*2.0
    pers = 52  # ~weekly
    gross = (sa.mean()/sd*np.sqrt(pers)) if sd>0 else float("nan")
    net   = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), len(sa), round(float(sa.mean())*100,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()
    print(f"=== SKEW PEEK (THIN DATA — ~10 weeks, hints only) h={args.horizon}d ===\n")
    print(f"  {'signal / direction':<38}{'gross':>8}{'net Sh':>9}{'n_dates':>9}{'mean%':>9}")
    print("  " + "-"*73)
    for col in ["skew_25d", "iv_rank"]:
        df = load(args.db, args.horizon, col)
        if df.empty:
            print(f"  {col}: no joinable data"); continue
        for sign, label in [(+1, f"{col} high->LONG (label as-is)"),
                            (-1, f"{col} high->SHORT (inverted)")]:
            r = decile_spread(df, sign)
            if r:
                g, nt, nd, mp = r
                mark = " <- pass(thin!)" if nt > 0.3 else ""
                print(f"  {label:<38}{g:>+8.2f}{nt:>+9.3f}{nd:>9d}{mp:>+9.3f}{mark}")
    print("\n  ⚠ ~10 effective windows — directional HINT only, not validated.")
    print("  If skew shows a strong consistent sign here -> options axis worth the")
    print("  Aug VRP wait. If noise -> deprioritize options until more data.")


if __name__ == "__main__":
    main()
