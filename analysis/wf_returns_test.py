#!/usr/bin/env python3
"""
analysis/wf_returns_test.py -- does the BUY filter make MONEY?

Reads reports/wf_rows_h{H}.csv (from wf_returns_dump.py) and answers the three
questions a hit rate cannot:

  1. RETURN SPREAD   mean fwd_ret of BUY vs the benchmark. Magnitude, not direction.
  2. BETA-STRIP      walk-forward beta per ticker, estimated on PRIOR folds only,
                     applied to the current fold. resid = fwd_ret - beta*spy_fwd_ret.
                     The SI brick lost 74% of raw return to beta. This is where
                     things die.
  3. COST            10 bps round-trip per position (ML_QUANT_COST_BPS, matching
                     fitness_scorer). Also reported at 20 bps as a stress.

Two books, side by side:
  LONG-ONLY   equal-weight BUY names,  benchmark = the fold's universe mean
  LONG/SHORT  long BUY, short non-BUY, dollar-neutral (benchmark = the short leg)

All t-stats are clustered on FOLD (9 folds -> k=9, or k=8 after beta-strip drops
fold 0, which has no prior history to estimate beta from). Pooling ticker-days as
if independent is what inflated the short-interest t-stat to -20.

USAGE
    python -m analysis.wf_returns_test --horizon 5
    python -m analysis.wf_returns_test --horizon 5 --thr 0.60
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COST_BPS = float(os.environ.get("ML_QUANT_COST_BPS", "10.0"))
MIN_BETA_OBS = 40      # prior-fold rows needed to trust a per-ticker beta


def tstat(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    k = len(x)
    if k < 2:
        return np.nan, np.nan, 0
    m, s = x.mean(), x.std(ddof=1)
    t = m / (s / np.sqrt(k)) if s > 0 else np.nan
    return m, t, k


def beta_strip(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward beta per ticker: fit on folds < i, apply at fold i.
    Fold 0 has no history and is dropped. No look-ahead."""
    out = []
    folds = sorted(df.fold.unique())
    for f in folds[1:]:
        hist = df[df.fold < f]
        cur  = df[df.fold == f].copy()
        betas = {}
        for t, g in hist.groupby("ticker"):
            g = g.dropna(subset=["fwd_ret", "spy_fwd_ret"])
            if len(g) >= MIN_BETA_OBS and g.spy_fwd_ret.std() > 0:
                betas[t] = float(np.polyfit(g.spy_fwd_ret, g.fwd_ret, 1)[0])
        cur["beta"] = cur.ticker.map(betas).fillna(1.0)
        cur["resid_ret"] = cur.fwd_ret - cur.beta * cur.spy_fwd_ret
        out.append(cur)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def book(df: pd.DataFrame, ret_col: str, thr: float, cost: float):
    """Per-fold long-only and long/short spreads, net of `cost` per leg."""
    lo, ls = [], []
    for f, g in df.groupby("fold"):
        g = g.dropna(subset=[ret_col])
        buy  = g[g.prob_up >= thr]
        rest = g[g.prob_up <  thr]
        if len(buy) == 0 or len(rest) == 0:
            continue
        mb, mu, mr = buy[ret_col].mean(), g[ret_col].mean(), rest[ret_col].mean()
        lo.append((mb - cost) - mu)                  # long BUY vs universe
        ls.append((mb - cost) - (mr + cost))         # long BUY, short rest
    return np.array(lo), np.array(ls)


def line(label, arr):
    m, t, k = tstat(arr)
    flag = "PASS" if (t == t and t >= 3.0 and m > 0) else ("fail" if t == t else "  --")
    print(f"  {label:<34} {m*100:+7.3f} %   t={t:+6.2f}   "
          f"{int((arr>0).sum())}/{k} folds+   [{flag}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--thr", type=float, default=0.55)
    a = ap.parse_args()

    path = ROOT / "reports" / f"wf_rows_h{a.horizon}.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["prob_up", "fwd_ret", "spy_fwd_ret"])

    print("=" * 78)
    print(f"  BUY FILTER -- DOES IT MAKE MONEY?   h={a.horizon}  thr={a.thr}")
    print("=" * 78)
    print(f"  rows {len(df):,} | tickers {df.ticker.nunique()} | "
          f"folds {sorted(df.fold.unique())}")

    # --- PARITY CHECK: must reproduce walk_forward_history's buy_hit ----------
    hits, ns, bases = [], [], []
    for f, g in df.groupby("fold"):
        buy = g[g.prob_up >= a.thr]
        if len(buy):
            hits.append(buy.y_true.mean()); ns.append(len(buy))
            bases.append(g.y_true.mean())
    hits, ns, bases = np.array(hits), np.array(ns), np.array(bases)
    edge = hits - bases
    m, t, k = tstat(edge)
    print(f"\n  PARITY (must match tonight's fold CSV):")
    print(f"    buy_hit {np.average(hits, weights=ns):.4f}   "
          f"base {bases.mean():.4f}   "
          f"edge {m*100:+.3f} pp   t={t:+.2f}   n_buy={int(ns.sum()):,}")
    print("    ^ if this does not match the fold-CSV number, the dump has "
          "diverged. STOP.")

    cost = COST_BPS / 10_000.0
    print(f"\n  cost = {COST_BPS:.0f} bps round-trip per position (per leg)")

    print("\n  --- 1. RAW RETURN (gross, includes market beta) ---")
    lo, ls = book(df, "fwd_ret", a.thr, 0.0)
    line("long-only  vs universe", lo)
    line("long/short vs non-BUY", ls)

    print("\n  --- 2. BETA-STRIPPED (walk-forward beta, prior folds only) ---")
    bs = beta_strip(df)
    if bs.empty:
        print("    not enough folds to estimate walk-forward beta")
    else:
        lo, ls = book(bs, "resid_ret", a.thr, 0.0)
        line("long-only  vs universe", lo)
        line("long/short vs non-BUY", ls)
        print(f"    median per-ticker beta: {bs.beta.median():.3f}")

    print(f"\n  --- 3. BETA-STRIPPED + NET OF {COST_BPS:.0f} BPS  <<< THE DECISION ---")
    if not bs.empty:
        lo, ls = book(bs, "resid_ret", a.thr, cost)
        line("long-only  vs universe", lo)
        line("long/short vs non-BUY", ls)

        print(f"\n  --- 3b. STRESS: 20 bps ---")
        lo, ls = book(bs, "resid_ret", a.thr, 0.0020)
        line("long-only  vs universe", lo)
        line("long/short vs non-BUY", ls)

    print("\n" + "=" * 78)
    print("  PASS = t >= 3.0 AND mean > 0. Anything else is not a position.")
    print("  Section 3 long-only is the number. Everything above it is diagnostic.")
    print("=" * 78)


if __name__ == "__main__":
    main()
