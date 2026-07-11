"""
analysis/c1_combiner.py - C1: combine validated decorrelated books (mom, gp, op, ep).
Input: data/qv_books.csv (monthly net book returns, Jun 10 validation).
Walk-forward weights only (expanding window, 36mo warmup). Pre-registered ship gate:
combined Sharpe beats mom-alone outright, OR within 0.10 with maxDD improvement >= 20pct.
Run: python -m analysis.c1_combiner
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WARMUP = 36
STREAMS = ["mom", "gp", "op", "ep"]


def hrp_weights(cov):
    """Hierarchical risk parity: corr-distance linkage, quasi-diag, recursive bisection."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            c1, c2 = cl[:half], cl[half:]
            def cvar(c):
                sub = cov[np.ix_(c, c)]
                ivp = 1.0 / np.diag(sub)
                ivp /= ivp.sum()
                return float(ivp @ sub @ ivp)
            v1, v2 = cvar(c1), cvar(c2)
            a = 1 - v1 / (v1 + v2)
            w[c1] *= a
            w[c2] *= (1 - a)
            nxt += [c1, c2]
        clusters = nxt
    return w.sort_index().values


def run_combo(df, method):
    """Walk-forward combined return series for a weighting method."""
    R = df[STREAMS]
    out = []
    for i in range(WARMUP, len(R)):
        hist = R.iloc[:i]
        if method == "ew":
            w = np.ones(4) / 4
        elif method == "ivol":
            iv = 1.0 / hist.std().values
            w = iv / iv.sum()
        elif method == "hrp":
            w = hrp_weights(hist.cov().values)
        elif method == "mom50":
            w = np.array([0.5, 1/6, 1/6, 1/6])
        elif method == "mom_only":
            w = np.array([1.0, 0, 0, 0])
        out.append((R.index[i], float(R.iloc[i].values @ w)))
    return pd.DataFrame(out, columns=["date", "ret"]).set_index("date")["ret"]


def stats(r, label):
    sh = float(np.sqrt(12) * r.mean() / r.std())
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    ann = float(eq.iloc[-1] ** (12 / len(r)) - 1)
    print(f"  {label:9s} Sharpe={sh:+.2f}  maxDD={dd:+.1%}  annRet={ann:+.1%}  n={len(r)}")
    return sh, dd


def main():
    df = pd.read_csv(ROOT / "data/qv_books.csv", index_col=0, parse_dates=True)
    print(f"books loaded: {df.shape[0]} months, streams {STREAMS}")
    print("full-period corr:\n", df[STREAMS].corr().round(2), "\n")
    base_sh, base_dd = None, None
    results = {}
    for m in ["mom_only", "ew", "ivol", "hrp", "mom50"]:
        r = run_combo(df, m)
        sh, dd = stats(r, m)
        results[m] = (sh, dd)
        if m == "mom_only":
            base_sh, base_dd = sh, dd
    print("\n=== PRE-REGISTERED SHIP GATE vs mom_only ===")
    best = None
    for m, (sh, dd) in results.items():
        if m == "mom_only":
            continue
        dd_impr = (abs(base_dd) - abs(dd)) / abs(base_dd)  # positive = drawdown got smaller
        ship = (sh > base_sh) or (sh > base_sh - 0.10 and dd_impr >= 0.20)
        print(f"  {m:9s} dSharpe={sh-base_sh:+.2f}  DDimpr={dd_impr:+.0%}  -> {'SHIP-CANDIDATE' if ship else 'no'}")
        if ship and (best is None or sh > results[best][0]):
            best = m
    print(f"\nVERDICT: {('C1 SHIPS via ' + best) if best else 'NO-SHIP — momentum-alone stays the system'}")
    print("(live deployment still gated on Jun 29 momentum verdict; this is the backtest leg)")


if __name__ == "__main__":
    main()
