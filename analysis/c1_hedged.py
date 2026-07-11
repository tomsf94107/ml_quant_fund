"""
analysis/c1_hedged.py - C1 redesign (pre-registered, one shot): beta-hedged combination.
Each book minus walk-forward-beta x SPY (3bps/mo cost on hedge notional), then combine.
GATE: best hedged combo Sharpe must EXCEED unhedged mom-alone (+1.53). Fail -> C1 closed.
Run: python -m analysis.c1_hedged
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WARMUP = 36
HEDGE_COST_M = 0.0003  # 3 bps/mo per unit hedge notional
STREAMS = ["mom", "gp", "op", "ep"]


def hedged_streams(df):
    """Walk-forward beta hedge: beta from months [0,i), applied at month i."""
    out = {s: [] for s in STREAMS}
    idx = []
    for i in range(WARMUP, len(df)):
        hist = df.iloc[:i]
        idx.append(df.index[i])
        for s in STREAMS:
            b = np.polyfit(hist["spy"], hist[s], 1)[0]
            r = df[s].iloc[i] - b * df["spy"].iloc[i] - abs(b) * HEDGE_COST_M
            out[s].append(r)
    return pd.DataFrame(out, index=idx)


def hrp_weights(cov):
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
                ivp = 1.0 / np.diag(sub); ivp /= ivp.sum()
                return float(ivp @ sub @ ivp)
            v1, v2 = cvar(c1), cvar(c2)
            a = 1 - v1 / (v1 + v2)
            w[c1] *= a; w[c2] *= (1 - a)
            nxt += [c1, c2]
        clusters = nxt
    return w.sort_index().values


def run_combo(H, method):
    out = []
    for i in range(12, len(H)):  # 12mo warmup on hedged history for weights
        hist = H.iloc[:i]
        if method == "ew":
            w = np.ones(4) / 4
        elif method == "ivol":
            iv = 1.0 / hist.std().values; w = iv / iv.sum()
        elif method == "hrp":
            w = hrp_weights(hist.cov().values)
        elif method == "mom50":
            w = np.array([0.5, 1/6, 1/6, 1/6])
        elif method == "mom_only_hedged":
            w = np.array([1.0, 0, 0, 0])
        out.append((H.index[i], float(H.iloc[i].values @ w)))
    return pd.DataFrame(out, columns=["date", "ret"]).set_index("date")["ret"]


def stats(r, label):
    sh = float(np.sqrt(12) * r.mean() / r.std())
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    ann = float(eq.iloc[-1] ** (12 / len(r)) - 1)
    print(f"  {label:16s} Sharpe={sh:+.2f}  maxDD={dd:+.1%}  annRet={ann:+.1%}  n={len(r)}")
    return sh


def main():
    df = pd.read_csv(ROOT / "data/qv_books.csv", index_col=0, parse_dates=True)
    H = hedged_streams(df)
    print(f"hedged streams: {len(H)} months (post {WARMUP}mo beta warmup)")
    print("hedged stream Sharpes:",
          {s: round(float(np.sqrt(12) * H[s].mean() / H[s].std()), 2) for s in STREAMS})
    print("hedged corr:\n", H.corr().round(2), "\n")
    BASELINE = 1.53  # unhedged mom-alone, pre-registered bar
    best, best_sh = None, -9
    for m in ["mom_only_hedged", "ew", "ivol", "hrp", "mom50"]:
        sh = stats(run_combo(H, m), m)
        if m != "mom_only_hedged" and sh > best_sh:
            best, best_sh = m, sh
    print(f"\n=== GATE: best hedged combo ({best}) Sharpe {best_sh:+.2f} vs unhedged mom-alone +{BASELINE:.2f} ===")
    print(f"VERDICT: {'C1-HEDGED SHIPS (paper/shadow first; live gated on Jun 29)' if best_sh > BASELINE else 'NO-SHIP — C1 closed until shorts/universe expansion'}")


if __name__ == "__main__":
    main()
