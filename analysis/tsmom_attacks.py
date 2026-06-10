"""
analysis/tsmom_attacks.py — pre-ship attack battery on v2 multi-asset TSMOM.
Pre-registered Jun 10 2026, thresholds locked BEFORE running:
  A1 extended history 2008->: full Sharpe > 0.4 AND each era (08-12,13-18,19-26) > -0.3
  A2 contribution: no single ETF > 50% cum P&L; SPY+QQQ combined < 50%
  A3 cost stress: Sharpe > 0.3 at 30bps
  A4 lookback basin: 3/6/9/12-month (skip 21d) ALL positive net Sharpe
Run: python -m analysis.tsmom_attacks
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FWD, STEP, VOL_LB = 20, 20, 40
ETFS = ["TLT","IEF","DBC","USO","GLD","SLV","UUP","FXE","SPY","QQQ"]
ANN = 252 / FWD


def book(panel, lb, skip=21, cost_bps=10.0, contrib=None):
    rets = panel.pct_change()
    dates = panel.index
    out, prev = [], set()
    for i in range(lb + 5, len(dates) - FWD, STEP):
        mom = panel.iloc[i - skip] / panel.iloc[i - lb] - 1.0
        hold = mom[mom > 0].dropna().index.tolist()
        d1 = dates[i + FWD]
        if not hold:
            out.append((d1, 0.0)); prev = set(); continue
        vol = rets[hold].iloc[max(0, i - VOL_LB): i].std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0)
        w = w / w.sum() if w.sum() > 0 else w
        fwd = panel[hold].iloc[i + FWD] / panel[hold].iloc[i] - 1.0
        gross = float((w * fwd).sum())
        turn = len(set(hold) ^ prev) / max(len(hold), 1)
        out.append((d1, gross - (cost_bps / 1e4) * turn))
        if contrib is not None:
            for t in hold:
                contrib[t] = contrib.get(t, 0.0) + float(w[t] * fwd[t])
        prev = set(hold)
    return pd.DataFrame(out, columns=["date", "net"]).set_index("date")["net"]


def sharpe(r):
    return float(np.sqrt(ANN) * r.mean() / r.std()) if len(r) > 5 and r.std() > 0 else np.nan


def main():
    from analysis.momentum_purged_wf import build_close_panel
    panel = build_close_panel(ETFS, "2006-01-01")
    print("history loaded per ETF:")
    for c in panel.columns:
        s = panel[c].dropna()
        print(f"  {c}: {s.index.min().date()} -> {s.index.max().date()} ({len(s)}d)")
    common_start = max(panel[c].dropna().index.min() for c in panel.columns)
    print(f"\ncommon start (all 10 live): {common_start.date()}  <- A1 window begins here")
    panel = panel[panel.index >= common_start]

    print("\n=== A1: EXTENDED HISTORY (6-1) ===")
    contrib = {}
    r = book(panel, 126, contrib=contrib)
    full = sharpe(r)
    eras = {"08-12": r[r.index < "2013"], "13-18": r[(r.index >= "2013") & (r.index < "2019")],
            "19-26": r[r.index >= "2019"]}
    era_sh = {k: sharpe(v) for k, v in eras.items()}
    print(f"  full-window net Sharpe = {full:+.2f}  (n={len(r)})")
    for k, v in era_sh.items():
        print(f"    {k}: Sharpe {v:+.2f} (n={len(eras[k])})")
    yr = r.groupby(r.index.year).apply(sharpe)
    print("  per-year:", {int(k): round(v, 2) for k, v in yr.items() if v == v})
    a1 = (full > 0.4) and all(v > -0.3 for v in era_sh.values() if v == v)
    print(f"  A1 {'PASS' if a1 else 'FAIL'} (full>0.4 + every era>-0.3)")

    print("\n=== A2: PER-ASSET P&L CONTRIBUTION (gross, full window) ===")
    tot = sum(contrib.values())
    shares = {k: v / tot for k, v in sorted(contrib.items(), key=lambda x: -x[1])} if tot > 0 else {}
    for k, v in shares.items():
        print(f"  {k}: {v:+.0%}")
    eq_share = shares.get("SPY", 0) + shares.get("QQQ", 0)
    a2 = (max(shares.values()) < 0.50 if shares else False) and eq_share < 0.50
    print(f"  max single = {max(shares.values()):.0%} | SPY+QQQ = {eq_share:.0%}")
    print(f"  A2 {'PASS' if a2 else 'FAIL'} (no asset>=50%, equity<50%)")

    print("\n=== A3: COST STRESS (6-1) ===")
    for bps in (10, 20, 30):
        sh = sharpe(book(panel, 126, cost_bps=bps))
        print(f"  {bps}bps: Sharpe {sh:+.2f}")
        if bps == 30: a3 = sh > 0.3
    print(f"  A3 {'PASS' if a3 else 'FAIL'} (>0.3 at 30bps)")

    print("\n=== A4: LOOKBACK BASIN ===")
    basin = {}
    for months, lb in [(3, 63), (6, 126), (9, 189), (12, 252)]:
        basin[months] = sharpe(book(panel, lb))
        print(f"  {months}m-1: Sharpe {basin[months]:+.2f}")
    a4 = all(v > 0 for v in basin.values() if v == v)
    print(f"  A4 {'PASS' if a4 else 'FAIL'} (all lookbacks positive = basin not spike)")

    n_pass = sum([a1, a2, a3, a4])
    print(f"\n=== ATTACK VERDICT: {n_pass}/4 passed ===")
    print("SHIP-TO-SHADOW requires 4/4. Anything less: documented kill or stated weakness.")


if __name__ == "__main__":
    main()
