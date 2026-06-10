"""
analysis/tsmom_validate.py — R3: time-series momentum (Moskowitz-Ooi-Pedersen 2012)
on the ETF set. ABSOLUTE trend (own past return sign), not cross-sectional rank.

SIGNAL: for each ETF, 12-1 momentum = sign(return over t-252..t-21). Long if positive,
flat if negative (long-only book, matching system constraints). Vol-scaled weights
(inverse trailing 40d vol, matching the shipped momentum sizing). Rebalance every 20d.
Variants tested: 12-1 (canonical) and 6-1 (faster).

GATES (pre-registered, the standard kill-bar):
  G1: net Sharpe > 0.3 over full history (10bps/turnover)
  G2: positive in MOST yearly regimes (not one-year artifact)
  G3: |corr| < 0.3 vs cross-sectional momentum book returns (else it's not a new signal)
  G4: beats buy-and-hold SPY risk-adjusted (else it's just beta)

Run: python -m analysis.tsmom_validate
"""
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0
FWD, STEP = 20, 20
LOOKBACKS = {"tsmom_12_1": (252, 21), "tsmom_6_1": (126, 21)}
ETF_SET = ["TLT","IEF","DBC","USO","GLD","SLV","UUP","FXE","SPY","QQQ"]  # MULTI-ASSET v2 (MOP-faithful): bonds/cmdty/FX/equity — pre-registered Jun 10, one shot


def book_returns(panel, lb, skip, vol_lb=40):
    """Long-only TSMOM book: hold ETFs with positive (lb-skip) own return, inv-vol weights.
    Non-overlapping FWD windows. Returns (per-window net series, turnover series)."""
    rets = panel.pct_change()
    dates = panel.index
    out, prev_hold = [], set()
    for i in range(lb + 5, len(dates) - FWD, STEP):
        mom = panel.iloc[i - skip] / panel.iloc[i - lb] - 1.0
        hold = mom[mom > 0].index.tolist()
        d1 = dates[i + FWD]
        if not hold:
            out.append((d1, 0.0, 0.0)); prev_hold = set(); continue
        vol = rets[hold].iloc[max(0, i - vol_lb): i].std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0)
        w = w / w.sum() if w.sum() > 0 else w
        fwd = panel[hold].iloc[i + FWD] / panel[hold].iloc[i] - 1.0
        gross = float((w * fwd).sum())
        turn = len(set(hold) ^ prev_hold) / max(len(hold), 1)
        net = gross - (COST_BPS / 1e4) * turn
        out.append((d1, net, turn)); prev_hold = set(hold)
    df = pd.DataFrame(out, columns=["date", "net", "turn"]).set_index("date")
    return df


def xs_momentum_book(panel_full, kind="mom_6_1"):
    """Reference: the existing cross-sectional momentum book returns (for G3 corr)."""
    from analysis.momentum_purged_wf import momentum
    rets = panel_full.pct_change()
    dates = panel_full.index
    out = []
    for i in range(260, len(dates) - FWD, STEP):
        sig = momentum(panel_full.iloc[: i + 1], kind).iloc[-1].dropna()
        if len(sig) < 30: continue
        top = sig[sig.rank(pct=True) >= 0.9].index
        vol = rets[top].iloc[max(0, i - 60): i].std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0); w = w / w.sum()
        fwd = panel_full[top].iloc[i + FWD] / panel_full[top].iloc[i] - 1.0
        out.append((dates[i + FWD], float((w * fwd).sum()) - 2 * COST_BPS / 1e4))
    return pd.DataFrame(out, columns=["date", "net"]).set_index("date")["net"]


def stats(r, label):
    ann = 252 / FWD
    sh = float(np.sqrt(ann) * r.mean() / r.std()) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    print(f"  {label:12s} n={len(r):3d} netSharpe={sh:+.2f} maxDD={dd:+.1%} tot={float(eq.iloc[-1]-1):+.1%}")
    return sh


def main():
    from analysis.momentum_purged_wf import build_close_panel
    uni = pd.read_sql("SELECT DISTINCT ticker FROM predictions",
                      sqlite3.connect(str(ROOT / "accuracy.db")))["ticker"].tolist()
    etfs = [t for t in ETF_SET if t in uni] or ETF_SET
    print(f"ETF set: {etfs}")
    panel_full = build_close_panel(sorted(set(uni) | set(etfs)), "2019-01-01")
    panel = panel_full[[e for e in etfs if e in panel_full.columns]].dropna(how="all")
    spy = panel_full["SPY"].dropna()

    xs = xs_momentum_book(panel_full)
    print("\nReference XS-momentum book loaded for G3.")

    spy_w = spy.resample("20D").last().pct_change().dropna()
    ann = 252 / FWD
    spy_sh = float(np.sqrt(ann) * spy_w.mean() / spy_w.std())
    print(f"G4 reference: SPY buy-hold ~Sharpe {spy_sh:+.2f}\n")

    for name, (lb, skip) in LOOKBACKS.items():
        print(f"=== {name} (long-only, inv-vol, {FWD}d rebal) ===")
        bk = book_returns(panel, lb, skip)
        sh = stats(bk["net"], name)
        yr = bk["net"].groupby(bk.index.year).apply(
            lambda r: float(np.sqrt(ann) * r.mean() / r.std()) if r.std() > 0 else np.nan)
        print("  per-year Sharpe:", {int(k): round(v, 2) for k, v in yr.items()})
        joined = pd.concat([bk["net"], xs], axis=1, keys=["ts", "xs"]).dropna()
        corr = float(joined["ts"].corr(joined["xs"])) if len(joined) > 10 else np.nan
        g1 = sh > 0.3
        g2 = (yr > 0).sum() >= max(2, int(0.6 * yr.notna().sum()))
        g3 = abs(corr) < 0.3 if corr == corr else False
        g4 = sh > spy_sh
        print(f"  G1 Sharpe>0.3: {g1} | G2 most-years+: {g2} ({int((yr>0).sum())}/{int(yr.notna().sum())}) "
              f"| G3 |corr|<0.3 vs XS-mom: {g3} (corr={corr:+.2f}) | G4 beats SPY: {g4}")
        print(f"  VERDICT: {'PASS — candidate 2nd signal' if (g1 and g2 and g3 and g4) else 'FAIL'}\n")

    print("HONEST NOTES: 9-ETF book = LOW breadth (TSMOM canon uses ~50 futures); long-only")
    print("halves the signal (no short leg); 2019-2026 has two stress regimes (2020, 2022).")
    print("G3 is the decisive gate — if TSMOM just tracks the XS book, it adds nothing.")


if __name__ == "__main__":
    main()
