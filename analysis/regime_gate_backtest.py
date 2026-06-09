"""
analysis/regime_gate_backtest.py — M16: does a macro/regime exposure gate improve the
momentum book? Gated vs ungated, same purged-WF discipline.

GATE (risk overlay only — picks unchanged, weights scaled):
  exposure 1.0 normal | 0.5 stressed | 0.25 crisis
TRIGGERS (computed from close panel only — no new data):
  - SPY 20d realized vol (annualized): > 25% -> stressed, > 35% -> crisis
  - credit stress: HYG/LQD 30d relative return < -2% -> stressed
  - both stressed simultaneously -> crisis
  (Recession-model trigger EXCLUDED from backtest: probs flat ~0 over 2022-2026,
   nothing to validate. If gate ships, it joins live as cheap untested insurance.)

PRE-REGISTERED SHIP RULE (decided before seeing results):
  SHIP only if max-drawdown improves >= 20% AND net Sharpe loss <= 0.2.
  Honest expectation: gates trade a little Sharpe for drawdown. One stress episode
  (2022) in window -> thin evidence either way; verdict will say so.

Run:  python -m analysis.regime_gate_backtest
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0
FWD = 20          # momentum hold, matches momentum_purged_wf
STEP = 20
VOL_STRESS, VOL_CRISIS = 0.25, 0.35      # annualized SPY 20d realized vol
CREDIT_STRESS = -0.02                     # HYG/LQD 30d relative return
EXPO = {0: 1.0, 1: 0.5, 2: 2.5e-1}       # level -> exposure


def regime_series(spy: pd.Series, hyg: pd.Series, lqd: pd.Series) -> pd.Series:
    rv = spy.pct_change().rolling(20).std() * np.sqrt(252)
    credit = (hyg.pct_change(30) - lqd.pct_change(30))
    vol_level = pd.Series(0, index=rv.index)
    vol_level[rv > VOL_STRESS] = 1
    vol_level[rv > VOL_CRISIS] = 2
    cred_level = (credit < CREDIT_STRESS).astype(int)
    level = pd.concat([vol_level, cred_level], axis=1).max(axis=1)
    level[(vol_level >= 1) & (cred_level >= 1)] = 2
    return level.reindex(spy.index).fillna(0).astype(int)


def run_gated(panel: pd.DataFrame, kind: str, level: pd.Series, gated: bool):
    """Momentum top-decile book, inverse-vol weights, exposure-scaled if gated.
    Non-overlapping FWD-day windows, net of cost. Returns per-window net returns."""
    from signals.momentum_signal import compute_momentum  # noqa — only for parity check
    from analysis.momentum_purged_wf import momentum
    rets = panel.pct_change()
    dates = panel.index
    out = []
    for i in range(260, len(dates) - FWD, STEP):
        d0, d1 = dates[i], dates[i + FWD]
        sig = momentum(panel.iloc[: i + 1], kind).iloc[-1].dropna()
        if len(sig) < 30:
            continue
        top = sig[sig.rank(pct=True) >= 0.9].index
        if len(top) == 0:
            continue
        vol = rets[top].iloc[max(0, i - 60): i].std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0)
        w = w / w.sum() if w.sum() > 0 else w
        fwd_ret = (panel[top].iloc[i + FWD] / panel[top].iloc[i] - 1.0)
        gross = float((w * fwd_ret).sum())
        expo = EXPO[int(level.loc[:d0].iloc[-1])] if gated else 1.0
        net = expo * gross - (COST_BPS / 1e4) * 2 * expo
        out.append((d1, net, expo))
    return pd.DataFrame(out, columns=["date", "net", "expo"]).set_index("date")


def stats(r: pd.Series, label: str):
    ann = 252 / FWD
    sh = float(np.sqrt(ann) * r.mean() / r.std()) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    tot = float(eq.iloc[-1] - 1)
    print(f"  {label:10s} n={len(r):3d}  netSharpe={sh:+.2f}  maxDD={dd:+.1%}  totRet={tot:+.1%}")
    return sh, dd


def main():
    from analysis.momentum_purged_wf import build_close_panel
    uni = pd.read_sql("SELECT DISTINCT ticker FROM predictions",
                      __import__("sqlite3").connect(str(ROOT / "accuracy.db")))["ticker"].tolist()
    need = sorted(set(uni) | {"SPY", "HYG", "LQD"})
    print(f"panel: {len(need)} tickers (universe + SPY/HYG/LQD)")
    panel = build_close_panel(need, "2020-01-01")
    spy, hyg, lqd = panel["SPY"].dropna(), panel["HYG"].dropna(), panel["LQD"].dropna()
    level = regime_series(spy, hyg, lqd)
    pct = level.value_counts(normalize=True).sort_index()
    print("regime distribution:", {int(k): f"{v:.0%}" for k, v in pct.items()})
    uni_panel = panel[[c for c in panel.columns if c not in ("HYG", "LQD")]]

    for kind in ["mom_6_1", "mom_12_1"]:
        print(f"\n=== {kind} ===")
        ung = run_gated(uni_panel, kind, level, gated=False)
        gat = run_gated(uni_panel, kind, level, gated=True)
        s0, d0 = stats(ung["net"], "UNGATED")
        s1, d1 = stats(gat["net"], "GATED")
        dd_impr = (d0 - d1) / abs(d0) if d0 < 0 else np.nan
        print(f"  -> DD improvement: {dd_impr:+.0%}  Sharpe delta: {s1 - s0:+.2f}")
        ship = (dd_impr >= 0.20) and ((s0 - s1) <= 0.20)
        print(f"  PRE-REGISTERED VERDICT: {'SHIP' if ship else 'NO-SHIP'}")
    print("\nHONEST CAVEAT: one stress episode (2022) in window; thin evidence either way.")


if __name__ == "__main__":
    main()
