"""
analysis/momentum_invvol_wf.py — confirm per-name INVERSE-VOL weighting on the
long-only momentum BOOK at the SAME purged-WF standard that validated momentum.
C2 sizing decision (Jun 1 2026).

Reuses the validated harness EXACTLY (momentum() score, purged_kfold_indices 5 folds
5d embargo, 20d rebalance, 10bps cost). Only the per-fold metric changes: long-only
top-decile BOOK return, equal vs inverse-vol, per OOS fold.

GATE: inv-vol beats equal in >=4/5 OOS folds AND higher pooled fold Sharpe -> ship.
2-3/5 -> inconclusive, keep equal-weight, revisit with more data.

    python -m analysis.momentum_invvol_wf
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.momentum_purged_wf import build_close_panel, momentum
from analysis.walk_forward import purged_kfold_indices, EMBARGO_DAYS, N_FOLDS

COST_BPS = 10.0
DECILE = 0.10


def fold_book(sig, ret_fwd, daily_ret, dates_in_fold, step_dates, vol_lb=40, winsor=0.40):
    use = [d for d in step_dates if d in dates_in_fold]
    eq_rets, iv_rets, eq_turn, iv_turn = [], [], [], []
    prev_eq = prev_iv = None
    for d in use:
        row = sig.loc[d].dropna()
        if len(row) < 10:
            continue
        fr = ret_fwd.loc[d]
        order = row.sort_values()
        k = max(1, int(len(order) * DECILE))
        picks = list(order.tail(k).index)
        rets = np.array([fr.get(t, np.nan) for t in picks])
        ok = ~np.isnan(rets)
        if ok.sum() == 0:
            continue
        picks_ok = [p for p, o in zip(picks, ok) if o]
        rets_ok = np.clip(rets[ok], -winsor, winsor)
        vol = np.array([daily_ret[t].loc[:d].tail(vol_lb).std() * np.sqrt(252) for t in picks_ok])
        med = np.nanmedian(vol)
        vol = np.where(np.isnan(vol) | (vol <= 0), med, vol)
        n = len(picks_ok)
        wE = np.ones(n) / n
        wI = (1.0 / vol); wI = wI / wI.sum()
        eq_rets.append((wE * rets_ok).sum())
        iv_rets.append((wI * rets_ok).sum())
        for w, prev, turn in ((wE, prev_eq, eq_turn), (wI, prev_iv, iv_turn)):
            ws = pd.Series(w, index=picks_ok)
            if prev is not None:
                idx = ws.index.union(prev.index)
                turn.append(float((ws.reindex(idx).fillna(0) - prev.reindex(idx).fillna(0)).abs().sum()))
        prev_eq = pd.Series(wE, index=picks_ok)
        prev_iv = pd.Series(wI, index=picks_ok)
    if len(eq_rets) < 2:
        return None
    pers = 252 / 20
    def _sharpe(rets, turn):
        a = np.array(rets); sd = a.std()
        cost = (np.mean(turn) if turn else 0.0) * (COST_BPS / 1e4)
        return ((a.mean() - cost) / sd * np.sqrt(pers)) if sd > 0 else float("nan")
    return round(_sharpe(eq_rets, eq_turn), 3), round(_sharpe(iv_rets, iv_turn), 3), len(eq_rets)


def run(panel, kind="mom_6_1", fwd=20, step=20, vol_lb=40):
    sig = momentum(panel, kind)
    ret_fwd = (panel.shift(-fwd) / panel - 1.0)
    daily_ret = panel.pct_change()
    daily = pd.Series(list(panel.index[252:len(panel) - fwd]))
    step_dates = set(daily[::step].tolist())
    print(f"\n  {kind}: long-only BOOK purged-WF ({N_FOLDS} folds, {EMBARGO_DAYS}d embargo, vol_lb={vol_lb})")
    print(f"  {'fold':<6}{'test window':<26}{'EQ Sh':>8}{'IV Sh':>8}{'IV-EQ':>8}{'n':>5}")
    eq_all, iv_all = [], []
    for fi, (tr, te) in enumerate(purged_kfold_indices(daily, n_folds=N_FOLDS, embargo=EMBARGO_DAYS)):
        fold_dates = set(daily.iloc[te].tolist())
        r = fold_book(sig, ret_fwd, daily_ret, fold_dates, step_dates, vol_lb=vol_lb)
        if r is None:
            continue
        eq, iv, n = r
        fd = [d for d in step_dates if d in fold_dates]
        win = f"{min(fd).date()}->{max(fd).date()}" if fd else "?"
        print(f"  {fi:<6}{win:<26}{eq:>8.2f}{iv:>8.2f}{iv-eq:>+8.2f}{n:>5}")
        eq_all.append(eq); iv_all.append(iv)
    if eq_all:
        iv_wins = sum(1 for e, i in zip(eq_all, iv_all) if i > e)
        print(f"\n  POOLED mean fold Sharpe:  EQUAL {np.mean(eq_all):+.3f}   INV-VOL {np.mean(iv_all):+.3f}")
        print(f"  INV-VOL beats EQUAL in {iv_wins}/{len(eq_all)} OOS folds")
        gate = iv_wins >= max(1, int(0.8 * len(eq_all))) and np.mean(iv_all) > np.mean(eq_all)
        print(f"  GATE (IV>EQ >=80% folds AND higher pooled): {'PASS -> ship inv-vol' if gate else 'NOT MET -> keep equal-weight'}")
    return eq_all, iv_all


def main():
    tickers = [t.strip().upper() for t in open(ROOT / "tickers.txt")
               if t.strip() and not t.startswith("#")]
    print(f"Loading {len(tickers)} tickers...")
    panel = build_close_panel(tickers, "2018-01-01")
    print(f"Panel: {panel.shape[1]} tickers, {panel.shape[0]} days")
    for kind in ("mom_6_1", "mom_12_1"):
        run(panel, kind=kind)
    print("\n  Per-fold consistency is the test: inv-vol must beat equal-weight in")
    print("  most embargoed OOS folds, not just full-sample.")


if __name__ == "__main__":
    main()
