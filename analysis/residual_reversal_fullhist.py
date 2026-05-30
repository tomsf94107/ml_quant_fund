"""
analysis/residual_reversal_fullhist.py — STEP 2 PROPER VALIDATION.

The 2-month accuracy.db backtest showed cs-demean(5d return) reversion at net
Sharpe +0.55, but the Rule #1 audit flagged 3 reasons not to trust it yet:
  (1) only ~45 dates over 2 months = ONE regime,
  (2) daily-overlapping 5d windows inflate Sharpe via autocorrelation,
  (3) raw returns had -74% / +446% outliers that can distort the spread.

This re-validates on FULL multi-year OHLCV (via features.builder._download, the
same Massive path the system uses), fixing all three:
  - full history 2020->now = many regimes
  - sample every 5th trading day = NON-overlapping 5d windows
  - winsorize forward returns at +/-25% to kill data-error outliers
Signal = -cs_demean(trailing 5d return) on the concentrated 149, no-trade band.
GATE: net-of-cost Sharpe > 0.3 across the full history -> trustworthy -> wire in.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.builder import _download
from features.alpha_transformations import cs_demean

COST_BPS = 10.0


def build_close_panel(tickers, start):
    closes = {}
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            d = _download(tk, start, None)
            d = d.set_index("date")["close"]
            closes[tk] = d
        except Exception as e:
            print(f"  skip {tk}: {type(e).__name__}", flush=True)
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def backtest(panel, step=5, winsor=0.25, exit_band=0.30, lookback=5, fwd=5):
    ret_lb  = panel.pct_change(lookback)          # trailing 5d (backward)
    ret_fwd = panel.shift(-fwd) / panel - 1.0     # forward 5d
    ret_fwd = ret_fwd.clip(-winsor, winsor)       # winsorize outliers
    score = -cs_demean(ret_lb)                    # oversold-relative -> long

    dates = panel.index[lookback : len(panel)-fwd : step]  # non-overlapping
    long_held, short_held = set(), set()
    spreads, turns = [], []
    for d in dates:
        row = score.loc[d].dropna()
        if len(row) < 10:
            continue
        fr = ret_fwd.loc[d]
        order = row.sort_values()
        ranks = {t: i/len(order) for i, t in enumerate(order.index)}
        k = max(1, len(order)//10)
        new_long  = {t for t in long_held  if t in ranks and ranks[t] >= 1-exit_band}
        new_short = {t for t in short_held if t in ranks and ranks[t] <= exit_band}
        new_long  |= set(order.tail(k).index)
        new_short |= set(order.head(k).index)
        if long_held or short_held:
            lt = 1.0 - len(new_long & long_held)/max(1,len(new_long))
            st = 1.0 - len(new_short & short_held)/max(1,len(new_short))
            turns.append((lt+st)/2)
        lr = np.nanmean([fr.get(t, np.nan) for t in new_long])  if new_long  else 0.0
        sr = np.nanmean([fr.get(t, np.nan) for t in new_short]) if new_short else 0.0
        spreads.append((lr if not np.isnan(lr) else 0) - (sr if not np.isnan(sr) else 0))
        long_held, short_held = new_long, new_short
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    turn = float(np.mean(turns)) if turns else 1.0
    cost = turn*(COST_BPS/1e4)*2.0
    # annualize: step=5 -> ~50 periods/yr
    pers = 252/step
    gross = (sa.mean()/sd*np.sqrt(pers)) if sd>0 else float("nan")
    net   = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), round(turn,3), len(sa), round(float(sa.mean())*100,4)


def backtest_clean(panel, step=5, winsor=0.25, lookback=5, fwd=5):
    """No band, no carryover — each sampled date is a fresh independent decile
    bet. The clean trust-check: do non-overlapping 5d reversion deciles pay net
    of cost across the full history? Turnover assumed ~100% per rebalance (fresh
    book each step) so cost is deducted at the per-rebalance decile turnover."""
    ret_lb  = panel.pct_change(lookback, fill_method=None)
    ret_fwd = (panel.shift(-fwd) / panel - 1.0).clip(-winsor, winsor)
    score = -cs_demean(ret_lb)
    dates = panel.index[lookback : len(panel)-fwd : step]
    spreads = []
    for d in dates:
        row = score.loc[d].dropna()
        if len(row) < 10:
            continue
        fr = ret_fwd.loc[d]
        order = row.sort_values()
        k = max(1, len(order)//10)
        longs  = order.tail(k).index   # most oversold
        shorts = order.head(k).index   # overextended
        lr = np.nanmean([fr.get(t, np.nan) for t in longs])
        sr = np.nanmean([fr.get(t, np.nan) for t in shorts])
        if not (np.isnan(lr) or np.isnan(sr)):
            spreads.append(lr - sr)
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    # fresh decile each rebalance ~ full turnover of both legs
    cost = 1.0 * (COST_BPS/1e4) * 2.0
    pers = 252/step
    gross = (sa.mean()/sd*np.sqrt(pers)) if sd>0 else float("nan")
    net   = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), len(sa), round(float(sa.mean())*100,4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== RESIDUAL REVERSAL full-hist validation: {len(tickers)} names, {args.start}->now ===")
    panel = build_close_panel(tickers, args.start)
    print(f"  close panel: {panel.shape[0]} days x {panel.shape[1]} names\n")
    print(f"  {'config':<40}{'turnover':>9}{'gross':>8}{'net Sh':>9}{'n':>6}{'mean%':>9}")
    print("  " + "-"*82)
    for label, step in [("non-overlap 5d step (primary)", 5), ("daily step (overlap, for compare)", 1)]:
        r = backtest(panel, step=step)
        if r:
            g, nt, tu, nd, mp = r
            print(f"  {label:<40}{tu*100:>8.1f}%{g:>+8.2f}{nt:>+9.3f}{nd:>6d}{mp:>+9.4f}")
    print("\n  --- CLEAN no-band non-overlap (the real trust check) ---")
    for label, step in [("clean non-overlap 5d", 5), ("clean non-overlap 10d", 10)]:
        rc = backtest_clean(panel, step=step)
        if rc:
            g, nt, nd, mp = rc
            print(f"  {label:<40}{'~100%':>9}{g:>+8.2f}{nt:>+9.3f}{nd:>6d}{mp:>+9.4f}")
    print("\n  GATE: net Sh (non-overlap, full history) > 0.3 -> trustworthy -> wire in.")
    print("  If it collapses vs the 2-month +0.55 -> that was a single-regime artifact.")


if __name__ == "__main__":
    main()
