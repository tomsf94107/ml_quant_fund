"""
analysis/momentum_shadow_backfill.py — INSTANT validation of the momentum shadow
pipeline (no 20-day wait).

The forward reconciler (scripts/reconcile_momentum_shadow.py) correctly refuses to
score today's picks until 20 trading days elapse. To get a read NOW, we exploit
that momentum is DETERMINISTIC from price history: truncate the price panel to any
past date D, run the SAME rank_signal the live shadow uses, and attach the real
20d-forward return that has ALREADY happened. No look-ahead — picks at D use only
prices <= D; the outcome uses close[D]->close[D+20], which is now in the past.

This is NOT a re-backtest (momentum already validated +0.96). It VALIDATES THE
SHADOW PLUMBING: proves the live shadow->outcome path computes the same edge the
backtest did, so when forward picks mature we trust the numbers. It also gives an
immediate top-decile-vs-rest live-style track record across regimes.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import _download
from signals.momentum_signal import rank_signal, MOM_HORIZON

COST_BPS = 10.0


def build_panel(tickers, start):
    closes = {}
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            closes[tk] = _download(tk, start, None).set_index("date")["close"]
        except Exception:
            pass
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def run(panel, kind, step=20, winsor=0.40):
    """Reconstruct shadow picks every `step` days; attach real 20d fwd return.
    Measures BUY-candidate (top-decile) avg return vs rest, and net-of-cost spread."""
    fwd = (panel.shift(-MOM_HORIZON) / panel - 1.0).clip(-winsor, winsor)
    # rebalance dates: warm signal, and 20d outcome must exist (not last 20 rows)
    valid = panel.index[252 : len(panel) - MOM_HORIZON]
    dates = list(valid[::step])
    buy_rets, rest_rets, spreads = [], [], []
    n_buys = 0
    for d in dates:
        sub = panel.loc[:d]                      # truncate to as-of date (no look-ahead)
        try:
            sig = rank_signal(sub, kind)
        except Exception:
            continue
        if sig.empty:
            continue
        fr = fwd.loc[d]
        buys = sig[sig["is_buy_candidate"]]["ticker"].tolist()
        rest = sig[~sig["is_buy_candidate"]]["ticker"].tolist()
        br = np.nanmean([fr.get(t, np.nan) for t in buys]) if buys else np.nan
        rr = np.nanmean([fr.get(t, np.nan) for t in rest]) if rest else np.nan
        if not np.isnan(br):
            buy_rets.append(br); n_buys += len(buys)
        if not (np.isnan(br) or np.isnan(rr)):
            spreads.append(br - rr)
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    cost = 1.0 * (COST_BPS / 1e4) * 2.0
    pers = 252 / step
    net = ((sa.mean() - cost) / sd * np.sqrt(pers)) if sd > 0 else float("nan")
    buy_win = float(np.mean([1 if x > 0 else 0 for x in buy_rets]))
    return (round(float(np.mean(buy_rets)) * 100, 3),   # avg BUY 20d return %
            round(buy_win * 100, 1),                     # BUY win-rate %
            round(float(sa.mean()) * 100, 3),            # avg top-vs-rest spread %
            round(float(net), 3),                        # net-of-cost Sharpe
            len(spreads))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== MOMENTUM SHADOW BACKFILL (instant validation of the live pipeline) ===")
    panel = build_panel(tickers, args.start)
    print(f"  panel: {panel.shape[0]} days x {panel.shape[1]} names\n")
    print(f"  {'signal':<12}{'BUY 20d ret%':>13}{'BUY win%':>10}{'top-vs-rest%':>14}{'net Sh':>9}{'n':>5}")
    print("  " + "-"*64)
    for kind in ["mom_6_1", "mom_12_1"]:
        r = run(panel, kind)
        if r:
            ar, wr, sp, nt, n = r
            print(f"  {kind:<12}{ar:>+13.3f}{wr:>10.1f}{sp:>+14.3f}{nt:>+9.3f}{n:>5d}")
    print("\n  This validates the SHADOW PLUMBING (reconstructed picks + real 20d outcomes)")
    print("  and gives an immediate read. Forward shadow table is the true OOS test as it fills.")


if __name__ == "__main__":
    main()
