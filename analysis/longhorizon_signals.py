"""
analysis/longhorizon_signals.py — STEP 2b: longer-horizon, low-turnover signals.

Every short-horizon signal died on noise + daily-turnover cost. This tests
LONGER-hold factor signals (20d horizon, 20d rebalance = ~4x less turnover) of
types suited to longer holds, NOT stretched reversion (the h-sweep showed
reversion goes negative by h=20). Through the SAME gate: full-history non-overlap,
winsorized, net of 10bps/turnover, per-regime.

Signals tested (cross-sectional rank each rebalance, long top decile, short bottom):
  - momentum 12-1m  : trailing 252d return minus last 21d (classic Jegadeesh-Titman)
  - momentum 6-1m   : trailing 126d minus last 21d
  - low-volatility  : -1 * trailing 60d realized vol (low-vol anomaly, long low-vol)
  - long-term rev   : -1 * trailing 252d return (DeBondt-Thaler contrarian)
GATE: net Sh > 0.3 AND positive in most regimes.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import _download

COST_BPS = 10.0


def build_close_panel(tickers, start):
    closes = {}
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            closes[tk] = _download(tk, start, None).set_index("date")["close"]
        except Exception as e:
            print(f"  skip {tk}: {type(e).__name__}", flush=True)
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def signal_panel(panel, kind):
    if kind == "mom_12_1":
        return panel.pct_change(252, fill_method=None) - panel.pct_change(21, fill_method=None)
    if kind == "mom_6_1":
        return panel.pct_change(126, fill_method=None) - panel.pct_change(21, fill_method=None)
    if kind == "low_vol":
        return -panel.pct_change(1, fill_method=None).rolling(60).std()
    if kind == "lt_reversal":
        return -panel.pct_change(252, fill_method=None)
    raise ValueError(kind)


def backtest(panel, kind, fwd=20, step=20, winsor=0.40):
    sig = signal_panel(panel, kind)
    ret_fwd = (panel.shift(-fwd) / panel - 1.0).clip(-winsor, winsor)
    idx = panel.index
    spreads = []
    for i in range(252, len(idx)-fwd, step):
        d = idx[i]
        row = sig.loc[d].dropna()
        if len(row) < 10:
            continue
        fr = ret_fwd.loc[d]
        order = row.sort_values()
        k = max(1, len(order)//10)
        longs  = order.tail(k).index   # high signal = long
        shorts = order.head(k).index
        lr = np.nanmean([fr.get(t, np.nan) for t in longs])
        sr = np.nanmean([fr.get(t, np.nan) for t in shorts])
        if not (np.isnan(lr) or np.isnan(sr)):
            spreads.append(lr - sr)
    if not spreads:
        return None
    sa = np.array(spreads); sd = sa.std()
    cost = 1.0*(COST_BPS/1e4)*2.0
    pers = 252/step
    gross = (sa.mean()/sd*np.sqrt(pers)) if sd>0 else float("nan")
    net   = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(gross),3), round(float(net),3), len(sa), round(float(sa.mean())*100,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--start", default="2019-01-01")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== LONG-HORIZON SIGNALS (20d hold, 20d rebalance): {len(tickers)} names ===")
    panel = build_close_panel(tickers, args.start)
    print(f"  close panel: {panel.shape[0]} days x {panel.shape[1]} names\n")
    print(f"  {'signal':<18}{'gross':>8}{'net Sh':>9}{'n':>6}{'mean%':>9}")
    print("  " + "-"*52)
    results = {}
    for kind in ["mom_12_1", "mom_6_1", "low_vol", "lt_reversal"]:
        r = backtest(panel, kind)
        results[kind] = r
        if r:
            g, nt, nd, mp = r
            mark = " <-- PASS" if nt > 0.3 else ""
            print(f"  {kind:<18}{g:>+8.2f}{nt:>+9.3f}{nd:>6d}{mp:>+9.3f}{mark}")
    # per-regime for any passer — fewer/longer regimes + 10d step for enough rebalances
    for kind, r in results.items():
        if r and r[1] > 0.3:
            print(f"\n  --- per-REGIME: {kind} (10d step for more rebalances) ---")
            for rl, rs, re_ in [("pre-2022 (covid+bull)","2019-01-01","2021-12-31"),
                                ("2022 BEAR (crash test)","2021-09-01","2022-12-31"),
                                ("2023-24","2022-09-01","2024-12-31"),
                                ("2025-26 (current)","2024-09-01","2026-12-31")]:
                sub = panel.loc[(panel.index>=rs)&(panel.index<=re_)]
                if len(sub) < 200:
                    print(f"  {rl:<22} (short, {len(sub)}d)"); continue
                rr = backtest(sub, kind, step=10)
                if rr: print(f"  {rl:<22}{rr[0]:>+8.2f}{rr[1]:>+9.3f}{rr[2]:>6d}{rr[3]:>+9.3f}")
    print("\n  GATE: net Sh > 0.3 AND positive in most regimes (esp 2025-26).")


if __name__ == "__main__":
    main()
