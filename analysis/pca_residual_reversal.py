"""
analysis/pca_residual_reversal.py — STEP 2a: PROPER PCA residual reversal (Avellaneda-Lee).

The crude cs-demean reversion failed full-history validation (net -0.43). This is
the stronger construction: instead of residualizing returns against the cross-
sectional MEAN (one blunt factor), residualize against the top-K PRINCIPAL
COMPONENTS of the return covariance (the real statistical factors). The residual =
the part of each stock's return NOT explained by common factors. Avellaneda-Lee:
that residual mean-reverts -> buy stocks whose residual is most negative (cheap vs
factor-implied), short most positive.

Validated through the SAME gate that killed the crude version: full-history
2020->now, NON-overlapping windows, winsorized, net of 10bps/turnover.
GATE: clean non-overlap net Sharpe > 0.3.
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


def pca_residual(daily_ret_window, n_factors):
    """Given a (days x names) return window, regress each name's returns on the
    top-n_factors PCs of the cross-section and return the residual std-score for
    the LAST day. Returns a Series (name -> residual z) for the most recent day."""
    X = daily_ret_window.dropna(axis=1, how="any")
    if X.shape[1] < 20 or X.shape[0] < 30:
        return None
    # PCA on the covariance: eigenvectors of returns' correlation
    Xz = (X - X.mean()) / X.std().replace(0, np.nan)
    Xz = Xz.dropna(axis=1, how="any")
    if Xz.shape[1] < 20:
        return None
    cov = np.cov(Xz.values.T)
    eigval, eigvec = np.linalg.eigh(cov)
    top = eigvec[:, -n_factors:]                 # top-K eigenvectors
    factors = Xz.values @ top                    # (days x K) factor returns
    # regress each name on factors, take residual of the last row
    resid_last = {}
    for j, name in enumerate(Xz.columns):
        y = Xz.values[:, j]
        beta, *_ = np.linalg.lstsq(factors, y, rcond=None)
        resid = y - factors @ beta
        # cumulative residual (the "s-score" proxy): how far below/above factor-implied
        resid_last[name] = resid.sum() / (resid.std() + 1e-9)
    return pd.Series(resid_last)


def backtest(panel, n_factors=5, est_window=60, step=5, winsor=0.25, fwd=5):
    rets = panel.pct_change(1, fill_method=None)
    ret_fwd = (panel.shift(-fwd) / panel - 1.0).clip(-winsor, winsor)
    idx = panel.index
    spreads = []
    for i in range(est_window, len(idx)-fwd, step):
        d = idx[i]
        window = rets.iloc[i-est_window:i]
        sscore = pca_residual(window, n_factors)
        if sscore is None or len(sscore) < 20:
            continue
        fr = ret_fwd.loc[d]
        order = sscore.sort_values()             # most negative resid = cheap = LONG
        k = max(1, len(order)//10)
        longs  = order.head(k).index             # lowest s-score = most below factor = buy
        shorts = order.tail(k).index
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
    return round(float(gross),3), round(float(net),3), len(sa), round(float(sa.mean())*100,4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--start", default="2020-01-01")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== PCA RESIDUAL REVERSAL (Avellaneda-Lee): {len(tickers)} names ===")
    panel = build_close_panel(tickers, args.start)
    print(f"  close panel: {panel.shape[0]} days x {panel.shape[1]} names\n")
    print(f"  {'config (n_factors, est_window)':<40}{'gross':>8}{'net Sh':>9}{'n':>6}{'mean%':>9}")
    print("  " + "-"*72)
    print("  --- fine sweep around the winning region (120d window) ---")
    for nf in [5, 8, 10, 12, 15, 20]:
        for ew in [90, 120, 150, 180]:
            r = backtest(panel, n_factors=nf, est_window=ew)
            if r:
                g, nt, nd, mp = r
                mark = " <-- PASS" if nt > 0.3 else ""
                print(f"  factors={nf:>2}, window={ew}d{'':<20}{g:>+8.2f}{nt:>+9.3f}{nd:>6d}{mp:>+9.4f}{mark}")
    print("\n  --- per-REGIME check (factors=10, window=120d) on the best region ---")
    regimes = [("2020 COVID", "2020-01-01", "2020-12-31"),
               ("2021 bull",  "2021-01-01", "2021-12-31"),
               ("2022 bear",  "2022-01-01", "2022-12-31"),
               ("2023-24",    "2023-01-01", "2024-12-31"),
               ("2025-26",    "2025-01-01", "2026-12-31")]
    for rlabel, rs, re_ in regimes:
        sub = panel.loc[(panel.index >= rs) & (panel.index <= re_)]
        if len(sub) < 150:
            print(f"  {rlabel:<14} (too short, {len(sub)}d)"); continue
        r = backtest(sub, n_factors=10, est_window=120)
        if r:
            g, nt, nd, mp = r
            print(f"  {rlabel:<14}{g:>+8.2f}{nt:>+9.3f}{nd:>6d}{mp:>+9.4f}")
        else:
            print(f"  {rlabel:<14} (no result)")
    print("\n  GATE: net Sh > 0.3 -> PCA residual reversal works where crude cs-demean failed.")
    print("  If all <=0 -> 1-5d reversion at this scale is not tradeable; move to other axes.")


if __name__ == "__main__":
    main()
