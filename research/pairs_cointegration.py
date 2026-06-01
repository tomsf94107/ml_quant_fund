"""
Pairs / cointegration — Alpha Hunt Queue #2, PHASE 1 (Jun 1 2026).

KILL-OR-CONTINUE GATE before any backtest: do within-bucket pairs that COINTEGRATE
in a formation window still MEAN-REVERT out-of-sample, across REGIMES? PCA-residual
reversal was killed (COVID artifact); pairs has the same failure mode (in-sample
cointegration dying forward). Test the prerequisite first, cheaply.

Strictly OOS: formation half selects cointegrated pairs (Engle-Granger p<0.05) + sets
spread mean/std; trading half (unseen) measures whether |z|>2 entries CONVERGE. Roll
non-overlapping windows across history = regimes. No P&L/costs yet — only "does the
relationship hold forward." Holds across regimes -> build Phase 2 backtest. Doesn't ->
KILL, move to Hunt Queue #3.

    python -m research.pairs_cointegration --start 2018-01-01
"""
from __future__ import annotations
import argparse, itertools
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import coint
from features.builder import _download


def _closes(ticker, start):
    df = _download(ticker, start, None)
    col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        raise ValueError(f"{ticker}: no close column ({list(df.columns)})")
    dcol = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if dcol is None:
        raise ValueError(f"{ticker}: no date column ({list(df.columns)})")
    s = pd.Series(df[col].values, index=pd.to_datetime(df[dcol].values)).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _bucket_pairs(meta_path="tickers_metadata.csv", tickers_file="tickers.txt"):
    meta = pd.read_csv(meta_path)
    uni = {t.strip().upper() for t in open(tickers_file) if t.strip() and not t.startswith("#")}
    meta = meta[meta["ticker"].str.upper().isin(uni)]
    pairs = []
    for bucket, g in meta.groupby("bucket"):
        names = sorted(g["ticker"].str.upper().unique())
        if len(names) < 2:
            continue
        for a, b in itertools.combinations(names, 2):
            pairs.append((a, b, bucket))
    return pairs


def _spread_z(pa, pb, ref_idx=None):
    la, lb = np.log(pa), np.log(pb)
    beta = np.polyfit(lb, la, 1)[0]
    spread = la - beta * lb
    if ref_idx is not None:
        mu, sd = spread.loc[ref_idx].mean(), spread.loc[ref_idx].std()
    else:
        mu, sd = spread.mean(), spread.std()
    if sd == 0 or np.isnan(sd):
        return None
    return (spread - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--form-days", type=int, default=252)
    ap.add_argument("--trade-days", type=int, default=126)
    ap.add_argument("--coint-p", type=float, default=0.05)
    ap.add_argument("--entry-z", type=float, default=2.0)
    args = ap.parse_args()

    pairs = _bucket_pairs()
    print(f"Within-bucket candidate pairs: {len(pairs)}")
    need = sorted({t for p in pairs for t in p[:2]})
    px = {}
    for t in need:
        try:
            px[t] = _closes(t, args.start)
        except Exception as e:
            print(f"  skip {t}: {e}")
    print(f"Loaded closes for {len(px)}/{len(need)} tickers\n")
    pairs = [(a, b, bk) for (a, b, bk) in pairs if a in px and b in px]

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(px[t].index) for t in px])))
    win = args.form_days + args.trade_days
    starts = range(0, len(all_dates) - win, win)

    window_results = []
    for s0 in starts:
        form_idx = all_dates[s0 : s0 + args.form_days]
        trade_idx = all_dates[s0 + args.form_days : s0 + win]
        if len(trade_idx) < 20:
            continue
        label = f"{form_idx[0].date()}->{trade_idx[-1].date()}"
        n_coint, n_reverted = 0, 0
        for a, b, bk in pairs:
            pa, pb = px[a], px[b]
            both = pa.index.intersection(pb.index)
            f = form_idx.intersection(both)
            tr = trade_idx.intersection(both)
            if len(f) < args.form_days * 0.8 or len(tr) < 20:
                continue
            try:
                _, pval, _ = coint(pa.loc[f], pb.loc[f])
            except Exception:
                continue
            if pval >= args.coint_p:
                continue
            n_coint += 1
            full = pd.concat([pa, pb], axis=1).dropna()
            full = full.loc[full.index.isin(f.union(tr))]
            z = _spread_z(full.iloc[:, 0], full.iloc[:, 1], ref_idx=f)
            if z is None:
                continue
            ztr = z.loc[z.index.isin(tr)]
            if ztr.empty:
                continue
            entries = ztr[abs(ztr) > args.entry_z]
            if entries.empty:
                continue
            if abs(ztr.iloc[-1]) < abs(entries.iloc[0]):
                n_reverted += 1
        if n_coint >= 5:
            rate = n_reverted / n_coint
            window_results.append((label, n_coint, n_reverted, rate))
            print(f"  {label}:  cointegrated={n_coint:3d}  reverted_OOS={n_reverted:3d}  rate={rate*100:.0f}%")

    print("\n" + "=" * 64)
    if not window_results:
        print("PHASE 1: no windows with >=5 cointegrated pairs. KILL -> Hunt Queue #3.")
        return
    rates = [r for *_, r in window_results]
    mean_rate = np.mean(rates)
    pos = sum(1 for r in rates if r > 0.5)
    print(f"PAIRS PHASE 1 -- {len(window_results)} regimes/windows")
    print(f"  mean OOS reversion rate: {mean_rate*100:.0f}%")
    print(f"  windows with >50% reversion: {pos}/{len(window_results)}")
    print("-" * 64)
    print("READ (random reversion baseline ~50%):")
    print("  - reversion >55% in MOST windows incl recent -> REAL, build Phase 2")
    print("  - reversion ~50% or only some regimes -> doesn't hold forward -> KILL -> #3")
    print("=" * 64)


if __name__ == "__main__":
    main()
