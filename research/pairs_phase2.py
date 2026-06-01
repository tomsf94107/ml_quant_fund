"""
Pairs / cointegration — Alpha Hunt Queue #2, PHASE 2 (Jun 1 2026).

Real net-of-cost backtest with the literature's fixes (Gatev/Do-Faff decline, Chan
non-convergence, convergence-rate filters, regime-dependence):
  1. ROLLING RE-SELECTION — pairs re-chosen each formation window, never static.
  2. REAL ENTRY/EXIT P&L — enter |z|>entry, exit |z|<exit, stop |z|>stop. Market-
     neutral, dollar-matched. Beta fit on FORMATION ONLY (no leak).
  3. CONVERGENCE-RATE FILTER — keep only pairs whose OU spread half-life is short
     enough to revert within the trading window.
  4. NET-OF-COST — same 10bps/turnover as momentum, charged on BOTH legs.
  5. PER-REGIME + DECORRELATION — judged as a DIVERSIFIER vs momentum, not standalone.

VERDICT: net-positive across MOST regimes (incl recent) AND |corr|<0.3 vs momentum
-> SURVIVES as signal #2 (unlocks combiner). Negative after costs / high corr -> KILL.

    python -m research.pairs_phase2 --start 2018-01-01
"""
from __future__ import annotations
import argparse, itertools
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import coint
from features.builder import _download

COST_RATE = 10.0 / 10_000.0


def _closes(ticker, start):
    df = _download(ticker, start, None)
    col = "close" if "close" in df.columns else "Close"
    dcol = "date" if "date" in df.columns else "Date"
    s = pd.Series(df[col].values, index=pd.to_datetime(df[dcol].values)).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _bucket_pairs(meta_path="tickers_metadata.csv", tickers_file="tickers.txt"):
    meta = pd.read_csv(meta_path)
    uni = {t.strip().upper() for t in open(tickers_file) if t.strip() and not t.startswith("#")}
    meta = meta[meta["ticker"].str.upper().isin(uni)]
    out = []
    for bucket, g in meta.groupby("bucket"):
        names = sorted(g["ticker"].str.upper().unique())
        for a, b in itertools.combinations(names, 2):
            out.append((a, b))
    return out


def _half_life(spread):
    s = spread.dropna()
    if len(s) < 30: return np.inf
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag = lag.loc[delta.index]
    beta = np.polyfit(lag.values, delta.values, 1)[0]
    if beta >= 0: return np.inf
    return -np.log(2) / beta


def _trade_pair(la_form, lb_form, la_trade, lb_trade, entry, exit_, stop):
    beta = np.polyfit(lb_form.values, la_form.values, 1)[0]
    spread_form = la_form - beta * lb_form
    mu, sd = spread_form.mean(), spread_form.std()
    if sd == 0 or np.isnan(sd): return None, beta
    z = ((la_trade - beta * lb_trade) - mu) / sd
    ra = la_trade.diff(); rb = lb_trade.diff()
    pos = 0; positions = []
    for zi in z:
        if pos == 0:
            if zi > entry: pos = -1
            elif zi < -entry: pos = 1
        else:
            if abs(zi) < exit_ or abs(zi) > stop: pos = 0
        positions.append(pos)
    pos_s = pd.Series(positions, index=z.index)
    spread_ret = pos_s.shift(1) * (ra - beta * rb) / (1 + abs(beta))
    turn = pos_s.diff().abs().fillna(0)
    cost = turn * COST_RATE * 2
    return (spread_ret.fillna(0) - cost), beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--form-days", type=int, default=252)
    ap.add_argument("--trade-days", type=int, default=126)
    ap.add_argument("--coint-p", type=float, default=0.05)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--max-half-life", type=int, default=30)
    args = ap.parse_args()

    pairs = _bucket_pairs()
    need = sorted({t for p in pairs for t in p})
    px = {}
    for t in need:
        try: px[t] = _closes(t, args.start)
        except Exception: pass
    pairs = [(a, b) for a, b in pairs if a in px and b in px]
    print(f"{len(pairs)} within-bucket pairs, {len(px)} tickers loaded\n")

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(px[t].index) for t in px])))
    win = args.form_days + args.trade_days
    starts = range(0, len(all_dates) - win, args.trade_days)

    allpx = pd.DataFrame({t: px[t] for t in px}).sort_index()
    mkt_ret = np.log(allpx).diff().mean(axis=1)

    regime_rows = []
    pairs_daily = pd.Series(dtype=float)
    for s0 in starts:
        form_idx = all_dates[s0 : s0 + args.form_days]
        trade_idx = all_dates[s0 + args.form_days : s0 + win]
        if len(trade_idx) < 20: continue
        label = f"{trade_idx[0].date()}->{trade_idx[-1].date()}"
        selected = []
        for a, b in pairs:
            pa, pb = px[a], px[b]
            both = pa.index.intersection(pb.index)
            f = form_idx.intersection(both)
            if len(f) < args.form_days * 0.8: continue
            la_f, lb_f = np.log(pa.loc[f]), np.log(pb.loc[f])
            try:
                _, pval, _ = coint(pa.loc[f], pb.loc[f])
            except Exception:
                continue
            if pval >= args.coint_p: continue
            beta = np.polyfit(lb_f.values, la_f.values, 1)[0]
            hl = _half_life(la_f - beta * lb_f)
            if not (0 < hl <= args.max_half_life): continue
            selected.append((a, b))
        if not selected: continue
        day_rets = []
        for a, b in selected:
            pa, pb = px[a], px[b]
            both = pa.index.intersection(pb.index)
            f = form_idx.intersection(both); tr = trade_idx.intersection(both)
            if len(tr) < 20: continue
            net, _ = _trade_pair(np.log(pa.loc[f]), np.log(pb.loc[f]),
                                  np.log(pa.loc[tr]), np.log(pb.loc[tr]),
                                  args.entry_z, args.exit_z, args.stop_z)
            if net is not None: day_rets.append(net)
        if not day_rets: continue
        port = pd.concat(day_rets, axis=1).mean(axis=1)
        pairs_daily = pd.concat([pairs_daily, port])
        tot = (1 + port).prod() - 1
        ann = port.mean() * 252; vol = port.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        regime_rows.append((label, len(selected), tot * 100, sharpe))
        print(f"  {label}:  pairs={len(selected):3d}  net_ret={tot*100:+6.2f}%  sharpe={sharpe:+.2f}")

    print("\n" + "=" * 64)
    if not regime_rows:
        print("PHASE 2: no tradeable pairs after convergence filter. KILL -> #3.")
        return
    rets = [r for *_, r, _ in regime_rows]
    sharpes = [s for *_, s in regime_rows]
    pos_windows = sum(1 for r in rets if r > 0)
    aligned = pd.concat([pairs_daily.rename("pairs"), mkt_ret.rename("mkt")], axis=1).dropna()
    corr = aligned["pairs"].corr(aligned["mkt"]) if len(aligned) > 20 else float("nan")
    overall_sharpe = (pairs_daily.mean() * 252) / (pairs_daily.std() * np.sqrt(252)) if pairs_daily.std() > 0 else 0
    print(f"PAIRS PHASE 2 (net of 10bps/leg) -- {len(regime_rows)} regimes")
    print(f"  windows net-positive: {pos_windows}/{len(regime_rows)}")
    print(f"  mean window sharpe:   {np.mean(sharpes):+.2f}")
    print(f"  pooled net sharpe:    {overall_sharpe:+.2f}")
    print(f"  corr vs mkt/momentum proxy: {corr:+.2f}")
    print("-" * 64)
    print("VERDICT: net-positive MOST regimes (incl recent) AND |corr|<0.3 -> SURVIVES #2")
    print("         negative after costs / |corr|>0.3 -> KILL -> Hunt #3")
    print("=" * 64)


if __name__ == "__main__":
    main()
