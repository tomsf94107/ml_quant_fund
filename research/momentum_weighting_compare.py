"""
Momentum weighting comparison — C2 sizing test (Jun 1 2026).

Does #4's vol info, used to size the momentum book, beat validated equal-weight
(+0.96 net Sharpe) NET OF COST? Research (Barroso-Santa-Clara, Daniel-Moskowitz):
PORTFOLIO-level vol-targeting ~doubles momentum Sharpe; PER-NAME scaling is weaker
(matches our Opt-3 rejection); costs can 15x turnover and kill it (Barroso-Detzel).
Test all three net-of-cost with turnover, strictly-backward vol at each rebalance.

Reuses the validated backfill engine logic (truncate panel -> rank_signal -> real 20d
fwd return). Three weightings, same picks/dates/cost:
  A EQUAL-WEIGHT (baseline; must reproduce ~+0.96)
  B PER-NAME INV-VOL (weight ~ 1/vol_i)
  C PORTFOLIO VOL-TARGET (equal picks, scale TOTAL exposure by target/basket vol)

    python -m research.momentum_weighting_compare --start 2018-01-01
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import _download
from signals.momentum_signal import rank_signal, MOM_HORIZON

COST_BPS = 10.0
TARGET_VOL = 0.15


def build_panel(tickers, start):
    closes = {}
    for tk in tickers:
        try:
            d = _download(tk, start, None)
            closes[tk] = pd.Series(d["close"].values, index=pd.to_datetime(d["date"].values))
        except Exception:
            pass
    panel = pd.DataFrame(closes).sort_index()
    return panel[~panel.index.duplicated(keep="last")]


def run(panel, kind="mom_6_1", step=20, winsor=0.40):
    fwd = (panel.shift(-MOM_HORIZON) / panel - 1.0).clip(-winsor, winsor)
    daily_ret = panel.pct_change()
    valid = panel.index[252 : len(panel) - MOM_HORIZON]
    dates = list(valid[::step])
    recs = {"A": [], "B": [], "C": []}
    prev_w = {"A": None, "B": None, "C": None}
    turn = {"A": [], "B": [], "C": []}
    for d in dates:
        sub = panel.loc[:d]
        try:
            sig = rank_signal(sub, kind)
        except Exception:
            continue
        if sig.empty:
            continue
        buys = sig[sig["is_buy_candidate"]]["ticker"].tolist()
        if not buys:
            continue
        fr = fwd.loc[d]
        rets = np.array([fr.get(t, np.nan) for t in buys])
        ok = ~np.isnan(rets)
        if ok.sum() == 0:
            continue
        buys_ok = [b for b, o in zip(buys, ok) if o]
        rets_ok = rets[ok]
        vol = {}
        for t in buys_ok:
            r = daily_ret[t].loc[:d].tail(20)
            vol[t] = r.std() * np.sqrt(252) if r.notna().sum() >= 10 else np.nan
        vol_arr = np.array([vol[t] for t in buys_ok])
        med = np.nanmedian(vol_arr)
        vol_arr = np.where(np.isnan(vol_arr) | (vol_arr <= 0), med, vol_arr)
        n = len(buys_ok)
        wA = np.ones(n) / n
        wB = (1.0 / vol_arr); wB = wB / wB.sum()
        wC_base = np.ones(n) / n
        basket_vol = np.sqrt((wC_base**2 * vol_arr**2).sum())
        scale = min(TARGET_VOL / basket_vol, 1.5) if basket_vol > 0 else 1.0
        wC = wC_base * scale
        recs["A"].append((wA * rets_ok).sum())
        recs["B"].append((wB * rets_ok).sum())
        recs["C"].append((wC * rets_ok).sum())
        for scheme, w in (("A", wA), ("B", wB), ("C", wC)):
            wser = pd.Series(w, index=buys_ok)
            if prev_w[scheme] is not None:
                allidx = wser.index.union(prev_w[scheme].index)
                tnow = wser.reindex(allidx).fillna(0)
                tprev = prev_w[scheme].reindex(allidx).fillna(0)
                turn[scheme].append(float((tnow - tprev).abs().sum()))
            prev_w[scheme] = wser
    out = {}
    pers = 252 / step
    for s in ("A", "B", "C"):
        arr = np.array(recs[s])
        if len(arr) < 5:
            out[s] = None; continue
        avg_turn = np.mean(turn[s]) if turn[s] else 0.0
        cost_per = avg_turn * (COST_BPS / 1e4)
        net_mean = arr.mean() - cost_per
        sd = arr.std()
        sharpe = (net_mean / sd * np.sqrt(pers)) if sd > 0 else float("nan")
        gross_sharpe = (arr.mean() / sd * np.sqrt(pers)) if sd > 0 else float("nan")
        out[s] = {"n": len(arr), "net_sharpe": round(sharpe,3),
                  "gross_sharpe": round(gross_sharpe,3), "avg_turnover": round(avg_turn,3),
                  "total_net_ret": round(((1+arr-cost_per).prod()-1)*100,1)}
    return out, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--kind", default="mom_6_1")
    ap.add_argument("--step", type=int, default=20)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in open(ROOT / "tickers.txt")
               if t.strip() and not t.startswith("#")]
    print(f"Loading {len(tickers)} tickers...")
    panel = build_panel(tickers, args.start)
    print(f"Panel: {panel.shape[1]} tickers, {panel.shape[0]} days "
          f"({panel.index[0].date()} -> {panel.index[-1].date()})\n")
    out, dates = run(panel, kind=args.kind, step=args.step)
    names = {"A": "EQUAL-WEIGHT (baseline)", "B": "PER-NAME INV-VOL (Opt-3 redux)",
             "C": "PORTFOLIO VOL-TARGET (research winner)"}
    print(f"{'scheme':<40} {'net_Sh':>7} {'gross_Sh':>8} {'turnover':>8} {'tot_net%':>9}")
    print("-" * 76)
    for s in ("A", "B", "C"):
        r = out[s]
        if r is None:
            print(f"{names[s]:<40} {'n/a':>7}"); continue
        print(f"{names[s]:<40} {r['net_sharpe']:>7.2f} {r['gross_sharpe']:>8.2f} "
              f"{r['avg_turnover']:>8.2f} {r['total_net_ret']:>8.1f}%")
    print("\n" + "=" * 76)
    print("READ:")
    print("  - A should reproduce ~+0.96 net Sharpe; else engine off.")
    print("  - C > A on net Sharpe -> SHIP C (research: vol-target ~doubles momentum Sharpe)")
    print("  - B weak/high-turnover -> expected (Opt-3 rejected)")
    print("  - C turnover huge & net<<gross -> costs kill it (Barroso-Detzel)")
    print("=" * 76)


if __name__ == "__main__":
    main()
