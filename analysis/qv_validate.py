"""
analysis/qv_validate.py — Quality + Value cross-sectional signal validation (Track C).
Uses fundamentals.db (PIT, filed_date < as_of) joined to the close panel.

SIGNALS (each: monthly rebalance, top-decile long, inv-vol, 10bps):
  quality_gp   = (revenue - cogs) / total_assets      [OSAP GP prior: +17.8% 2020+ gross]
  quality_op   = operating_income / equity            [OSAP OperProf: +8.0%]
  value_bm     = equity / (close * shares_out)        [OSAP BM: +17.4%]
  value_ep     = net_income / (close * shares_out)    [OSAP EP: +7.4%]

GATES (pre-registered, same bar as TSMOM): G1 netSharpe>0.3 | G2 positive most years |
  G3 |corr|<0.3 vs momentum book (decisive: NEW axis?) | G4 beats SPY.
HONEST: ~110-136 tickers with fundamentals = thin cross-section; a pass = "C1 candidate",
  not standalone edge. Fundamentals are a different axis from price -> G3 should pass clean.
PIT: a fact is usable only on dates AFTER its filed_date. Strict.
Run: python -m analysis.qv_validate
"""
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
COST_BPS, FWD_M, ANN = 10.0, 21, 12  # monthly hold


def latest_pit(facts, concept, as_of, ticker):
    """Most recent value of `concept` for `ticker` with filed_date strictly < as_of."""
    d = facts[(facts.ticker == ticker) & (facts.concept == concept) &
              (facts.filed_date < as_of)]
    if d.empty:
        return np.nan
    return d.sort_values("filed_date").iloc[-1]["value"]


def build_fundamental_signal(facts, panel, rebal_dates, signal_fn, tickers):
    """For each monthly rebalance date, compute cross-sectional signal from PIT facts."""
    rows = {}
    for d in rebal_dates:
        as_of = d.strftime("%Y-%m-%d")
        vals = {}
        for t in tickers:
            v = signal_fn(facts, as_of, t, panel, d)
            if v == v and np.isfinite(v):
                vals[t] = v
        if len(vals) >= 10:
            rows[d] = pd.Series(vals)
    return rows


def sig_gp(facts, as_of, t, panel, d):
    rev = latest_pit(facts, "revenue", as_of, t)
    cogs = latest_pit(facts, "cogs", as_of, t)
    assets = latest_pit(facts, "total_assets", as_of, t)
    return (rev - cogs) / assets if assets and assets > 0 else np.nan


def sig_op(facts, as_of, t, panel, d):
    oi = latest_pit(facts, "operating_income", as_of, t)
    eq = latest_pit(facts, "equity", as_of, t)
    return oi / eq if eq and eq > 0 else np.nan


def _mktcap(facts, as_of, t, panel, d):
    sh = latest_pit(facts, "shares_out", as_of, t)
    if not (sh and sh > 0) or t not in panel.columns:
        return np.nan
    px = panel[t].loc[:d].dropna()
    return px.iloc[-1] * sh if len(px) else np.nan


def sig_bm(facts, as_of, t, panel, d):
    eq = latest_pit(facts, "equity", as_of, t)
    mc = _mktcap(facts, as_of, t, panel, d)
    return eq / mc if mc and mc > 0 else np.nan


def sig_ep(facts, as_of, t, panel, d):
    ni = latest_pit(facts, "net_income", as_of, t)
    mc = _mktcap(facts, as_of, t, panel, d)
    return ni / mc if mc and mc > 0 else np.nan


def book_returns(sig_by_date, panel, rebal_dates):
    """Top-decile long, inv-vol, net of cost. Returns per-rebalance net series."""
    rets = panel.pct_change()
    out, prev = [], set()
    dl = sorted(sig_by_date.keys())
    for k, d in enumerate(dl[:-1]):
        sig = sig_by_date[d].dropna()
        top = sig[sig.rank(pct=True) >= 0.9].index
        top = [t for t in top if t in panel.columns]
        if len(top) < 2:
            out.append((dl[k + 1], 0.0)); prev = set(); continue
        i = panel.index.get_indexer([d], method="ffill")[0]
        j = panel.index.get_indexer([dl[k + 1]], method="ffill")[0]
        if i < 60 or j <= i:
            continue
        vol = rets[top].iloc[i - 60:i].std()
        w = (1.0 / vol.replace(0, np.nan)).fillna(0); w = w / w.sum() if w.sum() > 0 else w
        fwd = panel[top].iloc[j] / panel[top].iloc[i] - 1.0
        turn = len(set(top) ^ prev) / max(len(top), 1)
        out.append((dl[k + 1], float((w * fwd).sum()) - COST_BPS / 1e4 * turn))
        prev = set(top)
    return pd.DataFrame(out, columns=["date", "net"]).set_index("date")["net"]


def sharpe(r):
    return float(np.sqrt(ANN) * r.mean() / r.std()) if len(r) > 3 and r.std() > 0 else np.nan


def main():
    from analysis.momentum_purged_wf import build_close_panel, momentum
    facts = pd.read_sql("SELECT ticker,concept,value,filed_date FROM xbrl_facts",
                        sqlite3.connect(str(ROOT / "fundamentals.db")))
    tickers = sorted(facts.ticker.unique())
    print(f"fundamentals: {len(tickers)} tickers")
    uni = pd.read_sql("SELECT DISTINCT ticker FROM predictions",
                      sqlite3.connect(str(ROOT / "accuracy.db")))["ticker"].tolist()
    panel = build_close_panel(sorted(set(uni) | {"SPY"}), "2015-01-01")
    rebal = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    rebal = [d for d in rebal if d >= panel.index.min()]

    # momentum book for G3
    mb_rebal = {}
    for d in rebal:
        s = momentum(panel.loc[:d], "mom_6_1")
        if len(s) and s.iloc[-1].dropna().shape[0] >= 10:
            mb_rebal[d] = s.iloc[-1].dropna()
    mb = book_returns(mb_rebal, panel, rebal)
    mb_m = (1 + mb).resample("M").prod() - 1

    spy = panel["SPY"].resample("M").last().pct_change().dropna()
    spy_sh = sharpe(spy)
    print(f"momentum book + SPY ref (Sharpe {spy_sh:+.2f}) ready\n")

    sigs = {"quality_gp": sig_gp, "quality_op": sig_op, "value_bm": sig_bm, "value_ep": sig_ep}
    priors = {"quality_gp": "GP +17.8% 2020+", "quality_op": "OperProf +8.0%",
              "value_bm": "BM +17.4%", "value_ep": "EP +7.4%"}
    for name, fn in sigs.items():
        sbd = build_fundamental_signal(facts, panel, rebal, fn, tickers)
        bk = book_returns(sbd, panel, rebal)
        sh = sharpe(bk)
        yr = bk.groupby(bk.index.year).apply(sharpe)
        bkm = (1 + bk).resample("M").prod() - 1
        j = pd.concat([bkm, mb_m], axis=1, keys=["q", "m"]).dropna()
        corr = float(j["q"].corr(j["m"])) if len(j) > 6 else np.nan
        g1, g2 = sh > 0.3, (yr > 0).sum() >= max(2, int(0.6 * yr.notna().sum()))
        g3, g4 = (abs(corr) < 0.3 if corr == corr else False), sh > spy_sh
        print(f"=== {name}  [OSAP prior {priors[name]}] ===")
        print(f"  netSharpe={sh:+.2f}  n_rebal={len(bk)}  yrs+={int((yr>0).sum())}/{int(yr.notna().sum())}  corr_vs_mom={corr:+.2f}")
        print(f"  G1>{0.3}:{g1} G2:{g2} G3|corr|<.3:{g3} G4>SPY:{g4} -> {'PASS (C1 candidate)' if all([g1,g2,g3,g4]) else 'FAIL'}\n")
    print("Reminder: ~120-name cross-section is thin; pass = combine-candidate, not standalone.")


if __name__ == "__main__":
    main()
