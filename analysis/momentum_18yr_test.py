#!/usr/bin/env python3
"""
analysis/momentum_18yr_test.py -- the VALIDATED momentum signal on 18 years.

WHY
    momentum passed strict purged-WF 4/4 OOS folds (net ~+1.0 Sharpe) -- but that
    window is ~2022-26, one regime, and shadow mode is now waiting 12 weeks of
    live data to confirm it. daily_prices holds 2008->2026 adjusted closes for
    334+ tickers. This runs the SAME definition (signals/momentum_signal.py,
    lifted from analysis/momentum_purged_wf.py) through the SAME gate that
    killed the direction model: return spread -> beta-strip -> net of cost,
    t clustered by YEAR (~18 clusters, not 200 overlapping months).

WHAT IT IS NOT
    - Not a new signal definition. compute_momentum is imported, not re-derived.
    - Not survivorship-clean: daily_prices is the CURRENT ticker list back-
      extended (pead_survivorship.py documents this). Survivor bias INFLATES
      momentum (the delisted losers are missing). Therefore:
          FAIL here is conclusive. PASS is provisional, pending shadow.
    - Beta-strip uses the EQUAL-WEIGHT UNIVERSE as the market factor, not SPY
      (SPY only exists 2022+ in daily_prices). Hedging vs your own investable
      universe; stated, not hidden.

SANITY CHECK BUILT IN
    2009 contains the worst momentum crash on record. If the yearly table does
    not show momentum bleeding in 2009, the test is broken -- distrust it.

USAGE
    python -m analysis.momentum_18yr_test                # both kinds
    python -m analysis.momentum_18yr_test --kind mom_12_1
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signals.momentum_signal import compute_momentum, LOOKBACKS, SKIP, MOM_HORIZON

HOLD      = MOM_HORIZON          # 20 trading days, matches validation
DECILE    = 0.10
COST_RT   = float(os.environ.get("ML_QUANT_COST_BPS", "10.0")) / 10_000.0
FACTOR    = os.environ.get("ML_QUANT_FACTOR", "ff").lower()    # ff | ew
NULL_RUN  = os.environ.get("ML_QUANT_NULL", "0") == "1"
BUCKET_CAP = int(os.environ.get("ML_QUANT_BUCKET_CAP", "0"))   # 3 = mirror the shadow
_RNG      = np.random.default_rng(int(os.environ.get("ML_QUANT_NULL_SEED", "42")))
MIN_NAMES = 60                   # skip a rebalance date with fewer scored names
BETA_MIN  = 12                   # expanding-window obs needed before stripping


def load_panel() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{ROOT/'prices.db'}?mode=ro", uri=True)
    df = pd.read_sql("SELECT ticker, date, adj_close FROM daily_prices "
                     "WHERE adj_close IS NOT NULL", con)
    con.close()
    panel = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def load_ff() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{ROOT/'prices.db'}?mode=ro", uri=True)
    ff = pd.read_sql("SELECT date, mkt_rf, rf FROM ff_factors_daily", con,
                     index_col="date")
    con.close()
    ff.index = pd.to_datetime(ff.index)
    return ff.sort_index()


def run_kind(panel: pd.DataFrame, kind: str) -> pd.DataFrame:
    lb = LOOKBACKS[kind]
    first = lb + SKIP + 5
    dates = list(range(first, len(panel) - HOLD, HOLD))
    rows, prev_basket = [], set()

    for i in dates:
        window = panel.iloc[: i + 1]
        try:
            score = compute_momentum(window.tail(lb + 60), kind)
        except ValueError:
            continue
        if len(score) < MIN_NAMES:
            continue

        if NULL_RUN:
            score = pd.Series(_RNG.permutation(score.values), index=score.index)
        k = max(1, int(len(score) * DECILE))
        ranked = score.sort_values(ascending=False)
        if BUCKET_CAP:
            from signals.momentum_signal import _load_bucket_map
            _bmap = _load_bucket_map()
            _top, _per = [], {}
            for _t in ranked.index:
                _b = _bmap.get(str(_t).upper(), "UNK")
                if _per.get(_b, 0) < BUCKET_CAP:
                    _top.append(_t)
                    _per[_b] = _per.get(_b, 0) + 1
                if len(_top) >= k:
                    break
            top = set(_top)
        else:
            top = set(ranked.head(k).index)
        bot = set(ranked.tail(k).index)   # L/S diag only; shadow has no short leg

        fwd = panel.iloc[i + HOLD] / panel.iloc[i] - 1.0
        fwd = fwd.reindex(score.index).dropna()
        t_in = [t for t in top if t in fwd.index]
        b_in = [t for t in bot if t in fwd.index]
        if len(t_in) < 5 or len(fwd) < MIN_NAMES:
            continue

        turnover = 1.0 - (len(top & prev_basket) / len(top)) if prev_basket else 1.0
        prev_basket = top

        rows.append({
            "date":     panel.index[i],
            "end_date": panel.index[i + HOLD],
            "year":     panel.index[i].year,
            "n_scored": len(fwd),
            "top_ret":  fwd[t_in].mean(),
            "bot_ret":  fwd[b_in].mean() if b_in else np.nan,
            "univ_ret": fwd.mean(),
            "turnover": turnover,
        })

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    # walk-forward beta of the top basket vs the EW universe (expanding, no lookahead)
    if FACTOR == "ff":
        ff = load_ff()
        _mkt, _rf = [], []
        for _, r in d.iterrows():
            w = ff.loc[(ff.index > r["date"]) & (ff.index <= r["end_date"])]
            if len(w) < HOLD - 3:
                _mkt.append(np.nan); _rf.append(np.nan)
            else:
                _mkt.append(float((1 + w.mkt_rf).prod() - 1))
                _rf.append(float((1 + w.rf).prod() - 1))
        d["fac_ret"], d["rf_ret"] = _mkt, _rf
        d["dep_ret"] = d.top_ret - d.rf_ret
        _nd = int(d.fac_ret.isna().sum())
        if _nd:
            print(f"  [note] {_nd} rebalance(s) lack full FF coverage "
                  f"(table ends {ff.index.max().date()}); dropped from stripped lines")
    else:
        d["fac_ret"] = d.univ_ret
        d["dep_ret"] = d.top_ret
    d["beta"] = np.nan
    for j in range(len(d)):
        if j >= BETA_MIN:
            h = d.iloc[:j]
            if h.univ_ret.std() > 0:
                hh = h.dropna(subset=["fac_ret", "dep_ret"])
                if len(hh) >= BETA_MIN and hh.fac_ret.std() > 0:
                    d.iloc[j, d.columns.get_loc("beta")] = float(
                        np.polyfit(hh.fac_ret, hh.dep_ret, 1)[0])
    d["raw_spread"]   = d.top_ret - d.univ_ret
    # residual of the top basket vs the EW-universe factor; the factor's own
    # residual is 0 by construction, so this IS the beta-stripped spread
    d["resid_spread"] = d.dep_ret - d.beta * d.fac_ret
    d["net_spread"]   = d.resid_spread - COST_RT * d.turnover
    d["ls_spread"]    = d.top_ret - d.bot_ret
    return d


def yearly_t(x: pd.Series, years: pd.Series):
    g = x.groupby(years).mean().dropna()
    k = len(g)
    if k < 2:
        return np.nan, np.nan, 0, 0
    m, s = g.mean(), g.std(ddof=1)
    t = m / (s / np.sqrt(k)) if s > 0 else np.nan
    return m, t, int((g > 0).sum()), k


def line(label, x, years):
    m, t, pos, k = yearly_t(x, years)
    flag = "PASS" if (t == t and t >= 3.0 and m > 0) else "fail"
    print(f"  {label:<38} {m*100:+7.3f} %   t={t:+6.2f}   {pos}/{k} yrs+   [{flag}]")


def report(d: pd.DataFrame, kind: str):
    print("\n" + "=" * 78)
    print(f"  MOMENTUM {kind}  --  {d.date.min().date()} -> {d.date.max().date()}"
          f"   ({len(d)} rebalances, {HOLD}td hold, top decile EW)")
    print("=" * 78)
    print(f"  bucket_cap = {BUCKET_CAP or 'none (plain decile)'}")
    print(f"  factor = {'FF daily Mkt-RF, excess' if FACTOR == 'ff' else 'EW universe'}"
          + ("   *** NULL RUN: scores shuffled ***" if NULL_RUN else ""))
    print(f"  cost = {COST_RT*1e4:.0f} bps round-trip x measured turnover "
          f"(median turnover {d.turnover.median():.0%})")

    print("\n  --- FULL 18 YEARS, t clustered by year ---")
    line("raw spread (top - universe)", d.raw_spread, d.year)
    db = d.dropna(subset=["beta"])
    line("beta-stripped (vs EW universe)", db.resid_spread, db.year)
    line("beta-stripped NET of cost  <<<", db.net_spread, db.year)
    line("long/short (top - bottom), diag", d.ls_spread, d.year)
    print(f"    median walk-forward beta: {db.beta.median():.3f}")

    sub = db[db.date >= "2022-01-01"]
    if len(sub):
        print("\n  --- 2022+ SUB-WINDOW (reconcile vs the 4/4 purged-WF) ---")
        line("beta-stripped NET of cost", sub.net_spread, sub.year)

    print("\n  --- YEARLY (the 2009 sanity check lives here) ---")
    y = d.groupby("year").agg(
        rebals=("raw_spread", "size"),
        raw=("raw_spread", "mean"),
        ls=("ls_spread", "mean"),
        net=("net_spread", "mean"),
    )
    print(f"  {'year':<6}{'rebals':>7}{'raw%':>9}{'L/S%':>9}{'net%':>9}")
    for yr, r in y.iterrows():
        mark = "  <-- crash-check" if yr in (2009, 2020, 2022) else ""
        print(f"  {yr:<6}{int(r.rebals):>7}{r.raw*100:>+9.3f}{r.ls*100:>+9.3f}"
              f"{(r.net*100 if r.net == r.net else float('nan')):>+9.3f}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=list(LOOKBACKS))
    a = ap.parse_args()

    panel = load_panel()
    print(f"panel: {panel.shape[1]} tickers x {len(panel)} sessions "
          f"({panel.index[0].date()} -> {panel.index[-1].date()})")
    deep = (panel.notna().sum() > 3000).sum()
    print(f"tickers with >3000 bars (true 18yr names): {deep}")
    print("SURVIVOR-ONLY DATA: fail = conclusive, pass = provisional.")

    for kind in ([a.kind] if a.kind else list(LOOKBACKS)):
        d = run_kind(panel, kind)
        if d.empty:
            print(f"\n{kind}: no valid rebalances -- data problem, stop.")
            continue
        report(d, kind)
        _tag = (("_null" if NULL_RUN else "") + ("_ew" if FACTOR == "ew" else "")
                + (f"_cap{BUCKET_CAP}" if BUCKET_CAP else ""))
        out = ROOT / "reports" / f"momentum_18yr_{kind}{_tag}.csv"
        d.to_csv(out, index=False)
        print(f"\n  per-rebalance rows -> {out}")

    print("\n" + "=" * 78)
    print("  GATE: beta-stripped NET t>=3 and mean>0, across ~18 year-clusters")
    print("  including 2009. FAIL kills the 12-week shadow wait tonight.")
    print("=" * 78)


if __name__ == "__main__":
    main()
