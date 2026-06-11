"""
analysis/insider_event_study.py - R2 insider build: CMP-adapted EVENT STUDY.
(Cross-sectional monthly book is sample-starved at 137 mega-caps: only ~2.2K
open-market buys in 7yr. Pre-registered pivot: event-level study.)

DESIGN:
  1. ROUTINE classification (CMP): a buyer is routine if they bought in the same
     calendar month in 3+ consecutive years. Their buys are discarded.
  2. EVENTS: remaining (opportunistic) open-market buys (code P, acquired A),
     collapsed to (ticker, trade_date) with total notional + n_insiders that day.
  3. FORWARD EXCESS RETURN: ticker fwd 20d/60d return MINUS SPY same-window
     (market-excess; PIT uses filing_date+1 as entry, not trade_date — we only
     know about a buy once it is FILED).
  4. CLUSTER variant: 2+ distinct insiders buying same ticker within 21 calendar
     days — literature's strongest flavor.
GATES (pre-registered): mean excess t-stat gt 2 at 20d or 60d; sign consistent
  in most years; n gte 300 events post-classification. Cluster reported separately.
Run: python -m analysis.insider_event_study
"""
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def classify_routine(buys):
    """CMP: routine buyer = same calendar month in 3+ consecutive years."""
    routine_keys = set()
    g = buys.copy()
    g["yr"] = g.trade_date.str[:4].astype(int)
    g["mo"] = g.trade_date.str[5:7].astype(int)
    for (tk, ins, mo), sub in g.groupby(["ticker", "insider_name", "mo"]):
        yrs = sorted(sub.yr.unique())
        run = 1
        for a, b in zip(yrs, yrs[1:]):
            run = run + 1 if b == a + 1 else 1
            if run >= 3:
                routine_keys.add((tk, ins))
                break
    return routine_keys


def fwd_excess(panel, spy, ticker, entry_date, horizon):
    if ticker not in panel.columns:
        return np.nan
    px = panel[ticker].dropna()
    loc = px.index.searchsorted(entry_date)
    if loc >= len(px) - horizon:
        return np.nan
    r_t = px.iloc[loc + horizon] / px.iloc[loc] - 1.0
    sloc = spy.index.searchsorted(entry_date)
    if sloc >= len(spy) - horizon:
        return np.nan
    r_s = spy.iloc[sloc + horizon] / spy.iloc[sloc] - 1.0
    return float(r_t - r_s)


def report(ev, label):
    print(f"\n=== {label} (n={len(ev)}) ===")
    if len(ev) < 30:
        print("  too few events"); return
    for h in ("ex20", "ex60"):
        x = ev[h].dropna()
        t = float(x.mean() / (x.std() / np.sqrt(len(x))))
        print(f"  {h}: mean={x.mean():+.2%}  median={x.median():+.2%}  t={t:+.2f}  n={len(x)}")
    yr = ev.groupby(ev.entry.str[:4])["ex20"].mean()
    pos = int((yr > 0).sum())
    print(f"  20d sign by year: {pos}/{len(yr)} positive ->",
          {k: f"{v:+.1%}" for k, v in yr.items()})


def main():
    from analysis.momentum_purged_wf import build_close_panel
    raw = pd.read_sql(
        "SELECT ticker, insider_name, trade_date, filing_date, notional_usd, is_csuite "
        "FROM insider_filings_raw WHERE transaction_code = 'P' AND acquired_disposed = 'A'",
        sqlite3.connect(str(ROOT / "insider_trades.db")))
    print(f"open-market buys: {len(raw)} rows, {raw.ticker.nunique()} tickers")

    routine = classify_routine(raw)
    print(f"routine buyers classified: {len(routine)} (ticker,insider) pairs")
    raw["key"] = list(zip(raw.ticker, raw.insider_name))
    opp = raw[~raw.key.isin(routine)].copy()
    print(f"opportunistic buys: {len(opp)} rows")

    # collapse to (ticker, filing_date) events; entry = filing date (PIT)
    ev = (opp.groupby(["ticker", "filing_date"])
          .agg(notional=("notional_usd", "sum"), n_ins=("insider_name", "nunique"),
               csuite=("is_csuite", "max")).reset_index())
    ev = ev.rename(columns={"filing_date": "entry"})
    print(f"events: {len(ev)}")

    tickers = sorted(ev.ticker.unique())
    panel = build_close_panel(tickers + ["SPY"], "2018-06-01")
    spy = panel["SPY"].dropna()
    ev["ex20"] = [fwd_excess(panel, spy, r.ticker, r.entry, 20) for r in ev.itertuples()]
    ev["ex60"] = [fwd_excess(panel, spy, r.ticker, r.entry, 60) for r in ev.itertuples()]
    ev.to_csv(ROOT / "data/insider_events.csv", index=False)

    report(ev, "ALL opportunistic buys")
    report(ev[ev.csuite == 1], "C-SUITE opportunistic buys")
    # cluster: 2+ distinct insiders within 21 days (count distinct insiders per window)
    opp["fd"] = pd.to_datetime(opp.filing_date)
    cl_keys = set()
    for tk, sub in opp.groupby("ticker"):
        sub = sub.sort_values("fd")
        for i, r in sub.iterrows():
            win = sub[(sub.fd >= r.fd - pd.Timedelta(days=21)) & (sub.fd <= r.fd)]
            if win.insider_name.nunique() >= 2:
                cl_keys.add((tk, r.filing_date))
    ev["cluster"] = [ (r.ticker, r.entry) in cl_keys for r in ev.itertuples() ]
    report(ev[ev.cluster], "CLUSTER (2+ insiders / 21d)")

    print("\nGATES: t gt 2 (20d or 60d) + most-years-positive + n gte 300.")
    print("If PASS -> event OVERLAY (like pre-earnings layer), NOT a standalone book.")
    print("NOTE: expansion adds SP400 names where insider buying is denser -> re-test then.")


if __name__ == "__main__":
    main()
