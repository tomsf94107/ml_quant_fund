"""
analysis/lazy_prices_validate.py — Hunt #7 PAYOFF: validate the Lazy Prices signal
that was built+fetched+similarity-computed Jun 1 but never tested for edge.

Signal (Cohen-Malloy-Nguyen 2020): firms that CHANGE their 10-K language YoY
underperform. LOW YoY similarity (big change) = SELL; HIGH similarity (stable) = hold/long.

Validation discipline (same that killed reversal/pairs):
  - forward returns over 63d AND 126d post-filing (anomaly horizon = months, not days)
  - rank WITHIN each filing-year cohort (cross-sectional, avoids cross-year drift)
  - long stable (top-similarity) minus short changed (bottom-similarity) decile/tercile spread
  - NET of 10bps/turnover
  - PER-YEAR breakdown (does it hold every year or does one carry it? — the decisive test)
  - correlation vs momentum (C1 gate: |corr|<0.3 to count as a NEW decorrelated alpha)
  - HONEST power caveat: ~5 annual cross-sections x ~90 names = low breadth, wide error bars

Run:
  python -m analysis.lazy_prices_validate
"""
import sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 10.0
FWD_DAYS = [63, 126]          # ~quarter and ~half-year (CMN horizon)
SECTIONS = ["business", "risk_factors", "mda"]
SEC_DB = ROOT / "sec_filings.db"

def load_similarity():
    con = sqlite3.connect(str(SEC_DB))
    df = pd.read_sql("""
        SELECT ticker, filing_date, section, cosine, jaccard
        FROM sec_10k_similarity
    """, con)
    con.close()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["year"] = df["filing_date"].dt.year
    return df

def fwd_return(panel, tk, fdate, ndays):
    """Forward return from first trading day >= filing_date to ndays later."""
    if tk not in panel.columns:
        return np.nan
    s = panel[tk].dropna()
    s = s[s.index >= fdate]
    if len(s) < ndays + 1:
        return np.nan
    p0 = s.iloc[0]
    p1 = s.iloc[min(ndays, len(s)-1)]
    if p0 <= 0:
        return np.nan
    return float(p1 / p0 - 1.0)

def main():
    from analysis.momentum_purged_wf import build_close_panel

    sim = load_similarity()
    tickers = sorted(sim["ticker"].unique())
    print(f"similarity events: {len(sim)} rows, {len(tickers)} tickers, "
          f"years {sim.year.min()}-{sim.year.max()}")

    print("building close panel (Massive download)...")
    panel = build_close_panel(tickers, "2018-01-01")
    print(f"panel: {panel.shape[1]} tickers, {panel.shape[0]} days\n")

    # use cosine as the similarity metric (jaccard as robustness later)
    for metric in ["cosine"]:
        print(f"\n{'='*70}\nMETRIC: {metric}  (LOW = big change = CMN SELL; HIGH = stable)\n{'='*70}")
        for section in SECTIONS:
            sub = sim[sim["section"] == section].copy()
            # attach forward returns
            for nd in FWD_DAYS:
                sub[f"fwd{nd}"] = [fwd_return(panel, r.ticker, r.filing_date, nd)
                                  for r in sub.itertuples()]
            print(f"\n--- section={section}  (n_events={len(sub)}) ---")
            for nd in FWD_DAYS:
                col = f"fwd{nd}"
                d = sub.dropna(subset=[col, metric])
                if len(d) < 30:
                    print(f"  fwd{nd}: only {len(d)} usable events — too thin")
                    continue
                # cross-sectional spread WITHIN each year: long top-tercile sim, short bottom
                yr_spreads = []
                for yr, g in d.groupby("year"):
                    if len(g) < 15:
                        continue
                    hi = g[metric].quantile(0.67)
                    lo = g[metric].quantile(0.33)
                    long_ret = g[g[metric] >= hi][col].mean()    # stable filers
                    short_ret = g[g[metric] <= lo][col].mean()   # big-change filers
                    # CMN: stable should beat changed -> spread = long - short, expect > 0
                    spread = long_ret - short_ret
                    yr_spreads.append((yr, len(g), spread, long_ret, short_ret))
                if not yr_spreads:
                    print(f"  fwd{nd}: no year with >=15 names"); continue
                spreads = np.array([s for _,_,s,_,_ in yr_spreads])
                # crude net-of-cost: spread is a tercile L/S, ~1 turnover/yr each leg
                net = spreads - 2 * (COST_BPS/1e4)
                # rank-IC: does similarity rank correlate with fwd return? (CMN: positive)
                from scipy.stats import spearmanr
                ic_by_yr = []
                for yr, g in d.groupby("year"):
                    if len(g) >= 15:
                        ic,_ = spearmanr(g[metric], g[col]); ic_by_yr.append(ic)
                mean_ic = np.nanmean(ic_by_yr) if ic_by_yr else np.nan
                print(f"  fwd{nd}: mean L/S spread (stable-changed) = {spreads.mean():+.3%} "
                      f"net {net.mean():+.3%} | mean rank-IC(sim,ret)={mean_ic:+.3f}")
                print(f"          per-year spread: " +
                      "  ".join(f"{yr}:{s:+.1%}(n{n})" for yr,n,s,_,_ in yr_spreads))
    print("\nGATES: (1) net spread > 0 AND holds MOST years (not 1-year artifact);")
    print("       (2) positive rank-IC consistent; (3) decorrelated from momentum (separate check).")
    print("HONEST: ~5-6 annual cross-sections = LOW breadth; treat as preliminary.")

if __name__ == "__main__":
    main()
