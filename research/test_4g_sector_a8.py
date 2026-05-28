"""
4G experiment: does sector-relative A8 ranking beat universe-wide ranking?
Computes 5d-forward returns across the full A8 panel period, joins sector,
compares top-picks selected universe-wide vs within-sector.
READ-ONLY analysis — writes nothing to production.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
import numpy as np
from features.builder import SECTOR_ETF_MAP, SECTOR_ETF

# 1. Load A8 panel
panel = pd.read_parquet("data/a8_oos_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
print(f"A8 panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
      f"{panel['date'].min().date()} to {panel['date'].max().date()}")

# 2. Attach sector
panel["sector"] = panel["ticker"].map(lambda t: SECTOR_ETF_MAP.get(t, SECTOR_ETF))

# 3. Get 5d-forward returns from Polygon (cached macro fetch per ticker)
from features.builder import _get_macro_cached
tickers = panel["ticker"].unique().tolist()
print(f"Fetching prices for {len(tickers)} tickers...")

fwd_returns = {}  # (ticker, date) -> 5d fwd return
for i, t in enumerate(tickers):
    try:
        px = _get_macro_cached(t, "2020-01-01", "2026-05-27")
        if px is None or len(px) == 0:
            continue
        # px is a Series or DataFrame of close prices indexed by date
        close = px["Close"]
        close = close.sort_index()
        fwd = close.shift(-5) / close - 1.0  # 5-day forward return
        for d, r in fwd.items():
            if pd.notna(r):
                fwd_returns[(t, pd.Timestamp(d).normalize())] = r
    except Exception as e:
        print(f"  {t}: {str(e)[:40]}")
    if (i+1) % 30 == 0:
        print(f"  ...{i+1}/{len(tickers)}")

print(f"Forward returns computed: {len(fwd_returns):,} (ticker,date) points")

# 4. Join forward returns to panel
panel["fwd_5d"] = panel.apply(
    lambda row: fwd_returns.get((row["ticker"], row["date"].normalize()), np.nan), axis=1)
panel = panel.dropna(subset=["fwd_5d"])
print(f"Panel with forward returns: {len(panel):,} rows")

# 5. For each date: universe top-10 vs sector top (1 per sector)
TOP_N = 10
uni_rets, sec_rets = [], []
for date, g in panel.groupby("date"):
    if len(g) < 20:
        continue
    # Universe: top 10 by a8_prob
    uni_top = g.nlargest(TOP_N, "a8_prob")
    uni_rets.append(uni_top["fwd_5d"].mean())
    # Sector-relative: top 1 per sector, then take top 10 of those by a8_prob
    sec_top = g.loc[g.groupby("sector")["a8_prob"].idxmax()]
    sec_top = sec_top.nlargest(TOP_N, "a8_prob")
    sec_rets.append(sec_top["fwd_5d"].mean())

uni_mean = np.mean(uni_rets)
sec_mean = np.mean(sec_rets)
print("\n" + "="*55)
print(f"4G RESULT (over {len(uni_rets)} trading days, full panel period):")
print(f"  Universe-wide top-{TOP_N} mean 5d fwd return: {uni_mean:+.4f} ({uni_mean*100:+.2f}%)")
print(f"  Sector-relative top-{TOP_N} mean 5d fwd return: {sec_mean:+.4f} ({sec_mean*100:+.2f}%)")
print(f"  Difference (sector - universe): {(sec_mean-uni_mean)*100:+.3f}pp per 5d")
print("="*55)
if sec_mean > uni_mean:
    print("  → Sector-relative WINS. 4G hypothesis supported.")
else:
    print("  → Universe-wide wins or ties. 4G hypothesis NOT supported.")
