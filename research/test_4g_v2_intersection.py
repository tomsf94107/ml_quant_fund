"""
4G v2: does INTERSECTION (universe-top AND sector-top) beat pure universe-wide?
Dual-confirmation conviction filter. Reuses 4G's price-fetch logic.
READ-ONLY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd, numpy as np
from features.builder import SECTOR_ETF_MAP, SECTOR_ETF, _get_macro_cached

panel = pd.read_parquet("data/a8_oos_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
panel["sector"] = panel["ticker"].map(lambda t: SECTOR_ETF_MAP.get(t, SECTOR_ETF))

tickers = panel["ticker"].unique().tolist()
print(f"Fetching prices for {len(tickers)} tickers...")
fwd = {}
for i, t in enumerate(tickers):
    try:
        px = _get_macro_cached(t, "2020-01-01", "2026-05-27")
        if px is None or len(px) == 0: continue
        c = px["Close"].sort_index()
        f = c.shift(-5)/c - 1.0
        for d, r in f.items():
            if pd.notna(r): fwd[(t, pd.Timestamp(d).normalize())] = r
    except Exception: pass
    if (i+1) % 30 == 0: print(f"  ...{i+1}/{len(tickers)}")

panel["fwd_5d"] = panel.apply(lambda r: fwd.get((r["ticker"], r["date"].normalize()), np.nan), axis=1)
panel = panel.dropna(subset=["fwd_5d"])

TOP_N = 10
uni_rets, inter_rets, inter_counts = [], [], []
for date, g in panel.groupby("date"):
    if len(g) < 20: continue
    uni_top = set(g.nlargest(TOP_N, "a8_prob")["ticker"])
    uni_rets.append(g[g["ticker"].isin(uni_top)]["fwd_5d"].mean())
    # sector-top: top 2 per sector (so intersection isn't too thin)
    sec_top = set(g.groupby("sector", group_keys=False).apply(
        lambda x: x.nlargest(2, "a8_prob"))["ticker"])
    inter = uni_top & sec_top
    inter_counts.append(len(inter))
    if inter:
        inter_rets.append(g[g["ticker"].isin(inter)]["fwd_5d"].mean())

uni_m = np.mean(uni_rets)
inter_m = np.mean(inter_rets)
print("\n" + "="*55)
print(f"4G v2 INTERSECTION ({len(uni_rets)} days):")
print(f"  Universe-wide top-{TOP_N}:        {uni_m*100:+.2f}%")
print(f"  Intersection (uni AND sector): {inter_m*100:+.2f}%  (avg {np.mean(inter_counts):.1f} names/day)")
print(f"  Difference: {(inter_m-uni_m)*100:+.3f}pp")
print("="*55)
print("  → Intersection WINS" if inter_m > uni_m else "  → Universe-wide still wins")
