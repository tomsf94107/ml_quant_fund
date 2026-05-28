"""4G v2 rigor: Sharpe + sub-period stability + thin-day check for intersection."""
import sys; from pathlib import Path
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
        if px is None or len(px)==0: continue
        c = px["Close"].sort_index(); f = c.shift(-5)/c - 1.0
        for d, r in f.items():
            if pd.notna(r): fwd[(t, pd.Timestamp(d).normalize())] = r
    except Exception: pass
    if (i+1)%30==0: print(f"  ...{i+1}/{len(tickers)}")

panel["fwd_5d"] = panel.apply(lambda r: fwd.get((r["ticker"], r["date"].normalize()), np.nan), axis=1)
panel = panel.dropna(subset=["fwd_5d"]).sort_values("date")

rows = []
for date, g in panel.groupby("date"):
    if len(g) < 20: continue
    uni = set(g.nlargest(10, "a8_prob")["ticker"])
    sec = set(g.groupby("sector", group_keys=False).apply(lambda x: x.nlargest(2,"a8_prob"), include_groups=False)["ticker"])
    inter = uni & sec
    rows.append({
        "date": date,
        "uni_ret": g[g["ticker"].isin(uni)]["fwd_5d"].mean(),
        "inter_ret": g[g["ticker"].isin(inter)]["fwd_5d"].mean() if inter else np.nan,
        "inter_n": len(inter),
    })
r = pd.DataFrame(rows).dropna(subset=["inter_ret"])

def stats(x):
    return f"mean={x.mean()*100:+.2f}% std={x.std()*100:.2f}% sharpe={x.mean()/x.std():.3f}"

print("\n" + "="*60)
print("4G v2 RIGOR CHECK")
print("="*60)
print(f"Days: {len(r)}, avg intersection size: {r['inter_n'].mean():.1f}")
print(f"Thin days (<=3 names): {(r['inter_n']<=3).sum()} ({(r['inter_n']<=3).mean()*100:.0f}%)")
print(f"\nFULL PERIOD:")
print(f"  Universe:     {stats(r['uni_ret'])}")
print(f"  Intersection: {stats(r['inter_ret'])}")
# Sub-period split
mid = r["date"].quantile(0.5)
h1, h2 = r[r["date"]<=mid], r[r["date"]>mid]
print(f"\nH1 ({h1['date'].min().date()} to {h1['date'].max().date()}):")
print(f"  Universe:     {stats(h1['uni_ret'])}")
print(f"  Intersection: {stats(h1['inter_ret'])}")
print(f"H2 ({h2['date'].min().date()} to {h2['date'].max().date()}):")
print(f"  Universe:     {stats(h2['uni_ret'])}")
print(f"  Intersection: {stats(h2['inter_ret'])}")
print("="*60)
won = (r['inter_ret'].mean() > r['uni_ret'].mean())
h1_won = (h1['inter_ret'].mean() > h1['uni_ret'].mean())
h2_won = (h2['inter_ret'].mean() > h2['uni_ret'].mean())
print(f"Intersection beats universe: full={won} H1={h1_won} H2={h2_won}")
print("STABLE WIN" if (won and h1_won and h2_won) else "INCONSISTENT — be cautious")
