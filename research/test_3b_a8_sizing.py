"""
3B backtest: does scaling position size by A8 prob improve Sharpe?
Spec gate: Sharpe improvement >= 0.1 to ship.
Compares equal-weight top-N vs A8-multiplier-weighted top-N.
READ-ONLY — proves/disproves before any production wiring.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd, numpy as np
from features.builder import _get_macro_cached

panel = pd.read_parquet("data/a8_oos_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
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

def a8_mult(p):
    if p > 0.5: return 2.0
    if p < 0.2: return 0.5
    return 1.0

TOP_N = 10
base_rets, a8_rets = [], []
for date, g in panel.groupby("date"):
    if len(g) < 20: continue
    top = g.nlargest(TOP_N, "a8_prob").copy()
    # Baseline: equal weight
    base_rets.append(top["fwd_5d"].mean())
    # A8-scaled: weight by a8_mult, normalized
    top["w"] = top["a8_prob"].map(a8_mult)
    top["w"] /= top["w"].sum()
    a8_rets.append((top["fwd_5d"] * top["w"]).sum())

def sharpe(x): 
    x = np.array(x)
    return x.mean()/x.std() if x.std()>0 else 0.0

bs, a8s = sharpe(base_rets), sharpe(a8_rets)
import numpy as np
print("\n" + "="*55)
print(f"3B A8-SIZING BACKTEST ({len(base_rets)} days, per-5d Sharpe)")
print("="*55)
print(f"  Baseline (equal-weight):  mean={np.mean(base_rets)*100:+.2f}%  sharpe={bs:.4f}")
print(f"  A8-scaled weights:        mean={np.mean(a8_rets)*100:+.2f}%  sharpe={a8s:.4f}")
print(f"  Sharpe delta: {a8s-bs:+.4f}")
print(f"  GATE (>= 0.1 per-5d... but spec likely means annualized):")
print(f"    per-5d delta {a8s-bs:+.4f} → annualized ~{(a8s-bs)*7.07:+.3f}")
print("="*55)
# Sub-period
mid = panel["date"].quantile(0.5)
print("(check sub-period stability next if base result is promising)")
