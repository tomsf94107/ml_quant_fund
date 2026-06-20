import sqlite3
import numpy as np, pandas as pd

con = sqlite3.connect('accuracy.db')
q = """
SELECT p.horizon, p.prediction_date, p.ticker, o.actual_return
FROM predictions p
JOIN outcomes o ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.signal != 'HOLD' AND p.prediction_date >= date('now','-90 days')
"""
df = pd.read_sql(q, con); con.close()
df['prediction_date'] = pd.to_datetime(df['prediction_date'])

for h in (1, 3, 5):
    g = df[df.horizon == h]
    if len(g) == 0: continue
    print(f"\n=== h={h} ===")
    # 1. is the per-trade return driven by outliers? (median vs mean, top contributor)
    r = g['actual_return']
    print(f"  n={len(r)}  mean={r.mean()*100:+.3f}%  MEDIAN={r.median()*100:+.3f}%  std={r.std()*100:.2f}%")
    print(f"  min={r.min()*100:+.1f}%  max={r.max()*100:+.1f}%  top-3 trades: {sorted(r*100, reverse=True)[:3]}")
    # if mean >> median, a few big winners carry it
    # 2. how many DISTINCT days had signals? (breadth / deployability)
    print(f"  distinct signal-days: {g['prediction_date'].nunique()} (over ~63 trading days)")
    print(f"  avg BUYs per active day: {len(g)/g['prediction_date'].nunique():.1f}")
    # 3. PORTFOLIO daily return: equal-weight all BUYs each day (realistic, no compounding magic)
    daily = g.groupby('prediction_date')['actual_return'].mean()  # EW portfolio per day
    print(f"  EW-portfolio per-signal-day: mean={daily.mean()*100:+.3f}% median={daily.median()*100:+.3f}%")
    print(f"  days positive: {(daily>0).sum()}/{len(daily)} ({100*(daily>0).mean():.0f}%)")
    # net of cost at portfolio level
    net_daily = daily - 0.002
    print(f"  net EW-portfolio mean/active-day: {net_daily.mean()*100:+.3f}%")
