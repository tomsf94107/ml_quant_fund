import sqlite3
import numpy as np, pandas as pd

con = sqlite3.connect('accuracy.db')
# h5 BUYs + their returns + the SAME-DAY universe return (market proxy) to subtract
q = """
SELECT p.prediction_date, p.ticker, p.prob_up, o.actual_return
FROM predictions p
JOIN outcomes o ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.signal='BUY' AND p.horizon=5 AND p.prediction_date>=date('now','-90 days')
"""
buys = pd.read_sql(q, con)

# market proxy = average h5 return across the WHOLE universe on each prediction_date
q2 = """
SELECT o.prediction_date, AVG(o.actual_return) AS mkt_ret, COUNT(*) AS n_univ
FROM outcomes o
WHERE o.horizon=5 AND o.prediction_date>=date('now','-90 days')
GROUP BY o.prediction_date
"""
mkt = pd.read_sql(q2, con)
con.close()

df = buys.merge(mkt, on='prediction_date', how='left')
df['excess'] = df['actual_return'] - df['mkt_ret']   # market-neutral excess return

def summ(g, label, col):
    if len(g) < 10: print(f"  {label}: too few"); return
    r = g[col]
    t = r.mean()/r.std()*np.sqrt(len(r)) if r.std()>0 else 0
    print(f"  {label:24s} n={len(g):>4} mean={r.mean()*100:+.3f}% median={r.median()*100:+.3f}% win={100*(r>0).mean():.0f}% t={t:+.2f}")

print("=== h5 BUYs: RAW vs MARKET-NEUTRAL (excess over universe avg) ===")
print("\n-- all h5 BUYs --")
summ(df, "RAW return", "actual_return")
summ(df, "EXCESS (mkt-neutral)", "excess")
print("\n-- h5 BUYs at prob_up >= 0.68 (the better threshold) --")
hi = df[df.prob_up >= 0.68]
summ(hi, "RAW return", "actual_return")
summ(hi, "EXCESS (mkt-neutral)", "excess")

# how much of raw return was beta?
print(f"\n  avg market (universe h5) return over window: {df['mkt_ret'].mean()*100:+.3f}%")
print(f"  -> if EXCESS mean ~0 and t<2: edge was BETA. if EXCESS positive and t>2: real ALPHA.")
