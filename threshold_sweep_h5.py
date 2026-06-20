import sqlite3
import numpy as np, pandas as pd
con = sqlite3.connect('accuracy.db')
q = """SELECT p.prob_up, o.actual_return, o.actual_up
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.signal='BUY' AND p.horizon=5 AND p.prediction_date>=date('now','-90 days')"""
df = pd.read_sql(q, con); con.close()
print(f"h5 BUYs total: {len(df)}")
print(f"{'cutoff':>7} {'n':>5} {'acc%':>6} {'avg_ret%':>9} {'net/trade%':>11}")
for cut in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    g = df[df.prob_up >= cut]
    if len(g) < 10: 
        print(f"{cut:>7} {len(g):>5}  (too few)"); continue
    acc = g.actual_up.mean()*100
    avg = g.actual_return.mean()*100
    net = avg - 0.20
    print(f"{cut:>7} {len(g):>5} {acc:>6.1f} {avg:>9.3f} {net:>11.3f}")
