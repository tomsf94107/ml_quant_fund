import sqlite3
import numpy as np, pandas as pd
con = sqlite3.connect('accuracy.db')
q = """SELECT p.prob_up, o.actual_return, o.actual_up
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.signal='BUY' AND p.horizon=5 AND p.prediction_date>=date('now','-90 days')"""
df = pd.read_sql(q, con); con.close()

def stats(g, label):
    if len(g)<10: print(f"  {label}: too few (n={len(g)})"); return
    print(f"  {label:22s} n={len(g):>4} acc={g.actual_up.mean()*100:>5.1f}% net/trade={g.actual_return.mean()*100-0.2:>6.2f}% median={g.actual_return.median()*100:+.2f}%")

print("=== h5 BUY: current default vs proposed band ===")
stats(df[df.prob_up>=0.60], "current (>=0.60)")
stats(df[df.prob_up>=0.70], "floor (>=0.70)")
stats(df[(df.prob_up>=0.65)&(df.prob_up<=0.75)], "BAND [0.65,0.75]")
stats(df[(df.prob_up>=0.65)&(df.prob_up<0.78)], "BAND [0.65,0.78)")
stats(df[df.prob_up>0.78], "overconfident tail (>0.78)")
