import sqlite3
import numpy as np, pandas as pd

con = sqlite3.connect('accuracy.db')
# BUY signals (long-only) + their realized returns, last 90 days
q = """
SELECT p.horizon, p.prediction_date, o.actual_return
FROM predictions p
JOIN outcomes o ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.signal != 'HOLD' AND p.prediction_date >= date('now','-90 days')
"""
df = pd.read_sql(q, con)
con.close()

# round-trip cost assumption: 20bps (10 each way). conservative for liquid US.
COST = 0.0020
# trading days per year, approx trades/yr if continuously redeployed at this horizon
TRADES_PER_YR = {1: 252, 3: 84, 5: 50}

print(f"{'h':>2} {'n':>5} {'win%':>6} {'avg_ret%':>9} {'net_ret%':>9} {'ann_net%':>9} {'perTradeSharpe':>14}")
for h in (1, 3, 5):
    g = df[df.horizon == h]['actual_return'].dropna()
    if len(g) == 0:
        print(f"{h:>2}   no data"); continue
    n = len(g)
    win = (g > 0).mean() * 100
    avg = g.mean()                      # gross avg return per trade
    net = avg - COST                    # net of round-trip cost
    tpy = TRADES_PER_YR[h]
    ann_net = (1 + net) ** tpy - 1      # annualized if continuously redeployed
    # per-trade Sharpe (not annualized) + annualized Sharpe
    sharpe_trade = g.mean() / g.std() if g.std() > 0 else 0
    sharpe_ann = sharpe_trade * np.sqrt(tpy)
    print(f"{h:>2} {n:>5} {win:>6.1f} {avg*100:>9.3f} {net*100:>9.3f} {ann_net*100:>9.1f} {sharpe_ann:>14.2f}")

print()
print("COST assumption: 20bps round-trip. TRADES/YR: h1=252, h3=84, h5=50.")
print("ann_net = net-per-trade compounded at trades/yr (assumes continuous redeploy).")
print("Sharpe = annualized (per-trade Sharpe x sqrt(trades/yr)).")
print("NOTE: h1 n is small -> wide error bars; treat h1 ann figures with caution.")
