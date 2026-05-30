"""
Expanded spot-check for institutional features.

Pulls 20 BUY HIGH signals (8 winners, 8 losers, 4 controls), runs
features.institutional_features.get_institutional_features() on each
as of prediction_date, joins to outcomes, and prints:
  - win vs loss means per feature (the discrimination test)
  - feature-feature correlation matrix (the redundancy test)
"""
import sqlite3
import pandas as pd
import numpy as np
from features.institutional_features import get_institutional_features

ACC_DB = "accuracy.db"

# 1. Pull stratified sample from predictions JOIN outcomes
con = sqlite3.connect(ACC_DB)
sql = """
SELECT p.ticker, p.prediction_date, p.horizon, p.prob_up,
       o.actual_return
FROM predictions p
JOIN outcomes o
  ON p.ticker = o.ticker
 AND p.prediction_date = o.prediction_date
 AND p.horizon = o.horizon
WHERE p.prediction_date BETWEEN '2026-04-15' AND '2026-05-14'
  AND p.prob_up >= 0.70
  AND p.signal = 'BUY'
ORDER BY p.prediction_date
"""
df = pd.read_sql(sql, con)
con.close()

print(f"Universe of BUY HIGH signals in window: {len(df)}")

# Stratified sample
winners = df[df['actual_return'] >= 0.05].sample(min(8, len(df[df['actual_return'] >= 0.05])), random_state=42)
losers  = df[df['actual_return'] <= -0.03].sample(min(8, len(df[df['actual_return'] <= -0.03])), random_state=42)
mids    = df[(df['actual_return'] > -0.03) & (df['actual_return'] < 0.05)].sample(min(4, len(df[(df['actual_return'] > -0.03) & (df['actual_return'] < 0.05)])), random_state=42)

sample = pd.concat([winners, losers, mids]).reset_index(drop=True)
sample['outcome_bucket'] = sample['actual_return'].apply(
    lambda r: 'WIN' if r >= 0.05 else ('LOSS' if r <= -0.03 else 'MID')
)
print(f"Sample: {len(winners)} wins, {len(losers)} losses, {len(mids)} mids")

# 2. Compute features for each
FEAT_COLS = [
    'inst_signed_flow_5d', 'inst_block_buy_sell_7d', 'inst_sweep_count_7d',
    'inst_auction_imbal_5d', 'inst_signed_flow_30d',
    'inst_block_notional_7d', 'inst_block_count_7d',
]
# NOTE: dp_signed_flow_5d intentionally excluded — redundant per audit

rows = []
for _, r in sample.iterrows():
    feats = get_institutional_features(r['ticker'], r['prediction_date'])
    row = {'ticker': r['ticker'], 'date': r['prediction_date'],
           'outcome_bucket': r['outcome_bucket'], 'actual_ret': r['actual_return']}
    for f in FEAT_COLS:
        row[f] = feats.get(f)
    rows.append(row)
feat_df = pd.DataFrame(rows)
print("\n=== Raw feature values ===")
print(feat_df.to_string(index=False))

# 3. Win vs Loss discrimination
print("\n=== Win vs Loss means (the 'is it signal?' test) ===")
summary = feat_df.groupby('outcome_bucket')[FEAT_COLS].agg(['mean', 'std', 'count'])
print(summary.to_string())

# 4. Win-loss delta (positive = feature higher in winners, what we'd want)
wins = feat_df[feat_df['outcome_bucket']=='WIN'][FEAT_COLS].mean()
losses = feat_df[feat_df['outcome_bucket']=='LOSS'][FEAT_COLS].mean()
delta = (wins - losses).sort_values(ascending=False)
print("\n=== Mean delta (winners − losers) ===")
print("Positive delta = feature is higher in winners (signal)")
print("Near zero = no discrimination")
print(delta.round(4).to_string())

# 5. Correlation matrix
print("\n=== Feature-feature correlation (redundancy check) ===")
corr = feat_df[FEAT_COLS].corr()
print(corr.round(2).to_string())
print("\nPairs with |corr| > 0.85 (candidates to drop):")
for i, c1 in enumerate(FEAT_COLS):
    for c2 in FEAT_COLS[i+1:]:
        v = corr.loc[c1, c2]
        if abs(v) > 0.85:
            print(f"  {c1:30s} ~ {c2:30s} ρ={v:+.2f}")
