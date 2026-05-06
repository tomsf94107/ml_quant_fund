"""
Feature correlation cull diagnostic.
Identifies redundant features (rho > 0.7) in prediction_features table.
"""
import sqlite3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
DB_PATH = "accuracy.db"
OUTPUT_DIR = Path("logs/feature_audits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

print("Loading prediction_features table...")
df = pd.read_sql("SELECT * FROM prediction_features", conn)
conn.close()

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['prediction_date'].min()} -> {df['prediction_date'].max()}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Horizons: {sorted(df['horizon'].unique())}")
print()

metadata_cols = {'ticker', 'prediction_date', 'horizon', 'id', 'created_at',
                 'prediction_id', 'feature_id', 'rowid'}
feat_cols = [c for c in df.columns if c not in metadata_cols]
print(f"Feature columns: {len(feat_cols)}")
print()

nan_only = [c for c in feat_cols if df[c].isna().all()]
constant = [c for c in feat_cols if df[c].nunique() <= 1]
if nan_only:
    print(f"All-NaN columns (excluded): {nan_only}")
if constant:
    print(f"Constant columns (excluded): {constant}")

feat_cols_clean = [c for c in feat_cols if c not in nan_only and c not in constant]
print(f"Features used in correlation matrix: {len(feat_cols_clean)}")
print()

print("Computing correlation matrix...")
corr = df[feat_cols_clean].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

pairs = []
for col in upper.columns:
    for other in upper.index:
        c = upper.loc[other, col]
        if pd.notna(c) and c > THRESHOLD:
            pairs.append((col, other, c))

pairs.sort(key=lambda x: -x[2])

print(f"\n{'='*70}")
print(f"FEATURE PAIRS WITH |rho| > {THRESHOLD}")
print(f"{'='*70}")
print(f"Total redundant pairs: {len(pairs)}")
print()

if not pairs:
    print("No redundant pairs found above threshold. Feature set is healthy.")
else:
    print(f"{'Feature A':<32s} {'Feature B':<32s} {'rho':>6s}")
    print("-" * 72)
    for a, b, c in pairs:
        print(f"{a:<32s} {b:<32s} {c:>6.3f}")

print(f"\n{'='*70}")
print("FEATURES MOST INVOLVED IN REDUNDANCY")
print(f"{'='*70}")
involvement = {}
for a, b, c in pairs:
    involvement[a] = involvement.get(a, 0) + 1
    involvement[b] = involvement.get(b, 0) + 1

ranked = sorted(involvement.items(), key=lambda x: -x[1])
for feat, count in ranked[:20]:
    print(f"  {feat:<32s} appears in {count} redundant pair(s)")

ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"feature_correlation_cull_{ts}.txt"
with open(output_file, 'w') as f:
    f.write(f"Feature correlation cull diagnostic\n")
    f.write(f"Run time: {pd.Timestamp.now()}\n")
    f.write(f"DB: {DB_PATH}\n")
    f.write(f"Rows: {len(df):,}\n")
    f.write(f"Features analyzed: {len(feat_cols_clean)}\n")
    f.write(f"Threshold: {THRESHOLD}\n")
    f.write(f"Redundant pairs: {len(pairs)}\n\n")
    f.write(f"{'='*70}\n")
    f.write(f"ALL REDUNDANT PAIRS\n")
    f.write(f"{'='*70}\n\n")
    for a, b, c in pairs:
        f.write(f"{a:<32s}  {b:<32s}  {c:>6.3f}\n")
    f.write(f"\n{'='*70}\n")
    f.write(f"INVOLVEMENT COUNTS\n")
    f.write(f"{'='*70}\n\n")
    for feat, count in ranked:
        f.write(f"{feat:<32s}  {count}\n")

print(f"\nFull report saved: {output_file}")
