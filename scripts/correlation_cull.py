#!/usr/bin/env python3
"""
scripts/correlation_cull.py

Identifies redundant features via pairwise correlation. Reports pairs with
|ρ| > threshold (default 0.7) so you can decide which to drop.

Per Finding Alphas Ch. 10 / Gap_Check_and_Roadmap(04292026).md Lever 4.

Does NOT auto-drop. Output is a ranked console report + CSV. You review.

Two modes:
  --from-db (default): reads prediction_features table (27-col subset)
  --from-builder: builds full 83-feature panel via features.builder
                   (slower, covers ALL production features)

Usage:
    python scripts/correlation_cull.py --from-builder --builder-tickers 30
    python scripts/correlation_cull.py --threshold 0.8 --top 20
    python scripts/correlation_cull.py --csv-out reports/correlation_cull.csv
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "accuracy.db"
ID_COLS = {"id", "ticker", "prediction_date", "horizon", "created_at"}
NON_FEATURE_COLS = {
    "date", "ticker", "__ticker__",
    "open", "high", "low", "close", "volume", "adj_close",
    "Open", "High", "Low", "Close", "Volume", "Adj Close",
} | ID_COLS


def load_features_from_db(limit: int | None = None) -> pd.DataFrame:
    """Load prediction_features table (27-col subset)."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        q = "SELECT * FROM prediction_features"
        if limit:
            q += f" LIMIT {limit}"
        df = pd.read_sql(q, conn)
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    return df[feat_cols]


def load_features_from_builder(
    tickers: list[str],
    start_date: str = "2024-01-01",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build full 83-feature panel via features.builder.build_feature_dataframe
    for the given tickers. Stacks them into one long DataFrame.

    Slower than load_features_from_db() but covers ALL production features.
    """
    from features.builder import build_feature_dataframe
    panels = []
    for i, t in enumerate(tickers, 1):
        try:
            df = build_feature_dataframe(t, start_date=start_date)
            if df is None or df.empty:
                if verbose:
                    print(f"  [{i}/{len(tickers)}] {t}: empty, skip")
                continue
            df = df.copy()
            df["__ticker__"] = t
            panels.append(df)
            if verbose:
                print(f"  [{i}/{len(tickers)}] {t}: {len(df)} rows × {len(df.columns)} cols")
        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(tickers)}] {t}: FAILED {e}")
    if not panels:
        return pd.DataFrame()
    big = pd.concat(panels, ignore_index=True)
    feat_cols = [c for c in big.columns if c not in NON_FEATURE_COLS]
    return big[feat_cols]


def compute_correlation_pairs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Returns DataFrame of (feat_a, feat_b, abs_corr, signed_corr, n_obs)
    for pairs where |corr| >= threshold. Sorted by abs_corr desc.
    """
    df_clean = df.dropna(axis=1, how="all")
    df_clean = df_clean.loc[:, df_clean.std() > 1e-10]
    if df_clean.empty:
        return pd.DataFrame(columns=["feat_a", "feat_b", "abs_corr", "signed_corr", "n_obs"])

    corr = df_clean.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            c = corr.loc[a, b]
            if pd.isna(c):
                continue
            if abs(c) >= threshold:
                n_obs = int(df_clean[[a, b]].dropna().shape[0])
                pairs.append({
                    "feat_a":      a,
                    "feat_b":      b,
                    "abs_corr":    abs(c),
                    "signed_corr": c,
                    "n_obs":       n_obs,
                })

    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def suggest_drops(pairs_df: pd.DataFrame) -> list[str]:
    """
    Greedy: from each correlated pair, drop the feature appearing in MORE
    high-correlation pairs (most redundant globally). Tie-break alphabetical.
    """
    if pairs_df.empty:
        return []

    appearances = {}
    for _, row in pairs_df.iterrows():
        appearances[row["feat_a"]] = appearances.get(row["feat_a"], 0) + 1
        appearances[row["feat_b"]] = appearances.get(row["feat_b"], 0) + 1

    drops = set()
    keep = set()

    for _, row in pairs_df.iterrows():
        a, b = row["feat_a"], row["feat_b"]
        if a in drops or b in drops:
            continue
        if a in keep and b in keep:
            continue

        if appearances[a] > appearances[b]:
            drops.add(a); keep.add(b)
        elif appearances[b] > appearances[a]:
            drops.add(b); keep.add(a)
        else:
            if a < b:
                drops.add(b); keep.add(a)
            else:
                drops.add(a); keep.add(b)

    return sorted(drops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="abs correlation threshold (default: 0.7)")
    parser.add_argument("--top", type=int, default=30,
                        help="show top N correlated pairs (default: 30)")
    parser.add_argument("--csv-out", type=str, default="reports/correlation_cull.csv",
                        help="output CSV path")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit DB rows for testing (--from-db mode)")
    parser.add_argument("--from-builder", action="store_true",
                        help="Build full 83-feature panel from features.builder")
    parser.add_argument("--builder-tickers", type=int, default=30,
                        help="Number of tickers to sample for --from-builder (default 30)")
    parser.add_argument("--builder-start", type=str, default="2024-01-01",
                        help="Start date for --from-builder (default 2024-01-01)")
    args = parser.parse_args()

    if args.from_builder:
        tickers_file = Path(__file__).parent.parent / "tickers.txt"
        all_tickers = [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]
        import random
        random.seed(42)
        sampled = random.sample(all_tickers, min(args.builder_tickers, len(all_tickers)))
        print(f"[correlation_cull] Building full 83-feature panel from "
              f"{len(sampled)} tickers (start={args.builder_start})...")
        df = load_features_from_builder(sampled, start_date=args.builder_start)
        print(f"[correlation_cull] {len(df):,} rows × {len(df.columns)} feature cols")
    else:
        print(f"[correlation_cull] Loading prediction_features from {DB_PATH}...")
        df = load_features_from_db(limit=args.limit)
        print(f"[correlation_cull] {len(df):,} rows × {len(df.columns)} feature cols")

    if df.empty:
        print("[correlation_cull] No data. Aborting.")
        return 1

    print(f"[correlation_cull] Computing pairwise correlations "
          f"(threshold |ρ|>={args.threshold})...")
    pairs = compute_correlation_pairs(df, args.threshold)
    print(f"[correlation_cull] Found {len(pairs)} correlated pairs")

    if pairs.empty:
        print("[correlation_cull] No pairs above threshold. Features look diverse.")
        return 0

    print(f"\n=== Top {min(args.top, len(pairs))} most-correlated pairs ===")
    print(f"{'feat_a':25s}  {'feat_b':25s}  {'|ρ|':>6s}  {'signed':>7s}  {'n_obs':>6s}")
    print("-" * 80)
    for _, row in pairs.head(args.top).iterrows():
        print(f"{row['feat_a']:25s}  {row['feat_b']:25s}  "
              f"{row['abs_corr']:6.3f}  {row['signed_corr']:+7.3f}  {row['n_obs']:6d}")

    drops = suggest_drops(pairs)
    if drops:
        print(f"\n=== Suggested drops ({len(drops)} features) ===")
        print("Greedy: feature appearing in most high-corr pairs.")
        print("REVIEW BEFORE DROPPING — some 'redundant' features encode different info.")
        for d in drops:
            count = sum(1 for _, r in pairs.iterrows() if r["feat_a"] == d or r["feat_b"] == d)
            print(f"  - {d:25s} (in {count} high-corr pairs)")

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)
    print(f"\n[correlation_cull] Full report → {out_path}")
    print(f"[correlation_cull] Remaining if drops applied: "
          f"{len(df.columns) - len(drops)}/{len(df.columns)} features")

    return 0


if __name__ == "__main__":
    sys.exit(main())
