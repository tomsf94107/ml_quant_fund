"""
fitness_scorer.py — WorldQuant-style alpha quality metric for ML Quant Fund.

Computes for each (ticker, horizon) ensemble model:
    fitness = sqrt(|annualized_return| / max(turnover, 0.125)) * sharpe

UPDATE: now supports --by-sector flag that aggregates per-ticker fitness up
to per-sector fitness, weighted by sample size. Uses tickers_metadata.csv
(bucket column) — same metadata file the sector accuracy work reads from.

Drop this in: ml_quant_fund/analysis/fitness_scorer.py
Run from project root with the ml_quant_310 env active:

    # Per-ticker leaderboard (default)
    python -m analysis.fitness_scorer --db accuracy.db

    # Per-sector leaderboard (aggregates ticker fitness, weighted by n_obs)
    python -m analysis.fitness_scorer --db accuracy.db --by-sector

    # Both, with output
    python -m analysis.fitness_scorer --db accuracy.db --write-table --csv fitness.csv
    python -m analysis.fitness_scorer --db accuracy.db --by-sector --csv fitness_sector.csv
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HORIZON_DAYS = {1: 1, 3: 3, 5: 5}
TRADING_DAYS_PER_YEAR = 252
MIN_OBS = 30
TURNOVER_FLOOR = 0.125


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FitnessRow:
    ticker: str
    horizon: int
    n_obs: int
    win_rate: float
    avg_return_per_bar: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    turnover: float
    fitness: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(db_path: Path, since: Optional[str] = None) -> pd.DataFrame:
    """Join predictions and outcomes; one row per (ticker, date, horizon)."""
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT p.ticker, p.prediction_date, p.horizon, p.prob_up,
               o.actual_return, o.actual_up
        FROM predictions p
        JOIN outcomes o
          ON p.ticker = o.ticker
         AND p.prediction_date = o.prediction_date
         AND p.horizon = o.horizon
        WHERE o.actual_return IS NOT NULL
    """
    params: list = []
    if since:
        query += " AND p.prediction_date >= ?"
        params.append(since)
    query += " ORDER BY p.ticker, p.horizon, p.prediction_date"
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
        return df
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df = df.dropna(subset=["prob_up", "actual_return"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Position construction
# ---------------------------------------------------------------------------

def position_from_prob(prob_up: pd.Series, mode: str) -> pd.Series:
    """Convert prob_up into a [-1, +1] position."""
    if mode == "long_only":
        return (prob_up > 0.5).astype(float)
    if mode == "long_short":
        return (2.0 * prob_up - 1.0).clip(lower=-1.0, upper=1.0)
    if mode == "binary_ls":
        return np.sign(prob_up - 0.5)
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-group fitness
# ---------------------------------------------------------------------------

def compute_group_fitness(g: pd.DataFrame, mode: str) -> Optional[FitnessRow]:
    """Compute fitness metrics for one (ticker, horizon) group."""
    if len(g) < MIN_OBS:
        return None

    horizon = int(g["horizon"].iloc[0])
    horizon_days = HORIZON_DAYS.get(horizon, horizon if horizon > 0 else 1)

    g = g.sort_values("prediction_date").reset_index(drop=True).copy()
    g["position"] = position_from_prob(g["prob_up"], mode)
    g["strat_ret"] = g["position"] * g["actual_return"]

    active = g[g["position"] != 0]
    if len(active) == 0:
        return None
    win_rate = float((active["strat_ret"] > 0).mean())

    avg_return_per_bar = float(g["strat_ret"].mean())
    bar_vol = float(g["strat_ret"].std(ddof=1)) if len(g) > 1 else 0.0
    bars_per_year = TRADING_DAYS_PER_YEAR / horizon_days
    annualized_return = avg_return_per_bar * bars_per_year
    annualized_vol = bar_vol * math.sqrt(bars_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    pos_diff = g["position"].diff().abs().dropna()
    turnover = float(pos_diff.mean()) if len(pos_diff) > 0 else 0.0

    fitness = math.sqrt(abs(annualized_return) / max(turnover, TURNOVER_FLOOR)) * sharpe

    return FitnessRow(
        ticker=str(g["ticker"].iloc[0]),
        horizon=horizon,
        n_obs=len(g),
        win_rate=win_rate,
        avg_return_per_bar=avg_return_per_bar,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe=sharpe,
        turnover=turnover,
        fitness=fitness,
    )


# ---------------------------------------------------------------------------
# Driver — per-ticker
# ---------------------------------------------------------------------------

def score_all(db_path: Path, mode: str = "long_only",
              since: Optional[str] = None) -> pd.DataFrame:
    """Score every (ticker, horizon) model. Returns ranked DataFrame."""
    df = load_predictions(db_path, since=since)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, group in df.groupby(["ticker", "horizon"], sort=False):
        result = compute_group_fitness(group, mode=mode)
        if result is not None:
            rows.append(asdict(result))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("fitness", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Driver — per-sector (NEW)
# ---------------------------------------------------------------------------

def aggregate_to_sector(
    ticker_fitness: pd.DataFrame,
    metadata_csv: Path,
) -> pd.DataFrame:
    """Aggregate per-(ticker, horizon) fitness to per-(sector, horizon).

    Each metric is weighted by n_obs so sectors with high-volume tickers
    don't get drowned out by tickers with very few observations.

    Returns a DataFrame ranked by sector fitness descending.
    """
    if ticker_fitness.empty:
        return pd.DataFrame()

    if not Path(metadata_csv).exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_csv}. "
            f"Expected schema: ticker,bucket"
        )

    meta = pd.read_csv(metadata_csv)
    meta["ticker"] = meta["ticker"].str.upper().str.strip()
    if "bucket" not in meta.columns:
        raise ValueError(
            f"Column 'bucket' not in {metadata_csv}. Available: {list(meta.columns)}"
        )

    df = ticker_fitness.copy()
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df = df.merge(meta[["ticker", "bucket"]], on="ticker", how="left")
    df["bucket"] = df["bucket"].fillna("UNKNOWN")

    # Weighted aggregation: each metric weighted by that row's n_obs
    def weighted_avg(group: pd.DataFrame, col: str) -> float:
        weights = group["n_obs"]
        values = group[col]
        if weights.sum() == 0:
            return float("nan")
        return float((values * weights).sum() / weights.sum())

    sector_rows = []
    for (bucket, horizon), group in df.groupby(["bucket", "horizon"]):
        sector_rows.append({
            "bucket": bucket,
            "horizon": int(horizon),
            "n_tickers": len(group),
            "total_n_obs": int(group["n_obs"].sum()),
            "win_rate": weighted_avg(group, "win_rate"),
            "annualized_return": weighted_avg(group, "annualized_return"),
            "annualized_vol": weighted_avg(group, "annualized_vol"),
            "sharpe": weighted_avg(group, "sharpe"),
            "turnover": weighted_avg(group, "turnover"),
            "fitness": weighted_avg(group, "fitness"),
        })

    out = pd.DataFrame(sector_rows)
    return out.sort_values("fitness", ascending=False).reset_index(drop=True)


def write_to_db(df: pd.DataFrame, db_path: Path,
                table: str = "fitness_scores") -> None:
    """Persist fitness results to accuracy.db."""
    if df.empty:
        return
    conn = sqlite3.connect(str(db_path))
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_fitness "
            f"ON {table}(fitness DESC)"
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WorldQuant-style fitness scoring for ML Quant Fund models.",
    )
    parser.add_argument("--db", default="accuracy.db", type=Path)
    parser.add_argument("--mode", default="long_only",
                        choices=["long_only", "long_short", "binary_ls"])
    parser.add_argument("--since", default=None,
                        help="ISO date (YYYY-MM-DD) — only score predictions on or after")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--write-table", action="store_true",
                        help="Persist to accuracy.db (table: fitness_scores or fitness_scores_sector)")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--by-sector", action="store_true",
                        help="Aggregate per-ticker fitness to per-sector fitness")
    parser.add_argument("--metadata", default="tickers_metadata.csv", type=Path,
                        help="Path to tickers_metadata.csv (for --by-sector)")
    args = parser.parse_args()

    df = score_all(args.db, mode=args.mode, since=args.since)
    if df.empty:
        print(f"No (ticker, horizon) groups with >= {MIN_OBS} observations.")
        return

    pd.set_option("display.float_format", "{:.4f}".format)

    if args.by_sector:
        sec_df = aggregate_to_sector(df, args.metadata)
        print(f"\nSector fitness · mode={args.mode} · n_sectors={sec_df['bucket'].nunique()}")
        if args.since:
            print(f"  (predictions since {args.since})")
        print()
        print(sec_df.head(args.top).to_string(index=False))

        print("\n--- Median sector fitness by horizon ---")
        print(sec_df.groupby("horizon")[["fitness", "sharpe", "turnover"]]
              .median()
              .to_string())

        if args.csv:
            sec_df.to_csv(args.csv, index=False)
            print(f"\nWrote {len(sec_df)} rows to {args.csv}")
        if args.write_table:
            write_to_db(sec_df, args.db, table="fitness_scores_sector")
            print(f"Wrote to {args.db} (table: fitness_scores_sector)")
        return

    # Default per-ticker output
    cols = ["ticker", "horizon", "n_obs", "win_rate",
            "annualized_return", "annualized_vol", "sharpe",
            "turnover", "fitness"]

    print(f"\nFitness scoring · mode={args.mode} · n_models={len(df)}")
    if args.since:
        print(f"  (predictions since {args.since})")
    print()
    print(df[cols].head(args.top).to_string(index=False))

    print("\n--- Summary stats across all scored models ---")
    summary_cols = ["sharpe", "turnover", "fitness", "win_rate", "annualized_return"]
    print(df[summary_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())

    print("\n--- Median fitness by horizon ---")
    print(df.groupby("horizon")[["fitness", "sharpe", "turnover"]]
          .median()
          .to_string())

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {len(df)} rows to {args.csv}")
    if args.write_table:
        write_to_db(df, args.db)
        print(f"Wrote {len(df)} rows to {args.db} (table: fitness_scores)")


if __name__ == "__main__":
    main()
