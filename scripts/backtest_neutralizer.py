"""
scripts/backtest_neutralizer.py
─────────────────────────────────────────────────────────────────
Backtest sector-neutral / dollar-neutral / no-neutralization modes
on historical predictions + outcomes.

Sprint 2 Stage 4a validation (May 14 2026).

Reads predictions + outcomes from accuracy.db, builds portfolios
under each mode using portfolio.neutralizer, computes per-date
realized returns, then aggregates Sharpe / drawdown / win rate.

Usage:
    python scripts/backtest_neutralizer.py --days 30 --horizon 3
    python scripts/backtest_neutralizer.py --days 60 --all-horizons
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from portfolio.neutralizer import build_research_portfolio  # noqa: E402

DB_PATH = ROOT / "accuracy.db"
METADATA_PATH = ROOT / "tickers_metadata.csv"

MIN_TICKERS_PER_DATE = 20  # skip dates with thin coverage
HORIZON_DAYS = {1: 1, 3: 3, 5: 5}
TRADING_DAYS_PER_YEAR = 252


def load_data(horizon: int, n_days: int) -> tuple[pd.DataFrame, dict]:
    """Load predictions+outcomes joined on (ticker, prediction_date, horizon)."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT p.prediction_date, p.ticker, p.horizon, p.prob_up,
                   o.actual_return
            FROM predictions p
            JOIN outcomes o
              ON p.ticker=o.ticker
             AND p.prediction_date=o.prediction_date
             AND p.horizon=o.horizon
            WHERE p.horizon=?
            ORDER BY p.prediction_date DESC, p.ticker
            """,
            conn,
            params=(horizon,),
        )

    # Keep most recent n_days of unique prediction dates
    unique_dates = sorted(df["prediction_date"].unique(), reverse=True)
    keep_dates = unique_dates[:n_days]
    df = df[df["prediction_date"].isin(keep_dates)]

    # Load sector map
    meta = pd.read_csv(METADATA_PATH)
    sector_map = dict(zip(meta["ticker"], meta["bucket"]))

    return df, sector_map


def compute_portfolio_returns(
    data: pd.DataFrame, sector_map: dict, mode: str
) -> pd.DataFrame:
    """Return DataFrame [date, portfolio_return] across all backtest dates."""
    daily_returns = []

    for date, group in data.groupby("prediction_date"):
        if len(group) < MIN_TICKERS_PER_DATE:
            continue

        # Build portfolio: input is predictions_df with prediction_date/ticker/horizon/prob_up
        preds = group[["prediction_date", "ticker", "horizon", "prob_up"]].copy()
        pf = build_research_portfolio(
            preds, sector_map=sector_map, mode=mode, long_only=False
        )

        if pf.empty:
            continue

        # Merge with actual_return
        merged = pf.merge(
            group[["ticker", "actual_return"]], on="ticker", how="inner"
        )

        # Portfolio return = sum(weight × actual_return)
        port_return = (merged["weight"] * merged["actual_return"]).sum()

        daily_returns.append(
            {
                "date": date,
                "portfolio_return": port_return,
                "n_positions": len(merged),
                "gross_exposure": merged["weight"].abs().sum(),
            }
        )

    return pd.DataFrame(daily_returns).sort_values("date").reset_index(drop=True)


def aggregate_stats(returns_df: pd.DataFrame, horizon: int) -> dict:
    """Compute summary stats from per-date portfolio returns."""
    if returns_df.empty:
        return {
            "n_days": 0,
            "mean_pnl": np.nan,
            "std_pnl": np.nan,
            "sharpe": np.nan,
            "ann_sharpe": np.nan,
            "cum_return": np.nan,
            "max_dd": np.nan,
            "win_rate": np.nan,
            "avg_gross": np.nan,
        }

    r = returns_df["portfolio_return"]
    n = len(r)
    mean_r = r.mean()
    std_r = r.std()
    sharpe = mean_r / std_r if std_r > 0 else np.nan
    # Each prediction holds for `horizon` days; bets per year ≈ 252/horizon
    ann_sharpe = sharpe * np.sqrt(TRADING_DAYS_PER_YEAR / horizon) if not np.isnan(sharpe) else np.nan

    # Cumulative return: sum-of-returns (additive, since bets overlap)
    # Use sum NOT product to avoid double-counting overlapping horizons
    cum_return = r.sum()

    # Max drawdown on cumulative path
    cum = r.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    win_rate = (r > 0).mean()
    avg_gross = returns_df["gross_exposure"].mean()

    return {
        "n_days": n,
        "mean_pnl": mean_r,
        "std_pnl": std_r,
        "sharpe": sharpe,
        "ann_sharpe": ann_sharpe,
        "cum_return": cum_return,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "avg_gross": avg_gross,
    }


def run_backtest(horizon: int, n_days: int):
    """Run backtest for one horizon and print results."""
    data, sector_map = load_data(horizon, n_days)

    if data.empty:
        print(f"No data for horizon={horizon}")
        return

    print(f"\n{'=' * 78}")
    print(f"BACKTEST — horizon={horizon}d, last {n_days} prediction days")
    print(f"Total rows: {len(data)}, unique dates: {data['prediction_date'].nunique()}")
    print(f"Date range: {data['prediction_date'].min()} → {data['prediction_date'].max()}")
    print(f"{'=' * 78}")

    results = []
    for mode in ["none", "sector", "dollar"]:
        returns_df = compute_portfolio_returns(data, sector_map, mode)
        stats = aggregate_stats(returns_df, horizon)
        stats["mode"] = mode
        results.append(stats)

    df = pd.DataFrame(results).set_index("mode")
    cols = ["n_days", "mean_pnl", "std_pnl", "sharpe", "ann_sharpe",
            "cum_return", "max_dd", "win_rate", "avg_gross"]
    df = df[cols]

    # Format for display
    print()
    print(df.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:.1f}"))

    # Decision rule
    best_mode = df["ann_sharpe"].idxmax()
    print(f"\n→ Highest annualized Sharpe: {best_mode!r} ({df.loc[best_mode, 'ann_sharpe']:+.3f})")


def main():
    parser = argparse.ArgumentParser(description="Backtest neutralizer modes")
    parser.add_argument("--days", type=int, default=30, help="Recent N prediction days")
    parser.add_argument("--horizon", type=int, default=3, choices=[1, 3, 5])
    parser.add_argument("--all-horizons", action="store_true", help="Run all 3 horizons")
    args = parser.parse_args()

    horizons = [1, 3, 5] if args.all_horizons else [args.horizon]
    for h in horizons:
        run_backtest(h, args.days)


if __name__ == "__main__":
    main()
