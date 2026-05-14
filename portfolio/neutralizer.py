"""
portfolio/neutralizer.py
─────────────────────────────────────────────────────────────────
Sector-neutral and dollar-neutral portfolio construction.
Sprint 2 Stage 4 (May 14 2026).

Pure functions — no DB, no Streamlit. Tested in isolation, called
from CLI wrapper or Streamlit research page.

ENTRY POINT:
    neutralize_signals(signals_df, sector_map, mode, long_only)
        -> DataFrame of [date, ticker, weight]

    build_research_portfolio(predictions_df, sector_map, mode,
                             long_only, top_n, negative_fitness_tickers)
        -> DataFrame of [date, ticker, weight, horizon]

INPUT (signals_df):
    DataFrame with columns:
      - date         (any sortable date type)
      - ticker       (str)
      - signal_value (float, typically prob_up - 0.5)

INPUT (predictions_df):
    DataFrame with columns:
      - prediction_date (str YYYY-MM-DD)
      - ticker          (str)
      - horizon         (int: 1, 3, or 5)
      - prob_up         (float in [0, 1])

MODES:
    'sector' — weights sum to 0 within each >=2-member sector
               (singleton sectors pass through unchanged)
    'dollar' — weights sum to 0 globally per date
    'none'   — raw signal values, no neutralization (baseline)

LONG-ONLY CONVERSION:
    When long_only=True: clip(weight, 0, inf) then normalize so
    sum-of-weights = 1.0 per date.

WRAPS:
    features.alpha_transformations.group_neutralize for sector math.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from features.alpha_transformations import group_neutralize

logger = logging.getLogger(__name__)

VALID_MODES = ("sector", "dollar", "none")


# ─────────────────────────────────────────────────────────────────────────────
#  Coverage diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def coverage_report(signals_df: pd.DataFrame, sector_map: dict) -> dict:
    """Report neutralization coverage given a sector map."""
    tickers = signals_df["ticker"].unique()
    bucket_counts: dict = {}
    for t in tickers:
        b = sector_map.get(t, "UNKNOWN")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    neutralizable = sum(c for c in bucket_counts.values() if c >= 2)
    pass_through = sum(c for c in bucket_counts.values() if c < 2)

    return {
        "total_tickers": len(tickers),
        "neutralizable": neutralizable,
        "pass_through": pass_through,
        "buckets": bucket_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Core neutralization
# ─────────────────────────────────────────────────────────────────────────────
def neutralize_signals(
    signals_df: pd.DataFrame,
    sector_map: dict,
    mode: str = "sector",
    long_only: bool = False,
) -> pd.DataFrame:
    """Apply neutralization to prediction signals.

    Raises:
        ValueError: if mode not in {'sector','dollar','none'}.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode: {mode!r}. Use one of {VALID_MODES}."
        )

    if signals_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "weight"])

    # Pivot to panel: rows=date, cols=ticker
    panel = signals_df.pivot(
        index="date", columns="ticker", values="signal_value"
    )

    if mode == "none":
        neutralized = panel
    elif mode == "sector":
        neutralized = group_neutralize(panel, sector_map)
    elif mode == "dollar":
        neutralized = panel.subtract(panel.mean(axis=1), axis=0)

    if long_only:
        neutralized = neutralized.clip(lower=0)
        row_sums = neutralized.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        neutralized = neutralized.div(row_sums, axis=0)

    # Unpivot to long form
    weights = (
        neutralized.stack(future_stack=True)
        .rename("weight")
        .reset_index()
    )

    # Keep all weights including exact zeros — they are meaningful signals

    return weights[["date", "ticker", "weight"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-end portfolio builder
# ─────────────────────────────────────────────────────────────────────────────
def build_research_portfolio(
    predictions_df: pd.DataFrame,
    sector_map: dict,
    mode: str = "sector",
    long_only: bool = False,
    top_n: Optional[int] = None,
    negative_fitness_tickers: Optional[set] = None,
) -> pd.DataFrame:
    """End-to-end: predictions -> neutralized portfolio weights."""
    if predictions_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "weight", "horizon"])

    df = predictions_df.copy()
    df["signal_value"] = df["prob_up"] - 0.5

    if negative_fitness_tickers:
        before = len(df)
        df = df[~df.apply(
            lambda r: (r["ticker"], r["horizon"]) in negative_fitness_tickers,
            axis=1,
        )]
        logger.info(f"Fitness filter: {before - len(df)} rows excluded")

    portfolios = []
    for horizon, sub in df.groupby("horizon"):
        sub_signals = sub.rename(columns={"prediction_date": "date"})[
            ["date", "ticker", "signal_value"]
        ]
        weights = neutralize_signals(sub_signals, sector_map, mode, long_only)
        weights["horizon"] = horizon
        portfolios.append(weights)

    if not portfolios:
        return pd.DataFrame(columns=["date", "ticker", "weight", "horizon"])

    portfolio = pd.concat(portfolios, ignore_index=True)

    if top_n is not None:
        portfolio = (
            portfolio.assign(_abs=portfolio["weight"].abs())
            .sort_values("_abs", ascending=False)
            .groupby(["date", "horizon"])
            .head(top_n)
            .drop(columns="_abs")
            .sort_values(
                ["date", "horizon", "weight"], ascending=[True, True, False]
            )
            .reset_index(drop=True)
        )

    return portfolio
