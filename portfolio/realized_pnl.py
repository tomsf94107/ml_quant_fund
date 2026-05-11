"""
portfolio/realized_pnl.py
─────────────────────────────────────────────────────────────────
FIFO realized P&L computation from manual_trade_log entries.
Sprint 1 Day 6 (May 11 2026).

Pure functions — no DB, no Streamlit. Tested in isolation, called
from ui/pages/16_Trade_Log.py.

ENTRY POINT:
    compute_fifo_pnl(trades_df) -> (closed_df, open_df, warnings)

INPUT (trades_df):
    DataFrame with columns:
      - ticker, side (BUY/SELL/TRIM), shares, price, fill_time
    Sorted by fill_time ASC (oldest first). Sorting done internally
    so caller doesn't need to pre-sort.

OUTPUT:
    closed_df: DataFrame of closed positions
      - ticker, entry_time, exit_time, shares, entry_price, exit_price,
        gross_pnl, return_pct, hold_days, exit_side

    open_df: DataFrame of remaining open BUY lots (unmatched)
      - ticker, entry_time, shares, entry_price, cost_basis

    warnings: list[str] of validation issues (oversold, sell-without-buy)

LOGIC:
    For each ticker independently:
      1. Process trades in chronological order
      2. Maintain a FIFO queue of open BUY lots
      3. On SELL/TRIM:
         - Dequeue oldest BUYs first until SELL shares consumed
         - Each matched (BUY_lot, SELL_slice) pair → one closed position
         - If SELL exceeds available BUYs → warn, match what we can
      4. After all trades processed, remaining queue = open positions

EDGE CASES:
    A) Partial sell (TRIM): handled as fractional dequeue of oldest lot
    B) Multiple BUYs / one SELL: FIFO splits the SELL across BUY lots
    C) Oversold (SELL > total BUYs): match max available, warn the rest
    D) SELL with no prior BUY: emit warning, skip
    E) SELL and TRIM treated IDENTICALLY for P&L purposes
─────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List

import pandas as pd


@dataclass
class _OpenLot:
    """One open BUY lot waiting to be matched against future SELL/TRIM."""
    entry_time: str
    shares: float
    entry_price: float


def _parse_time(ts: str) -> datetime:
    """Parse ISO 8601 timestamp; tolerate missing seconds."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        # Try alternate parses if needed (future-proof)
        return datetime.strptime(str(ts)[:19], "%Y-%m-%dT%H:%M:%S")


def _hold_days(entry_ts: str, exit_ts: str) -> float:
    """Days between entry and exit (float for fractional days)."""
    e = _parse_time(entry_ts)
    x = _parse_time(exit_ts)
    return round((x - e).total_seconds() / 86400.0, 3)


def compute_fifo_pnl(
    trades_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    FIFO match BUYs to SELL/TRIM trades per-ticker.

    Returns
    -------
    closed_df : pd.DataFrame
        One row per closed position (matched BUY portion → SELL portion).
        Columns: ticker, entry_time, exit_time, shares, entry_price,
                 exit_price, gross_pnl, return_pct, hold_days, exit_side

    open_df : pd.DataFrame
        One row per remaining open lot (unmatched BUY).
        Columns: ticker, entry_time, shares, entry_price, cost_basis

    warnings : list[str]
        Validation issues encountered during matching.
    """
    closed_rows = []
    open_rows = []
    warnings = []

    if trades_df.empty:
        return (
            pd.DataFrame(columns=[
                "ticker", "entry_time", "exit_time", "shares",
                "entry_price", "exit_price", "gross_pnl",
                "return_pct", "hold_days", "exit_side",
            ]),
            pd.DataFrame(columns=[
                "ticker", "entry_time", "shares", "entry_price", "cost_basis",
            ]),
            warnings,
        )

    # Sort by fill_time ASC (oldest first); per-ticker grouping handled below
    df = trades_df.sort_values("fill_time", kind="stable").reset_index(drop=True)

    # Process each ticker independently
    for ticker, group in df.groupby("ticker"):
        open_queue: deque[_OpenLot] = deque()

        for _, row in group.iterrows():
            side = row["side"]
            shares = float(row["shares"])
            price = float(row["price"])
            ts = row["fill_time"]

            if side == "BUY":
                open_queue.append(_OpenLot(
                    entry_time=ts, shares=shares, entry_price=price,
                ))

            elif side in ("SELL", "TRIM"):
                remaining_to_sell = shares

                while remaining_to_sell > 0 and open_queue:
                    oldest = open_queue[0]
                    match_shares = min(oldest.shares, remaining_to_sell)

                    gross_pnl = match_shares * (price - oldest.entry_price)
                    cost_basis = match_shares * oldest.entry_price
                    return_pct = (
                        (price - oldest.entry_price) / oldest.entry_price * 100.0
                        if oldest.entry_price > 0 else 0.0
                    )

                    closed_rows.append({
                        "ticker":      ticker,
                        "entry_time":  oldest.entry_time,
                        "exit_time":   ts,
                        "shares":      round(match_shares, 4),
                        "entry_price": round(oldest.entry_price, 4),
                        "exit_price":  round(price, 4),
                        "gross_pnl":   round(gross_pnl, 4),
                        "return_pct":  round(return_pct, 3),
                        "hold_days":   _hold_days(oldest.entry_time, ts),
                        "exit_side":   side,
                    })

                    oldest.shares -= match_shares
                    remaining_to_sell -= match_shares

                    if oldest.shares <= 1e-9:  # floating-point tolerance
                        open_queue.popleft()

                if remaining_to_sell > 1e-9:
                    if not open_queue and shares == remaining_to_sell:
                        warnings.append(
                            f"⚠ {ticker} @ {ts}: SELL/TRIM of {shares} with no prior BUY — skipped"
                        )
                    else:
                        warnings.append(
                            f"⚠ {ticker} @ {ts}: SELL/TRIM oversold — "
                            f"{remaining_to_sell:.4f} shares unmatched (no inventory)"
                        )

            else:
                warnings.append(f"⚠ Unknown side '{side}' for {ticker} @ {ts} — skipped")

        # After processing all trades for this ticker, queue holds open positions
        for lot in open_queue:
            open_rows.append({
                "ticker":      ticker,
                "entry_time":  lot.entry_time,
                "shares":      round(lot.shares, 4),
                "entry_price": round(lot.entry_price, 4),
                "cost_basis":  round(lot.shares * lot.entry_price, 2),
            })

    closed_df = pd.DataFrame(closed_rows) if closed_rows else pd.DataFrame(columns=[
        "ticker", "entry_time", "exit_time", "shares", "entry_price",
        "exit_price", "gross_pnl", "return_pct", "hold_days", "exit_side",
    ])
    open_df = pd.DataFrame(open_rows) if open_rows else pd.DataFrame(columns=[
        "ticker", "entry_time", "shares", "entry_price", "cost_basis",
    ])

    return closed_df, open_df, warnings


def summarize_realized(closed_df: pd.DataFrame) -> dict:
    """
    Compute summary stats from closed positions.

    Returns
    -------
    dict with keys:
        n_closed:        number of closed positions
        total_gross_pnl: sum of gross_pnl across all closed
        total_cost:      sum of (entry_price * shares) — money put at risk
        weighted_return_pct: total_gross_pnl / total_cost * 100
        win_rate_pct:    fraction with return_pct > 0
        avg_return_pct:  simple mean of return_pct (each closed = 1 unit)
        avg_hold_days:   mean hold_days
        best_trade_pct:  max return_pct
        worst_trade_pct: min return_pct
    """
    if closed_df.empty:
        return {
            "n_closed": 0, "total_gross_pnl": 0.0, "total_cost": 0.0,
            "weighted_return_pct": 0.0, "win_rate_pct": 0.0,
            "avg_return_pct": 0.0, "avg_hold_days": 0.0,
            "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
        }

    total_gross_pnl = float(closed_df["gross_pnl"].sum())
    total_cost = float((closed_df["shares"] * closed_df["entry_price"]).sum())
    weighted_return = (total_gross_pnl / total_cost * 100.0) if total_cost > 0 else 0.0
    wins = int((closed_df["return_pct"] > 0).sum())
    n = len(closed_df)

    return {
        "n_closed":            n,
        "total_gross_pnl":     round(total_gross_pnl, 2),
        "total_cost":          round(total_cost, 2),
        "weighted_return_pct": round(weighted_return, 3),
        "win_rate_pct":        round(wins / n * 100.0, 1) if n > 0 else 0.0,
        "avg_return_pct":      round(float(closed_df["return_pct"].mean()), 3),
        "avg_hold_days":       round(float(closed_df["hold_days"].mean()), 2),
        "best_trade_pct":      round(float(closed_df["return_pct"].max()), 3),
        "worst_trade_pct":     round(float(closed_df["return_pct"].min()), 3),
    }
