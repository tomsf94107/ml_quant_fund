"""
Institutional / dark-pool feature extraction (DuckDB backend, Session F).

Source: institutional_trades.duckdb (migrated from SQLite 2026-05-22).
PIT discipline: all queries use trade_date STRICTLY less than as_of_date.

Public API unchanged from SQLite version:
  - get_institutional_features(ticker, as_of_date)
  - get_institutional_features_batch(tickers, as_of_date)
  - load_institutional_features_pit(ticker, date_index)

NEW: load_institutional_features_pit_fast computes all dates for a ticker
in ONE DuckDB query, ~100x faster than per-row loop.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import numpy as np

DB_PATH = Path(os.environ.get("INST_DB_PATH", "institutional_trades.duckdb"))

WINDOW_5D = 5
WINDOW_7D = 7
WINDOW_30D = 30


def _connect(db_path: Path = DB_PATH):
    return duckdb.connect(str(db_path), read_only=True)


def _trading_days_back(as_of_date: str, n_calendar_days: int) -> str:
    start_dt = datetime.fromisoformat(as_of_date) - timedelta(days=n_calendar_days)
    return start_dt.date().isoformat()


def _signed_flow_pct(conn, ticker, start_date, as_of_date, dark_pool_only=False):
    dp_clause = "AND is_dark_pool = TRUE" if dark_pool_only else ""
    q = f"""
    SELECT
      SUM(CASE WHEN side = 'BUY'  THEN notional_usd ELSE 0 END) AS buy_n,
      SUM(CASE WHEN side = 'SELL' THEN notional_usd ELSE 0 END) AS sell_n,
      SUM(notional_usd) AS total_n
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      {dp_clause}
      AND side IN ('BUY', 'SELL')
    """
    row = conn.execute(q, [ticker.upper(), start_date, as_of_date]).fetchone()
    if not row or row[2] is None or row[2] == 0:
        return None
    buy_n, sell_n, total_n = row
    return float((buy_n - sell_n) / total_n)


def _block_buy_sell_ratio(conn, ticker, start_date, as_of_date):
    q = """
    SELECT
      SUM(CASE WHEN side = 'BUY'  THEN notional_usd ELSE 0 END) AS buy_block,
      SUM(CASE WHEN side = 'SELL' THEN notional_usd ELSE 0 END) AS sell_block
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      AND is_block = TRUE
      AND side IN ('BUY', 'SELL')
    """
    row = conn.execute(q, [ticker.upper(), start_date, as_of_date]).fetchone()
    if not row or row[0] is None or row[1] is None or row[1] == 0:
        return None
    buy_block, sell_block = row
    ratio = float(buy_block / sell_block)
    return max(0.01, min(100.0, ratio))


def _sweep_count(conn, ticker, start_date, as_of_date):
    q = """
    SELECT COUNT(*) FROM institutional_trades
    WHERE ticker = ? AND trade_date >= ? AND trade_date < ?
      AND is_sweep = TRUE
    """
    row = conn.execute(q, [ticker.upper(), start_date, as_of_date]).fetchone()
    return int(row[0]) if row else 0


def _closing_auction_imbalance(conn, ticker, start_date, as_of_date):
    q = """
    SELECT
      SUM(CASE WHEN side = 'BUY'  THEN notional_usd ELSE 0 END) AS buy_n,
      SUM(CASE WHEN side = 'SELL' THEN notional_usd ELSE 0 END) AS sell_n,
      SUM(notional_usd) AS total_n
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
      AND is_closing_auction = TRUE
      AND side IN ('BUY', 'SELL')
    """
    row = conn.execute(q, [ticker.upper(), start_date, as_of_date]).fetchone()
    if not row or row[2] is None or row[2] == 0:
        return None
    buy_n, sell_n, total_n = row
    return float((buy_n - sell_n) / total_n)


def _block_notional_norm(conn, ticker, start_date, as_of_date):
    q = """
    SELECT
      SUM(CASE WHEN is_block = TRUE THEN notional_usd ELSE 0 END) AS block_n,
      SUM(notional_usd) AS total_n
    FROM institutional_trades
    WHERE ticker = ?
      AND trade_date >= ?
      AND trade_date < ?
    """
    row = conn.execute(q, [ticker.upper(), start_date, as_of_date]).fetchone()
    if not row or row[1] is None or row[1] == 0:
        return None
    return float(row[0] / row[1])


def _block_count(conn, ticker, start_date, as_of_date):
    q = """
    SELECT COUNT(*) FROM institutional_trades
    WHERE ticker = ? AND trade_date >= ? AND trade_date < ?
      AND is_block = TRUE
    """
    row = conn.execute(q, [ticker.upper(), start_date, as_of_date]).fetchone()
    return int(row[0]) if row else 0


def get_institutional_features(ticker: str, as_of_date: str, db_path=None) -> dict:
    if db_path is None:
        db_path = DB_PATH
    ticker = ticker.upper().strip()
    out = {
        "inst_signed_flow_5d":      None,
        "inst_block_buy_sell_7d":   None,
        "inst_dp_signed_flow_5d":   None,
        "inst_sweep_count_7d":      0,
        "inst_auction_imbal_5d":    None,
        "inst_signed_flow_30d":     None,
        "inst_block_notional_7d":   None,
        "inst_block_count_7d":      0,
    }
    try:
        conn = _connect(db_path)
    except duckdb.Error:
        return out
    try:
        start_5d  = _trading_days_back(as_of_date, WINDOW_5D)
        start_7d  = _trading_days_back(as_of_date, WINDOW_7D)
        start_30d = _trading_days_back(as_of_date, WINDOW_30D)
        out["inst_signed_flow_5d"]    = _signed_flow_pct(conn, ticker, start_5d,  as_of_date)
        out["inst_block_buy_sell_7d"] = _block_buy_sell_ratio(conn, ticker, start_7d, as_of_date)
        out["inst_dp_signed_flow_5d"] = _signed_flow_pct(conn, ticker, start_5d,  as_of_date, dark_pool_only=True)
        out["inst_sweep_count_7d"]    = _sweep_count(conn, ticker, start_7d,  as_of_date)
        out["inst_auction_imbal_5d"]  = _closing_auction_imbalance(conn, ticker, start_5d, as_of_date)
        out["inst_signed_flow_30d"]   = _signed_flow_pct(conn, ticker, start_30d, as_of_date)
        out["inst_block_notional_7d"] = _block_notional_norm(conn, ticker, start_7d, as_of_date)
        out["inst_block_count_7d"]    = _block_count(conn, ticker, start_7d, as_of_date)
    finally:
        conn.close()
    return out


def get_institutional_features_batch(tickers, as_of_date, db_path=None):
    return {t: get_institutional_features(t, as_of_date, db_path) for t in tickers}


def load_institutional_features_pit_fast(ticker, date_index, db_path=None):
    """
    Vectorized PIT loader: 1 DuckDB query computes 4 inst features for all dates.
    PIT-safe: trade_date < as_of_date strictly.
    Returns DataFrame indexed by date_index with 4 columns matching builder.py.
    """
    if db_path is None:
        db_path = DB_PATH
    ticker = ticker.upper().strip()
    cols = [
        "inst_block_buy_sell_7d",
        "inst_signed_flow_30d",
        "inst_auction_imbal_5d",
        "inst_signed_flow_5d",
    ]
    out = pd.DataFrame(index=pd.Index(date_index), columns=cols, dtype=float)
    out[:] = np.nan

    try:
        conn = _connect(db_path)
    except duckdb.Error:
        return out

    try:
        asof_dates = []
        for d in date_index:
            if hasattr(d, "strftime"):
                asof_dates.append(d.strftime("%Y-%m-%d"))
            else:
                asof_dates.append(str(d))
        if not asof_dates:
            return out

        sql = """
        WITH daily AS (
            SELECT
                trade_date,
                SUM(CASE WHEN side='BUY'  THEN notional_usd ELSE 0 END) AS buy_n,
                SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END) AS sell_n,
                SUM(CASE WHEN side IN ('BUY','SELL') THEN notional_usd ELSE 0 END) AS total_n_signed,
                SUM(CASE WHEN is_block=TRUE AND side='BUY'  THEN notional_usd ELSE 0 END) AS block_buy_n,
                SUM(CASE WHEN is_block=TRUE AND side='SELL' THEN notional_usd ELSE 0 END) AS block_sell_n,
                SUM(CASE WHEN is_closing_auction=TRUE AND side='BUY'  THEN notional_usd ELSE 0 END) AS auct_buy_n,
                SUM(CASE WHEN is_closing_auction=TRUE AND side='SELL' THEN notional_usd ELSE 0 END) AS auct_sell_n,
                SUM(CASE WHEN is_closing_auction=TRUE AND side IN ('BUY','SELL') THEN notional_usd ELSE 0 END) AS auct_total_n
            FROM institutional_trades
            WHERE ticker = ?
            GROUP BY trade_date
        ),
        asof_dates_cte AS (
            SELECT UNNEST(?::DATE[]) AS asof_date
        )
        SELECT
            a.asof_date,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 5 DAY AND d.trade_date < a.asof_date THEN d.buy_n - d.sell_n ELSE 0 END) AS net_5d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 5 DAY AND d.trade_date < a.asof_date THEN d.total_n_signed ELSE 0 END) AS total_5d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 30 DAY AND d.trade_date < a.asof_date THEN d.buy_n - d.sell_n ELSE 0 END) AS net_30d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 30 DAY AND d.trade_date < a.asof_date THEN d.total_n_signed ELSE 0 END) AS total_30d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 7 DAY AND d.trade_date < a.asof_date THEN d.block_buy_n ELSE 0 END) AS block_buy_7d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 7 DAY AND d.trade_date < a.asof_date THEN d.block_sell_n ELSE 0 END) AS block_sell_7d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 5 DAY AND d.trade_date < a.asof_date THEN d.auct_buy_n - d.auct_sell_n ELSE 0 END) AS auct_net_5d,
            SUM(CASE WHEN d.trade_date >= a.asof_date - INTERVAL 5 DAY AND d.trade_date < a.asof_date THEN d.auct_total_n ELSE 0 END) AS auct_total_5d
        FROM asof_dates_cte a
        LEFT JOIN daily d ON TRUE
        GROUP BY a.asof_date
        ORDER BY a.asof_date
        """

        rows = conn.execute(sql, [ticker, asof_dates]).fetchall()

        # Build a string->index map so we can match by ISO date string regardless
        # of whether date_index contains datetime.date, pd.Timestamp, or strings
        idx_str_to_pos = {}
        for pos, k in enumerate(out.index):
            if hasattr(k, "strftime"):
                idx_str_to_pos[k.strftime("%Y-%m-%d")] = pos
            else:
                idx_str_to_pos[str(k)] = pos

        for asof_date, net_5d, total_5d, net_30d, total_30d, block_buy_7d, block_sell_7d, auct_net_5d, auct_total_5d in rows:
            # asof_date from DuckDB is a datetime.date object
            key_str = asof_date.strftime("%Y-%m-%d") if hasattr(asof_date, "strftime") else str(asof_date)
            pos = idx_str_to_pos.get(key_str)
            if pos is None:
                continue
            if total_5d and total_5d != 0:
                out.iloc[pos, out.columns.get_loc("inst_signed_flow_5d")] = float(net_5d / total_5d)
            if total_30d and total_30d != 0:
                out.iloc[pos, out.columns.get_loc("inst_signed_flow_30d")] = float(net_30d / total_30d)
            if block_sell_7d and block_sell_7d != 0:
                ratio = float(block_buy_7d / block_sell_7d)
                out.iloc[pos, out.columns.get_loc("inst_block_buy_sell_7d")] = max(0.01, min(100.0, ratio))
            if auct_total_5d and auct_total_5d != 0:
                out.iloc[pos, out.columns.get_loc("inst_auction_imbal_5d")] = float(auct_net_5d / auct_total_5d)
    finally:
        conn.close()
    return out


def load_institutional_features_pit(ticker, date_index, db_path=None):
    """Backward-compat alias to the fast path."""
    return load_institutional_features_pit_fast(ticker, date_index, db_path)
