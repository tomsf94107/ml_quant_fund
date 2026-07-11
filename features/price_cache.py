"""
features/price_cache.py — persistent RAW daily-bar cache with backward split
adjustment on read. Store unadjusted immutable bars + splits locally, adjust
BACKWARD on read. Never adjust forward. Raw 2022 close never changes -> cache
is pure append-only; only the splits table grows.

Wraps the daily-bar path of massive_client.download for SINGLE-TICKER '1d'
STOCK requests. Returned frame == mc.download(ticker,start,end,auto_adjust=True).
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)
_DB = Path("prices.db")


def _conn():
    c = sqlite3.connect(_DB, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS raw_bars (
        ticker TEXT, d TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (ticker, d))""")
    c.execute("""CREATE TABLE IF NOT EXISTS splits (
        ticker TEXT, exec_date TEXT, split_from REAL, split_to REAL,
        PRIMARY KEY (ticker, exec_date))""")
    return c


def _read_raw(con, ticker, start, end) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT d, open, high, low, close, volume FROM raw_bars "
        "WHERE ticker=? AND d>=? AND d<=? ORDER BY d",
        con, params=(ticker, start, end))
    if df.empty:
        return df
    df["d"] = pd.to_datetime(df["d"])
    df = df.set_index("d")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index.name = None
    return df


def _write_raw(con, ticker, df: pd.DataFrame):
    if df is None or df.empty:
        return
    rows = [(ticker, pd.Timestamp(i).strftime("%Y-%m-%d"),
             float(r["Open"]), float(r["High"]), float(r["Low"]),
             float(r["Close"]), float(r["Volume"]))
            for i, r in df.iterrows()]
    con.executemany("INSERT OR REPLACE INTO raw_bars VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()


def _store_splits(con, ticker, splits_list):
    rows = [(ticker, s["execution_date"], float(s["split_from"]), float(s["split_to"]))
            for s in splits_list]
    if rows:
        con.executemany("INSERT OR REPLACE INTO splits VALUES (?,?,?,?)", rows)
        con.commit()


def _read_splits(con, ticker) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT exec_date, split_from, split_to FROM splits WHERE ticker=? ORDER BY exec_date",
        con, params=(ticker,))


def _apply_backward_adjustment(raw: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or splits.empty:
        return raw.copy()
    adj = raw.copy()
    factor = pd.Series(1.0, index=adj.index)
    for _, s in splits.iterrows():
        sdate = pd.to_datetime(s["exec_date"])
        if sdate > adj.index.max():
            continue  # not yet effective; would divide current pre-split bars
        ratio = float(s["split_to"]) / float(s["split_from"])
        mask = adj.index < sdate
        factor[mask] *= ratio
    for col in ["Open", "High", "Low", "Close"]:
        adj[col] = adj[col] / factor
    adj["Volume"] = adj["Volume"] * factor
    return adj


def cached_daily(ticker, start, end, fetch_raw_fn, fetch_splits_fn) -> pd.DataFrame:
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    con = _conn()
    try:
        have = _read_raw(con, ticker, start_s, end_s)
        if have.empty:
            raw = fetch_raw_fn(ticker, start_s, end_s)
            _write_raw(con, ticker, raw)
        else:
            last = have.index.max()
            REFETCH_TAIL_DAYS = 5
            gap_start = (last - pd.Timedelta(days=REFETCH_TAIL_DAYS)).strftime("%Y-%m-%d")
            if gap_start <= end_s:
                gap = fetch_raw_fn(ticker, gap_start, end_s)
                if gap is not None and not gap.empty:
                    _write_raw(con, ticker, gap)
        try:
            sl = fetch_splits_fn(ticker)
            if sl:
                _store_splits(con, ticker, sl)
        except Exception as e:
            log.warning(f"price_cache: split fetch failed {ticker}: {e}")
        raw_full = _read_raw(con, ticker, start_s, end_s)
        splits = _read_splits(con, ticker)
        return _apply_backward_adjustment(raw_full, splits)
    finally:
        con.close()
