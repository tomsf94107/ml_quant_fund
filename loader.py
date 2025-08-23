# loader.py  —  SQLite-only data loaders
from __future__ import annotations

import os, sqlite3
from typing import Optional, List
import pandas as pd
import numpy as np

# === Config ===
INSIDER_SQLITE_PATH: str = os.getenv("INSIDER_SQLITE_PATH", "insider_trades.db")

# Try to import the MultiIndex-safe normalizer
try:
    from utils.price_norm import normalize_price_df
except Exception:
    # Minimal fallback (works for simple frames; your utils/price_norm is preferred)
    def normalize_price_df(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if px is None or px.empty:
            return pd.DataFrame(columns=["ticker","date","close","shares_outstanding","market_cap"])
        df = px.copy()
        # If columns are a MultiIndex, try to pick the ticker slice or flatten
        if isinstance(df.columns, pd.MultiIndex):
            # Try slice by any level
            sliced = False
            for lev in (0, 1, -1):
                try:
                    if ticker in df.columns.get_level_values(lev):
                        df = df.xs(ticker, axis=1, level=lev, drop_level=True)
                        sliced = True
                        break
                except Exception:
                    pass
            if not sliced:
                df.columns = ["_".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns.to_flat_index()]
        # Ensure a date column
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "date"})
            elif "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                df = df.reset_index()
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
                if "date" not in df.columns and "index" in df.columns:
                    df = df.rename(columns={"index": "date"})
        # Normalize close
        close_col = next((c for c in ["Adj Close","AdjClose","adj_close","Close","close"] if c in df.columns), None)
        if close_col:
            df = df.rename(columns={close_col: "close"})
        else:
            df["close"] = np.nan
        # Attach ticker, fundamentals placeholders
        if "ticker" not in df.columns:
            df["ticker"] = ticker
        for c in ["shares_outstanding","market_cap"]:
            if c not in df.columns:
                df[c] = np.nan
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df[["ticker","date","close","shares_outstanding","market_cap"]].dropna(subset=["date"])

# === Helpers ===
def _db_path() -> str:
    if not os.path.exists(INSIDER_SQLITE_PATH):
        raise FileNotFoundError(f"SQLite file not found: {INSIDER_SQLITE_PATH}")
    return INSIDER_SQLITE_PATH

def _sqlite_has_table(con: sqlite3.Connection, name: str) -> bool:
    try:
        q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;"
        return pd.read_sql_query(q, con, params=[name]).shape[0] > 0
    except Exception:
        return False

def insider_source_label() -> str:
    return f"SQLite ({INSIDER_SQLITE_PATH})"

# === Insider (SQLite) ===

def load_insider_from_sqlite_flows(db_path: str, ticker: str, start_date, end_date) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    if not _sqlite_has_table(con, "insider_flows"):
        con.close()
        return pd.DataFrame(columns=[
            "ticker","filed_date","net_shares","num_buy_tx","num_sell_tx",
            "num_exercise_like","max_txn_value_usd","any_exec_trade","large_tx_flag","holdings_delta",
            # keep slots for DB rollups even if empty
            "insider_7d","insider_21d",
        ])

    # IMPORTANT: reference the real column name "date" in WHERE/ORDER BY (not the alias)
    q = """
    SELECT
      UPPER(ticker) AS ticker,
      date AS filed_date,
      net_shares,
      insider_7d,
      insider_21d
    FROM insider_flows
    WHERE UPPER(ticker) = UPPER(?)
      AND date(date) BETWEEN date(?) AND date(?)
    ORDER BY date;
    """
    df = pd.read_sql_query(q, con, params=[ticker, str(start_date), str(end_date)])
    con.close()

    if df.empty:
        return pd.DataFrame(columns=[
            "ticker","filed_date","net_shares","num_buy_tx","num_sell_tx",
            "num_exercise_like","max_txn_value_usd","any_exec_trade","large_tx_flag","holdings_delta",
            "insider_7d","insider_21d",
        ])

    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce").dt.date
    # fill analytics not stored in DB
    df["num_buy_tx"]        = 0
    df["num_sell_tx"]       = 0
    df["num_exercise_like"] = 0
    df["max_txn_value_usd"] = 0.0
    df["any_exec_trade"]    = 0
    df["large_tx_flag"]     = 0
    df["holdings_delta"]    = 0.0

    # keep DB rollups on the frame
    for c in ["insider_7d","insider_21d"]:
        if c not in df.columns:
            df[c] = 0.0

    return df[[
        "ticker","filed_date","net_shares",
        "num_buy_tx","num_sell_tx","num_exercise_like",
        "max_txn_value_usd","any_exec_trade","large_tx_flag","holdings_delta",
        "insider_7d","insider_21d",
    ]].copy()

def insider_loader(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Public entry: always load from local SQLite flows."""
    return load_insider_from_sqlite_flows(_db_path(), ticker, start_date, end_date)

def list_insider_tickers(limit: int = 50) -> List[str]:
    """List distinct tickers present in insider_flows (uppercased)."""
    con = sqlite3.connect(_db_path())
    if not _sqlite_has_table(con, "insider_flows"):
        con.close()
        return []
    df = pd.read_sql_query(
        "SELECT DISTINCT UPPER(ticker) AS ticker FROM insider_flows ORDER BY ticker LIMIT ?;",
        con, params=[int(limit)]
    )
    con.close()
    return df["ticker"].dropna().astype(str).tolist()

def validate_insider_db() -> dict:
    """Quick validation summary of the local DB."""
    p = _db_path()
    con = sqlite3.connect(p)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", con)
    has_flows = _sqlite_has_table(con, "insider_flows")
    sample = pd.read_sql_query("SELECT * FROM insider_flows LIMIT 3;", con) if has_flows else pd.DataFrame()
    con.close()
    return {
        "path": p,
        "tables": tables["name"].tolist(),
        "has_insider_flows": bool(has_flows),
        "sample": sample.to_dict(orient="records")
    }

# === Price (yfinance) ===
def price_loader(ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Returns a normalized frame with at least:
      ['ticker','date','close','shares_outstanding','market_cap'].

    Robust to yfinance MultiIndex columns like ('Close','AAPL').
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is required for price_loader. Install it in your env.") from e

    t = str(ticker).strip().upper()

    # 1) Fetch raw prices (yfinance may return MultiIndex)
    try:
        if start_date and end_date:
            raw = yf.download(t, start=str(start_date), end=str(end_date), auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raw = yf.download(t, period="1y", auto_adjust=True, progress=False)
        else:
            raw = yf.download(t, period="1y", auto_adjust=True, progress=False)
    except Exception:
        raw = pd.DataFrame()

    # 2) Normalize to flat, consistent schema
    px = normalize_price_df(raw, t)
    if px is None or px.empty:
        return pd.DataFrame(columns=["ticker","date","close","shares_outstanding","market_cap"])

    # 3) Optional fundamentals (constant columns for the window)
    shares_out, mcap = np.nan, np.nan
    try:
        tk = yf.Ticker(t)
        fi = getattr(tk, "fast_info", None)
        if fi:
            # handle attr-style or dict-style fast_info
            shares_out = getattr(fi, "shares_outstanding", None) or (fi.get("shares_outstanding") if hasattr(fi, "get") else None) or (fi.get("sharesOutstanding") if hasattr(fi, "get") else None)
            mcap       = getattr(fi, "market_cap", None)         or (fi.get("market_cap") if hasattr(fi, "get") else None)         or (fi.get("marketCap") if hasattr(fi, "get") else None)
        if shares_out is None or mcap is None:
            try:
                info = tk.get_info()  # newer yfinance
            except Exception:
                info = getattr(tk, "info", {}) or {}
            shares_out = shares_out or info.get("sharesOutstanding")
            mcap       = mcap or info.get("marketCap")
    except Exception:
        pass

    if "shares_outstanding" not in px.columns:
        px["shares_outstanding"] = np.nan
    if "market_cap" not in px.columns:
        px["market_cap"] = np.nan

    if pd.isna(px["shares_outstanding"]).all() and shares_out is not None:
        try:
            px["shares_outstanding"] = float(shares_out)
        except Exception:
            pass

    if pd.isna(px["market_cap"]).all():
        if mcap is not None:
            try:
                px["market_cap"] = float(mcap)
            except Exception:
                pass
        else:
            # compute from shares * last close if both present
            try:
                so = float(shares_out) if shares_out is not None else np.nan
                last_close = float(pd.to_numeric(px["close"], errors="coerce").dropna().iloc[-1]) if px["close"].notna().any() else np.nan
                if np.isfinite(so) and np.isfinite(last_close):
                    px["market_cap"] = so * last_close
            except Exception:
                pass

    # 4) Final tidy/return (only the normalized columns)
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.date
    px = px.dropna(subset=["date"])
    return px[["ticker","date","close","shares_outstanding","market_cap"]]

