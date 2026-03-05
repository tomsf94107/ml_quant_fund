# data/etl_congress.py
# ─────────────────────────────────────────────────────────────────────────────
# Congressional trades ETL. Pulls from QuiverQuant API, writes to SQLite.
#
# What we fixed from v1_1_etl_congress.py:
#   ✗ REMOVED: st.secrets import at module level (crashed when no Streamlit)
#   ✗ FIXED: df["shares"].astype(int) — crashed on decimals/empty strings
#   ✓ ADDED: SQLite sink — same DB pattern as etl_insider.py
#   ✓ ADDED: load_congress_flows() read function for builder.py
#
# Zero Streamlit imports. Backend only.
# Run this script on a schedule (daily/weekly) to keep the DB fresh.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = Path(os.getenv("CONGRESS_DB_PATH", "congress_trades.db"))
QUIVER_API_KEY = os.getenv("QUIVER_API_KEY", "")
QUIVER_BASE    = "https://api.quiverquant.com/beta"


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create congress_flows table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS congress_flows (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker                   TEXT    NOT NULL,
            ds                       TEXT    NOT NULL,
            congress_net_shares      REAL    NOT NULL DEFAULT 0,
            congress_active_members  INTEGER NOT NULL DEFAULT 0,
            UNIQUE(ticker, ds)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_congress_ticker_ds "
        "ON congress_flows(ticker, ds)"
    )
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  QUIVERQUANT FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_quiver(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Pull congressional trades for `ticker` from QuiverQuant API.
    Returns clean DataFrame or empty DataFrame on any failure.
    """
    url = f"{QUIVER_BASE}/historical/congresstrading?ticker={ticker.upper()}"
    headers = {"Authorization": f"Token {api_key}"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return pd.DataFrame()   # ticker not covered
        print(f"  ⚠ QuiverQuant HTTP error for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠ QuiverQuant fetch failed for {ticker}: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # ── Normalize date ────────────────────────────────────────────────────────
    date_col = next(
        (c for c in df.columns if "date" in c.lower() or "transaction" in c.lower()),
        None
    )
    if date_col is None:
        print(f"  ⚠ No date column found in QuiverQuant response for {ticker}")
        return pd.DataFrame()

    df["ds"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=["ds"])

    # ── Normalize shares ──────────────────────────────────────────────────────
    # FIX: old code used .astype(int) which crashes on decimals or empty strings
    shares_col = next(
        (c for c in df.columns if "share" in c.lower() or "amount" in c.lower()),
        None
    )
    if shares_col:
        df["shares"] = pd.to_numeric(df[shares_col], errors="coerce").fillna(0)
    else:
        # QuiverQuant returns "Transaction" = "Purchase" or "Sale"
        tx_col = next((c for c in df.columns if "transaction" in c.lower()), None)
        if tx_col:
            df["shares"] = df[tx_col].str.lower().map(
                lambda x: 1 if "purchase" in str(x) or "buy" in str(x)
                else -1 if "sale" in str(x) or "sell" in str(x)
                else 0
            )
        else:
            df["shares"] = 0

    # ── Member column ─────────────────────────────────────────────────────────
    member_col = next(
        (c for c in df.columns
         if any(k in c.lower() for k in ["member", "name", "senator", "representative"])),
        None
    )

    # ── Aggregate per day ─────────────────────────────────────────────────────
    agg_cols = {"congress_net_shares": ("shares", "sum")}
    if member_col:
        agg_cols["congress_active_members"] = (member_col, "nunique")

    agg = df.groupby("ds").agg(**agg_cols).reset_index()

    if "congress_active_members" not in agg.columns:
        agg["congress_active_members"] = 1

    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  SQLITE SINK
# ══════════════════════════════════════════════════════════════════════════════

def _upsert_to_db(
    ticker: str,
    df: pd.DataFrame,
    conn: sqlite3.Connection,
) -> int:
    """Write congress flow rows to SQLite. Returns rows written."""
    if df.empty:
        return 0

    df = df.copy()
    df["ticker"] = ticker.upper()
    df["ds"]     = df["ds"].astype(str)

    if "congress_net_shares" not in df.columns:
        df["congress_net_shares"] = 0
    if "congress_active_members" not in df.columns:
        df["congress_active_members"] = 0

    rows = df[["ticker", "ds", "congress_net_shares",
               "congress_active_members"]].to_dict("records")

    conn.executemany("""
        INSERT OR REPLACE INTO congress_flows
            (ticker, ds, congress_net_shares, congress_active_members)
        VALUES
            (:ticker, :ds, :congress_net_shares, :congress_active_members)
    """, rows)
    conn.commit()
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_congress_etl(
    tickers:  list[str],
    api_key:  str | None = None,
    db_path:  Path = DB_PATH,
    verbose:  bool = True,
) -> dict[str, int]:
    """
    Run the full congressional trades ETL for a list of tickers.
    Writes results to SQLite at db_path.

    Returns dict: {ticker: rows_written}
    """
    key = api_key or QUIVER_API_KEY
    if not key:
        raise RuntimeError(
            "QUIVER_API_KEY is not set. "
            "Set it in your environment or .streamlit/secrets.toml:\n"
            "  QUIVER_API_KEY = 'your_key_here'"
        )

    conn    = _init_db(db_path)
    results = {}

    for ticker in tickers:
        ticker = ticker.upper().strip()
        if verbose:
            print(f"  {ticker} — fetching congressional trades...")

        df = _fetch_quiver(ticker, key)

        if df.empty:
            if verbose:
                print(f"    ⚠ No data for {ticker}")
            results[ticker] = 0
            continue

        n = _upsert_to_db(ticker, df, conn)
        results[ticker] = n

        if verbose:
            print(f"    ✓ {n} rows written for {ticker}")

    conn.close()
    return results


def load_congress_flows(
    ticker:     str,
    start_date: str | date | None = None,
    end_date:   str | date | None = None,
    db_path:    Path = DB_PATH,
) -> pd.DataFrame:
    """
    Read congressional flows from SQLite.
    Called by features/builder.py — do not rename this function.

    Returns DataFrame with columns:
        ds, ticker, congress_net_shares, congress_active_members
    """
    if not db_path.exists():
        return pd.DataFrame()

    conn  = sqlite3.connect(db_path)
    query = "SELECT * FROM congress_flows WHERE ticker = ?"
    params: list = [ticker.upper()]

    if start_date:
        query += " AND ds >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND ds <= ?"
        params.append(str(end_date))

    query += " ORDER BY ds"

    try:
        df = pd.read_sql(query, conn, params=params, parse_dates=["ds"])
        conn.close()
        return df
    except Exception as e:
        conn.close()
        print(f"  ⚠ Congress DB read failed for {ticker}: {e}")
        return pd.DataFrame()


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from models.train_all import DEFAULT_TICKERS

    parser = argparse.ArgumentParser(description="Run congressional trades ETL")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    print(f"\nRunning congress ETL for {len(tickers)} tickers...")
    results = run_congress_etl(tickers)
    total   = sum(results.values())
    print(f"\nDone. {total} total rows written to {DB_PATH}")
