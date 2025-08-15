# data/etl_holdings.py
from __future__ import annotations

import os
import sqlite3
import pandas as pd

# Prefer package import; fall back to repo-root style if running directly
try:
    from ml_quant_fund.core.feature_utils import finalize_features
except ModuleNotFoundError:
    from core.feature_utils import finalize_features


def _db_path() -> str | None:
    """Resolve DB path from env or Streamlit secrets."""
    url = os.getenv("INSIDER_DB_URL")
    if not url:
        try:
            import streamlit as st  # optional dependency
            url = st.secrets.get("insider_db_url")
        except Exception:
            url = None
    if not url:
        return None
    return url.replace("sqlite:///", "") if url.startswith("sqlite:///") else url


def _conn():
    p = _db_path()
    return sqlite3.connect(p) if p else None


def fetch_insider_holdings(ticker: str) -> pd.DataFrame:
    """
    Returns: ds (datetime[UTC]), hold_shares (float), hold_net_change (float)
    Expects SQLite table 'holdings' with columns: ticker, date, shares
    """
    cols = ["ds", "hold_shares", "hold_net_change"]
    con = _conn()
    if not con:
        return pd.DataFrame(columns=cols)

    q = """
        SELECT
            date   AS ds,
            shares AS hold_shares
        FROM holdings
        WHERE UPPER(ticker) = UPPER(?)
        ORDER BY date;
    """
    try:
        df = pd.read_sql(q, con, params=(str(ticker).strip().upper(),))
        if df.empty:
            return pd.DataFrame(columns=cols)

        # Parse and sanitize
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True)
        df["hold_shares"] = pd.to_numeric(df["hold_shares"], errors="coerce")
        df = df.dropna(subset=["ds"]).sort_values("ds")

        # If multiple rows per day, keep the latest
        df = df.drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)

        # Compute daily net change
        df["hold_net_change"] = df["hold_shares"].diff().fillna(0.0)

        # Finalize on a datetime index, then restore 'ds'
        finalized = finalize_features(df.set_index("ds"))
        finalized.index.name = "ds"
        out = finalized.reset_index()[["ds", "hold_shares", "hold_net_change"]]
        return out
    except Exception as e:
        print(f"insider holdings db error: {e}")
        return pd.DataFrame(columns=cols)
    finally:
        try:
            con.close()
        except Exception:
            pass
