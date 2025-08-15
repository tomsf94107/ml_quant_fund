# data/etl_insider.py
import os
import pandas as pd
from sqlalchemy import create_engine, inspect, text

try:
    import streamlit as st
    ST_SECRETS = st.secrets
except Exception:
    ST_SECRETS = {}

# ────────────────────────────────────────────────────────────────────────────
# Robust imports (works with or without package prefix)
# ────────────────────────────────────────────────────────────────────────────
try:
    from ml_quant_fund.core.feature_utils import finalize_features
except ModuleNotFoundError:
    from core.feature_utils import finalize_features

# ----------------------------- engine ---------------------------------
def _get_engine():
    url = (
        os.environ.get("INSIDER_DB_URL")
        or ST_SECRETS.get("insider_db_url")
        or "sqlite:///insider_trades.db"
    )
    return create_engine(url)

def _empty():
    return pd.DataFrame(
        columns=["ds", "insider_net_shares", "insider_buy_count", "insider_sell_count"]
    )

def _normalize_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

# ------------------------ source: insider_flows -----------------------
def _from_insider_flows(engine, ticker: str) -> pd.DataFrame:
    q = text("""
        SELECT date, net_shares
          FROM insider_flows
         WHERE UPPER(ticker)=UPPER(:tkr)
    """)
    try:
        df = pd.read_sql(q, engine, params={"tkr": ticker})
    except Exception as e:
        print(f"insider_flows query error: {e}")
        return _empty()

    if df.empty or "net_shares" not in df.columns:
        return _empty()

    df = df.rename(columns=str.lower)
    df["ds"] = _normalize_date(df["date"])
    df["signed_shares"] = pd.to_numeric(df["net_shares"], errors="coerce").fillna(0.0)

    # If flows are never negative, this source is not signed → signal to caller
    all_pos = (df["signed_shares"] >= 0).all()
    out = (
        df.dropna(subset=["ds"])
          .groupby("ds", as_index=False)
          .agg(insider_net_shares=("signed_shares","sum"))
          .sort_values("ds")
          .reset_index(drop=True)
    )
    # provisional counts (will be corrected if we detect signed data)
    out["insider_buy_count"]  = (out["insider_net_shares"] > 0).astype(int)
    out["insider_sell_count"] = (out["insider_net_shares"] < 0).astype(int)
    out.attrs["all_positive"] = bool(all_pos)
    return out

# ------------------------ source: holdings (diff) ---------------------
def _from_holdings_diff(engine, ticker: str) -> pd.DataFrame:
    q = text("""
        SELECT date, shares
          FROM holdings
         WHERE UPPER(ticker)=UPPER(:tkr)
    """)
    try:
        df = pd.read_sql(q, engine, params={"tkr": ticker})
    except Exception as e:
        print(f"holdings query error: {e}")
        return _empty()

    if df.empty or "shares" not in df.columns:
        return _empty()

    df = df.rename(columns=str.lower)
    df["ds"] = _normalize_date(df["date"])
    # Aggregate per day in case multiple rows exist
    daily = (
        df.dropna(subset=["ds"])
          .groupby("ds", as_index=False)
          .agg(total_shares=("shares","sum"))
          .sort_values("ds")
          .reset_index(drop=True)
    )
    if daily.empty:
        return _empty()

    daily["total_shares"] = pd.to_numeric(daily["total_shares"], errors="coerce").fillna(0.0)
    daily["insider_net_shares"] = daily["total_shares"].diff().fillna(0.0)

    out = daily[["ds","insider_net_shares"]].copy()
    out["insider_buy_count"]  = (out["insider_net_shares"] > 0).astype(int)
    out["insider_sell_count"] = (out["insider_net_shares"] < 0).astype(int)
    return out

# ------------------------ source: transactions (neutral) --------------
def _from_transactions_neutral(engine, ticker: str) -> pd.DataFrame:
    # No type/sign info in your table → avoid introducing bias
    q = text("""
        SELECT DISTINCT date
          FROM transactions
         WHERE UPPER(ticker)=UPPER(:tkr)
    """)
    try:
        df = pd.read_sql(q, engine, params={"tkr": ticker})
    except Exception as e:
        print(f"transactions query error: {e}")
        return _empty()

    if df.empty:
        return _empty()

    out = pd.DataFrame({
        "ds": _normalize_date(df["date"]),
        "insider_net_shares": 0.0,
        "insider_buy_count":  0,
        "insider_sell_count": 0,
    }).dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    return out

# ----------------------------- insider  -----------------------------
def _finish_insider_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ETL output: sort/clean via finalize_features and standardize columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "insider_net_shares", "insider_buy_count", "insider_sell_count"])

    df = df.copy()

    # Ensure required columns exist
    for c in ["ds", "insider_net_shares", "insider_buy_count", "insider_sell_count"]:
        if c not in df.columns:
            df[c] = 0

    # Parse ds and drop bad rows
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True)
    df = df.dropna(subset=["ds"])

    # Apply finalize_features on a datetime index, then restore 'ds'
    finalized = finalize_features(df.set_index("ds"))
    df = finalized.reset_index().rename(columns={"index": "ds"})

    # Keep schema/order tidy and strictly sorted
    df = df[["ds", "insider_net_shares", "insider_buy_count", "insider_sell_count"]]
    df = df.sort_values("ds").reset_index(drop=True)
    return df


# ----------------------------- public API -----------------------------
def fetch_insider_trades(ticker: str) -> pd.DataFrame:
    """
    Standardized output:
      ds, insider_net_shares, insider_buy_count, insider_sell_count
    Priority:
      1) insider_flows  — use if it has signed negatives
      2) holdings (diff of daily totals) — signed net flows
      3) transactions (neutral zeros) — last resort
    """
    tkr = str(ticker or "").upper().strip()
    if not tkr:
        return _finish_insider_df(pd.DataFrame())

    try:
        eng  = _get_engine()
        insp = inspect(eng)
        tables = set(insp.get_table_names())
    except Exception as e:
        print(f"insider db error: {e}")
        return _finish_insider_df(pd.DataFrame())

    # 1) insider_flows first
    if "insider_flows" in tables:
        flows = _from_insider_flows(eng, tkr)
        if not flows.empty and not flows.attrs.get("all_positive", False):
            return _finish_insider_df(flows)  # already signed (has negatives)

    # 2) fallback to holdings diff for signed net flows
    if "holdings" in tables:
        hold = _from_holdings_diff(eng, tkr)
        if not hold.empty:
            return _finish_insider_df(hold)

    # 3) final neutral fallback
    if "transactions" in tables:
        tx = _from_transactions_neutral(eng, tkr)
        return _finish_insider_df(tx)

    return _finish_insider_df(pd.DataFrame())
