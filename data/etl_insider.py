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

# ============================== CONFIG ======================================
def _as_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}

INCLUDE_AWARDS_IN_NET = _as_bool(
    ST_SECRETS.get("INSIDER_INCLUDE_AWARDS_IN_NET", os.environ.get("INSIDER_INCLUDE_AWARDS_IN_NET")),
    default=False,
)
LARGE_TX_THRESHOLD_USD = float(
    ST_SECRETS.get("INSIDER_LARGE_TX_THRESHOLD_USD", os.environ.get("INSIDER_LARGE_TX_THRESHOLD_USD", 1_000_000))
)

# ============================= ENGINE =======================================
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

# ======================= Transaction Code Mapping ============================
TRANSACTION_CODE_MAP = {
    # BUY-like
    "P": {"category": "buy",   "include_in_net": True,  "direction": +1, "notes": "Open-market/Private purchase"},
    # SELL-like
    "S": {"category": "sell",  "include_in_net": True,  "direction": -1, "notes": "Open-market/Private sale"},
    "F": {"category": "sell",  "include_in_net": True,  "direction": -1, "notes": "Sell-to-cover"},
    "D": {"category": "sell",  "include_in_net": True,  "direction": -1, "notes": "Disposition"},
    # Neutral (structural)
    "M": {"category": "exercise",   "include_in_net": False, "direction": 0, "notes": "Option exercise"},
    "C": {"category": "conversion", "include_in_net": False, "direction": 0, "notes": "Conversion of security"},
    "X": {"category": "exercise",   "include_in_net": False, "direction": 0, "notes": "Exercise of derivative"},
    # Grants / awards / gifts / other (excluded by default)
    "A": {"category": "award", "include_in_net": False, "direction": +1, "notes": "Grant/Award"},
    "G": {"category": "gift",  "include_in_net": False, "direction": 0,  "notes": "Bona fide gift"},
    "I": {"category": "other", "include_in_net": False, "direction": 0,  "notes": "Discretionary"},
    # Fallback
    "":  {"category": "unknown","include_in_net": False, "direction": 0,  "notes": "Empty code"},
}

def _classify_transaction(code: str, include_awards_in_net: bool = INCLUDE_AWARDS_IN_NET):
    c = (code or "").strip().upper()
    meta = TRANSACTION_CODE_MAP.get(c, TRANSACTION_CODE_MAP[""])
    include_in_net = meta["include_in_net"]
    direction = meta["direction"]
    category = meta["category"]
    is_exercise_like = category in {"exercise", "conversion"}
    if c == "A" and include_awards_in_net:
        include_in_net = True  # treat awards as inflow if configured
    return category, include_in_net, direction, is_exercise_like

def _compute_net_shares_row(row, include_awards_in_net: bool = INCLUDE_AWARDS_IN_NET):
    code = row.get("transactionCode") or row.get("code") or ""
    shares = row.get("transactionShares") or row.get("shares") or 0
    try:
        shares = float(shares)
    except Exception:
        shares = 0.0
    category, include_in_net, direction, is_exercise_like = _classify_transaction(code, include_awards_in_net)
    net_shares = (direction * shares) if include_in_net else 0.0
    return {
        "category": category,
        "net_shares": net_shares,
        "is_buy": (category == "buy") and (net_shares > 0),
        "is_sell": (category == "sell") and (net_shares < 0),
        "is_exercise_like": bool(is_exercise_like),
    }

# ========================= source: insider_flows =============================
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

# ========================= source: holdings (diff) ===========================
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

# ===================== source: transactions (SIGNED) =========================
def _from_transactions_signed(engine, ticker: str) -> pd.DataFrame:
    """
    Expecting a transactions table with at least:
      date, transactionCode (or code), transactionShares (or shares),
      transactionPrice (optional), officerTitle (optional)
    Produces daily signed net_shares + counts.
    """
    q = text("""
        SELECT
            date,
            transactionCode,
            transactionShares,
            transactionPrice,
            officerTitle
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

    df = df.rename(columns=str.lower)
    if "transactioncode" in df.columns and "transactionCode" not in df.columns:
        df = df.rename(columns={"transactioncode": "transactionCode"})
    if "transactionshares" in df.columns and "transactionShares" not in df.columns:
        df = df.rename(columns={"transactionshares": "transactionShares"})
    if "transactionprice" in df.columns and "transactionPrice" not in df.columns:
        df = df.rename(columns={"transactionprice": "transactionPrice"})

    df["ds"] = _normalize_date(df["date"])
    df = df.dropna(subset=["ds"]).reset_index(drop=True)
    if df.empty:
        return _empty()

    # Annotate rows with signed net_shares and flags
    anno = df.apply(lambda r: _compute_net_shares_row(r, INCLUDE_AWARDS_IN_NET), axis=1, result_type="expand")
    anno = pd.DataFrame(anno, columns=["category","net_shares","is_buy","is_sell","is_exercise_like"])
    tmp = pd.concat([df, anno], axis=1)

    # Optional per-row value (for future use/flags)
    def _tx_value(r):
        try:
            p = float(r.get("transactionPrice") or 0.0)
            q = float(r.get("transactionShares") or 0.0)
            return p * q
        except Exception:
            return 0.0
    tmp["txn_value_usd"] = tmp.apply(_tx_value, axis=1)

    # Aggregate to daily
    daily = (
        tmp.groupby("ds", as_index=False)
           .agg(
               insider_net_shares=("net_shares", "sum"),
               insider_buy_count=("is_buy", "sum"),
               insider_sell_count=("is_sell", "sum"),
               # internal metrics (not returned, but useful for debugging)
               _num_exercise_like=("is_exercise_like", "sum"),
               _max_txn_value_usd=("txn_value_usd", "max"),
           )
           .sort_values("ds")
           .reset_index(drop=True)
    )

    # Nothing prevents all-positive if only awards and INCLUDE_AWARDS_IN_NET=True
    return daily

# ============================== FALLBACK: neutral ============================
def _from_transactions_neutral(engine, ticker: str) -> pd.DataFrame:
    # Last resort if schema lacks code/shares: produce zeros on filing days
    q = text("""
        SELECT DISTINCT date
          FROM transactions
         WHERE UPPER(ticker)=UPPER(:tkr)
    """)
    try:
        df = pd.read_sql(q, engine, params={"tkr": ticker})
    except Exception as e:
        print(f"transactions(neutral) query error: {e}")
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

# ============================== Finalizer ===================================
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

# ============================== Public API ==================================
def fetch_insider_trades(ticker: str) -> pd.DataFrame:
    """
    Standardized output:
      ds, insider_net_shares, insider_buy_count, insider_sell_count

    Priority:
      1) insider_flows        — use if it has signed negatives
      2) transactions SIGNED  — compute from transactionCode/shares
      3) holdings (diff)      — signed via daily holdings delta
      4) transactions NEUTRAL — zeros on filing days

    Config:
      - INSIDER_INCLUDE_AWARDS_IN_NET (env/secret) to treat grants 'A' as inflow.
      - INSIDER_LARGE_TX_THRESHOLD_USD (env/secret) currently internal only.
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

    # 1) insider_flows first (only if it's truly signed i.e., has negatives)
    if "insider_flows" in tables:
        flows = _from_insider_flows(eng, tkr)
        if not flows.empty and not flows.attrs.get("all_positive", False):
            return _finish_insider_df(flows)

    # 2) try signed transactions if table has the necessary columns
    if "transactions" in tables:
        # Peek columns to decide signed vs neutral path
        try:
            cols = set(pd.read_sql(text("SELECT * FROM transactions WHERE 1=0"), eng).columns.str.lower())
        except Exception:
            cols = set()

        needed_any = {"transactioncode", "code"} & cols
        shares_any = {"transactionshares", "shares"} & cols

        if needed_any and shares_any:
            tx_signed = _from_transactions_signed(eng, tkr)
            if not tx_signed.empty:
                return _finish_insider_df(tx_signed)

    # 3) fallback to holdings diff for signed net flows
    if "holdings" in tables:
        hold = _from_holdings_diff(eng, tkr)
        if not hold.empty:
            return _finish_insider_df(hold)

    # 4) final neutral fallback from transactions (dates only)
    if "transactions" in tables:
        tx = _from_transactions_neutral(eng, tkr)
        return _finish_insider_df(tx)

    return _finish_insider_df(pd.DataFrame())
