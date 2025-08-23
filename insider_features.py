# ml_quant_fund/insider_features.py
# Minimal, robust insider feature builders used by the UI.

from __future__ import annotations
import numpy as np
import pandas as pd

def build_daily_insider_features(final_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ETL 'final_daily' into one row per (ticker, date) with basic columns the UI expects.
    Expected (but optional) columns in final_daily:
      - filed_date/date, ticker
      - net_shares, num_buy_tx, num_sell_tx, holdings_delta
      - any_exec_trade, large_tx_flag, max_txn_value_usd
    """
    if final_daily is None or final_daily.empty:
        return pd.DataFrame(columns=[
            "ticker","date","ins_net_shares","ins_buy_ct","ins_sell_ct",
            "ins_holdings_delta","any_exec_trade","large_tx_flag","max_txn_value_usd"
        ])

    df = final_daily.copy()

    # Ensure date & ticker
    if "filed_date" in df.columns:
        df["date"] = pd.to_datetime(df["filed_date"], errors="coerce").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        raise ValueError("final_daily must include 'filed_date' or 'date'.")

    if "ticker" not in df.columns:
        raise ValueError("final_daily must include 'ticker'.")

    # Ensure the numeric/flag columns exist
    defaults = {
        "net_shares": 0.0,
        "num_buy_tx": 0.0,
        "num_sell_tx": 0.0,
        "holdings_delta": 0.0,
        "any_exec_trade": 0,
        "large_tx_flag": 0,
        "max_txn_value_usd": 0.0,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    # Coerce numerics & flags
    num_cols = ["net_shares","num_buy_tx","num_sell_tx","holdings_delta","max_txn_value_usd"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["any_exec_trade","large_tx_flag"]:
        df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int)

    # Aggregate to daily
    agg = (
        df.groupby(["ticker","date"], as_index=False)
          .agg(
              net_shares        = ("net_shares","sum"),
              ins_buy_ct        = ("num_buy_tx","sum"),
              ins_sell_ct       = ("num_sell_tx","sum"),
              ins_holdings_delta= ("holdings_delta","sum"),
              any_exec_trade    = ("any_exec_trade","max"),
              large_tx_flag     = ("large_tx_flag","max"),
              max_txn_value_usd = ("max_txn_value_usd","max"),
          )
    )
    agg = agg.rename(columns={"net_shares": "ins_net_shares"})
    return agg

def _roll_sum(s: pd.Series, window: int, minp: int = 1) -> pd.Series:
    return s.rolling(window, min_periods=minp).sum()

def add_rolling_insider_features(px_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price calendar with daily insider features and add rolling aggregates:
      - ins_net_shares_{7d,30d}
      - ins_buy_ct_{7d,30d}, ins_sell_ct_{7d,30d}
      - ins_large_or_exec_7d
      - ins_pressure_30d, ins_pressure_30d_z
      - PLUS derived daily columns expected elsewhere:
        * ins_exec_or_large_flag  (alias of any_exec_trade OR large_tx_flag)
        * ins_buy_minus_sell_ct   (ins_buy_ct - ins_sell_ct)
        * ins_abs_net_shares      (abs(ins_net_shares))
    """
    if px_df is None or px_df.empty:
        return pd.DataFrame()

    cal = px_df.copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.date
    if "ticker" not in cal.columns:
        cal["ticker"] = cal.get("Ticker", cal.get("symbol", ""))

    d = daily_df.copy() if isinstance(daily_df, pd.DataFrame) else pd.DataFrame()
    if d.empty:
        d = pd.DataFrame(columns=[
            "ticker","date","ins_net_shares","ins_buy_ct","ins_sell_ct",
            "ins_holdings_delta","any_exec_trade","large_tx_flag","max_txn_value_usd"
        ])
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date

    m = cal.merge(d, on=["ticker","date"], how="left")

    # Ensure numeric/flag columns exist
    for c in ["ins_net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta","any_exec_trade","large_tx_flag"]:
        if c not in m.columns:
            m[c] = 0
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    # Base flags
    m["is_large_or_exec"] = ((m["any_exec_trade"] > 0) | (m["large_tx_flag"] > 0)).astype(int)

    # --- Derived daily columns REQUIRED elsewhere ---
    m["ins_exec_or_large_flag"] = m["is_large_or_exec"].astype(int)
    m["ins_buy_minus_sell_ct"]  = (m["ins_buy_ct"] - m["ins_sell_ct"]).astype(float)
    m["ins_abs_net_shares"]     = m["ins_net_shares"].abs().astype(float)

    m = m.sort_values(["ticker","date"])

    def _roll_sum(s: pd.Series, window: int, minp: int = 1) -> pd.Series:
        return s.rolling(window, min_periods=minp).sum()

    def _by_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # Rolling sums (~trading-day windows by row count)
        g["ins_net_shares_7d"]    = _roll_sum(g["ins_net_shares"], 7)
        g["ins_net_shares_30d"]   = _roll_sum(g["ins_net_shares"], 30)
        g["ins_buy_ct_7d"]        = _roll_sum(g["ins_buy_ct"], 7)
        g["ins_buy_ct_30d"]       = _roll_sum(g["ins_buy_ct"], 30)
        g["ins_sell_ct_7d"]       = _roll_sum(g["ins_sell_ct"], 7)
        g["ins_sell_ct_30d"]      = _roll_sum(g["ins_sell_ct"], 30)
        g["ins_large_or_exec_7d"] = _roll_sum(g["is_large_or_exec"], 7)

        # Pressure: z-score of 30d flow (optionally scaled by shares_outstanding)
        base = g["ins_net_shares_30d"].astype(float)
        if "shares_outstanding" in g.columns and g["shares_outstanding"].notna().any():
            denom = pd.to_numeric(g["shares_outstanding"], errors="coerce").replace(0, np.nan)
            base = base / denom

        mean180 = base.rolling(180, min_periods=30).mean()
        std180  = base.rolling(180, min_periods=30).std(ddof=0)
        g["ins_pressure_30d"]   = base
        g["ins_pressure_30d_z"] = (base - mean180) / std180.replace(0, np.nan)

        return g

    out = m.groupby("ticker", group_keys=False).apply(_by_ticker)

    # Guarantee the three expected columns exist even if groupby path changed
    for c in ["ins_exec_or_large_flag","ins_buy_minus_sell_ct","ins_abs_net_shares"]:
        if c not in out.columns:
            out[c] = 0

    return out

