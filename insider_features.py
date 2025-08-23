# insider_features.py
import pandas as pd
import numpy as np

REQUIRED_INSIDER_COLS = [
    "ticker","filed_date","net_shares","num_buy_tx","num_sell_tx",
    "num_exercise_like","max_txn_value_usd","any_exec_trade","large_tx_flag","holdings_delta"
]

def _ensure_dt(df, col="filed_date"):
    d = df.copy()
    if col in d.columns:
        d[col] = pd.to_datetime(d[col], errors="coerce").dt.date
    return d

def build_daily_insider_features(final_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Input (from ETL merge step):
      final_daily columns:
        ['ticker','filed_date','net_shares','num_buy_tx','num_sell_tx','num_exercise_like',
         'max_txn_value_usd','any_exec_trade','large_tx_flag','holdings_delta']
    Output: daily insider feature frame (same grain: ticker+date)
    """
    if final_daily is None or len(final_daily) == 0:
        return pd.DataFrame(columns=["ticker","date"])

    d = final_daily.copy()
    # normalize schema
    d = _ensure_dt(d, "filed_date").rename(columns={"filed_date":"date"})
    # Safety: keep only known columns if present
    keep = ["ticker","date"] + [c for c in REQUIRED_INSIDER_COLS if c in final_daily.columns and c not in ["ticker","filed_date"]]
    d = d[keep].copy()

    # Base daily signals
    d["exec_or_large_tx_flag"] = (d.get("any_exec_trade", False).astype(bool) | d.get("large_tx_flag", False).astype(bool)).astype(int)
    d["buy_minus_sell_ct"] = d.get("num_buy_tx", 0) - d.get("num_sell_tx", 0)
    d["abs_net_shares"] = d.get("net_shares", 0).abs()

    # Done—return daily level; rolling will be computed after joining with a full date index
    return d

def add_rolling_insider_features(price_df: pd.DataFrame, insider_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
      price_df: columns ['ticker','date', ...] daily calendar for each ticker (no gaps preferred)
      insider_daily_df: output of build_daily_insider_features()
    Returns:
      price_df with insider features merged + rolling aggregates.
    """
    if insider_daily_df is None or len(insider_daily_df) == 0:
        # return original with empty cols to keep downstream stable
        out = price_df.copy()
        for c in [
            "ins_net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta",
            "ins_exec_or_large_flag","ins_buy_minus_sell_ct","ins_abs_net_shares",
            "ins_net_shares_7d","ins_net_shares_30d",
            "ins_buy_ct_7d","ins_sell_ct_7d",
            "ins_holdings_delta_7d","ins_holdings_delta_30d",
            "ins_pressure_7d","ins_pressure_30d",
            "ins_large_or_exec_7d","ins_large_or_exec_30d"
        ]:
            out[c] = 0
        return out

    d = insider_daily_df.copy()
    d = _ensure_dt(d, "date")

    # Rename to namespaced columns before merge
    rename_map = {
        "net_shares": "ins_net_shares",
        "num_buy_tx": "ins_buy_ct",
        "num_sell_tx": "ins_sell_ct",
        "holdings_delta": "ins_holdings_delta",
        "exec_or_large_tx_flag": "ins_exec_or_large_flag",
        "buy_minus_sell_ct": "ins_buy_minus_sell_ct",
        "abs_net_shares": "ins_abs_net_shares",
    }
    d = d.rename(columns=rename_map)

    # Merge onto price_df
    out = price_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date

    out = out.merge(d, on=["ticker","date"], how="left")

    # Fill NA after merge
    fill_zeros = [
        "ins_net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta",
        "ins_exec_or_large_flag","ins_buy_minus_sell_ct","ins_abs_net_shares"
    ]
    out[fill_zeros] = out[fill_zeros].fillna(0)

    # Optional normalization if you have float shares or market cap.
    # Provide either 'shares_outstanding' or 'market_cap' in price_df.
    if "shares_outstanding" in out.columns:
        so = out["shares_outstanding"].replace({0:np.nan})
        out["ins_net_shares_norm"] = out["ins_net_shares"] / so
        out["ins_holdings_delta_norm"] = out["ins_holdings_delta"] / so
    elif "market_cap" in out.columns and "close" in out.columns:
        # Convert shares via market_cap/price as rough proxy
        shares_proxy = (out["market_cap"] / out["close"]).replace({0:np.nan})
        out["ins_net_shares_norm"] = out["ins_net_shares"] / shares_proxy
        out["ins_holdings_delta_norm"] = out["ins_holdings_delta"] / shares_proxy
    else:
        out["ins_net_shares_norm"] = 0.0
        out["ins_holdings_delta_norm"] = 0.0

    # Rolling features per ticker
    out = out.sort_values(["ticker","date"])
    def roll(group):
        g = group.copy()
        g["ins_net_shares_7d"] = g["ins_net_shares"].rolling(7, min_periods=1).sum()
        g["ins_net_shares_30d"] = g["ins_net_shares"].rolling(30, min_periods=1).sum()
        g["ins_buy_ct_7d"] = g["ins_buy_ct"].rolling(7, min_periods=1).sum()
        g["ins_sell_ct_7d"] = g["ins_sell_ct"].rolling(7, min_periods=1).sum()
        g["ins_holdings_delta_7d"] = g["ins_holdings_delta"].rolling(7, min_periods=1).sum()
        g["ins_holdings_delta_30d"] = g["ins_holdings_delta"].rolling(30, min_periods=1).sum()
        g["ins_large_or_exec_7d"] = g["ins_exec_or_large_flag"].rolling(7, min_periods=1).sum()
        g["ins_large_or_exec_30d"] = g["ins_exec_or_large_flag"].rolling(30, min_periods=1).sum()

        # Insider pressure = net_shares + 0.5*holdings_delta + 0.25*buy_minus_sell_ct (tunable)
        base = g["ins_net_shares"] + 0.5*g["ins_holdings_delta"] + 0.25*g["ins_buy_minus_sell_ct"]
        g["ins_pressure_7d"] = base.rolling(7, min_periods=1).sum()
        g["ins_pressure_30d"] = base.rolling(30, min_periods=1).sum()

        # Optionally z-score normalize rolling insider pressure per ticker
        g["ins_pressure_30d_z"] = (g["ins_pressure_30d"] - g["ins_pressure_30d"].rolling(120, min_periods=20).mean()) / (
            g["ins_pressure_30d"].rolling(120, min_periods=20).std(ddof=0)
        )
        return g

    out = out.groupby("ticker", group_keys=False).apply(roll)

    # Final NA safety
    out = out.fillna(0)
    return out
