# data/etl_congress.py

import os
import requests
import pandas as pd
from datetime import datetime
try:
    import streamlit as st
    SECRETS = st.secrets
except ImportError:
    SECRETS = {}

# Grab your Quiver API key from env or Streamlit Secrets
QUIVER_API_KEY = os.getenv("QUIVER_API_KEY") or SECRETS.get("QUIVER_API_KEY")

def fetch_congress_trades(ticker: str) -> pd.DataFrame:
    """
    Pull all recent congressional trades for `ticker`.
    Returns a DataFrame with columns:
      - ds: date of trade
      - congress_net_shares: +shares for buys, -shares for sells
      - congress_active_members: number of unique members trading that day
    """
    if not QUIVER_API_KEY:
        raise RuntimeError("Missing QUIVER_API_KEY for QuiverCongress ETL")

    url = (
        "https://api.quiverquant.com/beta/historical/congresstrading"
        f"?ticker={ticker}"
    )
    headers = {"Authorization": f"Token {QUIVER_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=8)
    resp.raise_for_status()
    data = resp.json()  # list of trade dicts

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # standardize date
    df["ds"] = pd.to_datetime(df["transactionDate"]).dt.date

    # shares is signed: positive = buy, negative = sell
    df["shares"] = df["shares"].astype(int)

    # aggregate per day
    agg = (
        df.groupby("ds")
          .agg(
             congress_net_shares = ("shares", "sum"),
             congress_active_members = ("memberName", "nunique")
          )
          .reset_index()
    )
    return agg
