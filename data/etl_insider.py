import os
import pandas as pd
import feedparser
from datetime import datetime
try:
    import streamlit as st
    ST_SECRETS = st.secrets
except ImportError:
    ST_SECRETS = {}

SEC_HEADERS = {
    "User-Agent": "MLQuantFund/0.1 (contact: your@email.com)"
}

def fetch_insider_trades(ticker: str, count: int = 40) -> pd.DataFrame:
    """
    Fetch recent Form 4 insider trades from SEC RSS feed.
    Returns a DataFrame with:
      - ds: date
      - net_shares: total shares bought - sold
      - num_buy_tx: number of buy transactions
      - num_sell_tx: number of sell transactions
    """
    cik = ticker.upper()
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={cik}&type=4&owner=only&count={count}&output=atom"
    )

    try:
        feed = feedparser.parse(url)
        trades = []
        for entry in feed.entries:
            title = entry.get("title", "").lower()
            date = entry.get("updated", "") or entry.get("published", "")
            ds = pd.to_datetime(date).date() if date else None
            if not ds:
                continue
            buy = "purchase" in title or "buy" in title
            sell = "sale" in title or "sell" in title
            net = 0
            if buy and not sell:
                net = 1
            elif sell and not buy:
                net = -1
            trades.append({"ds": ds, "net_shares": net, "buy": int(buy), "sell": int(sell)})

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        agg = (
            df.groupby("ds")
              .agg(
                  net_shares=("net_shares", "sum"),
                  num_buy_tx=("buy", "sum"),
                  num_sell_tx=("sell", "sum"),
              )
              .reset_index()
        )
        return agg

    except Exception as e:
        print(f"❌ Insider ETL error for {ticker}: {e}")
        return pd.DataFrame()
