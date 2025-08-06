# data/etl_insider.py

import os
import sys
from datetime import datetime
import pandas as pd
import feedparser

# ensure project root is on sys.path for imports
ROOT = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, os.pardir))
if PROJ not in sys.path:
    sys.path.append(PROJ)

try:
    import streamlit as st
    ST_SECRETS = st.secrets
except ImportError:
    ST_SECRETS = {}

# Local Excel fallback
LOCAL_XLSX     = os.path.join("data", "Insider_Trades_Data.xlsx")
# Google Sheet config
GSHEET_NAME      = "Insider_Trades_Data"
TAB_TRANSACTIONS = "Insider_Transactions"

def fetch_insider_trades(ticker: str, mode: str = "sheet-first") -> pd.DataFrame:
    """
    Fetch insider trades for a ticker.
      mode:
        - sheet-first: try Google Sheet, then RSS, then Excel
        - rss:         RSS then Excel
        - excel:       Excel only
    """
    ticker = ticker.upper().strip()

    # 1) sheet-first
    if mode in ("sheet-first", "sheet"):
        df = _fetch_from_sheet(ticker)
        if not df.empty:
            return df

    # 2) RSS
    if mode in ("sheet-first", "rss"):
        df = _fetch_from_rss(ticker)
        if not df.empty:
            return df

    # 3) Excel fallback
    return _fetch_from_excel(ticker)


def _fetch_from_sheet(ticker: str) -> pd.DataFrame:
    """Pulls data from your Google Sheet tab."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(
            ST_SECRETS["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open(GSHEET_NAME)
        ws = sh.worksheet(TAB_TRANSACTIONS)
        df = pd.DataFrame(ws.get_all_records())

        # normalize headers to uppercase snake
        df.columns = [
            c.strip().upper().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        # helper to find a column containing all subs
        def find_col(subs: list[str]) -> str | None:
            for col in df.columns:
                if all(s in col for s in subs):
                    return col
            return None

        date_col   = find_col(["TRANS", "DATE"])
        shares_col = find_col(["TRANS", "SHARE"])
        code_col   = find_col(["TRANS", "CODE"])
        ticker_col = find_col(["ISSUER", "SYMBOL"]) or find_col(["TICKER"])

        if not all([date_col, shares_col, code_col, ticker_col]):
            print("❌ Missing sheet columns:", date_col, shares_col, code_col, ticker_col)
            return pd.DataFrame()

        df = df[df[ticker_col].str.upper().str.strip() == ticker]
        if df.empty:
            return pd.DataFrame()

        df["ds"]     = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df[shares_col], errors="coerce").fillna(0)
        df["code"]   = df[code_col].str.strip().str.upper()

        def classify(row):
            c, s = row["code"], row["shares"]
            if c in ("P", "M", "A", "G", "F"):
                return ( s, 1, 0 )
            if c == "S":
                return (-s, 0, 1)
            return ( 0, 0, 0 )

        triples = df.apply(classify, axis=1).tolist()
        df[["net_shares","num_buy_tx","num_sell_tx"]] = pd.DataFrame(triples, index=df.index)

        return (
            df.groupby("ds")
              .agg(
                net_shares  = ("net_shares",  "sum"),
                num_buy_tx  = ("num_buy_tx",  "sum"),
                num_sell_tx = ("num_sell_tx", "sum"),
              )
              .reset_index()
        )

    except Exception as e:
        print("❌ Exception during Google Sheet fetch:", e)
        return pd.DataFrame()


def _fetch_from_rss(ticker: str, count: int = 40) -> pd.DataFrame:
    """Pulls latest Form 4 filings via the SEC RSS feed."""
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?"
        f"action=getcompany&CIK={ticker}&type=4&owner=only"
        f"&count={count}&output=atom"
    )
    try:
        feed = feedparser.parse(url)
        trades = []
        for e in feed.entries:
            title = e.get("title","").lower()
            date  = e.get("updated","") or e.get("published","")
            ds    = pd.to_datetime(date, errors="coerce").date()
            if pd.isna(ds):
                continue

            buy  = "purchase" in title or "buy" in title
            sell = "sale"    in title or "sell" in title
            net  = 1 if buy and not sell else -1 if sell and not buy else 0
            trades.append({"ds":ds, "net_shares":net, "num_buy_tx":int(buy), "num_sell_tx":int(sell)})

        if not trades:
            return pd.DataFrame()

        return (
            pd.DataFrame(trades)
              .groupby("ds")
              .sum()
              .reset_index()
        )
    except Exception as e:
        print("❌ RSS fetch error for", ticker, e)
        return pd.DataFrame()


def _fetch_from_excel(ticker: str) -> pd.DataFrame:
    """Reads your local Excel fallback file."""
    try:
        df = pd.read_excel(LOCAL_XLSX, sheet_name=TAB_TRANSACTIONS)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[df["ISSUERTRADINGSYMBOL"].str.upper().str.strip() == ticker]
        if df.empty:
            return pd.DataFrame()

        df["ds"]     = pd.to_datetime(df["TRANS_DATE"], errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df["TRANSACTIONSHARES"], errors="coerce").fillna(0)
        df["code"]   = df["TRANSACTIONCODE"].str.strip().str.upper()

        def classify(row):
            if row["code"] in ("P", "M", "A", "G", "F"):
                return ( row["shares"], 1, 0 )
            if row["code"] == "S":
                return (-row["shares"], 0, 1)
            return ( 0, 0, 0 )

        triples = df.apply(classify, axis=1).tolist()
        df[["net_shares","num_buy_tx","num_sell_tx"]] = pd.DataFrame(triples, index=df.index)

        return (
            df.groupby("ds")
              .agg(
                net_shares  = ("net_shares",  "sum"),
                num_buy_tx  = ("num_buy_tx",  "sum"),
                num_sell_tx = ("num_sell_tx", "sum"),
              )
              .reset_index()
        )

    except Exception as e:
        print(f"❌ Excel fallback error for {ticker}: {e}")
        return pd.DataFrame()
