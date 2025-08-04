import os
import pandas as pd
import feedparser
from datetime import datetime
from data.utils_cik import load_cik_to_ticker_map

try:
    import streamlit as st
    ST_SECRETS = st.secrets
except ImportError:
    ST_SECRETS = {}

SEC_HEADERS = {
    "User-Agent": "MLQuantFund/0.1 (contact: your@email.com)"
}

CIK_MAP = load_cik_to_ticker_map()
FORM4_TSV_PATH = "data/SEC/FULL_FORM4_COMBINED.tsv"

# Google Sheet config
GSHEET_NAME = "Insider_Trades_Data"
TAB_TRANSACTIONS = "Insider_Transactions"
TAB_HOLDINGS = "Insider_Holdings"

def fetch_insider_trades(ticker: str, mode: str = "sheet-first") -> pd.DataFrame:
    """
    Fetch insider trades:
    mode:
        - "sheet-first": use Insider_Trades_Data Google Sheet (preferred)
        - "rss": SEC RSS feed
        - "tsv": fallback to SEC Form 4 TSV file
        - "hybrid": try RSS then fallback
    """
    ticker = ticker.upper()

    if mode == "sheet-first":
        df = _fetch_from_sheet(ticker)
        if not df.empty:
            return df

    if mode == "rss" or mode == "hybrid":
        df = _fetch_from_rss(ticker)
        if not df.empty or mode == "rss":
            return df

    if mode == "tsv" or mode == "hybrid":
        return _fetch_from_tsv(ticker)

    return pd.DataFrame()


def _fetch_from_sheet(ticker: str) -> pd.DataFrame:
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            ST_SECRETS["gcp_service_account"], scope
        )
        gc = gspread.authorize(creds)
        sh = gc.open(GSHEET_NAME)

        ws = sh.worksheet(TAB_TRANSACTIONS)
        df = pd.DataFrame(ws.get_all_records())

        if "TRANS_DATE" not in df.columns or "ISSUERTRADINGSYMBOL" not in df.columns:
            return pd.DataFrame()

        df["ds"] = pd.to_datetime(df["TRANS_DATE"], errors="coerce").dt.date
        df = df[df["ISSUERTRADINGSYMBOL"].str.upper() == ticker]

        df["net_shares"] = 1   # You can refine this with quantity logic later
        df["num_buy_tx"] = 1   # Treat all as buys for now
        df["num_sell_tx"] = 0  # (Optional) you can improve this

        return (
            df.groupby("ds")
              .agg(net_shares=("net_shares", "sum"),
                   num_buy_tx=("num_buy_tx", "sum"),
                   num_sell_tx=("num_sell_tx", "sum"))
              .reset_index()
        )

    except Exception as e:
        print(f"❌ Google Sheet insider fetch error: {e}")
        return pd.DataFrame()


def _fetch_from_rss(ticker: str, count: int = 40) -> pd.DataFrame:
    """Try live RSS feed from SEC."""
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?"
        f"action=getcompany&CIK={ticker}&type=4&owner=only&count={count}&output=atom"
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
        return (
            df.groupby("ds")
              .agg(net_shares=("net_shares", "sum"),
                   num_buy_tx=("buy", "sum"),
                   num_sell_tx=("sell", "sum"))
              .reset_index()
        )
    except Exception as e:
        print(f"❌ RSS insider error for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_from_tsv(ticker: str) -> pd.DataFrame:
    try:
        cik_map = {v.upper(): k for k, v in CIK_MAP.items()}
        cik = cik_map.get(ticker)
        if not cik:
            print(f"⚠️ No CIK found for {ticker}")
            return pd.DataFrame()

        df = pd.read_csv(FORM4_TSV_PATH, sep="\t", dtype=str, low_memory=False)
        df = df[df["CIK"] == cik]

        df["ds"] = pd.to_datetime(df["transactionDate"], errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df["transactionShares"], errors="coerce").fillna(0)

        def classify(row):
            code = str(row["transactionCode"]).strip().upper()
            if code in ("P", "M"):
                return row["shares"], 1, 0
            elif code in ("S",):
                return -row["shares"], 0, 1
            else:
                return 0, 0, 0

        classified = df.apply(classify, axis=1, result_type="expand")
        df["net_shares"], df["num_buy_tx"], df["num_sell_tx"] = classified.T

        return (
            df.groupby("ds")
              .agg(net_shares=("net_shares", "sum"),
                   num_buy_tx=("num_buy_tx", "sum"),
                   num_sell_tx=("num_sell_tx", "sum"))
              .reset_index()
        )
    except Exception as e:
        print(f"❌ Insider TSV fallback error: {e}")
        return pd.DataFrame()
