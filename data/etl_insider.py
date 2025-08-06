# data/etl_insider.py_v1.3

import os, sys
# ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(__file__, os.pardir))
PROJ = os.path.abspath(os.path.join(ROOT, os.pardir))
if PROJ not in sys.path:
    sys.path.append(PROJ)

import pandas as pd
import feedparser
from data.utils_cik import load_cik_to_ticker_map

try:
    import streamlit as st
    ST_SECRETS = st.secrets
except ImportError:
    ST_SECRETS = {}

CIK_MAP         = load_cik_to_ticker_map()
LOCAL_XLSX      = os.path.join(PROJ, "data", "Insider_Trades_Data.xlsx")

GSHEET_NAME      = "Insider_Trades_Data"
TAB_TRANSACTIONS = "Insider_Transactions"


def fetch_insider_trades(ticker: str, mode: str = "sheet-first") -> pd.DataFrame:
    ticker = ticker.upper().strip()

    # 1) Sheet
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
        ws = gc.open(GSHEET_NAME).worksheet(TAB_TRANSACTIONS)
        df = pd.DataFrame(ws.get_all_records())

        # normalize headers
        df.columns = [
            c.strip().upper().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        def find_col(keywords):
            for col in df.columns:
                if all(k in col for k in keywords):
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

        def classify(r):
            c, s = r["code"], r["shares"]
            if c in ("P","M","A","G","F"): return (s, 1, 0)
            if c == "S":                   return (-s, 0, 1)
            return (0,0,0)

        triples = df.apply(classify, axis=1, result_type="expand")
        df["net_shares"], df["num_buy_tx"], df["num_sell_tx"] = triples[0], triples[1], triples[2]

        return (
            df.groupby("ds")
              .agg(net_shares=("net_shares","sum"),
                   num_buy_tx=("num_buy_tx","sum"),
                   num_sell_tx=("num_sell_tx","sum"))
              .reset_index()
        )

    except Exception as e:
        print("❌ Exception during Google Sheet fetch:", e)
        return pd.DataFrame()


def _fetch_from_rss(ticker: str, count: int = 40) -> pd.DataFrame:
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
            ds    = pd.to_datetime(date).date() if date else None
            if not ds: continue

            buy  = "purchase" in title or "buy"  in title
            sell = "sale"    in title or "sell" in title
            net  = 1 if buy and not sell else -1 if sell and not buy else 0
            trades.append({"ds":ds, "net_shares":net, "buy":int(buy), "sell":int(sell)})

        if not trades:
            return pd.DataFrame()

        dfx = pd.DataFrame(trades)
        return (
            dfx.groupby("ds")
               .agg(net_shares=("net_shares","sum"),
                    num_buy_tx=("buy","sum"),
                    num_sell_tx=("sell","sum"))
               .reset_index()
        )
    except Exception as e:
        print("❌ RSS fetch error for", ticker, e)
        return pd.DataFrame()


def _fetch_from_excel(ticker: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(LOCAL_XLSX, sheet_name=TAB_TRANSACTIONS)
        # normalize headers
        df.columns = [
            c.strip().upper().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        def find_col(keywords):
            for col in df.columns:
                if all(k in col for k in keywords):
                    return col
            return None

        date_col   = find_col(["TRANS", "DATE"])
        shares_col = find_col(["TRANS", "SHARE"])
        code_col   = find_col(["TRANS", "CODE"])
        ticker_col = find_col(["ISSUER", "SYMBOL"]) or find_col(["TICKER"])

        if not all([date_col, shares_col, code_col, ticker_col]):
            print("❌ Missing excel columns:", date_col, shares_col, code_col, ticker_col)
            return pd.DataFrame()

        df = df[df[ticker_col].str.upper().str.strip() == ticker]
        if df.empty:
            return pd.DataFrame()

        df["ds"]     = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df[shares_col], errors="coerce").fillna(0)
        df["code"]   = df[code_col].str.strip().str.upper()

        def classify(r):
            if r["code"] in ("P","M","A","G","F"): return (r["shares"],1,0)
            if r["code"] == "S":                   return (-r["shares"],0,1)
            return (0,0,0)

        triples = df.apply(classify, axis=1, result_type="expand")
        df["net_shares"], df["num_buy_tx"], df["num_sell_tx"] = triples[0], triples[1], triples[2]

        return (
            df.groupby("ds")
              .agg(net_shares=("net_shares","sum"),
                   num_buy_tx=("num_buy_tx","sum"),
                   num_sell_tx=("num_sell_tx","sum"))
              .reset_index()
        )
    except Exception as e:
        print("❌ Excel fallback error for", ticker, e)
        return pd.DataFrame()
