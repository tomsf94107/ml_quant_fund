# data/etl_insider.py

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

CIK_MAP        = load_cik_to_ticker_map()
FORM4_TSV_PATH = "data/SEC/FULL_FORM4_COMBINED.tsv"

GSHEET_NAME      = "Insider_Trades_Data"
TAB_TRANSACTIONS = "Insider_Transactions"


def fetch_insider_trades(ticker: str, mode: str = "sheet-first") -> pd.DataFrame:
    """
    Fetch insider trades for a ticker.
      mode:
        - sheet-first: try Google Sheet
        - rss: SEC RSS feed
        - tsv: fallback to local TSV
        - hybrid: RSS then TSV
    """
    ticker = ticker.upper().strip()

    if mode in ("sheet-first", "sheet"):
        df = _fetch_from_sheet(ticker)
        if not df.empty:
            return df

    if mode in ("rss", "hybrid"):
        df = _fetch_from_rss(ticker)
        if not df.empty or mode == "rss":
            return df

    if mode in ("tsv", "hybrid"):
        return _fetch_from_tsv(ticker)

    return pd.DataFrame()


def _fetch_from_sheet(ticker: str) -> pd.DataFrame:
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # authenticate
        creds = Credentials.from_service_account_info(
            ST_SECRETS["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)

        # open and pull
        sh = gc.open(GSHEET_NAME)
        ws = sh.worksheet(TAB_TRANSACTIONS)
        df = pd.DataFrame(ws.get_all_records())

        # normalize headers
        df.columns = [c.strip().upper().replace(" ", "_").replace("-", "_") for c in df.columns]

        # helper: find a single column containing all keys
        def find_col(keys):
            for col in df.columns:
                if all(k in col for k in keys):
                    return col
            return None

        date_col   = find_col(["TRANS", "DATE"])
        shares_col = find_col(["TRANS", "SHARE"])
        code_col   = find_col(["TRANS", "CODE"])
        ticker_col = find_col(["ISSUER", "SYMBOL"]) or find_col(["TICKER"])

        if not all([date_col, shares_col, code_col, ticker_col]):
            print("❌ Missing sheet columns:",
                  {
                    "TRANS_DATE": bool(date_col),
                    "TRANS_SHARES": bool(shares_col),
                    "TRANS_CODE": bool(code_col),
                    "TICKER_COL": bool(ticker_col),
                  })
            return pd.DataFrame()

        # filter rows
        df = df[df[ticker_col].str.upper().str.strip() == ticker]
        if df.empty:
            return pd.DataFrame()

        # parse and classify
        df["ds"]     = pd.to_datetime(df[date_col],   errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df[shares_col],  errors="coerce").fillna(0)
        df["code"]   = df[code_col].str.strip().str.upper()

        def classify(r):
            c, s = r["code"], r["shares"]
            if c in ("P", "M", "A", "G", "F"):
                return ( s, 1, 0 )
            if c == "S":
                return (-s, 0, 1)
            return ( 0, 0, 0 )

        triples = df.apply(classify, axis=1).tolist()
        df[["net_shares","num_buy_tx","num_sell_tx"]] = pd.DataFrame(triples, index=df.index)

        # aggregate
        return (
            df.groupby("ds")
              .agg(
                net_shares   = ("net_shares",   "sum"),
                num_buy_tx   = ("num_buy_tx",   "sum"),
                num_sell_tx  = ("num_sell_tx",  "sum"),
              )
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
               .agg(
                 net_shares   = ("net_shares","sum"),
                 num_buy_tx   = ("buy",      "sum"),
                 num_sell_tx  = ("sell",     "sum"),
               )
               .reset_index()
        )

    except Exception as e:
        print("❌ RSS fetch error for", ticker, e)
        return pd.DataFrame()


def _fetch_from_tsv(ticker: str) -> pd.DataFrame:
    try:
        cik_map = {v.upper(): k for k, v in CIK_MAP.items()}
        cik = cik_map.get(ticker)
        if not cik:
            print("⚠️ No CIK for", ticker)
            return pd.DataFrame()

        df = pd.read_csv(FORM4_TSV_PATH, sep="\t", dtype=str, low_memory=False)
        df = df[df["CIK"] == cik]

        df["ds"]     = pd.to_datetime(df["transactionDate"],   errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df["transactionShares"],  errors="coerce").fillna(0)
        df["code"]   = df["transactionCode"].str.strip().str.upper()

        def classify(r):
            if r["code"] in ("P","M","A","G","F"):
                return (r["shares"], 1, 0)
            if r["code"] == "S":
                return (-r["shares"],0,1)
            return (0,0,0)

        triples = df.apply(classify, axis=1).tolist()
        df[["net_shares","num_buy_tx","num_sell_tx"]] = pd.DataFrame(triples, index=df.index)

        return (
            df.groupby("ds")
              .agg(
                net_shares   = ("net_shares",  "sum"),
                num_buy_tx   = ("num_buy_tx",  "sum"),
                num_sell_tx  = ("num_sell_tx", "sum"),
              )
              .reset_index()
        )

    except Exception as e:
        print("❌ TSV fallback error:", e)
        return pd.DataFrame()
