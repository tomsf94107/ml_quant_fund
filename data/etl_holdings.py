# data/etl_holdings.py

import os, sys
# ensure project root is on sys.path so "data.utils_cik" resolves
ROOT = os.path.abspath(os.path.join(__file__, os.pardir))
if ROOT not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(ROOT, os.pardir)))

import pandas as pd
from datetime import datetime
from data.utils_cik import load_cik_to_ticker_map

try:
    import streamlit as st
    ST_SECRETS = st.secrets
except ImportError:
    ST_SECRETS = {}

CIK_MAP            = load_cik_to_ticker_map()
FORM4_HOLDING_TSV  = "data/SEC/NONDERIV_HOLDING.tsv"
GSHEET_NAME        = "Insider_Trades_Data"
TAB_HOLDINGS       = "Insider_Holdings"

def fetch_insider_holdings(ticker: str, mode: str = "sheet-first") -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - ds            (date of filing)
      - shares        (shares owned after filing)
      - net_change    (difference from prior filing)
    Modes:
      "sheet-first": load from Google Sheet
      "tsv":         load from local TSV
      "hybrid":      try sheet, then TSV
    """
    ticker = ticker.upper().strip()
    df = pd.DataFrame()

    if mode in ("sheet-first", "sheet"):
        df = _fetch_from_sheet(ticker)
        if not df.empty:
            return df

    if mode in ("tsv", "hybrid"):
        df = _fetch_from_tsv(ticker)

    return df


def _fetch_from_sheet(ticker: str) -> pd.DataFrame:
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # authenticate
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            ST_SECRETS["gcp_service_account"], scopes=scope
        )
        gc = gspread.authorize(creds)

        sh = gc.open(GSHEET_NAME)
        ws = sh.worksheet(TAB_HOLDINGS)
        records = ws.get_all_records()
        df = pd.DataFrame(records)

        # normalize headers
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

        # find ticker column
        if "ISSUERTRADINGSYMBOL" in df.columns:
            tcol = "ISSUERTRADINGSYMBOL"
        elif "TICKER" in df.columns:
            tcol = "TICKER"
        else:
            return pd.DataFrame()

        df = df[df[tcol].str.upper().str.strip() == ticker]
        if df.empty:
            return pd.DataFrame()

        # required sheet columns
        required = {"FILING_DATE", "SHRS_OWND_FOLWNG_TRANS"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        # parse
        df["ds"]     = pd.to_datetime(df["FILING_DATE"], errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df["SHRS_OWND_FOLWNG_TRANS"], errors="coerce").fillna(0)

        # keep relevant
        df = df[["ds", "shares"]].sort_values("ds")

        # compute daily change
        df["net_change"] = df["shares"].diff().fillna(0)

        return df.reset_index(drop=True)

    except Exception as e:
        # any auth / sheet issue → fallback
        print(f"❌ Google Sheet holdings fetch error: {e}")
        return pd.DataFrame()


def _fetch_from_tsv(ticker: str) -> pd.DataFrame:
    try:
        # map ticker→CIK then pull local TSV
        inv = {v.upper(): k for k, v in CIK_MAP.items()}
        cik = inv.get(ticker)
        if not cik or not os.path.exists(FORM4_HOLDING_TSV):
            return pd.DataFrame()

        df = pd.read_csv(FORM4_HOLDING_TSV, sep="\t", low_memory=False, dtype=str)
        df = df[df["ISSUERCIK"] == cik]

        # parse
        df["ds"]     = pd.to_datetime(df["FILING_DATE"], errors="coerce").dt.date
        df["shares"] = pd.to_numeric(df["SHRS_OWND_FOLWNG_TRANS"], errors="coerce").fillna(0)

        df = df[["ds", "shares"]].sort_values("ds")
        df["net_change"] = df["shares"].diff().fillna(0)

        return df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ TSV holdings fallback error: {e}")
        return pd.DataFrame()
