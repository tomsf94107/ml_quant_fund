# loader.py
from __future__ import annotations
from pathlib import Path
import os
import json
import sqlite3
import datetime as dt
from typing import Optional, Tuple, List

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Forecast accuracy DB loader (kept from your version, with tiny hardening)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_db_path() -> str:
    """
    Resolve the forecast accuracy SQLite DB path in this order:
    1) FORECAST_ACCURACY_DB env var
    2) <this_dir>/forecast_accuracy.db if exists
    3) 'forecast_accuracy.db' (CWD fallback)
    """
    env = os.getenv("FORECAST_ACCURACY_DB")
    if env:
        return env
    here = Path(__file__).resolve().parent
    p = here / "forecast_accuracy.db"
    return str(p) if p.exists() else "forecast_accuracy.db"

DEFAULT_DB = _resolve_db_path()
# Prefer materialized 'accuracy_unified' view if present, else fall back.
DEFAULT_TABLES: Tuple[str, ...] = ("accuracy_unified", "forecast_accuracy", "metrics")


def load_eval_logs_from_forecast_db(
    db_path: Optional[str] = None,
    tables: Tuple[str, ...] = DEFAULT_TABLES,
    dedupe: bool = True,
) -> pd.DataFrame:
    """
    Load model evaluation logs from a SQLite DB. Returns a unified DataFrame with:
        ['date','ticker','mae','mse','r2','model','confidence']
    """
    db_path = db_path or DEFAULT_DB
    frames: List[pd.DataFrame] = []

    try:
        with sqlite3.connect(db_path) as conn:
            try:
                existing = set(
                    pd.read_sql(
                        "SELECT name FROM sqlite_schema WHERE type IN ('table','view');",
                        conn,
                    )["name"].tolist()
                )
            except Exception:
                existing = set(
                    pd.read_sql(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view');",
                        conn,
                    )["name"].tolist()
                )

            for t in tables:
                if t not in existing:
                    continue
                try:
                    df = pd.read_sql(f"SELECT * FROM {t}", conn)
                except Exception:
                    continue
                if df.empty:
                    continue

                df.columns = df.columns.str.lower()

                # unify/keep 'date'
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                elif "timestamp" in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
                elif "ingested_at" in df.columns:
                    df["date"] = pd.to_datetime(df["ingested_at"], errors="coerce")
                else:
                    df["date"] = pd.NaT

                # normalize ticker
                if "ticker" in df.columns:
                    df["ticker"] = df["ticker"].astype(str).str.upper()
                else:
                    df["ticker"] = pd.NA

                # ensure required numeric columns exist
                for col in ("mae", "mse", "r2"):
                    if col not in df.columns:
                        df[col] = pd.NA
                    else:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # defaults for optional fields
                if "model" not in df.columns:
                    df["model"] = "XGBoost (Short Term)"
                if "confidence" not in df.columns:
                    df["confidence"] = 1.0

                frames.append(
                    df[["date", "ticker", "mae", "mse", "r2", "model", "confidence"]]
                )
    except Exception:
        # DB missing or unreadable → return empty with correct schema
        return pd.DataFrame(
            columns=["date", "ticker", "mae", "mse", "r2", "model", "confidence"]
        )

    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "mae", "mse", "r2", "model", "confidence"]
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date"]).sort_values("date")

    if dedupe:
        out = out.drop_duplicates(subset=["date", "ticker", "model"], keep="last")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Price loader (yfinance with safe fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_yf():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception:
        return None

def _coerce_date(x) -> dt.date:
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except Exception:
        return pd.NaT

def price_loader_yf(
    ticker: str,
    start_date: str | dt.date | None,
    end_date: str | dt.date | None,
    auto_adjust: bool = True,
    add_market_cap: bool = True,
) -> pd.DataFrame:
    """
    Return a daily price frame with at least: ['date','open','high','low','close','volume'].
    Adds 'shares_outstanding' (constant vector if available) and naive 'market_cap' = close * shares_outstanding.
    """
    yf = _try_import_yf()
    if yf is None:
        # Minimal empty frame with expected columns
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "shares_outstanding",
                "market_cap",
            ]
        )

    t = ticker.upper()
    # Resolve dates
    if start_date is None or end_date is None:
        # Default: 1 year
        hist = yf.download(t, period="1y", auto_adjust=auto_adjust)
    else:
        hist = yf.download(
            t,
            start=str(pd.to_datetime(start_date).date()),
            end=str(pd.to_datetime(end_date).date() + pd.Timedelta(days=1)),
            auto_adjust=auto_adjust,
        )

    if hist is None or hist.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "shares_outstanding",
                "market_cap",
            ]
        )

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist = hist.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        }
    ).reset_index().rename(columns={"Date": "date"})

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date
    hist["ticker"] = t

    # Shares outstanding via fast_info / info (constant across the frame)
    shares_outstanding = None
    try:
        tk = yf.Ticker(t)
        # fast_info is faster and safer
        so = getattr(getattr(tk, "fast_info", object()), "shares_outstanding", None)
        if so is None:
            so = getattr(getattr(tk, "info", {}), "get", lambda k, d=None: d)("sharesOutstanding", None)
        shares_outstanding = float(so) if so is not None else None
    except Exception:
        shares_outstanding = None

    if shares_outstanding is None:
        hist["shares_outstanding"] = None
    else:
        hist["shares_outstanding"] = float(shares_outstanding)

    if add_market_cap:
        if hist["shares_outstanding"].notna().any() and "close" in hist:
            hist["market_cap"] = hist["close"] * hist["shares_outstanding"]
        else:
            hist["market_cap"] = None

    return hist[["date", "ticker", "open", "high", "low", "close", "volume", "shares_outstanding", "market_cap"]]


# ─────────────────────────────────────────────────────────────────────────────
# Insider loaders (Google Sheets → final_daily) with robust auth & fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _try_get_service_account_dict() -> Optional[dict]:
    """
    Retrieve a Google service account JSON dict from either:
    - Streamlit secrets: st.secrets['gcp_service_account']
    - Environment var: GCP_SERVICE_ACCOUNT_JSON (raw JSON string)
    Returns None if unavailable.
    """
    # 1) Streamlit secrets (preferred in your Streamlit Cloud app)
    try:
        import streamlit as st  # type: ignore
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass

    # 2) Environment variable
    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None

    return None


def _open_gsheet(spreadsheet_name: str):
    """
    Open a Google Sheet by name using service account. Returns (gc, sh) or (None, None) on failure.
    """
    try:
        from google.oauth2.service_account import Credentials  # type: ignore
        import gspread  # type: ignore
    except Exception:
        return None, None

    sa_dict = _try_get_service_account_dict()
    if not sa_dict:
        return None, None

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(sa_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open(spreadsheet_name)
        return gc, sh
    except Exception:
        return None, None


def _read_worksheet_as_df(spreadsheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """
    Read a worksheet into a DataFrame. Returns empty DataFrame if anything fails.
    """
    gc, sh = _open_gsheet(spreadsheet_name)
    if gc is None or sh is None:
        return pd.DataFrame()

    try:
        ws = sh.worksheet(worksheet_name)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        return df
    except Exception:
        return pd.DataFrame()


def load_final_daily_from_gsheet(
    spreadsheet_name: str = "Insider_Trades_Data",
    worksheet_name: str = "Final_Daily",
) -> pd.DataFrame:
    """
    Load the merged insider daily output (your ETL result) from Google Sheets.
    Expected columns:
      ['ticker','filed_date','net_shares','num_buy_tx','num_sell_tx',
       'num_exercise_like','max_txn_value_usd','any_exec_trade','large_tx_flag','holdings_delta']
    Returns empty DataFrame if unavailable.
    """
    df = _read_worksheet_as_df(spreadsheet_name, worksheet_name)
    if df.empty:
        return df

    # Normalize schema
    cols_lower = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols_lower)
    # Coerce dates
    if "filed_date" in df.columns:
        df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce").dt.date
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    # Coerce numerics (defensive)
    for c in [
        "net_shares",
        "num_buy_tx",
        "num_sell_tx",
        "num_exercise_like",
        "max_txn_value_usd",
        "holdings_delta",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "any_exec_trade" in df.columns:
        df["any_exec_trade"] = df["any_exec_trade"].astype(bool)
    if "large_tx_flag" in df.columns:
        # allow 'TRUE'/'FALSE'/1/0
        df["large_tx_flag"] = df["large_tx_flag"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])

    return df


def insider_loader_gsheet(
    ticker: str,
    start_date: str | dt.date,
    end_date: str | dt.date,
    spreadsheet_name: str = "Insider_Trades_Data",
    worksheet_name: str = "Final_Daily",
) -> pd.DataFrame:
    """
    Filter the ETL daily insider output from Google Sheets to a single ticker and date window.
    Returns empty DataFrame if the sheet or auth isn't available.
    """
    t = str(ticker).upper()
    df = load_final_daily_from_gsheet(spreadsheet_name, worksheet_name)
    if df is None or df.empty:
        return pd.DataFrame()

    sd = pd.to_datetime(start_date).date() if start_date is not None else dt.date(1900, 1, 1)
    ed = pd.to_datetime(end_date).date() if end_date is not None else dt.date.today()

    if "filed_date" not in df.columns or "ticker" not in df.columns:
        return pd.DataFrame()

    mask = (df["ticker"] == t) & (df["filed_date"] >= sd) & (df["filed_date"] <= ed)
    return df.loc[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: unified feature-build input loaders used by forecast_utils.py
# ─────────────────────────────────────────────────────────────────────────────

def price_loader(ticker: str, start_date: str | dt.date, end_date: str | dt.date) -> pd.DataFrame:
    """
    Public shim used by your feature pipeline.
    """
    return price_loader_yf(ticker, start_date, end_date, auto_adjust=True, add_market_cap=True)


def insider_loader(ticker: str, start_date: str | dt.date, end_date: str | dt.date) -> pd.DataFrame:
    """
    Public shim used by your feature pipeline.
    Pulls from Google Sheets 'Insider_Trades_Data' / 'Final_Daily'.
    """
    return insider_loader_gsheet(ticker, start_date, end_date)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-tests (run `python loader.py`)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("— Forecast Accuracy DB —")
    df_logs = load_eval_logs_from_forecast_db()
    print("DB:", DEFAULT_DB)
    print("rows:", len(df_logs))
    if len(df_logs):
        print("range:", df_logs["date"].min(), "→", df_logs["date"].max())
        print(df_logs.head(3))

    print("\n— Price Loader (sample: AAPL, last month) —")
    try:
        today = dt.date.today()
        start = today - dt.timedelta(days=30)
        df_px = price_loader("AAPL", start, today)
        print("rows:", len(df_px), "cols:", list(df_px.columns))
        print(df_px.head(3))
    except Exception as e:
        print("price_loader error:", e)

    print("\n— Insider Loader (GSheet) —")
    try:
        today = dt.date.today()
        start = today - dt.timedelta(days=90)
        df_ins = insider_loader("AAPL", start, today)
        print("rows:", len(df_ins), "cols:", list(df_ins.columns) if len(df_ins) else [])
        print(df_ins.head(3) if len(df_ins) else "(empty)")
    except Exception as e:
        print("insider_loader error:", e)
