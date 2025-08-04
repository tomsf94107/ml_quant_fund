# ---------------------------------------------------------------------
# forecast_utils.py v4.7 – Supports dual GSheet tabs for insider trades
# ---------------------------------------------------------------------

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pandas_ta as ta
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from sentiment_utils import get_sentiment_scores
from send_email import send_email_alert
from core.helpers_xgb import train_xgb_predict

# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR, EVAL_DIR, INTRA_DIR = "forecast_logs", "forecast_eval", "logs"
GSHEET_EVAL_LOG              = "forecast_evaluation_log"
GSHEET_INSIDER_DATA          = "Insider_Trades_Data"
TICKER_FILE                  = "tickers.csv"
SPY_TICKER, SECTOR_ETF       = "SPY", "XLK"
VOL_LOOKBACK_Z               = 20

for _d in (LOG_DIR, EVAL_DIR, INTRA_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Google Sheets Utility
# -----------------------------------------------------------------------------
def _get_gsheet(sheet_name, tab_index=0):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name)
    return sheet.get_worksheet(tab_index)

def log_eval_to_gsheet(tkr, mae, mse, r2):
    try:
        sheet = _get_gsheet(GSHEET_EVAL_LOG)
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tkr,
            round(mae, 2),
            round(mse, 2),
            round(r2, 4),
        ])
    except Exception as e:
        st.warning(f"⚠️ Sheets logging failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Insider Trade Loader from GSheet
# -----------------------------------------------------------------------------
def fetch_insider_trades(ticker: str) -> pd.DataFrame:
    try:
        trans = _get_gsheet(GSHEET_INSIDER_DATA, tab_index=0).get_all_records()
        hold  = _get_gsheet(GSHEET_INSIDER_DATA, tab_index=1).get_all_records()
        df = pd.DataFrame(trans + hold)
        if df.empty or "ticker" not in df.columns or "TRANS_DATE" not in df.columns:
            return pd.DataFrame()

        df = df[df["ticker"].str.upper() == ticker.upper()]
        df["ds"] = pd.to_datetime(df["TRANS_DATE"], errors="coerce").dt.date
        df["net_shares"] = pd.to_numeric(df.get("net_shares", 0), errors="coerce").fillna(0)
        df["num_buy_tx"] = pd.to_numeric(df.get("num_buy_tx", 0), errors="coerce").fillna(0)
        df["num_sell_tx"] = pd.to_numeric(df.get("num_sell_tx", 0), errors="coerce").fillna(0)
        return (
            df.groupby("ds")
              .agg(net_shares=("net_shares", "sum"),
                   num_buy_tx=("num_buy_tx", "sum"),
                   num_sell_tx=("num_sell_tx", "sum"))
              .reset_index()
              .set_index("ds")
        )
    except Exception as e:
        st.error(f"❌ Insider GSheet load failed: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# -----------------------------------------------------------------------------
def build_feature_dataframe(ticker, start=None, end=None, lookback=180, min_rows=200):
    if start and end:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if len(df) < min_rows:
            extra = max(min_rows - len(df) + 30, lookback)
            df = yf.download(ticker, period=f"{extra}d", auto_adjust=True)
    else:
        df = yf.download(ticker, period=f"{lookback}d", auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or "Close" not in df:
        return pd.DataFrame()

    # Insider trades
    try:
        ins = fetch_insider_trades(ticker)
        df["insider_net_shares"] = ins["net_shares"].reindex(df.index, method="ffill").fillna(0)
        df["insider_buy_count"]  = ins["num_buy_tx"].reindex(df.index, method="ffill").fillna(0)
        df["insider_sell_count"] = ins["num_sell_tx"].reindex(df.index, method="ffill").fillna(0)
    except Exception as e:
        print(f"⚠️ Insider load error: {e}")
        df["insider_net_shares"] = 0
        df["insider_buy_count"]  = 0
        df["insider_sell_count"] = 0

    df["Return_1D"] = df["Close"].pct_change()
    for w in (5, 10, 20):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    try:
        df["RSI14"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        if not macd.empty:
            df["MACD"], df["MACD_sig"] = macd.iloc[:, 0], macd.iloc[:, 1]
        df["BB_width"] = ta.bbands(df["Close"])["BBP_20_2.0"]
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"])
    except Exception:
        pass

    if "Volume" in df and df["Volume"].notna().all():
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["OBV"]  = ta.obv(df["Close"], df["Volume"])
        df["vol_zscore"] = (
            df["Volume"] - df["Volume"].rolling(VOL_LOOKBACK_Z).mean()
        ) / df["Volume"].rolling(VOL_LOOKBACK_Z).std()

    for etf in (SPY_TICKER, SECTOR_ETF):
        try:
            ret = yf.download(etf, start=df.index.min(), end=df.index.max(), auto_adjust=True)["Close"].pct_change()
            df[f"{etf}_ret"] = ret
        except Exception:
            df[f"{etf}_ret"] = np.nan

    # Sentiment
    try:
        sent = get_sentiment_scores(ticker)
        for k, v in sent.items():
            df[f"sent_{k}"] = v
        if sent.get("positive", 0) > 70 or sent.get("negative", 0) > 70:
            send_email_alert(f"Sentiment Alert: {ticker}", f"{sent}")
    except Exception as e:
        print(f"⚠️ Sentiment error for {ticker}: {e}")
	
	# Final cleanup: ensure core MA columns exist
    return df.dropna(subset=["Close", "MA5", "MA10", "MA20"])
# ────────────────────────────────────────────────────────────────────────────
# XGBoost with sentiment & extra features
# ---------------------------------------------------------------------------
def safe_train_xgb_with_retries(tkr, start=None, end=None,
                                init_look=180, max_look=720, step=180):
    look = init_look
    while look <= max_look:
        try:
            df = build_feature_dataframe(tkr, start, end, look)
            mdl, Xts, yts, yhat, fb = train_xgb_predict(df)
            if yhat is None or len(yhat) == 0:
                raise ValueError("empty pred")
            return mdl, Xts, yts, yhat, fb
        except Exception as e:
            print(f"⚠️ Retry {look}d failed: {e}")
            look += step
    raise ValueError(f"❌ Model failed after {max_look}d")

# ────────────────────────────────────────────────────────────────────────────
# Prophet with optional regressors
# ---------------------------------------------------------------------------
def forecast_price_trend(tkr, start_date=None, end_date=None,
                         period_months: int = 3, log_results: bool = True):
    end_date   = end_date or datetime.today()
    start_date = start_date or end_date - timedelta(days=5*365)
    df = build_feature_dataframe(tkr, start_date, end_date, lookback=9999)
    if df.empty:
        return None, "No data"

    # prepare df for Prophet
    dfp = df[["Close"]].rename(columns={"Close": "y"})
    dfp = dfp.assign(ds=df.index).reset_index(drop=True)

    # ensure sentiment regs exist and no NaNs
    sent_cols = ["sent_positive", "sent_neutral", "sent_negative"]
    for c in sent_cols:
        dfp[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    # choose regressors: sentiment + macro/extras if present
    extra_regs = [
        "congress_net_shares", "congress_active_members",
        "insider_net_shares",   "insider_buy_count",   "insider_sell_count"
    ]
    regs = sent_cols + [r for r in extra_regs if r in dfp.columns]

    m = Prophet(daily_seasonality=True)
    for reg in regs:
        m.add_regressor(reg)

    m.fit(dfp)
    future = m.make_future_dataframe(periods=int(period_months*30))
    last = dfp[regs].iloc[-1]
    for reg in regs:
        future[reg] = last[reg]

    fcst = m.predict(future)
    res  = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        dfp[["ds", "y"]].rename(columns={"y": "actual"}),
        on="ds", how="left"
    )
    if log_results:
        res.to_csv(
            os.path.join(
                LOG_DIR,
                f"forecast_{tkr}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            ), index=False
        )
    return res, None

# ────────────────────────────────────────────────────────────────────────────
# Intraday + ML combo (unchanged)
# ---------------------------------------------------------------------------
def forecast_today_movement(ticker: str, start=None, end=None,
                            log_results: bool = True) -> tuple[str, str]:
    intraday_msg = ""
    try:
        df_intra = yf.download(ticker, period="7d", interval="1h", auto_adjust=True)
        if df_intra.empty or "Close" not in df_intra:
            raise ValueError("no intraday data")
        df_intra["Return"] = df_intra["Close"].pct_change()
        df_intra["Trend"]  = df_intra["Return"].rolling(3).mean()
        trend  = df_intra["Trend"].iloc[-1]
        latest = (df_intra["Return"].iloc[-1] or 0)*100
        if trend > 0.001:
            intraday_msg = f"📈 Likely Uptrend Today ({latest:.2f}%)"
        elif trend < -0.001:
            intraday_msg = f"📉 Likely Downtrend Today ({latest:.2f}%)"
        else:
            intraday_msg = f"🔄 Flat or Unclear Trend ({latest:.2f}%)"
        if log_results:
            df_intra.to_csv(
                os.path.join(
                    INTRA_DIR,
                    f"intraday_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv"
                )
            )
    except Exception as e:
        intraday_msg = f"⚠️ Intraday error: {e}"

    try:
        mdl, Xts, _, yhat, fb_note = safe_train_xgb_with_retries(ticker, start, end)
        if yhat is None:
            return intraday_msg, "⚠️ Model failed"
        if Xts.empty or "Close" not in Xts.columns:
            up = yhat[-1] > 0
        else:
            up = yhat[-1] > Xts.iloc[-1]["Close"]
        direction = "🟢 ML Forecast: Up" if up else "🔴 ML Forecast: Down"
        if fb_note:
            direction += f"\n{fb_note}"
        return intraday_msg + " + " + direction, ""
    except Exception as e:
        return intraday_msg, f"❌ Model error: {e}"

# ────────────────────────────────────────────────────────────────────────────
# Eval, logging & retrain wrappers (unchanged)
# ---------------------------------------------------------------------------
def _latest_log(ticker):
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{ticker}_")]
    return os.path.join(LOG_DIR, sorted(logs)[-1]) if logs else None


def auto_retrain_forecast_model(tkr):
    log_path = _latest_log(tkr)
    if not log_path:
        return
    df = pd.read_csv(log_path).dropna(subset=["yhat", "actual"])
    if df.empty:
        return
    mae = mean_absolute_error(df["actual"], df["yhat"])
    mse = mean_squared_error(df["actual"], df["yhat"])
    r2  = r2_score(df["actual"], df["yhat"])
    row = pd.DataFrame([[datetime.now(), tkr, mae, mse, r2]], columns=["timestamp","ticker","mae","mse","r2"])
    eval_path = os.path.join(EVAL_DIR, "forecast_evals.csv")
    row.to_csv(eval_path, mode="a", header=not os.path.exists(eval_path), index=False)
    log_eval_to_gsheet(tkr, mae, mse, r2)
    return row


def run_auto_retrain_all(ticker_list):
    evals = []
    for tkr in ticker_list:
        print(f"🔄 Retraining: {tkr}")
        df = auto_retrain_forecast_model(tkr)
        if isinstance(df, pd.DataFrame):
            evals.append(df)
    return pd.concat(evals, ignore_index=True) if evals else pd.DataFrame(columns=["ticker","mae","mse","r2"])

# ────────────────────────────────────────────────────────────────────────────
# SHAP Optional Plotter (unchanged)
# ---------------------------------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SHAP import failed: {e}")
    SHAP_AVAILABLE = False

def plot_shap(model, X, top_n: int = 10):
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not available")
        return
    try:
        expl = shap.Explainer(model, X)
        shap_values = expl(X)
        shap.plots.bar(shap_values[:, :top_n])
    except Exception as e:
        print(f"❌ SHAP plot failed: {e}")

# ────────────────────────────────────────────────────────────────────────────
# Load/save tickers & accuracy helpers (unchanged)
# ---------------------------------------------------------------------------
def get_latest_forecast_log(tkr, log_dir=LOG_DIR):
    logs = [f for f in os.listdir(log_dir) if f.startswith(f"forecast_{tkr.upper()}_")]
    return os.path.join(log_dir, sorted(logs)[-1]) if logs else None

def load_forecast_tickers():
    if not os.path.exists(TICKER_FILE):
        return ["AAPL","MSFT"]
    return [ln.strip().upper() for ln in open(TICKER_FILE) if ln.strip()]

def save_forecast_tickers(tkr_list):
    with open(TICKER_FILE, "w") as f:
        for t in tkr_list:
            f.write(t.strip().upper() + "\n")

def compute_rolling_accuracy(log_path):
    df = pd.read_csv(log_path, parse_dates=["ds"]).sort_values("ds")
    if {"yhat","actual"} - set(df.columns):
        return pd.DataFrame()
    df["pred_direction"]   = df["yhat"].diff().gt(0).mul(1).sub(df["yhat"].diff().le(0))
    df["actual_direction"] = df["actual"].diff().gt(0).mul(1).sub(df["actual"].diff().le(0))
    df["correct"]          = df["pred_direction"] == df["actual_direction"]
    df["7d_accuracy"]      = df["correct"].rolling(7).mean()
    df["30d_accuracy"]     = df["correct"].rolling(30).mean()
    return df[["ds","7d_accuracy","30d_accuracy","correct"]]

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost/Prophet/Intraday/Logging/SHAP remain unchanged from your v4.6
# Only insider trade sourcing changed from local TSV → Google Sheets
# ─────────────────────────────────────────────────────────────────────────────
