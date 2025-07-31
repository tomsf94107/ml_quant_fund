# ─────────────────────────────────────────────────────────────────────────────
# v4.3 – sentiment-aware XGB & Prophet · SHAP-safe · keeps all v4.2 helpers.  #
# ─────────────────────────────────────────────────────────────────────────────

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBRegressor
import pandas_ta as ta
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentiment_utils import get_sentiment_scores
from send_email import send_email_alert

# ─────────── PATHS / CONSTANTS ───────────────────────────────────────────────
LOG_DIR, EVAL_DIR, INTRA_DIR = "forecast_logs", "forecast_eval", "logs"
GSHEET_NAME, TICKER_FILE     = "forecast_evaluation_log", "tickers.csv"
SPY_TICKER, SECTOR_ETF       = "SPY", "XLK"
VOL_LOOKBACK_Z               = 20

for _d in (LOG_DIR, EVAL_DIR, INTRA_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────── GOOGLE-SHEETS LOGGER ────────────────────────────────────────────
def _get_gsheet_logger():
    try:
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["gcp_service_account"], scope
        )
        return gspread.authorize(creds).open(GSHEET_NAME).sheet1
    except Exception as e:
        st.error(f"❌ Sheets auth failed: {e}")
        return None

def log_eval_to_gsheet(tkr, mae, mse, r2):
    sh = _get_gsheet_logger()
    if not sh:
        return
    try:
        sh.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       tkr, round(mae, 2), round(mse, 2), round(r2, 4)])
    except Exception as e:
        st.warning(f"⚠️ Sheets logging failed: {e}")

# ─────────── FEATURE-ENGINEERING ─────────────────────────────────────────────
def build_feature_dataframe(ticker: str,
                            start=None,
                            end=None,
                            lookback: int = 180,
                            min_rows: int = 200) -> pd.DataFrame:
    """
    Download OHLCV data → add TA, market context & daily sentiment.
    Returns clean DataFrame (may be empty if data is inadequate).
    """
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

    # Basic TA
    df["Return_1D"] = df["Close"].pct_change()
    for w in (5, 10, 20):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    try:
        df["RSI14"]   = ta.rsi(df["Close"], length=14)
        macd          = ta.macd(df["Close"])
        if not macd.empty:
            df["MACD"], df["MACD_sig"] = macd.iloc[:, 0], macd.iloc[:, 1]
        df["BB_width"] = ta.bbands(df["Close"])["BBP_20_2.0"]
        df["ATR"]      = ta.atr(df["High"], df["Low"], df["Close"])
    except Exception:
        pass  # If pandas-ta throws, just keep NaNs

    # Volume & market context
    if "Volume" in df and df["Volume"].notna().all():
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["OBV"]  = ta.obv(df["Close"], df["Volume"])
        df["vol_zscore"] = (
            df["Volume"].sub(df["Volume"].rolling(VOL_LOOKBACK_Z).mean())
                        .div(df["Volume"].rolling(VOL_LOOKBACK_Z).std())
        )

    for etf in (SPY_TICKER, SECTOR_ETF):
        try:
            ret = yf.download(etf, start=df.index.min(), end=df.index.max(),
                              auto_adjust=True)["Close"].pct_change()
            df[f"{etf}_ret"] = ret
        except Exception:
            df[f"{etf}_ret"] = np.nan

    # Sentiment
    try:
        sent = get_sentiment_scores(ticker)                # {pos,neu,neg}
        for k, v in sent.items():                          # add as flat cols
            df[f"sent_{k}"] = v
        if sent.get("positive", 0) > 70 or sent.get("negative", 0) > 70:
            send_email_alert(f"Sentiment Alert: {ticker}", f"{sent}")
    except Exception as e:
        print(f"⚠️ Sentiment error for {ticker}: {e}")

    # Drop if any of the core MA columns still NaN
    return df.dropna(subset=["Close", "MA5", "MA10", "MA20"])

# ─────────── XGBOOST WITH SENTIMENT FEATURES ────────────────────────────────
def _ensure_return(df):
    if "Close" in df:
        df["Return"] = df["Close"].pct_change()
    elif "Return_1D" in df:
        df["Return"] = df["Return_1D"]
    else:
        raise ValueError("Missing price return")
    return df

def train_xgb_predict(df: pd.DataFrame, horizon: int = 1):
    df = _ensure_return(df)
    df["Target"] = df["Close"].shift(-horizon)

    base_feats = ["Close", "MA5", "MA10", "MA20", "Return"]
    sent_feats = [c for c in df.columns if c.startswith("sent_")]
    features   = base_feats + sent_feats
    df = df.dropna(subset=features + ["Target"])

    if len(df) < 10:
        lp = df["Close"].iloc[-1]
        return None, df[["Close"]].tail(3), None, [lp]*3, "⚠️ Fallback prediction"

    X, y = df[features], df["Target"]
    Xtr, Xts, ytr, yts = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBRegressor(n_estimators=100)
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xts)
    return model, Xts, yts, y_pred, None

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

# ─────────── PROPHET WITH SENTIMENT REGRESSORS ───────────────────────────────
def forecast_price_trend(tkr, start_date=None, end_date=None,
                         period_months: int = 3, log_results: bool = True):
    end_date   = end_date or datetime.today()
    start_date = start_date or end_date - timedelta(days=5*365)
    df = build_feature_dataframe(tkr, start_date, end_date, lookback=9999)
    if df.empty:
        return None, "No data"

    dfp = (df[["Close"]].rename(columns={"Close": "y"})
                     .assign(ds=df.index)
                     .reset_index(drop=True))
    for col in ["sent_positive", "sent_neutral", "sent_negative"]:
        dfp[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    dfp.fillna(method="ffill", inplace=True)

    m = Prophet(daily_seasonality=True)
    for reg in ["sent_positive", "sent_neutral", "sent_negative"]:
        m.add_regressor(reg)
    m.fit(dfp)

    future = m.make_future_dataframe(periods=int(period_months*30))
    last_sent = dfp[["sent_positive", "sent_neutral", "sent_negative"]].iloc[-1]
    for reg in last_sent.index:      # keep sentiment constant into future
        future[reg] = last_sent[reg]

    fcst = m.predict(future)
    res  = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        dfp[["ds", "y"]].rename(columns={"y": "actual"}), on="ds", how="left"
    )
    if log_results:
        res.to_csv(os.path.join(LOG_DIR,
                   f"forecast_{tkr}_{datetime.now():%Y%m%d_%H%M%S}.csv"),
                   index=False)
    return res, None

# ─────────── (UNCHANGED) INTRADAY + FORECAST COMBO ───────────────────────────
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
            df_intra.to_csv(os.path.join(
                INTRA_DIR, f"intraday_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv"))
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
        if fb_note: direction += f"\n{fb_note}"
        return intraday_msg + " + " + direction, ""
    except Exception as e:
        return intraday_msg, f"❌ Model error: {e}"

# ─────────── EVAL, LOGGING & RETRAIN WRAPPERS (UNCHANGED) ────────────────────
def _latest_log(ticker):  # same as v4.2
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
    row = pd.DataFrame([[datetime.now(), tkr, mae, mse, r2]],
                       columns=["timestamp", "ticker", "mae", "mse", "r2"])
    eval_path = os.path.join(EVAL_DIR, "forecast_evals.csv")
    row.to_csv(eval_path, mode="a", header=not os.path.exists(eval_path),
               index=False)
    log_eval_to_gsheet(tkr, mae, mse, r2)
    return row

def run_auto_retrain_all(ticker_list):
    evals = []
    for tkr in ticker_list:
        print(f"🔄 Retraining: {tkr}")
        df = auto_retrain_forecast_model(tkr)
        if isinstance(df, pd.DataFrame):
            evals.append(df)
    return (pd.concat(evals, ignore_index=True)
            if evals else pd.DataFrame(columns=["ticker", "mae", "mse", "r2"]))

# ─────────── SHAP OPTIONAL PLOTTER ───────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    print("⚠️ SHAP import failed:", e)
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

# ─────────── EVERYTHING ELSE (load/save tickers, accuracy helpers) UNCHANGED ─
def get_latest_forecast_log(tkr, log_dir=LOG_DIR):
    logs = [f for f in os.listdir(log_dir) if f.startswith(f"forecast_{tkr.upper()}_")]
    return os.path.join(log_dir, sorted(logs)[-1]) if logs else None

def load_forecast_tickers():
    if not os.path.exists(TICKER_FILE):
        return ["AAPL", "MSFT"]
    return [ln.strip().upper() for ln in open(TICKER_FILE) if ln.strip()]

def save_forecast_tickers(tkr_list):
    with open(TICKER_FILE, "w") as f:
        for t in tkr_list:
            f.write(t.strip().upper() + "\n")

def compute_rolling_accuracy(log_path):
    df = pd.read_csv(log_path, parse_dates=["ds"]).sort_values("ds")
    if {"yhat", "actual"} - set(df.columns):
        return pd.DataFrame()
    df["pred_direction"]   = df["yhat"].diff().gt(0).mul(1).sub(df["yhat"].diff().le(0))
    df["actual_direction"] = df["actual"].diff().gt(0).mul(1).sub(df["actual"].diff().le(0))
    df["correct"]          = df["pred_direction"] == df["actual_direction"]
    df["7d_accuracy"]      = df["correct"].rolling(7).mean()
    df["30d_accuracy"]     = df["correct"].rolling(30).mean()
    return df[["ds", "7d_accuracy", "30d_accuracy", "correct"]]
