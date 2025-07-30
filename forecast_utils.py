# v4.1 – Core TA & Market Context

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas_ta as ta
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentiment_utils import get_sentiment_scores
from send_email import send_email_alert

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# --- Paths / constants -------------------------------------------------------
LOG_DIR      = "forecast_logs"
EVAL_DIR     = "forecast_eval"
INTRA_DIR    = "logs"          # <—  keep it here
GSHEET_NAME  = "forecast_evaluation_log"
TICKER_FILE  = "tickers.csv"

SPY_TICKER   = "SPY"           # broad-market proxy
SECTOR_ETF   = "XLK"           # sector ETF for beta feature
VOL_LOOKBACK_Z = 20            # vol-Z look-back sessions
# ---------------------------------------------------------------------------

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)
os.makedirs(INTRA_DIR, exist_ok=True)   # <—  now INTRA_DIR is defined

# -----------------------------------------------------------------------------


# === Google-Sheets helpers ====================================================
def _get_gsheet_logger():
    try:
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open(GSHEET_NAME).sheet1
    except Exception as e:
        st.error(f"❌ Sheets auth failed: {e}")
        return None


def log_eval_to_gsheet(ticker, mae, mse, r2):
    sheet = _get_gsheet_logger()
    if not sheet:
        return
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, ticker, round(mae, 2), round(mse, 2), round(r2, 4)])
        st.success(f"✅ Logged to Google Sheets: {ticker}")
    except Exception as e:
        st.warning(f"⚠️ Sheets logging failed: {e}")
# ==============================================================================


# === Feature Engineering ======================================================

def build_feature_dataframe(
    ticker: str,
    start=None,
    end=None,
    lookback: int = 180,
    min_rows: int = 200,
) -> pd.DataFrame:
    """Download OHLCV & enrich with TA, volume + market context."""
    import yfinance as yf
    import pandas_ta as ta
    import numpy as np

    # 1) Price pull ----------------------------------------------------------------
    if start and end:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if len(df) < min_rows:
            extra = max(min_rows - len(df) + 30, lookback)
            df = yf.download(ticker, period=f"{extra}d", auto_adjust=True)
    else:
        df = yf.download(ticker, period=f"{lookback}d", auto_adjust=True)

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or "Close" not in df:
        return pd.DataFrame()

    # ——— TA + volume factors -------------------------------------------------------
    df["Return_1D"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # ——— Robust TA indicators using pandas_ta with safe fallback -------------------
    try:
        df["RSI14"] = ta.rsi(df["Close"], length=14)
    except Exception as e:
        print(f"⚠️ RSI error: {e}")
        df["RSI14"] = np.nan

    try:
        macd = ta.macd(df["Close"])
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            df["MACD"] = macd["MACD_12_26_9"] if "MACD_12_26_9" in macd else np.nan
            df["MACD_sig"] = macd["MACDs_12_26_9"] if "MACDs_12_26_9" in macd else np.nan
        else:
            df["MACD"] = df["MACD_sig"] = np.nan
    except Exception as e:
        print(f"⚠️ MACD error: {e}")
        df["MACD"] = df["MACD_sig"] = np.nan

    try:
        bb = ta.bbands(df["Close"])
        if isinstance(bb, pd.DataFrame) and "BBP_20_2.0" in bb:
            df["BB_width"] = bb["BBP_20_2.0"]
        else:
            df["BB_width"] = np.nan
    except Exception as e:
        print(f"⚠️ BBands error: {e}")
        df["BB_width"] = np.nan

    try:
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"])
    except Exception as e:
        print(f"⚠️ ATR error: {e}")
        df["ATR"] = np.nan

    # ——— Volume metrics -----------------------------------------------------------
    try:
        if "Volume" in df and df["Volume"].notna().all():
            df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
            df["OBV"] = ta.obv(df["Close"], df["Volume"])
            df["vol_zscore"] = (
                (df["Volume"] - df["Volume"].rolling(VOL_LOOKBACK_Z).mean())
                / df["Volume"].rolling(VOL_LOOKBACK_Z).std()
            )
        else:
            raise ValueError("Volume column missing or contains NaNs")
    except Exception as e:
        print(f"⚠️ Volume metrics error: {e}")
        df["VWAP"] = df["OBV"] = df["vol_zscore"] = np.nan

    # ——— Market / sector return join ----------------------------------------------
    def _get_ret(etf):
        try:
            tmp = yf.download(etf, start=df.index.min(), end=df.index.max(), auto_adjust=True)
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = tmp.columns.get_level_values(0)
            if "Close" not in tmp:
                raise ValueError("Missing Close column")
            return tmp["Close"].pct_change().rename(f"{etf}_ret")
        except Exception as e:
            print(f"⚠️ Market return fetch failed for {etf}: {e}")
            return pd.Series(index=df.index, name=f"{etf}_ret", dtype=float)

    df = df.join(_get_ret(SPY_TICKER), how="left")
    df = df.join(_get_ret(SECTOR_ETF), how="left")

    # ——— Sentiment block ----------------------------------------------------------
    try:
        sent = get_sentiment_scores(ticker)
        for k, v in sent.items():
            df[f"sent_{k}"] = v
        if sent.get("positive", 0) > 70 or sent.get("negative", 0) > 70:
            send_email_alert(f"Sentiment Alert: {ticker}",
                             f"Extreme sentiment detected: {sent}")
    except Exception as e:
        print(f"⚠️ Sentiment error for {ticker}: {e}")

    # ——— Final clean — only drop essential columns --------------------------------
    required_cols = ["Close", "MA5", "MA10", "MA20"]
    optional_cols = ["RSI14", "MACD", "MACD_sig"]

    # 🧪 Debug print — non-null counts for required and optional features
    print(f"\n✅ Total rows before dropna: {len(df)}")
    print("📊 Required non-null counts:")
    print(df[required_cols].notna().sum())
    print("📊 Optional non-null counts:")
    for col in optional_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].notna().sum()}")
        else:
            print(f"  {col}: MISSING")

    df = df.dropna(subset=required_cols)

    return df


# ==============================================================================


# === XGBoost helpers ==========================================================
def _ensure_return(df):
    if 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()
    elif 'Return_1D' in df.columns:
        df['Return'] = df['Return_1D']
    else:
        raise ValueError("Missing both 'Close' and 'Return_1D'")
    return df


def train_xgb_predict(df, horizon_days=1):
    df = _ensure_return(df)
    df["Target"] = df["Close"].shift(-horizon_days)

    # Required features only for training
    required_features = ["Close", "MA5", "MA10", "MA20", "Return"]
    df = df.dropna(subset=required_features + ["Target"])

    print(f"✅ Final rows after dropna: {len(df)}")

    if len(df) < 10:
        print("⚠️ Too few rows for XGBoost training. Using fallback.")
        last_price = df["Close"].iloc[-1]
        fallback_pred = [last_price] * min(len(df), 3)
        return None, df[["Close"]].tail(len(fallback_pred)), None, fallback_pred, "⚠️ Fallback prediction used"

    X = df[required_features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, None


def safe_train_xgb_with_retries(
    ticker: str,
    start=None,
    end=None,
    initial_lookback: int = 180,
    max_lookback: int = 720,
    retry_step: int = 180,
    horizon_days: int = 1
):
    lookback = initial_lookback
    while lookback <= max_lookback:
        try:
            df = build_feature_dataframe(ticker, start=start, end=end, lookback=lookback)
            print(f"🔁 Lookback {lookback}d – Raw rows: {len(df)}")
            if df.empty or len(df) < 30:
                raise ValueError(f"Too few rows returned after build: {len(df)}")

            model, X_test, y_test, y_pred, fallback_note = train_xgb_predict(df, horizon_days=horizon_days)

            if y_pred is None or len(y_pred) == 0:
                raise ValueError("Model prediction failed")

            return model, X_test, y_test, y_pred, fallback_note

        except Exception as ve:
            print(f"⚠️ Retry {lookback}d failed: {ve}")
            lookback += retry_step

    raise ValueError(f"❌ Model failed after {max_lookback}d lookback.")


# ==============================================================================


# === Prophet forecasting  (patched) ==========================================
def forecast_price_trend(ticker,
                         start_date=None,
                         end_date=None,
                         period_months: int = 3,
                         log_results: bool = True):
    """
    Long-horizon forecast using Facebook/Meta Prophet.
    Returns (forecast_df | None, err_msg | None)
    """
    from prophet import Prophet
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta
    import os

    if end_date is None:
        end_date = datetime.today()
    if start_date is None:
        start_date = end_date - timedelta(days=5 * 365)

    # ------------------------------------------------------------------ #
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    # 1️⃣  Flatten possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or "Close" not in df or len(df) < 60:
        return None, f"Not enough data or missing ‘Close’ for {ticker}"

    # ------------------------------------------------------------------ #
    # 2️⃣  Build Prophet-ready DataFrame
    df_prophet = (
        df[["Close"]]
        .copy()
        .rename(columns={"Close": "y"})
        .assign(ds=lambda d: d.index)               # keep the dates
        .reset_index(drop=True)[["ds", "y"]]        # enforce column order
    )
    df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
    df_prophet.dropna(subset=["y"], inplace=True)

    # ------------------------------------------------------------------ #
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future   = model.make_future_dataframe(periods=int(period_months * 30))
    forecast = model.predict(future)

    result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    result_df = result_df.merge(
        df_prophet.rename(columns={"y": "actual"}), on="ds", how="left"
    )

    if log_results:
        fn = f"forecast_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        result_df.to_csv(os.path.join(LOG_DIR, fn), index=False)

    return result_df, None

# ==============================================================================


# ------------------------------------------------------------------ function


def forecast_today_movement(
    ticker: str,
    start=None,
    end=None,
    log_results: bool = True
) -> tuple[str, str]:
    """
    Returns (msg, err) – *err* is '' on success.
    Combines a quick intraday trend signal with the 1-day XGBoost forecast.
    """
    intraday_msg = ""

    # ------------------------------ 1️⃣ Intraday part
    try:
        df_intra = yf.download(ticker, period="7d", interval="1h", auto_adjust=True)
        if df_intra.empty or "Close" not in df_intra:
            raise ValueError("no intraday data")

        df_intra["Return"] = df_intra["Close"].pct_change()
        df_intra["Trend"] = df_intra["Return"].rolling(3).mean()

        trend = df_intra["Trend"].iloc[-1]
        latest = df_intra["Return"].iloc[-1] * 100 if not np.isnan(df_intra["Return"].iloc[-1]) else 0

        if trend > 0.001:
            intraday_msg = f"📈 Likely Uptrend Today ({latest:.2f}%)"
        elif trend < -0.001:
            intraday_msg = f"📉 Likely Downtrend Today ({latest:.2f}%)"
        else:
            intraday_msg = f"🔄 Flat or Unclear Trend ({latest:.2f}%)"

        if log_results:
            fn = os.path.join(INTRA_DIR, f"intraday_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv")
            df_intra.to_csv(fn)

    except Exception as e:
        intraday_msg = f"⚠️ Intraday error: {e}"

    # ------------------------------ 2️⃣ ML Forecast part
    try:
        model, X_test, _, y_pred, fallback_note = safe_train_xgb_with_retries(
            ticker, start=start, end=end
        )

        if y_pred is None or len(y_pred) == 0:
            return intraday_msg, "⚠️ Model prediction failed"

        if X_test is None or len(X_test) == 0 or "Close" not in X_test.columns:
            up = y_pred[-1] > 0  # fallback heuristic
            direction = "🟢 ML Forecast: Up" if up else "🔴 ML Forecast: Down"
            direction += "\n⚠️ X_test empty – fallback logic used"
        else:
            up = y_pred[-1] > X_test.iloc[-1]["Close"]
            direction = "🟢 ML Forecast: Up" if up else "🔴 ML Forecast: Down"

        if fallback_note:
            direction += f"\n{fallback_note}"

        return f"{intraday_msg} + {direction}", ""

    except Exception as e:
        return intraday_msg, f"❌ Model error: {e}"

    # Optional fallback (will never be hit unless something silent fails)
    return intraday_msg, "⚠️ Unexpected model flow termination"

# ==============================================================================


# === Accuracy / logging =======================================================
def compute_rolling_accuracy(log_path):
    df = pd.read_csv(log_path, parse_dates=['ds']).sort_values('ds')
    if 'yhat' not in df or 'actual' not in df:
        return pd.DataFrame()

    df['pred_direction']   = df['yhat'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['actual_direction'] = df['actual'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['correct']          = df['pred_direction'] == df['actual_direction']
    df['7d_accuracy']      = df['correct'].rolling(7).mean()
    df['30d_accuracy']     = df['correct'].rolling(30).mean()
    return df[['ds', '7d_accuracy', '30d_accuracy', 'correct']]


def _latest_log(ticker):
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{ticker}_")]
    return os.path.join(LOG_DIR, sorted(logs)[-1]) if logs else None


def auto_retrain_forecast_model(ticker):
    log_path = _latest_log(ticker)
    if not log_path:
        return
    df = pd.read_csv(log_path).dropna(subset=['yhat', 'actual'])
    if df.empty:
        return

    mae = mean_absolute_error(df['actual'], df['yhat'])
    mse = mean_squared_error(df['actual'], df['yhat'])
    r2  = r2_score(df['actual'], df['yhat'])

    row = pd.DataFrame([[datetime.now(), ticker, mae, mse, r2]],
                       columns=["timestamp", "ticker", "mae", "mse", "r2"])
    eval_path = os.path.join(EVAL_DIR, "forecast_evals.csv")
    if os.path.exists(eval_path):
        row.to_csv(eval_path, mode="a", header=False, index=False)
    else:
        row.to_csv(eval_path, index=False)

    log_eval_to_gsheet(ticker, mae, mse, r2)
# ==============================================================================

# ----------------------------------------------------------------------
# 🗂️ Simple file helper – returns newest forecast_<ticker>_*.csv log
# ----------------------------------------------------------------------
def get_latest_forecast_log(ticker: str, log_dir: str = LOG_DIR) -> str | None:
    """Return full path to the most recent forecast log for `ticker`
    (files are named forecast_<TICKER>_YYYYMMDD_*.csv)."""
    logs = [f for f in os.listdir(log_dir)
            if f.startswith(f"forecast_{ticker.upper()}_")]
    if not logs:
        return None
    latest = sorted(logs)[-1]          # alphabetical sort ≈ chronological
    return os.path.join(log_dir, latest)


# === Convenience helpers ======================================================
def load_forecast_tickers():
    if not os.path.exists(TICKER_FILE):
        return ["AAPL", "MSFT"]
    with open(TICKER_FILE, "r") as f:
        return [ln.strip().upper() for ln in f if ln.strip()]


def save_forecast_tickers(ticker_list):
    with open(TICKER_FILE, "w") as f:
        for tkr in ticker_list:
            f.write(tkr.strip().upper() + "\n")


def run_auto_retrain_all(ticker_list=None):
    if ticker_list is None:
        ticker_list = load_forecast_tickers()
    for tkr in ticker_list:
        try:
            auto_retrain_forecast_model(tkr)
        except Exception as e:
            print(f"❌ Retrain error for {tkr}: {e}")
# ==============================================================================
