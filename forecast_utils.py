# forecast_utils.py v5.1 – local CSV eval & full feature set
# ---------------------------------------------------------------------------
import os
from datetime import datetime, timedelta
import contextlib, io

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas_ta as ta
import streamlit as st
from sqlalchemy import create_engine, text

from data.etl_insider  import fetch_insider_trades
from data.etl_holdings import fetch_insider_holdings
from sentiment_utils   import get_sentiment_scores
from send_email        import send_email_alert
from core.helpers_xgb  import train_xgb_predict

# ────────────────────────────────────────────────────────────────────────────
LOG_DIR     = "forecast_logs"
EVAL_DIR    = "forecast_eval"
INTRA_DIR   = "logs"
TICKER_FILE = "tickers.csv"

SPY_TICKER      = "SPY"
SECTOR_ETF      = "XLK"
VOL_LOOKBACK_Z  = 20

for d in (LOG_DIR, EVAL_DIR, INTRA_DIR):
    os.makedirs(d, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# Forecast accuracy loader (Database-based)
# ────────────────────────────────────────────────────────────────────────────

def load_forecast_accuracy() -> pd.DataFrame:
    db_url = st.secrets.get("accuracy_db_url")
    if not db_url:
        st.error("Database URL not configured (accuracy_db_url missing)")
        return pd.DataFrame(columns=["timestamp","ticker","mae","mse","r2"])

    engine = create_engine(db_url)

    # ◾ Ensure the table exists (no-op if it already does):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS forecast_accuracy (
                timestamp TEXT,
                ticker     TEXT,
                mae        FLOAT,
                mse        FLOAT,
                r2         FLOAT
            )
        """))

    query = """
        SELECT timestamp, ticker, mae, mse, r2
          FROM forecast_accuracy
      ORDER BY timestamp DESC
    """
    try:
        return pd.read_sql(query, engine, parse_dates=["timestamp"])
    except Exception as e:
        st.error(f"Failed to load accuracy from DB: {e}")
        return pd.DataFrame(columns=["timestamp","ticker","mae","mse","r2"])


# ────────────────────────────────────────────────────────────────────────────
# Core Feature-Engineering + Forecast Wrappers
# ────────────────────────────────────────────────────────────────────────────
def build_feature_dataframe(
    ticker: str,
    start=None,
    end=None,
    lookback: int = 180,
    min_rows: int = 200
) -> pd.DataFrame:
    """
    Download OHLCV → inject insider trades & holdings → TA →
    market context → sentiment → clean.
    """
    # 1) Price history
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

    # 2) Inject insider trades + holdings, then rename
    try:
        ins_trades = fetch_insider_trades(ticker).set_index("ds")
        ins_holds  = fetch_insider_holdings(ticker).set_index("ds")
        df = (
            df.join(ins_trades, how="left")
              .join(ins_holds.rename(columns={
                  "shares":     "hold_shares",
                  "net_change": "hold_net_change"
              }), how="left")
        )

        df["insider_net_shares"] = df.get("net_shares", 0).fillna(0)
        df["insider_buy_count"]  = df.get("num_buy_tx", 0).fillna(0)
        df["insider_sell_count"] = df.get("num_sell_tx", 0).fillna(0)

        df["hold_shares"]     = df.get("hold_shares", 0).fillna(0)
        df["hold_net_change"] = df.get("hold_net_change", 0).fillna(0)

        drop_cols = [c for c in ["net_shares", "num_buy_tx", "num_sell_tx"] if c in df]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
    except Exception as e:
        print(f"⚠️ Insider merge error: {e}")
        for col in [
            "insider_net_shares", "insider_buy_count", "insider_sell_count",
            "hold_shares", "hold_net_change"
        ]:
            df[col] = 0

    # 3) Technical indicators
    df["Return_1D"] = df["Close"].pct_change()
    for w in (5, 10, 20):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    try:
        df["RSI14"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        if not macd.empty:
            df["MACD"], df["MACD_sig"] = macd.iloc[:, 0], macd.iloc[:, 1]
        df["BB_width"] = ta.bbands(df["Close"])["BBP_20_2.0"]
        df["ATR"]      = ta.atr(df["High"], df["Low"], df["Close"])
    except Exception:
        pass

    # 4) Volume & market context
    if df.get("Volume") is not None and df["Volume"].notna().all():
        df["VWAP"]       = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["OBV"]        = ta.obv(df["Close"], df["Volume"])
        df["vol_zscore"] = (
            df["Volume"] - df["Volume"].rolling(VOL_LOOKBACK_Z).mean()
        ) / df["Volume"].rolling(VOL_LOOKBACK_Z).std()

    for etf in (SPY_TICKER, SECTOR_ETF):
        try:
            etf_ret = (
                yf.download(etf,
                            start=df.index.min(),
                            end=df.index.max(),
                            auto_adjust=True)["Close"]
                .pct_change()
            )
            df[f"{etf}_ret"] = etf_ret
        except Exception:
            df[f"{etf}_ret"] = np.nan

    # 5) Sentiment
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            sent = get_sentiment_scores(ticker)
        for k, v in sent.items():
            df[f"sent_{k}"] = v
        if sent.get("positive", 0) > 70 or sent.get("negative", 0) > 70:
            send_email_alert(f"Sentiment Alert: {ticker}", str(sent))
    except Exception:
        for k in ("positive", "neutral", "negative"):
            df[f"sent_{k}"] = 0

    # 6) Final cleanup
    return df.dropna(subset=["Close", "MA5", "MA10", "MA20"])


# ────────────────────────────────────────────────────────────────────────────
# XGBoost wrapper with retries
# ────────────────────────────────────────────────────────────────────────────
def safe_train_xgb_with_retries(
    tkr, start=None, end=None,
    init_look=180, max_look=720, step=180
):
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
# Prophet forecast
# ────────────────────────────────────────────────────────────────────────────
def forecast_price_trend(
    tkr, start_date=None, end_date=None,
    period_months: int = 3, log_results: bool = True
):
    end_date   = end_date or datetime.today()
    start_date = start_date or end_date - timedelta(days=5*365)
    df         = build_feature_dataframe(tkr, start_date, end_date, lookback=9999)
    if df.empty:
        return None, "No data"

    # prepare DataFrame for Prophet
    dfp = pd.DataFrame({"ds": df.index, "y": df["Close"].values})
    regs = []
    for col in [
        "sent_positive", "sent_neutral", "sent_negative",
        "insider_net_shares", "insider_buy_count", "insider_sell_count"
    ]:
        if col in df.columns:
            dfp[col] = df[col].values
            regs.append(col)

    m = Prophet(daily_seasonality=True)
    for r in regs:
        m.add_regressor(r)

    m.fit(dfp)
    future = m.make_future_dataframe(periods=int(period_months * 30))
    last   = dfp.iloc[-1]
    for r in regs:
        future[r] = last[r]

    fcst = m.predict(future)
    res  = (
        fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]
          .merge(
              dfp[["ds", "y"]].rename(columns={"y": "actual"}),
              on="ds", how="left"
          )
    )

    if log_results:
        out_path = os.path.join(
            LOG_DIR,
            f"forecast_{tkr}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        res.to_csv(out_path, index=False)

    return res, None


# ────────────────────────────────────────────────────────────────────────────
# Intraday + ML combo
# ────────────────────────────────────────────────────────────────────────────
def forecast_today_movement(
    ticker: str, start=None, end=None, log_results: bool = True
) -> tuple[str, str]:
    intraday_msg = ""
    try:
        df_i = yf.download(ticker, period="7d", interval="1h", auto_adjust=True)
        df_i["Return"] = df_i["Close"].pct_change()
        df_i["Trend"]  = df_i["Return"].rolling(3).mean()
        trend = df_i["Trend"].iloc[-1]
        pct   = (df_i["Return"].iloc[-1] or 0) * 100

        if trend > 0.001:
            intraday_msg = f"📈 Likely Uptrend ({pct:.2f}%)"
        elif trend < -0.001:
            intraday_msg = f"📉 Likely Downtrend ({pct:.2f}%)"
        else:
            intraday_msg = f"🔄 Flat Trend ({pct:.2f}%)"

        if log_results:
            df_i.to_csv(
                os.path.join(
                    INTRA_DIR,
                    f"intraday_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv"
                )
            )
    except Exception as e:
        intraday_msg = f"⚠️ Intraday error: {e}"

    try:
        mdl, Xts, _, yhat, fb = safe_train_xgb_with_retries(ticker, start, end)
        up = yhat[-1] > (Xts.iloc[-1]["Close"] if not Xts.empty else 0)
        direction = "🟢 ML Forecast Up" if up else "🔴 ML Forecast Down"
        if fb:
            direction += f" ({fb})"
        return f"{intraday_msg} + {direction}", ""
    except Exception as e:
        return intraday_msg, f"❌ Model error: {e}"


# ────────────────────────────────────────────────────────────────────────────
# Eval, logging & retrain wrappers
# ────────────────────────────────────────────────────────────────────────────
def _latest_log(tkr: str):
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{tkr}_")]
    return os.path.join(LOG_DIR, sorted(logs)[-1]) if logs else None


def auto_retrain_forecast_model(tkr: str):
    """
    Reads the latest forecast CSV, computes MAE/MSE/R2,
    appends to EVAL_DIR/forecast_evals.csv, and returns the new row.
    """
    path = _latest_log(tkr)
    if not path:
        return
    df_e = pd.read_csv(path).dropna(subset=["yhat", "actual"])
    if df_e.empty:
        return
    mae = mean_absolute_error(df_e["actual"], df_e["yhat"])
    mse = mean_squared_error(df_e["actual"], df_e["yhat"])
    r2  = r2_score(df_e["actual"], df_e["yhat"])
    row = pd.DataFrame([[datetime.now(), tkr, mae, mse, r2]],
                       columns=["timestamp", "ticker", "mae", "mse", "r2"])
    eval_path = os.path.join(EVAL_DIR, "forecast_evals.csv")
    row.to_csv(eval_path, mode="a", header=not os.path.exists(eval_path), index=False)
    return row


def run_auto_retrain_all(tickers: list[str]):
    dfs = []
    for t in tickers:
        out = auto_retrain_forecast_model(t)
        if isinstance(out, pd.DataFrame):
            dfs.append(out)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["timestamp", "ticker", "mae", "mse", "r2"])


# ────────────────────────────────────────────────────────────────────────────
# SHAP Optional Plotter
# ────────────────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def plot_shap(model, X, top_n: int = 10):
    if not SHAP_AVAILABLE:
        return
    expl = shap.Explainer(model, X)
    vals = expl(X)
    shap.plots.bar(vals[:, :top_n])


# ────────────────────────────────────────────────────────────────────────────
# Ticker List & Accuracy Helpers
# ────────────────────────────────────────────────────────────────────────────
def load_forecast_tickers() -> list[str]:
    if not os.path.exists(TICKER_FILE):
        return ["AAPL", "MSFT"]
    return [ln.strip().upper() for ln in open(TICKER_FILE) if ln.strip()]


def save_forecast_tickers(tkr_list: list[str]):
    with open(TICKER_FILE, "w") as f:
        for t in tkr_list:
            f.write(t.strip().upper() + "\n")


def compute_rolling_accuracy(log_path: str) -> pd.DataFrame:
    df = pd.read_csv(log_path, parse_dates=["ds"]).sort_values("ds")
    if {"yhat", "actual"} - set(df.columns):
        return pd.DataFrame()
    df["pred_direction"]   = df["yhat"].diff().gt(0).astype(int)
    df["actual_direction"] = df["actual"].diff().gt(0).astype(int)
    df["correct"]          = df["pred_direction"] == df["actual_direction"]
    df["7d_accuracy"]      = df["correct"].rolling(7).mean()
    df["30d_accuracy"]     = df["correct"].rolling(30).mean()
    return df[["ds", "7d_accuracy", "30d_accuracy", "correct"]]


def get_latest_forecast_log(tkr: str, log_dir: str = LOG_DIR) -> str | None:
    """
    Return the path to the most recent forecast CSV for a given ticker.
    """
    logs = [f for f in os.listdir(log_dir) if f.startswith(f"forecast_{tkr}_")]
    return os.path.join(log_dir, sorted(logs)[-1]) if logs else None
