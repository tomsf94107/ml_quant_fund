# forecast_utils.py v5.7 – env-guarded sentiment/insiders, smarter risk key use,
#                          Neon accuracy logging kept; SQLite loader unchanged
# ---------------------------------------------------------------------------
from __future__ import annotations

import os, sys, types, importlib.util, io, contextlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Optional TA; keep soft import to avoid hard failure
try:
    import pandas_ta as ta  # noqa
except Exception:
    ta = None  # finalize_features should tolerate missing TA columns

# Prophet (optional; used in forecast_price_trend)
try:
    from prophet import Prophet  # type: ignore
except Exception:
    Prophet = None  # graceful fallback below

# SQLAlchemy for SQLite access (for legacy accuracy loader below)
from sqlalchemy import create_engine, text

# ────────────────────────────────────────────────────────────────────────────
# Environment flags (let CLI disable network/secret-heavy bits)
# ────────────────────────────────────────────────────────────────────────────
NO_SENTIMENT = os.getenv("NO_SENTIMENT", "0") == "1"
NO_INSIDERS  = os.getenv("NO_INSIDERS",  "0") == "1"
DEBUG_FU     = os.getenv("FORECAST_UTILS_DEBUG", "0") == "1"

def _dbg(msg: str):
    if DEBUG_FU:
        print(f"[forecast_utils] {msg}")

# ────────────────────────────────────────────────────────────────────────────
# Dynamic imports / fallbacks for project modules (NO email import here)
# ────────────────────────────────────────────────────────────────────────────
def _find_file(base_dir: str, filename: str) -> str | None:
    """Return first path under base_dir whose basename == filename."""
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def _load_by_path(path: str, fullname: str):
    if not path or not os.path.exists(path):
        raise ModuleNotFoundError(fullname)
    spec = importlib.util.spec_from_file_location(fullname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

try:
    # 1) Relative (when imported as part of package)
    from .core.feature_utils import finalize_features
    from .data.etl_insider import fetch_insider_trades
    from .data.etl_holdings import fetch_insider_holdings
    from .core.helpers_xgb import train_xgb_predict
    from .events_risk import build_risk_features
    from .sentiment_utils import get_sentiment_scores
except ImportError:
    try:
        # 2) Absolute (ml_quant_fund.*)
        from ml_quant_fund.core.feature_utils import finalize_features
        from ml_quant_fund.data.etl_insider import fetch_insider_trades
        from ml_quant_fund.data.etl_holdings import fetch_insider_holdings
        from ml_quant_fund.core.helpers_xgb import train_xgb_predict
        from ml_quant_fund.events_risk import build_risk_features
        from ml_quant_fund.sentiment_utils import get_sentiment_scores
    except ImportError:
        # 3) Recursive path fallback
        PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # …/ml_quant_fund
        PARENT  = os.path.dirname(PKG_DIR)
        if PARENT not in sys.path:
            sys.path.insert(0, PARENT)
        if "ml_quant_fund" not in sys.modules:
            pkg = types.ModuleType("ml_quant_fund"); pkg.__path__ = [PKG_DIR]
            sys.modules["ml_quant_fund"] = pkg

        feat_mod   = _load_by_path(_find_file(PKG_DIR, "feature_utils.py"),
                                   "ml_quant_fund.core.feature_utils")
        finalize_features = getattr(feat_mod, "finalize_features")

        etli_path  = _find_file(PKG_DIR, "etl_insider.py")
        etlh_path  = _find_file(PKG_DIR, "etl_holdings.py")
        if not etli_path:
            raise ModuleNotFoundError("ml_quant_fund.data.etl_insider")
        if not etlh_path:
            raise ModuleNotFoundError("ml_quant_fund.data.etl_holdings")

        etli_mod   = _load_by_path(etli_path, "ml_quant_fund.data.etl_insider")
        etlh_mod   = _load_by_path(etlh_path, "ml_quant_fund.data.etl_holdings")
        fetch_insider_trades   = getattr(etli_mod, "fetch_insider_trades")
        fetch_insider_holdings = getattr(etlh_mod, "fetch_insider_holdings")

        hxgb_mod   = _load_by_path(_find_file(PKG_DIR, "helpers_xgb.py"),
                                   "ml_quant_fund.core.helpers_xgb")
        train_xgb_predict = getattr(hxgb_mod, "train_xgb_predict")

        risk_mod   = _load_by_path(_find_file(PKG_DIR, "events_risk.py"),
                                   "ml_quant_fund.events_risk")
        build_risk_features = getattr(risk_mod, "build_risk_features")

        sent_mod   = _load_by_path(_find_file(PKG_DIR, "sentiment_utils.py"),
                                   "ml_quant_fund.sentiment_utils")
        get_sentiment_scores = getattr(sent_mod, "get_sentiment_scores")

# Lazy email import to avoid pulling Streamlit into CI
def _maybe_email(subject: str, body: str):
    send = None
    try:
        from .send_email import send_email_alert as send  # type: ignore
    except Exception:
        try:
            from ml_quant_fund.send_email import send_email_alert as send  # type: ignore
        except Exception:
            send = None
    if send:
        try:
            send(subject, body)
        except Exception:
            pass

# ────────────────────────────────────────────────────────────────────────────
# Accuracy sink (Neon/Postgres via ACCURACY_DSN, fallback to SQLite)
# ────────────────────────────────────────────────────────────────────────────
try:
    from ml_quant_fund.accuracy_sink import log_accuracy  # writes to Neon if DSN set
except Exception:
    def log_accuracy(*args, **kwargs):  # no-op if sink unavailable
        return

def _compute_and_log_accuracy(ticker: str, y_true: np.ndarray, y_pred: np.ndarray,
                              model: str = "XGBoost (Short Term)", confidence: float = 1.0):
    """Compute MAE/MSE/R² and push one row to the accuracy sink."""
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        try:
            r2  = float(r2_score(y_true, y_pred))
        except Exception:
            r2  = float("nan")
        log_accuracy(ticker, mae, mse, r2, model=model, confidence=confidence)
    except Exception as e:
        print(f"[accuracy_sink] log failed: {e}")

# ────────────────────────────────────────────────────────────────────────────
# Paths & constants
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

# repo-local root for relative DB paths
ROOT = os.path.abspath(os.path.dirname(__file__))

# ────────────────────────────────────────────────────────────────────────────
# Accuracy DB loader (SQLite by default; overridable by caller)
# ────────────────────────────────────────────────────────────────────────────
def _resolve_sqlite_path(db_url: str) -> str:
    """
    Accepts 'sqlite:///relative_or_abs_path'. Returns absolute file path.
    """
    if not db_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite URLs are supported, e.g. sqlite:///forecast_accuracy.db")
    path = db_url.replace("sqlite:///", "", 1)
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)

def load_forecast_accuracy(db_url: str = "sqlite:///forecast_accuracy.db") -> pd.DataFrame:
    """
    Load accuracy rows from SQLite table forecast_accuracy.
    """
    try:
        abs_path = _resolve_sqlite_path(db_url)
        engine = create_engine(f"sqlite:///{abs_path}")

        # Ensure table exists
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS forecast_accuracy (
                    timestamp TEXT,
                    ticker    TEXT,
                    mae       FLOAT,
                    mse       FLOAT,
                    r2        FLOAT
                )
            """))

        # Load it
        query = """
            SELECT timestamp, ticker, mae, mse, r2
            FROM forecast_accuracy
            ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, engine, parse_dates=["timestamp"])
        return df
    except Exception as e:
        print(f"[load_forecast_accuracy] Failed: {e}")
        return pd.DataFrame(columns=["timestamp", "ticker", "mae", "mse", "r2"])

# ────────────────────────────────────────────────────────────────────────────
# Core Feature-Engineering + Forecast Wrappers
# ────────────────────────────────────────────────────────────────────────────
def _to_datetime(obj):
    if obj is None:
        return None
    try:
        return pd.to_datetime(obj)
    except Exception:
        return None

def build_feature_dataframe(
    ticker: str,
    start=None,
    end=None,
    lookback: int = 180,
    min_rows: int = 200
) -> pd.DataFrame:
    """
    Download OHLCV → (optionally) insiders → TA → market context →
    (optionally) sentiment → risk features → finalize.
    Use env flags to skip heavy bits during CLI runs:
      - NO_INSIDERS=1  → skip insider fetch/merge
      - NO_SENTIMENT=1 → skip news/sentiment
    """
    # 1) Price history
    start_dt = _to_datetime(start)
    end_dt   = _to_datetime(end)

    if start_dt is not None and end_dt is not None:
        df = yf.download(ticker, start=start_dt, end=end_dt, auto_adjust=True, progress=False)
        if len(df) < min_rows:
            extra = max(min_rows - len(df) + 30, lookback)
            df = yf.download(ticker, period=f"{extra}d", auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period=f"{lookback}d", auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    # 2) (Optional) Inject insider trades + holdings
    if NO_INSIDERS:
        _dbg("insiders disabled via NO_INSIDERS=1")
        for col in ["insider_net_shares","insider_buy_count","insider_sell_count","hold_shares","hold_net_change"]:
            df[col] = 0
    else:
        try:
            ins_trades = fetch_insider_trades(ticker)  # expects 'ds'
            ins_holds  = fetch_insider_holdings(ticker)

            # Ensure price index tz-naive
            try:
                df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)
            except Exception:
                pass

            def _prep(ins: pd.DataFrame) -> pd.DataFrame:
                if isinstance(ins, pd.DataFrame) and not ins.empty and "ds" in ins.columns:
                    ins = ins.copy()
                    ins["ds"] = pd.to_datetime(ins["ds"], errors="coerce", utc=True).dt.tz_localize(None)
                    ins = ins.dropna(subset=["ds"]).set_index("ds").sort_index()
                    return ins
                return pd.DataFrame()

            ins_trades = _prep(ins_trades)
            ins_holds  = _prep(ins_holds)

            if not ins_trades.empty:
                df = df.join(ins_trades, how="left")

            if not ins_holds.empty:
                df = df.join(
                    ins_holds.rename(columns={"shares": "hold_shares", "net_change": "hold_net_change"}),
                    how="left"
                )

            # ensure expected cols exist → fillna(0)
            need = ["insider_net_shares", "insider_buy_count", "insider_sell_count",
                    "hold_shares", "hold_net_change"]
            for col in need:
                if col not in df.columns:
                    df[col] = 0
            df[need] = df[need].fillna(0)
        except Exception as e:
            print(f"⚠️ Insider merge error: {e}")
            for col in ["insider_net_shares","insider_buy_count","insider_sell_count","hold_shares","hold_net_change"]:
                df[col] = 0

    # Ensure chronology after joins
    df.sort_index(inplace=True)

    # --- Insider feature engineering (after merge) -----------------------
    def _signed_log1p(x):
        return np.sign(x) * np.log1p(np.abs(x))

    # dollar-scaled flow (helps normalize across tickers)
    df["insider_flow_dollars"] = df["insider_net_shares"] * df["Close"]

    # stable, signed transform
    df["insider_flow_log"] = _signed_log1p(df["insider_net_shares"].fillna(0))

    # short/medium flow momentum
    df["insider_flow_7d"]  = df["insider_net_shares"].rolling(7,  min_periods=1).sum()
    df["insider_flow_21d"] = df["insider_net_shares"].rolling(21, min_periods=1).sum()

    # activity intensity
    df["insider_activity_7d"]  = (df["insider_net_shares"] != 0).rolling(7,  min_periods=1).sum()
    df["insider_activity_21d"] = (df["insider_net_shares"] != 0).rolling(21, min_periods=1).sum()

    # 3) Technical indicators
    df["Return_1D"] = df["Close"].pct_change()
    for w in (5, 10, 20):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    if ta is not None:
        try:
            df["RSI14"] = ta.rsi(df["Close"], length=14)

            macd = ta.macd(df["Close"])
            if macd is not None and not macd.empty:
                df["MACD"], df["MACD_sig"] = macd.iloc[:, 0], macd.iloc[:, 1]

            # Bollinger with fallback (finalize_features may also fill)
            bb = ta.bbands(df["Close"])
            if isinstance(bb, pd.DataFrame) and not bb.empty:
                if "BBP_20_2.0" in bb.columns:
                    df["BB_width"] = bb["BBP_20_2.0"]
                elif {"BBU_20_2.0", "BBL_20_2.0"}.issubset(bb.columns):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        df["BB_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df["Close"]
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"])
        except Exception:
            pass

    # 4) Volume & market context
    if "Volume" in df.columns and df["Volume"].notna().any():
        vol = df["Volume"].fillna(0)
        pv  = (df["Close"] * vol).cumsum()
        vc  = vol.cumsum().replace(0, np.nan)  # guard against early zeros
        df["VWAP"] = (pv / vc).bfill()

        if ta is not None:
            try:
                df["OBV"] = ta.obv(df["Close"], df["Volume"])
            except Exception:
                df["OBV"] = np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            roll_mean = df["Volume"].rolling(VOL_LOOKBACK_Z).mean()
            roll_std  = df["Volume"].rolling(VOL_LOOKBACK_Z).std()
            df["vol_zscore"] = (df["Volume"] - roll_mean) / roll_std

    for etf in (SPY_TICKER, SECTOR_ETF):
        try:
            etf_price = yf.download(
                etf,
                start=df.index.min(),
                end=df.index.max(),
                auto_adjust=True,
                progress=False
            )["Close"]
            df[f"{etf}_ret"] = etf_price.pct_change()
        except Exception:
            df[f"{etf}_ret"] = np.nan

    # 5) (Optional) Sentiment
    if NO_SENTIMENT:
        _dbg("sentiment disabled via NO_SENTIMENT=1")
        for k in ("positive", "neutral", "negative"):
            df[f"sent_{k}"] = 0
    else:
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                sent = get_sentiment_scores(ticker)  # expected dict of percentages
            for k in ("positive", "neutral", "negative"):
                df[f"sent_{k}"] = sent.get(k, 0)
            if sent.get("positive", 0) > 70 or sent.get("negative", 0) > 70:
                _maybe_email(f"Sentiment Alert: {ticker}", str(sent))
        except Exception as e:
            print(f"⚠️ Sentiment error: {e}")
            for k in ("positive", "neutral", "negative"):
                df[f"sent_{k}"] = 0

    # 6) Risk features (robust join on normalized date)
    try:
        risk_start = (start_dt.date() if start_dt is not None else df.index.min().date())
        risk_end   = (end_dt.date()   if end_dt   is not None else df.index.max().date())

        # auto-detect API availability unless explicitly disabled
        use_fmp      = (os.getenv("NO_FMP", "0") != "1") and bool(os.getenv("FMP_API_KEY"))
        use_finnhub  = (os.getenv("NO_FINNHUB", "0") != "1") and bool(os.getenv("FINNHUB_API_KEY"))

        risk = build_risk_features(
            risk_start - timedelta(days=3),
            risk_end   + timedelta(days=3),
            use_fmp=use_fmp, use_finnhub=use_finnhub
        )  # expects: ['date','risk_today','risk_next_1d','risk_next_3d','risk_prev_1d']

        # Normalize keys to tz-naive midnight
        risk = risk.copy()
        risk["date"] = pd.to_datetime(risk["date"], errors="coerce")\
                            .dt.tz_localize(None)\
                            .dt.normalize()

        df = df.copy()
        df["date"] = pd.to_datetime(df.index, errors="coerce")\
                        .tz_localize(None)\
                        .normalize()

        # Join on normalized date
        risk_idxed = risk.set_index("date")
        df = df.join(risk_idxed, on="date", how="left").drop(columns=["date"])

        # Ensure columns exist and fill NaN
        for c in ["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]:
            if c not in df.columns:
                df[c] = 0
        df[["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]] = \
            df[["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]].fillna(0)

        if os.getenv("RISK_DEBUG") == "1":
            nz = (df[["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]] != 0).sum().to_dict()
            print("risk nonzero counts:", nz)
    except Exception as e:
        print(f"⚠️ Risk features failed: {e}")
        for c in ["risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]:
            df[c] = 0

    # 7) Finalize & return
    df = finalize_features(df)
    return df.dropna(subset=["Close", "MA5", "MA10", "MA20"])

# ────────────────────────────────────────────────────────────────────────────
# XGBoost wrapper with retries  (logs accuracy on success)
# ────────────────────────────────────────────────────────────────────────────
def safe_train_xgb_with_retries(
    tkr, start=None, end=None,
    init_look=180, max_look=720, step=180
):
    look = init_look
    while look <= max_look:
        try:
            df = build_feature_dataframe(tkr, start, end, look)
            if df.empty:
                raise ValueError("no data")
            mdl, Xts, yts, yhat, fb = train_xgb_predict(df)
            if yhat is None or len(yhat) == 0:
                raise ValueError("empty pred")

            # ✅ log accuracy if we have a valid eval set
            if yts is not None and yhat is not None and len(yts) == len(yhat):
                _compute_and_log_accuracy(
                    tkr, np.asarray(yts), np.asarray(yhat),
                    model="XGBoost (Short Term)", confidence=1.0
                )

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
    if Prophet is None:
        return None, "Prophet not available"

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
          .merge(dfp[["ds", "y"]].rename(columns={"y": "actual"}),
                 on="ds", how="left")
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
        df_i = yf.download(ticker, period="7d", interval="1h", auto_adjust=True, progress=False)
        if df_i.empty:
            raise ValueError("no intraday data")
        df_i["Return"] = df_i["Close"].pct_change()
        df_i["Trend"]  = df_i["Return"].rolling(3).mean()
        trend = df_i["Trend"].iloc[-1]
        pct   = float(df_i["Return"].iloc[-1] or 0) * 100

        if trend > 0.001:
            intraday_msg = f"📈 Likely Uptrend ({pct:.2f}%)"
        elif trend < -0.001:
            intraday_msg = f"📉 Likely Downtrend ({pct:.2f}%)"
        else:
            intraday_msg = f"🔄 Flat Trend ({pct:.2f}%)"

        if log_results:
            df_i.to_csv(
                os.path.join(INTRA_DIR, f"intraday_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv")
            )
    except Exception as e:
        intraday_msg = f"⚠️ Intraday error: {e}"

    try:
        mdl, Xts, _, yhat, fb = safe_train_xgb_with_retries(ticker, start, end)
        last_close = Xts.iloc[-1]["Close"] if Xts is not None and not Xts.empty else 0
        up = yhat[-1] > last_close
        direction = "🟢 ML Forecast Up" if up else "🔴 ML Forecast Down"
        if fb:
            direction += f" ({fb})"
        return f"{intraday_msg} + {direction}", ""
    except Exception as e:
        return intraday_msg, f"❌ Model error: {e}"

# ────────────────────────────────────────────────────────────────────────────
# Eval, logging & retrain wrappers (Prophet CSV or XGB fallback)
# ────────────────────────────────────────────────────────────────────────────
def _latest_log(tkr: str):
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{tkr}_")]
    return os.path.join(LOG_DIR, sorted(logs)[-1]) if logs else None

def auto_retrain_forecast_model(tkr: str):
    """
    Prefer logging from the most recent Prophet forecast CSV if present.
    If not, run a quick XGB eval and log MAE/MSE/R² instead.
    """
    # 1) Prophet CSV path (if any)
    path = _latest_log(tkr)

    # 1a) If CSV exists, compute metrics from it
    if path and os.path.exists(path):
        df_e = pd.read_csv(path).dropna(subset=["yhat", "actual"])
        if not df_e.empty:
            err = df_e["actual"] - df_e["yhat"]
            mae = float(err.abs().mean())
            mse = float((err**2).mean())
            ss_res = float((err**2).sum())
            ss_tot = float(((df_e["actual"] - df_e["actual"].mean())**2).sum())
            r2 = float(1 - ss_res/ss_tot) if ss_tot else float("nan")

            row = pd.DataFrame([[datetime.now(), tkr, mae, mse, r2]],
                               columns=["timestamp", "ticker", "mae", "mse", "r2"])
            os.makedirs(EVAL_DIR, exist_ok=True)
            eval_path = os.path.join(EVAL_DIR, "forecast_evals.csv")
            row.to_csv(eval_path, mode="a", header=not os.path.exists(eval_path), index=False)

            # Prophet-generated forecast → log as Prophet
            try:
                log_accuracy(tkr, mae, mse, r2, model="Prophet", confidence=1.0)
            except Exception as e:
                print(f"[accuracy_sink] log failed: {e}")

            return row

    # 2) Fallback: run XGB quickly and log its metrics
    try:
        mdl, Xts, yts, yhat, _ = safe_train_xgb_with_retries(tkr, init_look=360, max_look=720, step=180)
        if yts is not None and yhat is not None and len(yts) == len(yhat):
            _compute_and_log_accuracy(
                tkr, np.asarray(yts), np.asarray(yhat),
                model="XGBoost (Short Term)", confidence=1.0
            )
            # return a small DataFrame for UI parity
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = float(mean_absolute_error(yts, yhat))
            mse = float(mean_squared_error(yts, yhat))
            try:
                r2 = float(r2_score(yts, yhat))
            except Exception:
                r2 = float("nan")
            return pd.DataFrame([[datetime.now(), tkr, mae, mse, r2]],
                                columns=["timestamp", "ticker", "mae", "mse", "r2"])
    except Exception as e:
        print(f"[auto_retrain_forecast_model] XGB fallback failed: {e}")

    # Nothing to log
    return pd.DataFrame(columns=["timestamp", "ticker", "mae", "mse", "r2"])

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
# SHAP Optional Plotter (kept minimal; use in UI helper)
# ────────────────────────────────────────────────────────────────────────────
try:
    import shap  # noqa
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def plot_shap(model, X, top_n: int = 10):
    if not SHAP_AVAILABLE or model is None or X is None or X.empty:
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
