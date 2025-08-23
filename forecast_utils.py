# forecast_utils.py v6.0 – robust insider import (pkg/local/fallback),
#                           insiders from local SQLite via loader.insider_loader,
#                           sanitize insider daily (ensure ticker/date columns),
#                           DB rollups merged (ins_net_shares_7d_db / _21d_db),
#                           safe merges (no MultiIndex ambiguity),
#                           flexible start/end args, env-guarded sentiment & risk,
#                           convenience columns preserved (ticker/date/close)
# ---------------------------------------------------------------------------

from __future__ import annotations

import os, sys, types, importlib.util, io, contextlib
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import yfinance as yf

# ────────────────────────────────────────────────────────────────────────────
# Insider feature builders (expect final_daily → daily → rolling)
# Robust import: package → local → minimal fallback implementations
# ────────────────────────────────────────────────────────────────────────────

def _insider_fallbacks():
    """Return minimal fallback functions if insider_features module is missing."""
    def build_daily_insider_features(final_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize 'final_daily' into a daily insider frame with canonical cols:
          ['ticker','date','net_shares','ins_buy_ct','ins_sell_ct','ins_holdings_delta']
        Assumes final_daily has at least ['ticker','filed_date','net_shares'].
        """
        if final_daily is None or final_daily.empty:
            return pd.DataFrame(columns=[
                "ticker","date","net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"
            ])
        df = final_daily.copy()

        # standardize date/ticker
        date_col = "filed_date" if "filed_date" in df.columns else ("date" if "date" in df.columns else None)
        if date_col is None:
            return pd.DataFrame(columns=[
                "ticker","date","net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"
            ])
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        if "ticker" not in df.columns:
            df["ticker"] = ""
        df["ticker"] = df["ticker"].astype(str).str.upper()

        # map columns
        df["net_shares"]         = pd.to_numeric(df.get("net_shares", 0), errors="coerce").fillna(0)
        df["ins_buy_ct"]         = pd.to_numeric(df.get("num_buy_tx", 0), errors="coerce").fillna(0)
        df["ins_sell_ct"]        = pd.to_numeric(df.get("num_sell_tx", 0), errors="coerce").fillna(0)
        df["ins_holdings_delta"] = pd.to_numeric(df.get("holdings_delta", 0.0), errors="coerce").fillna(0.0)

        # daily grain
        grp = df.groupby(["ticker","date"], as_index=False).agg({
            "net_shares":"sum",
            "ins_buy_ct":"sum",
            "ins_sell_ct":"sum",
            "ins_holdings_delta":"sum",
        })
        return grp

    def add_rolling_insider_features(price_df: pd.DataFrame, insider_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Join to price calendar and compute namespaced rolling features:
          ins_net_shares_{7d,30d}, ins_buy_ct_{7d,30d}, ins_sell_ct_{7d,30d},
          ins_holdings_delta_{7d,30d}, ins_buy_minus_sell_ct,
          ins_pressure_{7d,30d}, ins_pressure_30d_z, ins_large_or_exec_7d=0
        """
        if price_df is None or price_df.empty:
            return pd.DataFrame()
        p = price_df.copy()
        p["ticker"] = p.get("ticker", "").astype(str).str.upper()
        p["date"]   = pd.to_datetime(p["date"], errors="coerce").dt.date

        idf = insider_daily.copy() if insider_daily is not None else pd.DataFrame()
        if idf.empty:
            idf = pd.DataFrame(columns=["ticker","date","net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"])
        idf["ticker"] = idf.get("ticker", "").astype(str).str.upper()
        idf["date"]   = pd.to_datetime(idf["date"], errors="coerce").dt.date

        df = p.merge(idf, on=["ticker","date"], how="left")
        for c in ["net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"]:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

        # rolling helpers on calendar (treat NaN as 0)
        def roll_sum(s, w):
            return pd.Series(s, index=df.index).fillna(0).rolling(w, min_periods=1).sum()

        df["ins_net_shares"]        = df["net_shares"]
        df["ins_net_shares_7d"]     = roll_sum(df["net_shares"], 7)
        df["ins_net_shares_30d"]    = roll_sum(df["net_shares"], 30)

        df["ins_buy_ct"]            = df["ins_buy_ct"]
        df["ins_sell_ct"]           = df["ins_sell_ct"]
        df["ins_buy_ct_7d"]         = roll_sum(df["ins_buy_ct"], 7)
        df["ins_buy_ct_30d"]        = roll_sum(df["ins_buy_ct"], 30)
        df["ins_sell_ct_7d"]        = roll_sum(df["ins_sell_ct"], 7)
        df["ins_sell_ct_30d"]       = roll_sum(df["ins_sell_ct"], 30)

        df["ins_holdings_delta"]    = df["ins_holdings_delta"]
        df["ins_holdings_delta_7d"] = roll_sum(df["ins_holdings_delta"], 7)
        df["ins_holdings_delta_30d"]= roll_sum(df["ins_holdings_delta"], 30)

        df["ins_buy_minus_sell_ct"] = df["ins_buy_ct_30d"] - df["ins_sell_ct_30d"]

        # pressure proxies
        df["ins_pressure_7d"]   = df["ins_net_shares_7d"]  + 0.5*df["ins_holdings_delta_7d"]
        df["ins_pressure_30d"]  = df["ins_net_shares_30d"] + 0.5*df["ins_holdings_delta_30d"] + 0.25*df["ins_buy_minus_sell_ct"]
        mu = df["ins_pressure_30d"].mean(skipna=True)
        sd = df["ins_pressure_30d"].std(skipna=True)
        df["ins_pressure_30d_z"] = (df["ins_pressure_30d"] - mu) / (sd if sd and not np.isnan(sd) else 1.0)

        df["ins_exec_or_large_flag"] = 0
        df["ins_large_or_exec_7d"]   = 0
        df["ins_large_or_exec_30d"]  = 0
        df["ins_abs_net_shares"]     = np.abs(df["ins_net_shares"])
        df["ins_net_shares_norm"]    = df["ins_net_shares"]
        df["ins_holdings_delta_norm"]= df["ins_holdings_delta"]

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df

    return build_daily_insider_features, add_rolling_insider_features

# Try imports in order: package → top-level → fallback
try:
    from ml_quant_fund.insider_features import (
        build_daily_insider_features,
        add_rolling_insider_features,
    )
except Exception:
    try:
        from insider_features import (
            build_daily_insider_features,
            add_rolling_insider_features,
        )
    except Exception:
        build_daily_insider_features, add_rolling_insider_features = _insider_fallbacks()

# Optional TA; soft import
try:
    import pandas_ta as ta  # noqa
except Exception:
    ta = None

# Prophet (optional)
try:
    from prophet import Prophet  # type: ignore
except Exception:
    Prophet = None

# SQLAlchemy (for optional accuracy sinks)
try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = text = None

# ────────────────────────────────────────────────────────────────────────────
# Environment flags
# ────────────────────────────────────────────────────────────────────────────
NO_SENTIMENT = os.getenv("NO_SENTIMENT", os.getenv("DISABLE_SENTIMENT", "0")) == "1"
NO_INSIDERS  = os.getenv("NO_INSIDERS",  "0") == "1"
DEBUG_FU     = os.getenv("FORECAST_UTILS_DEBUG", "0") == "1"

def _dbg(msg: str):
    if DEBUG_FU:
        print(f"[forecast_utils] {msg}")

# ────────────────────────────────────────────────────────────────────────────
# Dynamic imports / fallbacks for project modules
# ────────────────────────────────────────────────────────────────────────────
def _find_file(base_dir: str, filename: str) -> str | None:
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
    # 1) Relative (package)
    from .core.feature_utils import finalize_features
    from .core.helpers_xgb import train_xgb_predict
    from .events_risk import build_risk_features
    from .sentiment_utils import get_sentiment_scores
except ImportError:
    try:
        # 2) Absolute
        from ml_quant_fund.core.feature_utils import finalize_features
        from ml_quant_fund.core.helpers_xgb import train_xgb_predict
        from ml_quant_fund.events_risk import build_risk_features
        from ml_quant_fund.sentiment_utils import get_sentiment_scores
    except ImportError:
        # 3) Fallback by path
        PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        PARENT  = os.path.dirname(PKG_DIR)
        if PARENT not in sys.path:
            sys.path.insert(0, PARENT)
        if "ml_quant_fund" not in sys.modules:
            pkg = types.ModuleType("ml_quant_fund"); pkg.__path__ = [PKG_DIR]
            sys.modules["ml_quant_fund"] = pkg

        feat_mod = _load_by_path(_find_file(PKG_DIR, "feature_utils.py"),
                                 "ml_quant_fund.core.feature_utils")
        finalize_features = getattr(feat_mod, "finalize_features")

        hxgb_mod = _load_by_path(_find_file(PKG_DIR, "helpers_xgb.py"),
                                 "ml_quant_fund.core.helpers_xgb")
        train_xgb_predict = getattr(hxgb_mod, "train_xgb_predict")

        risk_mod = _load_by_path(_find_file(PKG_DIR, "events_risk.py"),
                                 "ml_quant_fund.events_risk")
        build_risk_features = getattr(risk_mod, "build_risk_features")

        sent_mod = _load_by_path(_find_file(PKG_DIR, "sentiment_utils.py"),
                                 "ml_quant_fund.sentiment_utils")
        get_sentiment_scores = getattr(sent_mod, "get_sentiment_scores")

# Local DB insider loader (SQLite-only per your setup)
try:
    from loader import insider_loader  # returns 'final_daily' frame
except Exception:
    insider_loader = None

# Lazy email import (optional)
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

# Accuracy sink (Neon/Postgres via ACCURACY_DSN, fallback to no-op)
try:
    from ml_quant_fund.accuracy_sink import log_accuracy
except Exception:
    def log_accuracy(*args, **kwargs):
        return

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

ROOT = os.path.abspath(os.path.dirname(__file__))

# ────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ────────────────────────────────────────────────────────────────────────────
def _to_datetime(obj):
    if obj is None:
        return None
    try:
        return pd.to_datetime(obj)
    except Exception:
        return None

def _normalize_date_index(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx, errors="coerce")
    try:
        di = di.tz_localize(None)
    except Exception:
        pass
    return di.normalize()

def _norm_dt_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def _reset_index_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index but avoid duplicating columns when index names collide."""
    if isinstance(df.index, pd.RangeIndex):
        return df.copy()
    idx_names = []
    try:
        idx_names = [n for n in (df.index.names or []) if n]
    except Exception:
        if df.index.name:
            idx_names = [df.index.name]
    if any(n in df.columns for n in idx_names if n):
        return df.reset_index(drop=True)
    return df.reset_index()

def _sanitize_insider_daily(daily: pd.DataFrame, tkr: str) -> pd.DataFrame:
    """
    Ensure insider 'daily' has ['ticker','date'] as columns (not index) and the
    expected numeric fields (fill zeros if absent).
    """
    if daily is None:
        return pd.DataFrame(columns=["ticker","date","net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"])
    daily = _reset_index_safe(daily)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = ["__".join(str(x) for x in tup if x is not None) for tup in daily.columns]
    else:
        daily.columns = [str(c) for c in daily.columns]

    if "date" not in daily.columns and "filed_date" in daily.columns:
        daily = daily.rename(columns={"filed_date": "date"})
    if "ticker" not in daily.columns:
        if "Ticker" in daily.columns:
            daily = daily.rename(columns={"Ticker": "ticker"})
        else:
            daily["ticker"] = str(tkr).upper()

    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily["date"]   = pd.to_datetime(daily.get("date", pd.NaT), errors="coerce").dt.date

    need_zero_float = ["net_shares", "ins_buy_ct", "ins_sell_ct", "ins_holdings_delta"]
    for c in need_zero_float:
        if c not in daily.columns:
            if c == "ins_buy_ct" and "num_buy_tx" in daily.columns:
                daily[c] = pd.to_numeric(daily["num_buy_tx"], errors="coerce").fillna(0.0)
            elif c == "ins_sell_ct" and "num_sell_tx" in daily.columns:
                daily[c] = pd.to_numeric(daily["num_sell_tx"], errors="coerce").fillna(0.0)
            elif c == "ins_holdings_delta" and "holdings_delta" in daily.columns:
                daily[c] = pd.to_numeric(daily["holdings_delta"], errors="coerce").fillna(0.0)
            else:
                daily[c] = 0.0
        else:
            daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)

    keep = ["ticker","date","net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"]
    cols = [c for c in keep if c in daily.columns]
    return daily[cols].copy()

# ────────────────────────────────────────────────────────────────────────────
# Core Feature-Engineering + Forecast Wrappers
# ────────────────────────────────────────────────────────────────────────────

def build_feature_dataframe(
    ticker: str,
    start=None, end=None,
    start_date: str | date | None = None,
    end_date:   str | date | None = None,
    lookback: int = 180,
    min_rows: int = 200
) -> pd.DataFrame:
    """
    Download OHLCV → insiders from local DB (daily + rolling via insider_features) → TA → market context →
    (optional) sentiment → risk features → finalize.
    """
    # unify arg names
    if start_date is not None: start = start_date
    if end_date   is not None: end   = end_date

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

    # Ensure price index tz-naive midnight (important for joins)
    df.index = _normalize_date_index(df.index)

    # 2) (Optional) Inject insider signals (SQLite via loader.insider_loader)
    if NO_INSIDERS:
        _dbg("insiders disabled via NO_INSIDERS=1")
        zero_cols = [
            "insider_net_shares","insider_buy_count","insider_sell_count",
            "hold_shares","hold_net_change",
            "ins_net_shares","ins_net_shares_7d","ins_net_shares_30d",
            "ins_buy_ct","ins_sell_ct","ins_buy_ct_7d","ins_sell_ct_7d",
            "ins_holdings_delta","ins_holdings_delta_7d","ins_holdings_delta_30d",
            "ins_exec_or_large_flag","ins_large_or_exec_7d","ins_large_or_exec_30d",
            "ins_buy_minus_sell_ct","ins_abs_net_shares",
            "ins_net_shares_norm","ins_holdings_delta_norm",
            "ins_pressure_7d","ins_pressure_30d","ins_pressure_30d_z",
            "ins_net_shares_7d_db","ins_net_shares_21d_db",
        ]
        for c in zero_cols:
            df[c] = 0
    else:
        try:
            if insider_loader is None:
                raise RuntimeError("insider_loader unavailable (import failed)")

            price_start = df.index.min().date()
            price_end   = df.index.max().date()
            final_daily = insider_loader(ticker, price_start, price_end)

            # Normalize final_daily basics early
            if isinstance(final_daily, pd.DataFrame) and not final_daily.empty:
                if "filed_date" not in final_daily.columns and "date" in final_daily.columns:
                    final_daily = final_daily.rename(columns={"date":"filed_date"})
                if "ticker" not in final_daily.columns:
                    final_daily["ticker"] = str(ticker).upper()
                else:
                    final_daily["ticker"] = final_daily["ticker"].astype(str).str.upper()

            insider_daily_raw = (
                build_daily_insider_features(final_daily)
                if isinstance(final_daily, pd.DataFrame) and not final_daily.empty
                else pd.DataFrame(columns=["ticker","date","net_shares","ins_buy_ct","ins_sell_ct","ins_holdings_delta"])
            )
            insider_daily = _sanitize_insider_daily(insider_daily_raw, str(ticker))

            # Build a price calendar frame for insider feature builder
            price_df = pd.DataFrame({
                "ticker": str(ticker).upper(),
                "date":   df.index.date,
                "close":  df["Close"].values
            })
            price_df["shares_outstanding"] = np.nan
            price_df["market_cap"]         = np.nan

            enriched = add_rolling_insider_features(price_df, insider_daily)

            # Normalize 'date' to datetime (midnight) for safe column-merge
            enriched = enriched.copy()
            enriched["date"] = _norm_dt_col(pd.to_datetime(enriched["date"]))

            # Merge enriched namespaced columns back onto df by 'date'
            ins_cols = [c for c in enriched.columns if c.startswith("ins_")]
            left = df.copy()
            left["date"] = df.index
            df = left.merge(enriched[["date"] + ins_cols], on="date", how="left")
            df = df.drop(columns=["date"]).set_index(left.index)
            if ins_cols:
                df[ins_cols] = df[ins_cols].fillna(0)

            # Legacy compatibility columns: daily 'net_shares' aligned to price index
            if not insider_daily.empty:
                ns = (
                    pd.DataFrame({"date": _norm_dt_col(pd.to_datetime(insider_daily["date"])),
                                  "net_shares": insider_daily["net_shares"]})
                    .set_index("date")["net_shares"]
                )
                df["insider_net_shares"] = ns.reindex(df.index).fillna(0).values
            else:
                df["insider_net_shares"] = 0

            for legacy in ("insider_buy_count","insider_sell_count","hold_shares","hold_net_change"):
                if legacy not in df.columns:
                    df[legacy] = 0

            # Merge DB rollups (direct from final_daily if present)
            if isinstance(final_daily, pd.DataFrame) and not final_daily.empty and \
               {"ticker","filed_date"} <= set(final_daily.columns):

                roll_cols = [c for c in ("insider_7d","insider_21d") if c in final_daily.columns]
                if roll_cols:
                    roll = (
                        final_daily[["ticker","filed_date"] + roll_cols]
                        .rename(columns={"filed_date":"date"})
                        .copy()
                    )
                    roll["ticker"] = str(ticker).upper()
                    roll["date"]   = _norm_dt_col(pd.to_datetime(roll["date"]))

                    left = df.copy()
                    left["date"]   = df.index
                    left["ticker"] = str(ticker).upper()

                    df_roll = left.merge(roll, on=["ticker","date"], how="left") \
                                   .drop(columns=["date"]).set_index(left.index)

                    if "insider_7d" in df_roll.columns:
                        df_roll = df_roll.rename(columns={"insider_7d":"ins_net_shares_7d_db"})
                    if "insider_21d" in df_roll.columns:
                        df_roll = df_roll.rename(columns={"insider_21d":"ins_net_shares_21d_db"})

                    for extra in ("ins_net_shares_7d_db","ins_net_shares_21d_db"):
                        if extra in df_roll.columns:
                            df[extra] = pd.to_numeric(df_roll[extra], errors="coerce").fillna(0.0).values
                        else:
                            df[extra] = 0.0
                else:
                    df["ins_net_shares_7d_db"]  = 0.0
                    df["ins_net_shares_21d_db"] = 0.0
            else:
                df["ins_net_shares_7d_db"]  = 0.0
                df["ins_net_shares_21d_db"] = 0.0

        except Exception as e:
            print(f"⚠️ Insider merge error: {e}")
            for c in [
                "insider_net_shares","insider_buy_count","insider_sell_count","hold_shares","hold_net_change",
                "ins_net_shares","ins_net_shares_7d","ins_net_shares_30d",
                "ins_buy_ct","ins_sell_ct","ins_buy_ct_7d","ins_sell_ct_7d",
                "ins_holdings_delta","ins_holdings_delta_7d","ins_holdings_delta_30d",
                "ins_exec_or_large_flag","ins_large_or_exec_7d","ins_large_or_exec_30d",
                "ins_buy_minus_sell_ct","ins_abs_net_shares",
                "ins_net_shares_norm","ins_holdings_delta_norm",
                "ins_pressure_7d","ins_pressure_30d","ins_pressure_30d_z",
                "ins_net_shares_7d_db","ins_net_shares_21d_db",
            ]:
                df[c] = 0

    # Ensure chronology after joins
    df.sort_index(inplace=True)

    # --- Legacy insider feature engineering (kept for continuity) ---
    def _signed_log1p(x):
        return np.sign(x) * np.log1p(np.abs(x))

    df["insider_flow_dollars"] = df.get("insider_net_shares", 0) * df["Close"]
    df["insider_flow_log"]     = _signed_log1p(df.get("insider_net_shares", 0))
    df["insider_flow_7d"]      = df.get("insider_net_shares", 0).rolling(7,  min_periods=1).sum()
    df["insider_flow_21d"]     = df.get("insider_net_shares", 0).rolling(21, min_periods=1).sum()
    df["insider_activity_7d"]  = (df.get("insider_net_shares", 0) != 0).rolling(7,  min_periods=1).sum()
    df["insider_activity_21d"] = (df.get("insider_net_shares", 0) != 0).rolling(21, min_periods=1).sum()

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
        vc  = vol.cumsum().replace(0, np.nan)
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
            etf_price.index = _normalize_date_index(etf_price.index)
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

        use_fmp      = (os.getenv("NO_FMP", "0") != "1") and bool(os.getenv("FMP_API_KEY"))
        use_finnhub  = (os.getenv("NO_FINNHUB", "0") != "1") and bool(os.getenv("FINNHUB_API_KEY"))

        risk = build_risk_features(
            risk_start - timedelta(days=3),
            risk_end   + timedelta(days=3),
            use_fmp=use_fmp, use_finnhub=use_finnhub
        )  # expects: ['date','risk_today','risk_next_1d','risk_next_3d','risk_prev_1d']

        risk = risk.copy()
        risk["date"] = _norm_dt_col(pd.to_datetime(risk["date"]))

        left = df.copy(); left["date"] = df.index
        df = left.merge(risk[["date","risk_today","risk_next_1d","risk_next_3d","risk_prev_1d"]], on="date", how="left")
        df = df.drop(columns=["date"]).set_index(left.index)

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

    # 7) Finalize engineered features
    df = finalize_features(df)

    # Convenience: keep BOTH capitalized Close and a lowercase alias for UI/snippets
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]

    # Convenience: add easy-access columns for printing/joins (index remains DatetimeIndex)
    df["date"]   = df.index
    df["ticker"] = str(ticker).upper()

    # Ensure required basics present for downstream filters
    req = [c for c in ["Close","MA5","MA10","MA20"] if c in df.columns]
    if req:
        return df.dropna(subset=req)
    return df

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
            df = build_feature_dataframe(tkr, start, end, lookback=look)
            if df.empty:
                raise ValueError("no data")
            mdl, Xts, yts, yhat, fb = train_xgb_predict(df)
            if yhat is None or len(yhat) == 0:
                raise ValueError("empty pred")

            if yts is not None and yhat is not None and len(yts) == len(yhat):
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                mae = float(mean_absolute_error(yts, yhat))
                mse = float(mean_squared_error(yts, yhat))
                try:
                    r2  = float(r2_score(yts, yhat))
                except Exception:
                    r2  = float("nan")
                try:
                    from ml_quant_fund.accuracy_sink import log_accuracy
                    log_accuracy(tkr, mae, mse, r2, model="XGBoost (Short Term)", confidence=1.0)
                except Exception:
                    pass
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
    df         = build_feature_dataframe(tkr, start_date=start_date, end_date=end_date, lookback=9999)
    if df.empty:
        return None, "No data"

    dfp = pd.DataFrame({"ds": df.index, "y": df["Close"].values})
    regs = []
    for col in [
        "sent_positive", "sent_neutral", "sent_negative",
        "insider_net_shares", "insider_buy_count", "insider_sell_count",
        "ins_pressure_30d_z", "ins_large_or_exec_7d", "ins_net_shares_7d",
        "ins_net_shares_7d_db", "ins_net_shares_21d_db",
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
# Eval, logging & retrain wrappers
# ────────────────────────────────────────────────────────────────────────────
def _latest_log(tkr: str):
    logs = [f for f in os.listdir(LOG_DIR) if f.startswith(f"forecast_{tkr}_")]
    return os.path.join(LOG_DIR, sorted(logs)[-1]) if logs else None

def auto_retrain_forecast_model(tkr: str):
    path = _latest_log(tkr)
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

            try:
                log_accuracy(tkr, mae, mse, r2, model="Prophet", confidence=1.0)
            except Exception as e:
                print(f"[accuracy_sink] log failed: {e}")
            return row

    try:
        mdl, Xts, yts, yhat, _ = safe_train_xgb_with_retries(tkr, init_look=360, max_look=720, step=180)
        if yts is not None and yhat is not None and len(yts) == len(yhat):
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = float(mean_absolute_error(yts, yhat))
            mse = float(mean_squared_error(yts, yhat))
            try:
                r2 = float(r2_score(yts, yhat))
            except Exception:
                r2 = float("nan")
            try:
                log_accuracy(tkr, mae, mse, r2, model="XGBoost (Short Term)", confidence=1.0)
            except Exception:
                pass
            return pd.DataFrame([[datetime.now(), tkr, mae, mse, r2]],
                                columns=["timestamp", "ticker", "mae", "mse", "r2"])
    except Exception as e:
        print(f"[auto_retrain_forecast_model] XGB fallback failed: {e}")

    return pd.DataFrame(columns=["timestamp", "ticker", "mae", "mse", "r2"])

def run_auto_retrain_all(tickers: list[str]):
    dfs = []
    for t in tickers:
        out = auto_retrain_forecast_model(t)
        if isinstance(out, pd.DataFrame):
            dfs.append(out)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["timestamp","ticker","mae","mse","r2"])

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

# Constants used by helpers
TICKER_FILE = "tickers.csv"
LOG_DIR     = "forecast_logs"

def load_forecast_tickers() -> list[str]:
    """Load tickers from TICKER_FILE (one per line), falling back to defaults."""
    import os
    if not os.path.exists(TICKER_FILE):
        return ["AAPL", "MSFT"]
    return [ln.strip().upper() for ln in open(TICKER_FILE) if ln.strip()]

def save_forecast_tickers(tkr_list: list[str]):
    """Persist a list of tickers to TICKER_FILE (one per line, uppercased)."""
    with open(TICKER_FILE, "w") as f:
        for t in tkr_list:
            f.write(t.strip().upper() + "\n")

def get_latest_forecast_log(tkr: str, log_dir: str = LOG_DIR) -> str | None:
    """
    Return the absolute path of the newest forecast log CSV for a ticker.
    Looks for files named like 'forecast_{TICKER}_*.csv' and falls back to '*{TICKER}*.csv'.
    """
    import os, glob
    if not os.path.isdir(log_dir):
        return None
    # Primary pattern (recommended writer format)
    cands = glob.glob(os.path.join(log_dir, f"forecast_{tkr}_*.csv"))
    if not cands:
        # Fallback: any csv mentioning the ticker
        cands = glob.glob(os.path.join(log_dir, f"*{tkr}*.csv"))
    return os.path.abspath(max(cands, key=os.path.getmtime)) if cands else None


def compute_rolling_accuracy_compat(ticker_or_path, window_days=30, start_date=None, end_date=None, **kwargs):
    return compute_rolling_accuracy(
        ticker_or_path,
        window_days=window_days,
        start=start_date,
        end=end_date,
        **kwargs
    )

def compute_rolling_accuracy(ticker_or_path,
                             window_days: int = 30,
                             start=None,
                             end=None,
                             log_dir: str = LOG_DIR):
    """
    Compute rolling directional accuracy from a forecast log CSV.

    Accepts either:
      • a ticker (e.g., "AAPL") → resolves latest CSV via get_latest_forecast_log()
      • a direct CSV filepath

    Returns a DataFrame with columns: ['date','accuracy'].

    Optional:
      • start/end → date filters
      • window_days → rolling window size for the hit-rate mean
    """
    import os, glob
    import numpy as np
    import pandas as pd

    def _pick(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    # 1) Resolve the file path
    log_path = None
    if isinstance(ticker_or_path, str) and os.path.isfile(ticker_or_path):
        log_path = ticker_or_path
    else:
        # Treat input as ticker
        try:
            log_path = get_latest_forecast_log(str(ticker_or_path).upper(), log_dir=log_dir)
        except Exception:
            log_path = None
        # Fallback: latest matching CSV in log_dir (primary pattern first, then broad)
        if not log_path or not os.path.exists(log_path):
            pat1 = os.path.join(log_dir, f"forecast_{ticker_or_path}_*.csv")
            pat2 = os.path.join(log_dir, f"*{ticker_or_path}*.csv")
            cands = sorted(glob.glob(pat1) + glob.glob(pat2), key=lambda p: os.path.getmtime(p))
            log_path = cands[-1] if cands else None

    if not log_path or not os.path.exists(log_path):
        raise FileNotFoundError(f"No forecast log found for '{ticker_or_path}' in '{log_dir}'.")


    # 2) Load & normalize columns (use FULL history for rolling calc)
    df_full = pd.read_csv(log_path)
    dcol = _pick(df_full, ["ds", "date", "Date"])
    if dcol is None:
        raise ValueError("No date column found in forecast log.")
    df_full[dcol] = pd.to_datetime(df_full[dcol], errors="coerce")
    df_full = (df_full.rename(columns={dcol: "date"})
               .dropna(subset=["date"])
               .sort_values("date"))

    # Actual / prediction columns (be flexible)
    ycol = _pick(df_full, ["actual", "y", "close", "Close"])
    pcol = _pick(df_full, ["yhat", "pred", "prediction"])
    if ycol is None or pcol is None:
        raise ValueError("Could not find actual/prediction columns in forecast log.")

    y  = pd.to_numeric(df_full[ycol], errors="coerce")
    yh = pd.to_numeric(df_full[pcol], errors="coerce")

    # Directional hits and rolling mean on FULL history
    hits = (np.sign(y.diff()) == np.sign(yh.diff())).astype(float)
    min_periods = max(3, window_days // 3)  # tweak to 1 if you want values immediately
    acc_full = hits.rolling(window_days, min_periods=min_periods).mean()

    out = pd.DataFrame({"date": df_full["date"], "accuracy": acc_full})

    # Apply date filters AFTER rolling to avoid fresh-window NaNs
    if start is not None:
        out = out[out["date"] >= pd.to_datetime(start)]
    if end is not None:
        out = out[out["date"] <= pd.to_datetime(end)]

    return out.reset_index(drop=True)
