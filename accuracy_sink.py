from __future__ import annotations
import os, sqlite3
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

PG_DSN = os.getenv("ACCURACY_DSN")

CREATE_SQL_PG = """
CREATE TABLE IF NOT EXISTS metrics (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  ticker TEXT NOT NULL,
  mae DOUBLE PRECISION,
  mse DOUBLE PRECISION,
  r2  DOUBLE PRECISION,
  model TEXT DEFAULT 'XGBoost (Short Term)',
  confidence DOUBLE PRECISION DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_metrics_ticker_ts ON metrics(ticker, ts);
"""

CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS metrics (
  timestamp   TEXT,
  ticker      TEXT,
  mae         REAL,
  mse         REAL,
  r2          REAL,
  ingested_at TEXT
);
"""

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_accuracy(ticker: str, mae: float, mse: float, r2: float,
                 model: str = "XGBoost (Short Term)", confidence: float = 1.0,
                 sqlite_path: str = "forecast_accuracy.db"):
    """Write one row of metrics to Postgres (if ACCURACY_DSN set) else SQLite."""
    if PG_DSN:
        eng = create_engine(PG_DSN, pool_pre_ping=True)
        with eng.begin() as cx:
            cx.execute(text(CREATE_SQL_PG))
            cx.execute(
                text("""INSERT INTO metrics (ticker, mae, mse, r2, model, confidence)
                        VALUES (:t, :mae, :mse, :r2, :m, :c)"""),
                dict(t=ticker.upper(), mae=float(mae), mse=float(mse),
                     r2=float(r2), m=model, c=float(confidence))
            )
        return
    # SQLite fallback
    ts = _now_str()
    with sqlite3.connect(sqlite_path) as c:
        c.executescript(CREATE_SQLITE)
        c.execute(
            "INSERT INTO metrics (timestamp,ticker,mae,mse,r2,ingested_at) VALUES (?,?,?,?,?,?)",
            (ts, ticker.upper(), float(mae), float(mse), float(r2), ts),
        )

def load_accuracy_any(sqlite_path: str = "forecast_accuracy.db") -> pd.DataFrame:
    """Read metrics from Postgres if DSN set; otherwise from local SQLite/loader."""
    if PG_DSN:
        eng = create_engine(PG_DSN, pool_pre_ping=True)
        q = """SELECT ts AS date, upper(ticker) AS ticker, mae, mse, r2, model, confidence
               FROM metrics ORDER BY ts"""
        return pd.read_sql(q, eng, parse_dates=["date"])
    # SQLite fallback via loader if available
    try:
        from loader import load_eval_logs_from_forecast_db
        return load_eval_logs_from_forecast_db(db_path=sqlite_path)
    except Exception:
        # bare-bones read of local "metrics" table
        if not os.path.exists(sqlite_path):
            return pd.DataFrame(columns=["date","ticker","mae","mse","r2","model","confidence"])
        with sqlite3.connect(sqlite_path) as c:
            try:
                df = pd.read_sql("SELECT * FROM metrics", c)
            except Exception:
                return pd.DataFrame(columns=["date","ticker","mae","mse","r2","model","confidence"])
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["date"] = pd.NaT
        if "model" not in df.columns: df["model"] = "XGBoost (Short Term)"
        if "confidence" not in df.columns: df["confidence"] = 1.0
        keep = ["date","ticker","mae","mse","r2","model","confidence"]
        return df[keep].dropna(subset=["date"]).sort_values("date")
