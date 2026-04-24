# models/importance_tracker.py
# Tracks feature importance over time across retrains.

from __future__ import annotations
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

DB_PATH = Path("accuracy.db")

def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def init_importance_table() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT    NOT NULL,
                horizon       INTEGER NOT NULL,
                retrain_date  TEXT    NOT NULL,
                feature       TEXT    NOT NULL,
                importance    REAL    NOT NULL,
                rank          INTEGER NOT NULL,
                created_at    TEXT    NOT NULL,
                UNIQUE(ticker, horizon, retrain_date, feature)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fim_ticker_horizon ON feature_importance_history(ticker, horizon)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fim_date ON feature_importance_history(retrain_date)")
        conn.commit()

def save_feature_importance(ticker: str, horizon: int, importances: dict, retrain_date=None) -> int:
    if not importances:
        return 0
    retrain_date = retrain_date or date.today()
    date_str = str(retrain_date)
    from datetime import datetime
    now = datetime.now().isoformat()
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    init_importance_table()
    written = 0
    with _get_conn() as conn:
        for rank, (feature, importance) in enumerate(sorted_items, 1):
            conn.execute("""
                INSERT OR REPLACE INTO feature_importance_history
                    (ticker, horizon, retrain_date, feature, importance, rank, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, horizon, date_str, feature, float(importance), rank, now))
            written += 1
        # Update SHAP importance if provided
        if shap_importances:
            for feat, shap_val in shap_importances.items():
                try:
                    conn.execute("""
                        UPDATE feature_importance_history
                        SET shap_importance = ?
                        WHERE ticker=? AND horizon=? AND feature=? AND retrain_date=?
                    """, (float(shap_val), ticker, horizon, feat, date_str))
                except Exception:
                    pass
        conn.commit()
    return written

def get_top_features(horizon: int = 1, days: int = 30, top_n: int = 20) -> pd.DataFrame:
    cutoff = str(date.today() - timedelta(days=days))
    with _get_conn() as conn:
        df = pd.read_sql("""
            SELECT feature, AVG(importance) as avg_importance,
                   COUNT(*) as n_observations
            FROM feature_importance_history
            WHERE retrain_date >= ? AND horizon = ?
            GROUP BY feature
            ORDER BY avg_importance DESC
            LIMIT ?
        """, conn, params=[cutoff, horizon, top_n])
    return df

def get_declining_features(horizon: int = 1, days: int = 60, decline_threshold: float = 0.20) -> pd.DataFrame:
    cutoff = str(date.today() - timedelta(days=days))
    with _get_conn() as conn:
        df = pd.read_sql("""
            SELECT feature, retrain_date, importance
            FROM feature_importance_history
            WHERE retrain_date >= ? AND horizon = ?
            ORDER BY retrain_date
        """, conn, params=[cutoff, horizon])
    if df.empty:
        return pd.DataFrame()
    df["retrain_date"] = pd.to_datetime(df["retrain_date"])
    mid_date = df["retrain_date"].min() + (df["retrain_date"].max() - df["retrain_date"].min()) / 2
    early = df[df["retrain_date"] <= mid_date].groupby("feature")["importance"].mean()
    recent = df[df["retrain_date"] > mid_date].groupby("feature")["importance"].mean()
    comp = pd.DataFrame({"early": early, "recent": recent}).dropna()
    comp["decline_pct"] = (comp["early"] - comp["recent"]) / comp["early"]
    return comp[comp["decline_pct"] > decline_threshold].sort_values("decline_pct", ascending=False).reset_index()

def get_rising_features(horizon: int = 1, days: int = 60, rise_threshold: float = 0.20) -> pd.DataFrame:
    cutoff = str(date.today() - timedelta(days=days))
    with _get_conn() as conn:
        df = pd.read_sql("""
            SELECT feature, retrain_date, importance
            FROM feature_importance_history
            WHERE retrain_date >= ? AND horizon = ?
            ORDER BY retrain_date
        """, conn, params=[cutoff, horizon])
    if df.empty:
        return pd.DataFrame()
    df["retrain_date"] = pd.to_datetime(df["retrain_date"])
    mid_date = df["retrain_date"].min() + (df["retrain_date"].max() - df["retrain_date"].min()) / 2
    early = df[df["retrain_date"] <= mid_date].groupby("feature")["importance"].mean()
    recent = df[df["retrain_date"] > mid_date].groupby("feature")["importance"].mean()
    comp = pd.DataFrame({"early": early, "recent": recent}).dropna()
    comp["rise_pct"] = (comp["recent"] - comp["early"]) / comp["early"]
    return comp[comp["rise_pct"] > rise_threshold].sort_values("rise_pct", ascending=False).reset_index()
