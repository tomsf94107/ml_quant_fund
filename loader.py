# loader.py
from __future__ import annotations
from pathlib import Path
import os
import sqlite3
import pandas as pd

# Resolve DB path robustly:
# 1) FORECAST_ACCURACY_DB env var
# 2) a file named forecast_accuracy.db next to this loader.py
# 3) fallback to CWD
def _resolve_db_path() -> str:
    env = os.getenv("FORECAST_ACCURACY_DB")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve().parent
    p = here / "forecast_accuracy.db"
    if p.exists():
        return str(p)
    return "forecast_accuracy.db"

DEFAULT_DB = _resolve_db_path()
DEFAULT_TABLES = ("forecast_accuracy", "metrics")

def load_eval_logs_from_forecast_db(
    db_path: str | None = None,
    tables: tuple[str, ...] = DEFAULT_TABLES
) -> pd.DataFrame:
    db_path = db_path or DEFAULT_DB

    with sqlite3.connect(db_path) as conn:
        # discover existing tables
        existing = set(pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )["name"].tolist())

        frames: list[pd.DataFrame] = []
        for t in tables:
            if t not in existing:
                continue
            df = pd.read_sql(f"SELECT * FROM {t}", conn)
            if df.empty:
                continue

            df.columns = df.columns.str.lower()

            # unify a 'date' column
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
            elif "ingested_at" in df.columns:
                df["date"] = pd.to_datetime(df["ingested_at"], errors="coerce")
            else:
                df["date"] = pd.NaT

            # ensure required columns
            for col in ("ticker", "mae", "mse", "r2"):
                if col not in df.columns:
                    df[col] = pd.NA

            # dashboard expects these; we default if not present
            if "model" not in df.columns:
                df["model"] = "XGBoost (Short Term)"
            if "confidence" not in df.columns:
                df["confidence"] = 1.0

            frames.append(df[["date","ticker","mae","mse","r2","model","confidence"]])

    if not frames:
        return pd.DataFrame(columns=["date","ticker","mae","mse","r2","model","confidence"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

# Quick self-test (run `python loader.py`)
if __name__ == "__main__":
    df = load_eval_logs_from_forecast_db()
    print("DB:", DEFAULT_DB)
    print("rows:", len(df))
    if len(df):
        print("range:", df["date"].min(), "→", df["date"].max())
        print("sample:\n", df.head(5))
