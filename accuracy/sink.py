# accuracy/sink.py
# ─────────────────────────────────────────────────────────────────────────────
# Prediction logging and accuracy tracking. Replaces Google Sheets + CSV.
#
# What was wrong with the old system:
#   ✗ Logged MAE/MSE/R² — regression metrics on a classifier. Meaningless.
#   ✗ No prediction logging — only recorded how well model fit training data,
#     never checked if live predictions were actually right.
#   ✗ Google Sheets as datastore — rate limits, auth expiry, no transactions.
#
# What this module does:
#   ✓ Two tables:
#       predictions — "on date D, ticker T, horizon H, model said prob=X (BUY)"
#       outcomes    — "on date D+H, ticker T actually moved UP/DOWN"
#   ✓ reconcile() joins them daily to compute realized accuracy
#   ✓ SQLite by default (zero config), Postgres via DATABASE_URL env var
#   ✓ load_accuracy() read function for the dashboard
#
# Zero Streamlit imports. Backend only.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

# ── Database config ────────────────────────────────────────────────────────────
# Set DATABASE_URL in environment for Postgres:
#   postgresql://user:password@host:5432/dbname
# If not set, falls back to SQLite.
DATABASE_URL = os.getenv("DATABASE_URL", "")
SQLITE_PATH  = Path(os.getenv("ACCURACY_DB_PATH", "accuracy.db"))


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _is_postgres() -> bool:
    return bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))


@contextmanager
def _get_conn() -> Iterator:
    """
    Context manager that yields a DB connection.
    Uses Postgres if DATABASE_URL is set, otherwise SQLite.
    Commits on clean exit, rolls back on exception.
    """
    if _is_postgres():
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
        except ImportError:
            raise RuntimeError(
                "DATABASE_URL is set but psycopg2 is not installed. "
                "Run: pip install psycopg2-binary"
            )
    else:
        conn = sqlite3.connect(SQLITE_PATH)

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _placeholder() -> str:
    """SQL parameter placeholder: %s for Postgres, ? for SQLite."""
    return "%s" if _is_postgres() else "?"


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """
    Create tables if they don't exist.
    Safe to call repeatedly — idempotent.
    """
    with _get_conn() as conn:
        cur = conn.cursor()

        # ── predictions: one row per (ticker, horizon, prediction_date) ──────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              SERIAL PRIMARY KEY,
                ticker          TEXT    NOT NULL,
                prediction_date TEXT    NOT NULL,
                horizon         INTEGER NOT NULL,
                prob_up         REAL    NOT NULL,
                signal          TEXT    NOT NULL,
                confidence      TEXT    NOT NULL,
                model_version   TEXT    NOT NULL DEFAULT 'v1',
                created_at      TEXT    NOT NULL,
                UNIQUE(ticker, prediction_date, horizon)
            )
        """ if _is_postgres() else """
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                prediction_date TEXT    NOT NULL,
                horizon         INTEGER NOT NULL,
                prob_up         REAL    NOT NULL,
                signal          TEXT    NOT NULL,
                confidence      TEXT    NOT NULL,
                model_version   TEXT    NOT NULL DEFAULT 'v1',
                created_at      TEXT    NOT NULL,
                UNIQUE(ticker, prediction_date, horizon)
            )
        """)

        # ── outcomes: what actually happened h days after prediction_date ─────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id              SERIAL PRIMARY KEY,
                ticker          TEXT    NOT NULL,
                prediction_date TEXT    NOT NULL,
                horizon         INTEGER NOT NULL,
                outcome_date    TEXT    NOT NULL,
                actual_return   REAL    NOT NULL,
                actual_up       INTEGER NOT NULL,
                created_at      TEXT    NOT NULL,
                UNIQUE(ticker, prediction_date, horizon)
            )
        """ if _is_postgres() else """
            CREATE TABLE IF NOT EXISTS outcomes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                prediction_date TEXT    NOT NULL,
                horizon         INTEGER NOT NULL,
                outcome_date    TEXT    NOT NULL,
                actual_return   REAL    NOT NULL,
                actual_up       INTEGER NOT NULL,
                created_at      TEXT    NOT NULL,
                UNIQUE(ticker, prediction_date, horizon)
            )
        """)

        # ── accuracy_cache: pre-computed metrics (updated by reconcile()) ─────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_cache (
                id              SERIAL PRIMARY KEY,
                ticker          TEXT    NOT NULL,
                horizon         INTEGER NOT NULL,
                window_days     INTEGER NOT NULL,
                accuracy        REAL,
                roc_auc         REAL,
                brier_score     REAL,
                n_predictions   INTEGER NOT NULL DEFAULT 0,
                computed_at     TEXT    NOT NULL,
                UNIQUE(ticker, horizon, window_days)
            )
        """ if _is_postgres() else """
            CREATE TABLE IF NOT EXISTS accuracy_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                horizon         INTEGER NOT NULL,
                window_days     INTEGER NOT NULL,
                accuracy        REAL,
                roc_auc         REAL,
                brier_score     REAL,
                n_predictions   INTEGER NOT NULL DEFAULT 0,
                computed_at     TEXT    NOT NULL,
                UNIQUE(ticker, horizon, window_days)
            )
        """)

        # Indexes
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_pred_date   ON predictions(prediction_date)",
            "CREATE INDEX IF NOT EXISTS idx_out_ticker  ON outcomes(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_out_date    ON outcomes(prediction_date)",
        ]:
            cur.execute(stmt)


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE: LOG PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def log_prediction(
    ticker:          str,
    prediction_date: str | date,
    horizon:         int,
    prob_up:         float,
    signal:          str,
    confidence:      str,
    model_version:   str = "v1",
) -> None:
    """
    Log a single prediction. Called every time generate_signals() runs.

    Parameters
    ----------
    ticker          : e.g. "AAPL"
    prediction_date : the date the prediction was made (today)
    horizon         : 1, 3, or 5 days
    prob_up         : calibrated P(up) from the model
    signal          : "BUY" | "HOLD"
    confidence      : "HIGH" | "MEDIUM" | "LOW"
    model_version   : model identifier for tracking regressions
    """
    init_db()
    p = _placeholder()
    with _get_conn() as conn:
        conn.cursor().execute(f"""
            INSERT OR REPLACE INTO predictions
                (ticker, prediction_date, horizon, prob_up,
                 signal, confidence, model_version, created_at)
            VALUES ({p},{p},{p},{p},{p},{p},{p},{p})
        """, (
            ticker.upper(), str(prediction_date), horizon,
            float(prob_up), signal, confidence, model_version,
            datetime.utcnow().isoformat(),
        ))


def log_predictions_batch(rows: list[dict]) -> int:
    """
    Log multiple predictions at once. Each dict must have keys:
        ticker, prediction_date, horizon, prob_up, signal, confidence
    Optional key: model_version (default "v1")

    Returns number of rows written.
    """
    if not rows:
        return 0
    init_db()
    p = _placeholder()
    now = datetime.utcnow().isoformat()
    records = [
        (
            r["ticker"].upper(),
            str(r["prediction_date"]),
            int(r["horizon"]),
            float(r["prob_up"]),
            r["signal"],
            r["confidence"],
            r.get("model_version", "v1"),
            now,
        )
        for r in rows
    ]
    with _get_conn() as conn:
        conn.cursor().executemany(f"""
            INSERT OR REPLACE INTO predictions
                (ticker, prediction_date, horizon, prob_up,
                 signal, confidence, model_version, created_at)
            VALUES ({p},{p},{p},{p},{p},{p},{p},{p})
        """, records)
    return len(records)


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE: RECONCILE OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════

def reconcile_outcomes(
    tickers: list[str] | None = None,
    as_of:   date | None = None,
) -> int:
    """
    For each logged prediction where the outcome date has passed,
    fetch the actual price return and write to outcomes table.

    This is the function that makes your accuracy metrics real.
    Run it daily (cron job or Streamlit scheduled rerun).

    Returns number of new outcomes recorded.
    """
    import yfinance as yf

    init_db()
    as_of = as_of or date.today()
    p     = _placeholder()

    # Find predictions that don't yet have outcomes and are due
    with _get_conn() as conn:
        query = """
            SELECT p.ticker, p.prediction_date, p.horizon
            FROM predictions p
            LEFT JOIN outcomes o
              ON p.ticker = o.ticker
             AND p.prediction_date = o.prediction_date
             AND p.horizon = o.horizon
            WHERE o.id IS NULL
        """
        pending = pd.read_sql(query, conn)

    if pending.empty:
        return 0

    # Filter to predictions whose outcome date has passed
    pending["prediction_date"] = pd.to_datetime(pending["prediction_date"]).dt.date
    pending["outcome_date"] = pending.apply(
        lambda r: r["prediction_date"] + timedelta(days=int(r["horizon"])),
        axis=1
    )
    due = pending[pending["outcome_date"] <= as_of]

    if due.empty:
        return 0

    if tickers:
        due = due[due["ticker"].isin([t.upper() for t in tickers])]

    # Fetch actual returns
    written = 0
    now = datetime.utcnow().isoformat()

    for ticker, group in due.groupby("ticker"):
        min_date = group["prediction_date"].min() - timedelta(days=2)
        max_date  = group["outcome_date"].max() + timedelta(days=2)

        try:
            px = yf.download(
                ticker, start=str(min_date), end=str(max_date),
                auto_adjust=True, progress=False
            )
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            if px.empty:
                continue
            close = px["Close"].squeeze()
        except Exception as e:
            print(f"  ⚠ Could not fetch prices for {ticker}: {e}")
            continue

        with _get_conn() as conn:
            cur = conn.cursor()
            for _, row in group.iterrows():
                pred_date    = row["prediction_date"]
                outcome_date = row["outcome_date"]

                try:
                    close_pred   = float(close.asof(pd.Timestamp(pred_date)))
                    close_out    = float(close.asof(pd.Timestamp(outcome_date)))
                    actual_ret   = (close_out - close_pred) / close_pred
                    actual_up    = int(actual_ret > 0)
                except Exception:
                    continue

                cur.execute(f"""
                    INSERT OR REPLACE INTO outcomes
                        (ticker, prediction_date, horizon, outcome_date,
                         actual_return, actual_up, created_at)
                    VALUES ({p},{p},{p},{p},{p},{p},{p})
                """, (
                    ticker, str(pred_date), int(row["horizon"]),
                    str(outcome_date), actual_ret, actual_up, now,
                ))
                written += 1

    return written


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE: UPDATE ACCURACY CACHE
# ══════════════════════════════════════════════════════════════════════════════

def update_accuracy_cache(
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Compute accuracy metrics for all ticker×horizon combinations
    and write to accuracy_cache table.

    Call after reconcile_outcomes().
    Returns the metrics DataFrame.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    init_db()
    p = _placeholder()
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()

    with _get_conn() as conn:
        joined = pd.read_sql("""
            SELECT
                p.ticker, p.horizon, p.prob_up, p.signal,
                o.actual_up, o.actual_return, p.prediction_date
            FROM predictions p
            JOIN outcomes o
              ON p.ticker = o.ticker
             AND p.prediction_date = o.prediction_date
             AND p.horizon = o.horizon
        """, conn)

    if joined.empty:
        return pd.DataFrame()

    joined = joined[joined["prediction_date"] >= cutoff]

    rows = []
    now  = datetime.utcnow().isoformat()

    for (ticker, horizon), grp in joined.groupby(["ticker", "horizon"]):
        if len(grp) < 5:   # not enough data for meaningful metrics
            continue

        y_true = grp["actual_up"].astype(int)
        y_prob = grp["prob_up"].astype(float)
        y_pred = (y_prob >= 0.55).astype(int)

        accuracy = float((y_true == y_pred).mean())

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            brier = None

        rows.append({
            "ticker":        ticker,
            "horizon":       horizon,
            "window_days":   window_days,
            "accuracy":      round(accuracy, 4),
            "roc_auc":       round(auc, 4) if auc is not None else None,
            "brier_score":   round(brier, 4) if brier is not None else None,
            "n_predictions": len(grp),
            "computed_at":   now,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    with _get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(f"""
            INSERT OR REPLACE INTO accuracy_cache
                (ticker, horizon, window_days, accuracy, roc_auc,
                 brier_score, n_predictions, computed_at)
            VALUES ({p},{p},{p},{p},{p},{p},{p},{p})
        """, result[[
            "ticker", "horizon", "window_days", "accuracy", "roc_auc",
            "brier_score", "n_predictions", "computed_at"
        ]].values.tolist())

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  READ: DASHBOARD QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def load_accuracy(
    tickers:     list[str] | None = None,
    horizon:     int | None = None,
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Load accuracy metrics from the cache for the dashboard.

    Parameters
    ----------
    tickers     : filter to specific tickers (None = all)
    horizon     : filter to specific horizon (None = all)
    window_days : which rolling window to load (default 90d)

    Returns
    -------
    DataFrame with columns:
        ticker, horizon, accuracy, roc_auc, brier_score, n_predictions
    Sorted by roc_auc descending (best performers first).
    """
    init_db()

    with _get_conn() as conn:
        df = pd.read_sql(
            "SELECT * FROM accuracy_cache WHERE window_days = ?",
            conn, params=(window_days,)
        ) if not _is_postgres() else pd.read_sql(
            "SELECT * FROM accuracy_cache WHERE window_days = %s",
            conn, params=(window_days,)
        )

    if df.empty:
        return df

    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
    if horizon is not None:
        df = df[df["horizon"] == horizon]

    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def load_prediction_history(
    ticker:  str,
    horizon: int = 1,
    days:    int = 90,
) -> pd.DataFrame:
    """
    Load full prediction + outcome history for one ticker.
    Used by the accuracy trend chart on Page 2.

    Returns DataFrame with columns:
        prediction_date, prob_up, signal, actual_up, actual_return, correct
    """
    init_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    p = _placeholder()

    with _get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT
                p.prediction_date,
                p.prob_up,
                p.signal,
                p.confidence,
                o.actual_up,
                o.actual_return
            FROM predictions p
            JOIN outcomes o
              ON p.ticker = o.ticker
             AND p.prediction_date = o.prediction_date
             AND p.horizon = o.horizon
            WHERE p.ticker = {p}
              AND p.horizon = {p}
              AND p.prediction_date >= {p}
            ORDER BY p.prediction_date
        """, conn, params=(ticker.upper(), horizon, cutoff))

    if df.empty:
        return df

    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["correct"] = (
        ((df["signal"] == "BUY")  & (df["actual_up"] == 1)) |
        ((df["signal"] == "HOLD") & (df["actual_up"] == 0))
    ).astype(int)

    return df


def load_accuracy_any(
    tickers:     list[str] | None = None,
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Convenience alias — returns accuracy for any available horizon.
    Called by the old dashboard import: from ml_quant_fund.accuracy_sink import load_accuracy_any
    Keeps backward compatibility during transition.
    """
    return load_accuracy(tickers=tickers, window_days=window_days)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — run the daily reconciliation job
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run accuracy reconciliation")
    parser.add_argument("--reconcile", action="store_true",
                        help="Fetch outcomes for pending predictions")
    parser.add_argument("--cache",     action="store_true",
                        help="Recompute accuracy cache")
    parser.add_argument("--window",    type=int, default=90,
                        help="Rolling window in days (default 90)")
    args = parser.parse_args()

    if args.reconcile or (not args.reconcile and not args.cache):
        print("Reconciling outcomes...")
        n = reconcile_outcomes()
        print(f"  {n} new outcomes recorded")

    if args.cache or (not args.reconcile and not args.cache):
        print("Updating accuracy cache...")
        df = update_accuracy_cache(window_days=args.window)
        if not df.empty:
            print(df[["ticker", "horizon", "accuracy", "roc_auc",
                       "n_predictions"]].to_string(index=False))
        else:
            print("  No data yet — run predictions first")
