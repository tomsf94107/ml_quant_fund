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
from utils.timezone import now_et, today_et, ts_et
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

# ── Database config ────────────────────────────────────────────────────────────
# Set DATABASE_URL in environment for Postgres:
#   postgresql://user:password@host:5432/dbname
# If not set, falls back to SQLite.

# Module-level Session for connection reuse (DNS pool conservation).
# Added May 4 2026 to prevent DNS thread exhaustion in Pipeline B/C.
import requests as _requests_for_session
_session = _requests_for_session.Session()

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
        conn = sqlite3.connect(SQLITE_PATH, timeout=30)

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
    is_watchlist:    bool = False,
    tier:            str = "tactical",
    prob_raw:        float | None = None,
) -> None:
    """
    Log a single prediction. Called every time generate_signals() runs.

    Parameters
    ----------
    ticker          : e.g. "AAPL"
    prediction_date : the date the prediction was made (today)
    horizon         : 1, 3, or 5 days
    prob_up         : effective probability AFTER multipliers
    prob_raw        : raw calibrated P(up) BEFORE multipliers (None if not provided)
    signal          : "BUY" | "HOLD"
    confidence      : "HIGH" | "MEDIUM" | "LOW"
    model_version   : model identifier for tracking regressions
    is_watchlist    : True for watchlist tickers (excluded from main accuracy)
    tier            : "core" | "secondary" | "tactical" | "lotto"
    """
    init_db()
    p = _placeholder()
    with _get_conn() as conn:
        conn.cursor().execute(f"""
            INSERT OR REPLACE INTO predictions
                (ticker, prediction_date, horizon, prob_up, prob_raw,
                 signal, confidence, model_version, created_at, is_watchlist, tier)
            VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})
        """, (
            ticker.upper(), str(prediction_date), horizon,
            float(prob_up),
            float(prob_raw) if prob_raw is not None else None,
            signal, confidence, model_version,
            ts_et(), 1 if is_watchlist else 0, tier,
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
    now = ts_et()
    records = [
        (
            r["ticker"].upper(),
            str(r["prediction_date"]),
            int(r["horizon"]),
            float(r["prob_up"]),
            (float(r["prob_raw"]) if r.get("prob_raw") is not None else None),
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
                (ticker, prediction_date, horizon, prob_up, prob_raw,
                 signal, confidence, model_version, created_at)
            VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p})
        """, records)
    return len(records)


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE: RECONCILE OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_price_fallback(ticker: str, start_date, end_date) -> "pd.DataFrame | None":
    """
    Fallback price fetcher: tries Polygon.io then Alpha Vantage when yfinance fails.
    Returns DataFrame with Open/Close columns indexed by date, or None if all fail.
    """
    import requests
    import pandas as pd

    from config.keys import MASSIVE_API_KEY as POLYGON_KEY
    AV_KEY      = "XX4T4AIT98GEF3K8"

    # --- Polygon.io ---
    try:
        url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
               f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50&apiKey={POLYGON_KEY}")
        r = _session.get(url, timeout=10)
        data = r.json()
        if data.get("resultsCount", 0) > 0:
            rows = []
            for bar in data["results"]:
                rows.append({
                    "Date":  pd.Timestamp(bar["t"], unit="ms", tz="UTC").tz_localize(None).normalize(),
                    "Open":  bar["o"],
                    "Close": bar["c"],
                })
            df = pd.DataFrame(rows).set_index("Date")
            print(f"  ✅ Polygon fallback succeeded for {ticker}")
            return df
    except Exception as e:
        print(f"  ⚠ Polygon failed for {ticker}: {e}")

    # --- Alpha Vantage ---
    try:
        url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
               f"&symbol={ticker}&outputsize=compact&apikey={AV_KEY}")
        r = _session.get(url, timeout=10)
        data = r.json()
        ts = data.get("Time Series (Daily)", {})
        if ts:
            rows = []
            for date_str, vals in ts.items():
                d = pd.Timestamp(date_str)
                if pd.Timestamp(str(start_date)) <= d <= pd.Timestamp(str(end_date)):
                    rows.append({
                        "Date":  d,
                        "Open":  float(vals["1. open"]),
                        "Close": float(vals["5. adjusted close"]),
                    })
            if rows:
                df = pd.DataFrame(rows).set_index("Date").sort_index()
                print(f"  ✅ Alpha Vantage fallback succeeded for {ticker}")
                return df
    except Exception as e:
        print(f"  ⚠ Alpha Vantage failed for {ticker}: {e}")

    return None

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
    def _add_trading_days(start_date, n_days):
        """Add n trading days (Mon-Fri) to a date, skipping weekends."""
        d = start_date
        added = 0
        while added < n_days:
            d += timedelta(days=1)
            if d.weekday() < 5:  # Mon-Fri
                added += 1
        return d

    pending["outcome_date"] = pending.apply(
        lambda r: _add_trading_days(r["prediction_date"], int(r["horizon"])),
        axis=1
    )
    due = pending[pending["outcome_date"] <= as_of]

    if due.empty:
        return 0

    if tickers:
        due = due[due["ticker"].isin([t.upper() for t in tickers])]

    # Fetch actual returns
    written = 0
    now = ts_et()

    for ticker, group in due.groupby("ticker"):
        min_date = group["prediction_date"].min() - timedelta(days=2)
        max_date  = group["outcome_date"].max() + timedelta(days=2)

        try:
            from features import massive_client as mc
            px = mc.download(
                ticker, start=str(min_date), end=str(max_date),
                auto_adjust=True, progress=False
            )
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            if px.empty:
                raise ValueError("Empty Massive response")
            close = px["Close"].squeeze()
        except Exception as e:
            print(f"  ⚠ Massive primary failed for {ticker}: {e} — trying yfinance/fallbacks")
            px = _fetch_price_fallback(ticker, min_date, max_date)
            if px is None:
                print(f"  ⚠ All sources failed for {ticker}")
                continue
            close = px["Close"].squeeze()

        with _get_conn() as conn:
            cur = conn.cursor()
            for _, row in group.iterrows():
                pred_date    = row["prediction_date"]
                outcome_date = row["outcome_date"]

                try:
                    # Score using close[prediction_date] -> close[outcome_date].
                    # FIXED May 4 2026: previous version (commit 8bcb44c, Mar 31)
                    # scored same-day open->close for every horizon, making the
                    # horizon dimension meaningless and producing direction-wrong
                    # labels in many cases (e.g., NVDA Apr 15 stored +1.18% when
                    # actual close-to-close return was -0.26%). All outcomes
                    # written between Mar 31 - May 4 need to be regenerated.
                    try:
                        close_series = px["Close"]
                        if isinstance(close_series, pd.DataFrame):
                            close_series = close_series.iloc[:, 0]
                        close_series = close_series.copy()
                        close_series.index = pd.to_datetime(close_series.index).tz_localize(None)

                        pred_ts = pd.Timestamp(pred_date)
                        out_ts  = pd.Timestamp(outcome_date)
                        price_at_pred    = float(close_series.asof(pred_ts))
                        price_at_outcome = float(close_series.asof(out_ts))

                        if price_at_pred == 0 or price_at_pred != price_at_pred:
                            continue
                        if price_at_outcome != price_at_outcome:
                            continue
                        actual_ret = (price_at_outcome - price_at_pred) / price_at_pred
                        if actual_ret != actual_ret:
                            continue
                        actual_up = int(actual_ret > 0)
                    except Exception:
                        continue
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
            WHERE p.is_watchlist = 0
        """, conn)

    if joined.empty:
        return pd.DataFrame()

    joined = joined[joined["prediction_date"] >= cutoff]

    rows = []
    now  = ts_et()

    for (ticker, horizon), grp in joined.groupby(["ticker", "horizon"]):
        if len(grp) < 1:   # not enough data for meaningful metrics
            continue

        y_true = grp["actual_up"].astype(int)
        y_prob = grp["prob_up"].astype(float)
        y_pred = (y_prob >= 0.55).astype(int)

        # Score all predictions directionally using prob_up > 0.5
        acc_true = grp["actual_up"].astype(int)
        acc_pred = (grp["prob_up"] > 0.5).astype(int)
        accuracy = float((acc_true == acc_pred).mean())

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
            "accuracy":      round(accuracy, 4) if accuracy is not None else None,
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
    # Score directionally: prob_up > 0.5 means predicting UP
    df["correct"] = ((df["prob_up"] > 0.5) == (df["actual_up"] == 1)).astype(int)

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


def reconcile_intraday_outcomes():
    """
    Match intraday predictions with actual prices 1hr/2hr/4hr later.
    Run this periodically during market hours.
    """
    import sqlite3, yfinance as yf
    from datetime import datetime, timedelta
    from utils.timezone import now_et, ET
    now = now_et()
    db  = sqlite3.connect("accuracy.db", timeout=30)

    # Get unreconciled predictions
    rows = db.execute("""
        SELECT p.ticker, p.prediction_ts, p.horizon_hr, p.price_at_pred, p.signal
        FROM intraday_predictions p
        LEFT JOIN intraday_outcomes o
            ON p.ticker=o.ticker AND p.prediction_ts=o.prediction_ts
               AND p.horizon_hr=o.horizon_hr
        WHERE o.id IS NULL
    """).fetchall()

    reconciled = 0
    for ticker, pred_ts, horizon_hr, price_at_pred, signal in rows:
        try:
            pred_dt  = ET.localize(datetime.fromisoformat(pred_ts))
            outcome_dt = pred_dt + timedelta(hours=horizon_hr)
            if outcome_dt > now:
                continue  # not yet

            # Fetch actual price at outcome time
            import pandas as pd
            from features import massive_client as mc
            hist = mc.download(ticker,
                               start=outcome_dt.strftime("%Y-%m-%d"),
                               end=(outcome_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                               interval="1m", auto_adjust=True, progress=False)
            if hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist.index = hist.index.tz_convert(ET)
            close = hist["Close"].squeeze()
            if outcome_dt.tzinfo is None:
                outcome_dt = ET.localize(outcome_dt)
            if outcome_dt.hour >= 16:
                # After market close — use last available price
                price_at_outcome = float(close.iloc[-1])
            else:
                idx = close.index.get_indexer([outcome_dt], method="nearest")[0]
                if abs((close.index[idx] - outcome_dt).total_seconds()) > 3600:
                    continue
                price_at_outcome = float(close.iloc[idx])
            actual_return    = (price_at_outcome - price_at_pred) / price_at_pred
            actual_up        = 1 if actual_return > 0 else 0
            ts_now           = now.strftime("%Y-%m-%dT%H:%M:%S")

            db.execute("""
                INSERT OR IGNORE INTO intraday_outcomes
                (ticker, prediction_ts, horizon_hr, outcome_ts,
                 price_at_pred, price_at_outcome, actual_return, actual_up, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (ticker, pred_ts, horizon_hr,
                  outcome_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                  price_at_pred, price_at_outcome,
                  actual_return, actual_up, ts_now))
            reconciled += 1
        except Exception:
            pass

    db.commit()

    # Update accuracy cache
    tickers = [r[0] for r in db.execute("SELECT DISTINCT ticker FROM intraday_outcomes").fetchall()]
    for ticker in tickers:
        for hr in [1, 2, 4]:
            rows2 = db.execute("""
                SELECT p.signal, o.actual_up
                FROM intraday_predictions p
                JOIN intraday_outcomes o
                    ON p.ticker=o.ticker AND p.prediction_ts=o.prediction_ts
                       AND p.horizon_hr=o.horizon_hr
                WHERE p.ticker=? AND p.horizon_hr=?
            """, (ticker, hr)).fetchall()
            if len(rows2) < 3:
                continue
            correct = sum(1 for sig, up in rows2
                         if (sig=="UP" and up==1) or (sig=="DOWN" and up==0) or (sig=="NEUTRAL"))
            acc = correct / len(rows2)
            db.execute("""
                INSERT OR REPLACE INTO intraday_accuracy_cache
                (ticker, horizon_hr, accuracy, n_predictions, computed_at)
                VALUES (?,?,?,?,?)
            """, (ticker, hr, acc, len(rows2), datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%S")))

    db.commit()
    db.close()
    print(f"Reconciled {reconciled} intraday outcomes")
    return reconciled


def get_intraday_accuracy_summary() -> list[dict]:
    """Return accuracy summary for display in dashboard."""
    import sqlite3
    db  = sqlite3.connect("accuracy.db", timeout=30)
    rows = db.execute("""
        SELECT ticker, horizon_hr, accuracy, n_predictions, computed_at
        FROM intraday_accuracy_cache
        ORDER BY ticker, horizon_hr
    """).fetchall()
    db.close()
    return [{"ticker": r[0], "horizon_hr": r[1], "accuracy": r[2],
             "n_predictions": r[3], "computed_at": r[4]} for r in rows]


def get_eod_accuracy_summary() -> list[dict]:
    """Score all predictions: prob_up>0.5 = predicting UP."""
    import sqlite3
    db = sqlite3.connect("accuracy.db", timeout=30)
    rows = db.execute("""
        SELECT p.ticker,
               COUNT(*) as n,
               ROUND(AVG(CASE WHEN (p.prob_up > 0.5 AND o.actual_up=1)
                               OR (p.prob_up <= 0.5 AND o.actual_up=0) THEN 1.0
                          ELSE 0.0 END), 3) as accuracy,
               ROUND(AVG(o.actual_return), 4) as avg_return
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
            AND p.prediction_date=o.prediction_date
            AND p.horizon=o.horizon
        WHERE p.horizon=1
          AND p.is_watchlist=0
        GROUP BY p.ticker
        ORDER BY accuracy DESC NULLS LAST
    """).fetchall()
    db.close()
    return [{"ticker": r[0], "n": r[1], "accuracy": r[2], "avg_return": r[3]} for r in rows]


def get_accuracy_by_tier(horizon: int = 1, days: int = 90) -> list[dict]:
    """
    Compute accuracy breakdown by tier (core, secondary, tactical, lotto).
    Excludes watchlist tickers. Useful for evaluating whether lotto suppression
    rules are working.
    """
    import sqlite3
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    db = sqlite3.connect("accuracy.db", timeout=30)
    rows = db.execute("""
        SELECT p.tier,
               p.signal,
               COUNT(*) as n,
               SUM(CASE WHEN o.actual_return > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN o.actual_return > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as hit_pct,
               ROUND(AVG(o.actual_return) * 100, 2) as avg_ret_pct
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
            AND p.prediction_date=o.prediction_date
            AND p.horizon=o.horizon
        WHERE p.horizon=?
          AND p.is_watchlist=0
          AND p.prediction_date >= ?
        GROUP BY p.tier, p.signal
        ORDER BY p.tier, p.signal
    """, (horizon, cutoff)).fetchall()
    db.close()
    return [
        {"tier": r[0], "signal": r[1], "n": r[2], "wins": r[3],
         "hit_pct": r[4], "avg_ret_pct": r[5]}
        for r in rows
    ]


def get_spy_relative_accuracy() -> list[dict]:
    """
    Compare HOLD ticker returns vs SPY return each day.
    Shows whether the model identifies outperformers even when signal=HOLD.
    """
    import sqlite3, yfinance as yf
    import pandas as pd
    from utils.timezone import today_et

    db = sqlite3.connect("accuracy.db", timeout=30)
    rows = db.execute("""
        SELECT p.ticker, p.prediction_date, p.signal, p.prob_up,
               o.actual_return, o.actual_up
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker 
            AND p.prediction_date=o.prediction_date
            AND p.horizon=o.horizon
        WHERE p.horizon=1
          AND p.is_watchlist=0
        ORDER BY p.prediction_date
    """).fetchall()
    db.close()

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["ticker","date","signal","prob_up","actual_return","actual_up"])
    df["date"] = pd.to_datetime(df["date"])

    # Fetch SPY returns
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = (df["date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    min_date_ext = (df["date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        from features import massive_client as mc
        spy = mc.download("SPY", start=min_date_ext, end=max_date,
                          auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy["spy_ret"] = spy["Close"].pct_change()
        spy = spy[["spy_ret"]].reset_index()
        spy.columns = ["date", "spy_ret"]
        spy["date"] = pd.to_datetime(spy["date"])
    except Exception:
        return []

    df = df.merge(spy, on="date", how="left")
    df = df.dropna(subset=["spy_ret"])  # remove non-trading days
    df["vs_spy"] = df["actual_return"] - df["spy_ret"]

    # Summary by date
    results = []
    for date, grp in df.groupby("date"):
        spy_ret = grp["spy_ret"].iloc[0]
        avg_ret = grp["actual_return"].mean()
        avg_vs_spy = grp["vs_spy"].mean()
        pct_beat = (grp["vs_spy"] > 0).mean()
        buys = grp[grp["signal"] == "BUY"]
        buy_acc = (buys["actual_up"] == 1).mean() if len(buys) > 0 else None
        results.append({
            "date":        date.strftime("%Y-%m-%d"),
            "spy_ret":     round(spy_ret, 4),
            "avg_ret":     round(avg_ret, 4),
            "avg_vs_spy":  round(avg_vs_spy, 4),
            "pct_beat_spy": round(pct_beat, 3),
            "n_buy":       len(buys),
            "buy_acc":     round(buy_acc, 3) if buy_acc is not None else None,
        })
    return results
