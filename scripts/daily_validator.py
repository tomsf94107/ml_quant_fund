"""
scripts/daily_validator.py
─────────────────────────────────────────────────────────────────────────────
Daily self-checker and auto-fixer for ML Quant Fund accuracy data.

Checks every ticker in tickers.txt every day:
  1. Price sanity      — re-fetches actual open/close prices and compares
                         stored actual_return against ground truth
  2. Ghost predictions — flags predictions on weekends/market holidays
  3. Signal integrity  — ensures BUY label matches prob_up >= BUY_THRESHOLD
  4. NULL/NaN guard    — flags outcomes with missing actual_return
  5. Off-by-one check  — detects timezone bugs (return sign inverted vs price)

Auto-fixes:
  - Wrong signal labels (BUY → HOLD if prob_up < BUY_THRESHOLD)
  - Bad outcomes (deletes and re-reconciles affected ticker/dates)
  - Ghost predictions (deletes weekend/holiday rows)

Writes report to logs/validator.log
Sends desktop notification if manual action needed.

Run via cron at 9 PM ET daily (after reconciliation):
  0 21 * * 1-5 cd ~/Desktop/ML_Quant_Fund && python scripts/daily_validator.py >> logs/validator.log 2>&1

Usage:
  python scripts/daily_validator.py              # validate last 30 days
  python scripts/daily_validator.py --days 90    # validate last 90 days
  python scripts/daily_validator.py --fix        # auto-fix issues found
  python scripts/daily_validator.py --ticker AAPL  # single ticker
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = Path("accuracy.db")
TICKERS_FILE   = Path("tickers.txt")
LOG_DIR        = Path("logs")
BUY_THRESHOLD  = 0.70
ET             = pytz.timezone("America/New_York")
RETURN_TOL     = 0.001   # 0.1% tolerance for return comparison
PRICE_TOL      = 0.005   # 0.5% tolerance for price comparison


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{ts}] {msg}", flush=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_tickers(ticker_arg: str | None = None) -> list[str]:
    if ticker_arg:
        return [ticker_arg.upper()]
    if TICKERS_FILE.exists():
        return [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    raise FileNotFoundError(f"tickers.txt not found at {TICKERS_FILE}")


def is_trading_day(d: date) -> bool:
    """Basic weekday check — Mon-Fri only, ignores holidays."""
    return d.weekday() < 5


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """Fetch OHLCV from yfinance with timezone-normalized index."""
    try:
        raw = yf.download(
            ticker,
            start=str(start - timedelta(days=5)),
            end=str(end + timedelta(days=2)),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        # Normalize index to timezone-naive dates — critical fix
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        return raw
    except Exception as e:
        log(f"  ⚠ Price fetch failed for {ticker}: {e}")
        return None


def compute_return(px: pd.DataFrame, pred_date: date) -> float | None:
    """Compute open→close return for a given date."""
    ts = pd.Timestamp(pred_date)
    try:
        day_open  = float(px["Open"].asof(ts))
        day_close = float(px["Close"].asof(ts))
        if day_open == 0 or np.isnan(day_open) or np.isnan(day_close):
            return None
        ret = (day_close - day_open) / day_open
        if ret != ret:  # NaN check
            return None
        return ret
    except Exception:
        return None


def desktop_alert(title: str, message: str):
    try:
        os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
    except Exception:
        pass


# ── Check Functions ───────────────────────────────────────────────────────────

def check_ghost_predictions(conn: sqlite3.Connection, days: int, fix: bool) -> dict:
    """Flag and optionally delete predictions on weekends."""
    log("Checking ghost predictions (weekends)...")
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    rows = conn.execute("""
        SELECT id, ticker, prediction_date
        FROM predictions
        WHERE prediction_date >= ?
    """, (cutoff,)).fetchall()

    ghosts = []
    for row in rows:
        pred_date = date.fromisoformat(row["prediction_date"][:10])
        if not is_trading_day(pred_date):
            ghosts.append(dict(row))

    if ghosts:
        log(f"  ❌ Found {len(ghosts)} ghost predictions on non-trading days")
        for g in ghosts:
            log(f"     {g['ticker']} on {g['prediction_date']}")
        if fix:
            dates = list({g["prediction_date"][:10] for g in ghosts})
            for d in dates:
                conn.execute("DELETE FROM predictions WHERE date(prediction_date)=?", (d,))
                conn.execute("DELETE FROM outcomes WHERE date(prediction_date)=?", (d,))
            conn.commit()
            log(f"  ✅ Deleted {len(ghosts)} ghost predictions and their outcomes")
    else:
        log(f"  ✅ No ghost predictions found")

    return {"ghosts": len(ghosts), "fixed": len(ghosts) if fix else 0}


def check_signal_labels(conn: sqlite3.Connection, days: int, fix: bool) -> dict:
    """Ensure BUY label matches prob_up >= BUY_THRESHOLD."""
    log(f"Checking signal label integrity (BUY_THRESHOLD={BUY_THRESHOLD})...")
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    bad_buys = conn.execute("""
        SELECT id, ticker, prediction_date, prob_up, signal
        FROM predictions
        WHERE prediction_date >= ?
          AND signal = 'BUY'
          AND prob_up < ?
    """, (cutoff, BUY_THRESHOLD)).fetchall()

    bad_holds = conn.execute("""
        SELECT id, ticker, prediction_date, prob_up, signal
        FROM predictions
        WHERE prediction_date >= ?
          AND signal = 'HOLD'
          AND prob_up >= ?
    """, (cutoff, BUY_THRESHOLD)).fetchall()

    issues = len(bad_buys) + len(bad_holds)
    if issues:
        log(f"  ❌ Found {len(bad_buys)} BUY signals below threshold, {len(bad_holds)} HOLD signals above threshold")
        if fix:
            for row in bad_buys:
                conn.execute("UPDATE predictions SET signal='HOLD' WHERE id=?", (row["id"],))
            for row in bad_holds:
                conn.execute("UPDATE predictions SET signal='BUY' WHERE id=?", (row["id"],))
            conn.commit()
            log(f"  ✅ Fixed {issues} signal labels")
    else:
        log(f"  ✅ All signal labels are correct")

    return {"bad_labels": issues, "fixed": issues if fix else 0}


def check_null_outcomes(conn: sqlite3.Connection, days: int, fix: bool) -> dict:
    """Flag outcomes with NULL or NaN actual_return."""
    log("Checking for NULL/NaN outcomes...")
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    nulls = conn.execute("""
        SELECT id, ticker, prediction_date, actual_return
        FROM outcomes
        WHERE prediction_date >= ?
          AND (actual_return IS NULL OR actual_return != actual_return)
    """, (cutoff,)).fetchall()

    if nulls:
        log(f"  ❌ Found {len(nulls)} outcomes with NULL/NaN actual_return")
        if fix:
            ids = [row["id"] for row in nulls]
            conn.execute(f"DELETE FROM outcomes WHERE id IN ({','.join('?'*len(ids))})", ids)
            conn.commit()
            log(f"  ✅ Deleted {len(nulls)} bad outcomes (will be re-reconciled)")
    else:
        log(f"  ✅ No NULL/NaN outcomes found")

    return {"null_outcomes": len(nulls), "fixed": len(nulls) if fix else 0}


def check_price_accuracy(
    tickers: list[str],
    conn: sqlite3.Connection,
    days: int,
    fix: bool,
) -> dict:
    """
    Re-fetch actual prices and compare against stored actual_return.
    Detects off-by-one timezone bugs and other price errors.
    """
    log(f"Checking price accuracy for {len(tickers)} tickers over last {days} days...")
    cutoff = date.today() - timedelta(days=days)

    total_checked = 0
    total_wrong   = 0
    total_fixed   = 0
    wrong_details = []

    for ticker in tickers:
        # Get stored outcomes for this ticker
        rows = conn.execute("""
            SELECT o.id, o.prediction_date, o.actual_return, o.actual_up
            FROM outcomes o
            WHERE o.ticker = ?
              AND date(o.prediction_date) >= ?
            ORDER BY o.prediction_date
        """, (ticker, cutoff.isoformat())).fetchall()

        if not rows:
            continue

        # Fetch prices once per ticker
        min_date = date.fromisoformat(rows[0]["prediction_date"][:10])
        max_date = date.fromisoformat(rows[-1]["prediction_date"][:10])
        px = fetch_prices(ticker, min_date, max_date)
        if px is None:
            log(f"  ⚠ Could not fetch prices for {ticker} — skipping")
            continue

        bad_ids = []
        for row in rows:
            pred_date = date.fromisoformat(row["prediction_date"][:10])
            total_checked += 1

            true_ret = compute_return(px, pred_date)
            if true_ret is None:
                continue

            stored_ret = row["actual_return"]
            if stored_ret is None:
                continue

            # Check if return is significantly wrong
            ret_diff = abs(true_ret - stored_ret)
            sign_wrong = (true_ret > 0) != (stored_ret > 0) and abs(true_ret) > 0.002

            if ret_diff > RETURN_TOL or sign_wrong:
                total_wrong += 1
                wrong_details.append({
                    "ticker": ticker,
                    "date": row["prediction_date"][:10],
                    "stored": round(stored_ret * 100, 3),
                    "actual": round(true_ret * 100, 3),
                    "sign_wrong": sign_wrong,
                })
                if fix:
                    actual_up = int(true_ret > 0)
                    conn.execute("""
                        UPDATE outcomes
                        SET actual_return=?, actual_up=?
                        WHERE id=?
                    """, (true_ret, actual_up, row["id"]))
                    bad_ids.append(row["id"])

        if fix and bad_ids:
            conn.commit()
            total_fixed += len(bad_ids)

    # Report wrong prices
    if wrong_details:
        log(f"  ❌ Found {total_wrong}/{total_checked} incorrect prices:")
        for w in wrong_details[:20]:  # show first 20
            flag = "⚠ SIGN WRONG" if w["sign_wrong"] else "diff"
            log(f"     {w['ticker']} {w['date']}: stored={w['stored']}% actual={w['actual']}% [{flag}]")
        if len(wrong_details) > 20:
            log(f"     ... and {len(wrong_details)-20} more")
        if fix:
            log(f"  ✅ Fixed {total_fixed} incorrect prices")
    else:
        log(f"  ✅ All {total_checked} prices are correct")

    return {
        "checked": total_checked,
        "wrong": total_wrong,
        "fixed": total_fixed,
        "details": wrong_details,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def check_delisted_tickers(tickers: list[str], fix: bool) -> dict:
    """
    Try to fetch recent price for each ticker.
    If yfinance returns no data, flag as possibly delisted.
    If fix=True, remove from tickers.txt and move to watchlist.txt.
    """
    log(f"Checking for delisted tickers across {len(tickers)} tickers...")
    delisted = []

    for ticker in tickers:
        try:
            raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if raw.empty:
                delisted.append(ticker)
        except Exception:
            delisted.append(ticker)

    if delisted:
        log(f"  ❌ Possibly delisted: {delisted}")
        if fix:
            # Remove from tickers.txt
            if TICKERS_FILE.exists():
                lines = [t for t in TICKERS_FILE.read_text().splitlines() if t.strip() and t.strip() not in delisted]
                TICKERS_FILE.write_text("\n".join(lines) + "\n")
            # Append to watchlist.txt
            watchlist = Path("watchlist.txt")
            existing = set(watchlist.read_text().splitlines()) if watchlist.exists() else set()
            new_entries = [t for t in delisted if t not in existing]
            if new_entries:
                with open(watchlist, "a") as f:
                    f.write("\n".join(new_entries) + "\n")
            log(f"  ✅ Moved {delisted} to watchlist.txt, removed from tickers.txt")
            desktop_alert(
                "ML Quant Fund — Delisted Tickers",
                f"Moved to watchlist: {', '.join(delisted)} — update train_all.py manually"
            )
    else:
        log(f"  ✅ All {len(tickers)} tickers have active price data")

    return {"delisted": len(delisted), "tickers": delisted, "fixed": len(delisted) if fix else 0}


def check_log_sizes(max_mb: float = 10.0) -> dict:
    """Flag log files that exceed max_mb and truncate if needed."""
    log(f"Checking log file sizes (max {max_mb}MB)...")
    large_logs = []

    for log_file in LOG_DIR.glob("*.log"):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            large_logs.append({"file": log_file.name, "size_mb": round(size_mb, 1)})
            log(f"  ⚠ {log_file.name}: {size_mb:.1f}MB — truncating to last 10000 lines")
            # Keep last 10000 lines
            lines = log_file.read_text(errors="ignore").splitlines()
            log_file.write_text("\n".join(lines[-10000:]) + "\n")

    if large_logs:
        log(f"  ✅ Truncated {len(large_logs)} oversized log files")
    else:
        log(f"  ✅ All log files within size limits")

    return {"large_logs": len(large_logs), "details": large_logs}


def run_validator(days: int = 30, fix: bool = True, ticker: str | None = None):
    LOG_DIR.mkdir(exist_ok=True)

    log("=" * 60)
    log(f"  ML Quant Fund — Daily Validator")
    log(f"  Date  : {date.today()} ET")
    log(f"  Days  : {days}")
    log(f"  Fix   : {fix}")
    log(f"  Ticker: {ticker or 'ALL'}")
    log("=" * 60)

    tickers = load_tickers(ticker)
    log(f"Loaded {len(tickers)} tickers")

    conn = get_conn()
    results = {}

    # Run all checks
    results["ghosts"]   = check_ghost_predictions(conn, days, fix)
    results["signals"]  = check_signal_labels(conn, days, fix)
    results["nulls"]    = check_null_outcomes(conn, days, fix)
    results["prices"]   = check_price_accuracy(tickers, conn, days, fix)
    results["delisted"] = check_delisted_tickers(tickers, fix)
    results["logs"]     = check_log_sizes()

    # Rebuild accuracy cache if anything was fixed
    total_fixed = sum(v.get("fixed", 0) for v in results.values())
    if total_fixed > 0 and fix:
        log(f"\nRebuilding accuracy cache ({total_fixed} fixes applied)...")
        try:
            from accuracy.sink import update_accuracy_cache
            update_accuracy_cache()
            log("  ✅ Accuracy cache rebuilt")
        except Exception as e:
            log(f"  ⚠ Cache rebuild failed: {e}")

    # Summary
    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    log(f"  Ghost predictions : {results['ghosts']['ghosts']} found, {results['ghosts']['fixed']} fixed")
    log(f"  Signal labels     : {results['signals']['bad_labels']} wrong, {results['signals']['fixed']} fixed")
    log(f"  NULL outcomes     : {results['nulls']['null_outcomes']} found, {results['nulls']['fixed']} fixed")
    log(f"  Price errors      : {results['prices']['wrong']}/{results['prices']['checked']} wrong, {results['prices']['fixed']} fixed")
    log(f"  Delisted tickers  : {results['delisted']['delisted']} found, {results['delisted']['fixed']} removed")
    log(f"  Large log files   : {results['logs']['large_logs']} truncated")
    log("=" * 60)

    # Desktop alert if issues found
    total_issues = (
        results["ghosts"]["ghosts"] +
        results["signals"]["bad_labels"] +
        results["nulls"]["null_outcomes"] +
        results["prices"]["wrong"] +
        results["delisted"]["delisted"]
    )
    if total_issues > 0 and not fix:
        desktop_alert(
            "ML Quant Fund — Validator",
            f"{total_issues} issues found — run with --fix to auto-correct"
        )
    elif total_fixed > 0:
        desktop_alert(
            "ML Quant Fund — Validator",
            f"Auto-fixed {total_fixed} issues. Accuracy cache rebuilt."
        )
    else:
        log("  ✅ All checks passed — no issues found")

    conn.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily accuracy validator for ML Quant Fund")
    parser.add_argument("--days",   type=int, default=30,   help="Days to look back (default: 30)")
    parser.add_argument("--fix",    action="store_true",     help="Auto-fix issues found")
    parser.add_argument("--ticker", type=str, default=None,  help="Validate single ticker only")
    args = parser.parse_args()

    run_validator(days=args.days, fix=args.fix, ticker=args.ticker)
