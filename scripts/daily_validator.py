"""
scripts/daily_validator.py
─────────────────────────────────────────────────────────────────────────────
Daily self-checker and auto-fixer for ML Quant Fund accuracy data.

Checks every ticker in tickers.txt every day:
  1. Price sanity      — re-fetches actual open/close prices and compares
                         stored actual_return against ground truth
  2. Ghost predictions — flags predictions on weekends/market holidays
  3. Signal integrity  — flags BUY/HOLD inconsistencies (NON-DESTRUCTIVE
                         since May 8 2026 — see schema v2 fix)
  4. NULL/NaN guard    — flags outcomes with missing actual_return
  5. Off-by-one check  — detects timezone bugs (return sign inverted vs price)

Auto-fixes (DESTRUCTIVE):
  - Bad outcomes (deletes and re-reconciles affected ticker/dates)
  - Ghost predictions (deletes weekend/holiday rows)

Non-destructive (warning only, May 8 2026):
  - Signal labels — logs warnings but does NOT modify signal column.
    Validator cannot reliably reconstruct generator's BUY decision (which
    includes hysteresis from yesterday's signal). Logs mismatches for
    manual review only.

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
# yfinance removed for ticker prices (May 12 2026 — Issue B fix per memory #6).
# Architecture rule: Massive for ticker OHLCV, yfinance for indexes ONLY.
# Pre-fix: validator overwrote sink's Massive-sourced prices with yfinance
# values, "fixing" ~390/day correct outcomes into wrong ones.
from features import massive_client as _mc
from features.yf_resilient import safe_yf_download  # KEEP for any future index check

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = Path("accuracy.db")
TICKERS_FILE   = Path("tickers.txt")
LOG_DIR        = Path("logs")
# FIXED May 8 2026: was 0.70 — destroyed 70 legitimate BUYs (May 7 cron).
# Generator uses DEFAULT_CONFIDENCE_THRESHOLD=0.55. DB column "prob_up"
# actually stores prob_EFF (post-multiplier) per sink.py docstring.
# Now reads from generator for single-source-of-truth.
try:
    from signals.generator import DEFAULT_CONFIDENCE_THRESHOLD as BUY_THRESHOLD
    from signals.generator import HYSTERESIS_EXIT, _lookup_yesterday_signal
    _HYSTERESIS_AVAILABLE = True
except ImportError:
    BUY_THRESHOLD = 0.55  # fallback if import fails
    HYSTERESIS_EXIT = {1: 0.65, 3: 0.50, 5: 0.50}  # match generator constants
    _lookup_yesterday_signal = None
    _HYSTERESIS_AVAILABLE = False
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
    """Fetch OHLCV from Massive (May 12 2026 — Issue B fix per memory #6).

    Architecture rule: Massive for ticker prices, yfinance for indexes only.
    Pre-fix: validator used safe_yf_download which routes to raw yfinance.
    yfinance and sink's Massive source naturally diverge (different
    adjustment policies, snapshot times). Validator's auto-fix was
    overwriting sink's correct Massive prices with diverged yfinance values.
    """
    try:
        raw = _mc.download(
            ticker,
            start=str(start - timedelta(days=5)),
            end=str(end + timedelta(days=2)),
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        # Normalize index to timezone-naive dates — critical fix
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        return raw
    except Exception as e:
        log(f"  ⚠ Price fetch failed for {ticker}: {e}")
        return None


def _add_trading_days(start_date: date, n_days: int) -> date:
    """Add n trading days (Mon-Fri) to a date. Mirrors sink.reconcile_outcomes."""
    d = start_date
    added = 0
    while added < n_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def compute_return(px: pd.DataFrame, pred_date: date, horizon: int) -> float | None:
    """Compute close[T] → close[T + horizon trading days] return.

    Fixed May 12 2026 to match sink.reconcile_outcomes (memory #19, May 4).
    Pre-fix used same-day open→close, which disagreed with stored values
    (which are N-day forward close-to-close returns). Validator's auto-fix
    was overwriting correct stored values with wrong same-day returns.
    """
    try:
        outcome_date = _add_trading_days(pred_date, horizon)
        pred_ts    = pd.Timestamp(pred_date)
        outcome_ts = pd.Timestamp(outcome_date)
        close_pred    = float(px["Close"].asof(pred_ts))
        close_outcome = float(px["Close"].asof(outcome_ts))
        if close_pred == 0 or np.isnan(close_pred) or np.isnan(close_outcome):
            return None
        ret = (close_outcome - close_pred) / close_pred
        if ret != ret:
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
    """
    Flag signal label inconsistencies. NON-DESTRUCTIVE (May 8 2026).

    Background: Generator's BUY decision is:
        BUY if (prob_eff > threshold AND NOT gate_block) OR
               (yesterday=BUY AND prob_eff > exit_threshold)   # hysteresis

    Threshold subtlety (May 8 2026):
      - Generator default: DEFAULT_CONFIDENCE_THRESHOLD = 0.55
      - Production override in daily_runner: BUY_THRESHOLD = 0.70
      - Confidence cap on h=3/h=5: prob_eff capped at 0.65
      - This validator uses 0.55 (the library default) for the lower
        bound check. False positives are possible — for example,
        h=3/h=5 BUYs that survived via hysteresis with prob_eff=0.65
        will pass validation but the 0.70 production threshold check
        wouldn't match. This is acceptable since validator is
        non-destructive (warnings only).

    DB column 'prob_up' stores prob_eff (post-multiplier).
    DB column 'gate_block' stores 0/1 for event-risk block (added May 8 2026).
    Hysteresis state is NOT stored — cannot be reconstructed.

    What this check does:
      - For ROWS WITH gate_block populated (post-migration):
          Flag BUY rows where (prob_up < threshold OR gate_block == 1)
          → these are unambiguously inconsistent with generator's logic
      - For HOLDs with prob_up >= threshold: SKIP (could be hysteresis exit)
      - For ROWS WITH gate_block NULL (pre-migration): SKIP entirely
        (cannot reconstruct without gate_block info)
      - NEVER UPDATE the signal column (--fix flag does NOT modify rows)

    Why non-destructive: prior version (commit a1415e3) destroyed legitimate
    BUYs (70 on 2026-05-07, 212 cumulative). Validator cannot reliably
    reconstruct generator's full decision logic.
    """
    log(f"Checking signal label integrity (threshold={BUY_THRESHOLD}, "
        f"NON-DESTRUCTIVE since May 8 2026)...")
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    # Schema-aware: only check rows where gate_block is populated.
    # Pre-migration rows (gate_block IS NULL) are skipped.
    #
    # May 12 2026: hysteresis-aware filtering.
    # Generator uses asymmetric ENTRY/EXIT thresholds per commit 4376a13.
    # When yesterday=BUY for (ticker, h), today's prob_up only needs to
    # exceed HYSTERESIS_EXIT (0.65/0.50/0.50 for h=1/3/5) to stay BUY.
    # Without this, validator falsely flags every legitimate
    # hysteresis-stay as 'inconsistent' (was 122 daily false positives).
    candidate_buys = conn.execute("""
        SELECT id, ticker, prediction_date, horizon, prob_up, signal,
               gate_block
        FROM predictions
        WHERE prediction_date >= ?
          AND signal = 'BUY'
          AND gate_block IS NOT NULL
          AND (prob_up < ? OR gate_block = 1)
    """, (cutoff, BUY_THRESHOLD)).fetchall()

    # Second-pass filter: drop rows that are valid generator decisions
    #
    # Two categories of "false positives" from the OR clause above:
    #   1. Hysteresis stays: yesterday=BUY + today's prob_up >= EXIT thresh
    #   2. Confidence-capped BUYs: prob_up=0.65 exact (capped from higher
    #      uncapped value per generator L632-634 INVERSION_HORIZONS).
    #      Pre-cap value may have exceeded ENTRY threshold legitimately.
    #
    # We need prob_eff_uncapped to assess category 2. Re-fetch with that col.
    candidate_ids = [r["id"] for r in candidate_buys]
    if candidate_ids:
        placeholders = ",".join("?" * len(candidate_ids))
        rows_with_uncapped = conn.execute(f"""
            SELECT id, ticker, prediction_date, horizon, prob_up,
                   prob_eff_uncapped, signal, gate_block
            FROM predictions
            WHERE id IN ({placeholders})
        """, candidate_ids).fetchall()
    else:
        rows_with_uncapped = []

    inconsistent_buys = []
    hysteresis_stays = 0
    capped_legit = 0
    for row in rows_with_uncapped:
        # gate_block=1 is always inconsistent (event-risk override)
        if row["gate_block"] == 1:
            inconsistent_buys.append(row)
            continue

        horizon_int = int(row["horizon"])
        exit_thresh  = HYSTERESIS_EXIT.get(horizon_int, 0.50)
        entry_thresh = 0.80 if horizon_int == 1 else 0.60  # HYSTERESIS_ENTRY

        # Category 1: hysteresis stay (yesterday=BUY + prob_up >= EXIT)
        prior = None
        if _HYSTERESIS_AVAILABLE:
            try:
                prior = _lookup_yesterday_signal(
                    ticker=row["ticker"],
                    horizon=horizon_int,
                    as_of=row["prediction_date"][:10],
                )
            except Exception:
                pass
        if prior == "BUY" and row["prob_up"] >= exit_thresh:
            hysteresis_stays += 1
            continue

        # Category 2: confidence-capped BUY (prob_up == cap, uncapped >= ENTRY)
        # Generator caps at 0.65 for INVERSION_HORIZONS (h=3, h=5) when
        # uncapped > 0.65. Pre-cap value drives the decision.
        uncapped = row["prob_eff_uncapped"]
        if uncapped is not None:
            try:
                uncapped_f = float(uncapped)
                if uncapped_f >= entry_thresh and row["prob_up"] < BUY_THRESHOLD:
                    capped_legit += 1
                    continue  # legitimate fresh BUY, was capped down post-decision
            except (ValueError, TypeError):
                pass

        # Remaining: genuine inconsistency
        inconsistent_buys.append(row)

    skipped_old = conn.execute("""
        SELECT COUNT(*) AS n FROM predictions
        WHERE prediction_date >= ?
          AND gate_block IS NULL
    """, (cutoff,)).fetchone()
    skipped_old_count = skipped_old["n"] if skipped_old else 0

    if hysteresis_stays > 0:
        log(f"  ℹ Filtered out {hysteresis_stays} valid hysteresis-stay "
            f"BUYs (yesterday=BUY + prob_up>=HYSTERESIS_EXIT)")
    if capped_legit > 0:
        log(f"  ℹ Filtered out {capped_legit} confidence-capped BUYs "
            f"(prob_eff_uncapped>=ENTRY but stored prob_up<{BUY_THRESHOLD})")

    if inconsistent_buys:
        log(f"  ⚠ Found {len(inconsistent_buys)} BUY rows inconsistent with "
            f"generator logic (prob_up < {BUY_THRESHOLD} OR gate_block=1, "
            f"after hysteresis filter)")
        log(f"    First 5 inconsistencies:")
        for row in inconsistent_buys[:5]:
            log(f"      {row['ticker']:<8} h={row['horizon']} "
                f"{row['prediction_date']} prob_up={row['prob_up']:.4f} "
                f"gate_block={row['gate_block']}")
        log(f"  ℹ NON-DESTRUCTIVE: signal column NOT modified.")
        log(f"  ℹ If pattern persists, investigate generator → sink data path.")
    else:
        log(f"  ✅ All schema-v2 signal labels consistent with generator logic")

    if skipped_old_count > 0:
        log(f"  ℹ Skipped {skipped_old_count} pre-migration rows "
            f"(gate_block IS NULL — added May 8 2026)")

    # Always returns 0 for "fixed" since we never modify
    return {"bad_labels": len(inconsistent_buys), "fixed": 0,
            "skipped_old_schema": skipped_old_count,
            "hysteresis_stays_filtered": hysteresis_stays,
            "capped_legit_filtered": capped_legit}


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
        # Get stored outcomes for this ticker — include horizon for proper return computation
        # Fixed May 12 2026: was comparing same-day returns to N-day forward returns
        rows = conn.execute("""
            SELECT o.id, o.prediction_date, o.horizon, o.actual_return, o.actual_up
            FROM outcomes o
            WHERE o.ticker = ?
              AND date(o.prediction_date) >= ?
            ORDER BY o.prediction_date
        """, (ticker, cutoff.isoformat())).fetchall()

        if not rows:
            continue

        # Fetch prices once per ticker — extend max_date by max horizon (5d)
        # so we have close[T+horizon] for all outcomes. Trading days, with buffer.
        min_date = date.fromisoformat(rows[0]["prediction_date"][:10])
        max_pred_date = date.fromisoformat(rows[-1]["prediction_date"][:10])
        max_date = _add_trading_days(max_pred_date, 5)
        px = fetch_prices(ticker, min_date, max_date)
        if px is None:
            log(f"  ⚠ Could not fetch prices for {ticker} — skipping")
            continue

        bad_ids = []
        for row in rows:
            pred_date = date.fromisoformat(row["prediction_date"][:10])
            total_checked += 1

            true_ret = compute_return(px, pred_date, int(row["horizon"]))
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

    # May 12 2026: route through Massive per memory #6. Pre-fix used
    # safe_yf_download which falsely flagged all 125 tickers as delisted
    # when yfinance had transient outages (SAFETY GUARD was always saving us).
    from datetime import date as _date_chk, timedelta as _td_chk
    _chk_end = _date_chk.today().strftime("%Y-%m-%d")
    _chk_start = (_date_chk.today() - _td_chk(days=7)).strftime("%Y-%m-%d")
    for ticker in tickers:
        try:
            raw = _mc.download(ticker, start=_chk_start, end=_chk_end,
                               auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                delisted.append(ticker)
        except Exception:
            delisted.append(ticker)

    actual_removed = 0
    aborted = False
    if delisted:
        log(f"  ❌ Possibly delisted: {delisted}")
        # SAFETY GUARD (May 7 2026): refuse to mass-wipe.
        # Bug May 7 2026: yfinance returned empty for ALL 125 tickers due to
        # transient API issues, daily_validator wiped tickers.txt entirely.
        # Real delisting is rare — usually 1-2 tickers per quarter.
        wipe_pct = (len(delisted) / max(1, len(tickers))) * 100
        if fix and wipe_pct > 10.0:
            aborted = True
            log(f"  🛑 ABORT — would remove {wipe_pct:.0f}% of tickers ({len(delisted)}/{len(tickers)})")
            log(f"     This usually means a yfinance/network outage, not real delisting.")
            log(f"     NOT modifying tickers.txt. Re-run validator manually after verifying.")
            desktop_alert(
                "ML Quant Fund — Delisted Check ABORTED",
                f"Would remove {wipe_pct:.0f}% — likely API outage, not real delistings. Manual review required."
            )
        elif fix:
            # Remove from tickers.txt
            if TICKERS_FILE.exists():
                lines = [t for t in TICKERS_FILE.read_text().splitlines() if t.strip() and t.strip() not in delisted]
                TICKERS_FILE.write_text("\n".join(lines) + "\n")
            # FIXED filename (May 7 2026): was "watchlist.txt" (orphan), now correct file
            watchlist = Path("tickers_watchlist.txt")
            existing = set()
            if watchlist.exists():
                existing = set(line.strip() for line in watchlist.read_text().splitlines()
                               if line.strip() and not line.startswith("#"))
            new_entries = [t for t in delisted if t not in existing]
            if new_entries:
                with open(watchlist, "a") as f:
                    f.write("\n".join(new_entries) + "\n")
            actual_removed = len(delisted)
            log(f"  ✅ Moved {delisted} to tickers_watchlist.txt, removed from tickers.txt")
            desktop_alert(
                "ML Quant Fund — Delisted Tickers",
                f"Moved to watchlist: {', '.join(delisted)} — update train_all.py manually"
            )
    else:
        log(f"  ✅ All {len(tickers)} tickers have active price data")

    return {
        "delisted": len(delisted),
        "tickers": delisted,
        "fixed": actual_removed,
        "aborted": aborted,
    }


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
    if results['delisted'].get('aborted'):
        log(f"  Delisted tickers  : {results['delisted']['delisted']} flagged, 0 removed (safety guard ABORTED — likely API outage)")
    else:
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
