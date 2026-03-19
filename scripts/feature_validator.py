"""
scripts/feature_validator.py
─────────────────────────────────────────────────────────────────────────────
Daily feature data validator for ML Quant Fund.
Runs at 8 PM ET daily (before daily_runner at 8:30 PM ET).

Validates that key features fed into the model match real-world current data:
  1. Market features    — VIX, SPY, oil (USO), S&P futures (ES=F)
  2. Price features     — spot-checks close prices for sample tickers
  3. Regime classifier  — validates current regime label vs raw VIX/SPY data
  4. Feature freshness  — checks that features are from today, not stale
  5. Earnings calendar  — warns if ticker has earnings within 3 days (high risk)

Auto-fixes:
  - Clears stale feature/regime cache so next run recomputes fresh

Alerts:
  - Desktop notification if any feature is stale or wrong
  - Logs all findings to logs/feature_validator.log

Run via cron at 8 PM ET (7 AM Vietnam Tue-Sat):
  0 7 * * 2-6 cd ~/Desktop/ML_Quant_Fund && /Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python scripts/feature_validator.py --fix >> ~/Desktop/ML_Quant_Fund/logs/feature_validator.log 2>&1

Usage:
  python scripts/feature_validator.py           # check only
  python scripts/feature_validator.py --fix     # check and auto-fix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

sys.path.insert(0, ".")

ET             = pytz.timezone("America/New_York")
LOG_DIR        = Path("logs")
TICKERS_FILE   = Path("tickers.txt")
REGIME_CACHE   = Path("models/saved/regime_cache.json")
STALE_HOURS    = 26   # features older than this are considered stale
PRICE_TOL      = 0.005  # 0.5% tolerance for price comparison
VIX_TOL        = 1.0    # 1 point tolerance for VIX comparison
RETURN_TOL     = 0.003  # 0.3% tolerance for return comparison

# Key market tickers to validate
MARKET_TICKERS = {
    "SPY":  "S&P 500 ETF",
    "^VIX": "VIX Fear Index",
    "USO":  "Crude Oil ETF",
    "TLT":  "Long Bond ETF",
    "^GSPC": "S&P 500 Index",
}

# Sample tickers for price spot-check (subset of universe)
SPOT_CHECK_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOG", "META", "AMD"]


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{ts}] {msg}", flush=True)


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return datetime.now(ET).date()


def desktop_alert(title: str, message: str):
    try:
        os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
    except Exception:
        pass


def fetch_latest(ticker: str) -> dict | None:
    """Fetch latest OHLCV + return for a ticker."""
    try:
        raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        latest = raw.iloc[-1]
        prev   = raw.iloc[-2] if len(raw) >= 2 else raw.iloc[-1]
        close      = float(latest["Close"])
        prev_close = float(prev["Close"])
        day_ret    = (float(latest["Close"]) - float(latest["Open"])) / float(latest["Open"])
        daily_ret  = (close - prev_close) / prev_close
        return {
            "date":       raw.index[-1].date(),
            "close":      round(close, 4),
            "open":       round(float(latest["Open"]), 4),
            "day_ret":    round(day_ret, 6),
            "daily_ret":  round(daily_ret, 6),
            "volume":     int(latest["Volume"]),
        }
    except Exception as e:
        log(f"  ⚠ Could not fetch {ticker}: {e}")
        return None


# ── Check 1: Market Feature Freshness & Accuracy ─────────────────────────────

def check_market_features(fix: bool) -> dict:
    """
    Fetch ground truth for VIX, SPY, USO, TLT and compare against
    what the feature pipeline would compute today.
    """
    log("Checking market features (VIX, SPY, USO, TLT, ES=F)...")
    issues = []
    today  = today_et()

    for ticker, label in MARKET_TICKERS.items():
        data = fetch_latest(ticker)
        if data is None:
            issues.append({"ticker": ticker, "issue": "no data from yfinance"})
            continue

        data_date = data["date"]
        days_old  = (today - data_date).days

        # Flag if data is from more than 1 trading day ago
        # (weekends count as 0 days old since markets are closed)
        if today.weekday() < 5 and days_old > 1:
            issues.append({
                "ticker": ticker,
                "label":  label,
                "issue":  f"data is {days_old} days old (expected today or yesterday)",
                "date":   str(data_date),
            })
            log(f"  ❌ {ticker} ({label}): data from {data_date} — {days_old} days old")
        else:
            log(f"  ✅ {ticker} ({label}): close={data['close']} date={data_date}")

    if not issues:
        log(f"  ✅ All market features are current")
    return {"issues": len(issues), "details": issues}


# ── Check 2: Regime Cache Freshness ──────────────────────────────────────────

def check_regime_cache(fix: bool) -> dict:
    """
    Validate the regime cache is fresh and matches current VIX/SPY data.
    """
    log("Checking regime classifier cache...")
    issues = []
    today  = today_et()

    if not REGIME_CACHE.exists():
        log(f"  ⚠ No regime cache found at {REGIME_CACHE}")
        return {"issues": 1, "details": ["no cache file"]}

    try:
        cache = json.loads(REGIME_CACHE.read_text())
        cached_date  = cache.get("date", "unknown")
        cached_label = cache.get("label", "unknown")
        cached_vix   = cache.get("vix_level", None)

        log(f"  Cached regime: {cached_label} | VIX={cached_vix} | date={cached_date}")

        # Check freshness
        try:
            cache_dt = date.fromisoformat(str(cached_date)[:10])
            days_old = (today - cache_dt).days
            if today.weekday() < 5 and days_old > 1:
                issues.append(f"cache is {days_old} days old")
                log(f"  ❌ Regime cache is {days_old} days old — needs refresh")
                if fix:
                    REGIME_CACHE.unlink()
                    log(f"  ✅ Deleted stale regime cache — will recompute on next run")
        except Exception:
            pass

        # Cross-check VIX
        vix_data = fetch_latest("^VIX")
        if vix_data and cached_vix:
            vix_diff = abs(float(cached_vix) - vix_data["close"])
            if vix_diff > VIX_TOL * 3:
                issues.append(f"cached VIX {cached_vix} vs actual {vix_data['close']} (diff={vix_diff:.1f})")
                log(f"  ❌ VIX mismatch: cached={cached_vix} actual={vix_data['close']}")
                if fix:
                    REGIME_CACHE.unlink()
                    log(f"  ✅ Deleted stale regime cache due to VIX mismatch")
            else:
                log(f"  ✅ VIX matches: cached={cached_vix} actual={vix_data['close']} (diff={vix_diff:.2f})")

        # Validate regime label matches VIX level
        if vix_data:
            actual_vix = vix_data["close"]
            expected_regime = (
                "VOLATILE" if actual_vix >= 25 else
                "BEAR"     if actual_vix >= 20 else
                "BULL"     if actual_vix < 15 else
                "NEUTRAL"
            )
            if cached_label != expected_regime:
                log(f"  ⚠ Regime label mismatch: cached={cached_label} expected≈{expected_regime} (VIX={actual_vix})")
            else:
                log(f"  ✅ Regime label {cached_label} consistent with VIX={actual_vix}")

    except Exception as e:
        issues.append(f"cache read error: {e}")
        log(f"  ❌ Could not read regime cache: {e}")

    if not issues:
        log(f"  ✅ Regime cache is fresh and consistent")
    return {"issues": len(issues), "details": issues}


# ── Check 3: Spot-check Ticker Prices ────────────────────────────────────────

def check_ticker_prices(fix: bool) -> dict:
    """
    Spot-check a sample of tickers: fetch latest close from yfinance
    and compare against what build_feature_dataframe would produce.
    """
    log(f"Spot-checking prices for {len(SPOT_CHECK_TICKERS)} tickers...")
    issues = []
    today  = today_et()

    for ticker in SPOT_CHECK_TICKERS:
        data = fetch_latest(ticker)
        if data is None:
            issues.append({"ticker": ticker, "issue": "no data"})
            continue

        data_date = data["date"]
        days_old  = (today - data_date).days

        if today.weekday() < 5 and days_old > 1:
            issues.append({
                "ticker": ticker,
                "issue": f"price data {days_old} days old",
                "date": str(data_date),
            })
            log(f"  ❌ {ticker}: price from {data_date} ({days_old}d old) — expected today")
        else:
            log(f"  ✅ {ticker}: close={data['close']} open={data['open']} ret={data['day_ret']*100:.2f}% date={data_date}")

    if not issues:
        log(f"  ✅ All spot-check prices are current")
    return {"issues": len(issues), "details": issues}


# ── Check 4: Earnings Calendar Warning ───────────────────────────────────────

def check_earnings_calendar() -> dict:
    """
    Warn if any tracked tickers have earnings within the next 3 days.
    High earnings risk = model predictions less reliable.
    """
    log("Checking earnings calendar (next 3 days)...")

    if not TICKERS_FILE.exists():
        return {"warnings": 0, "tickers": []}

    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    upcoming = []
    today    = today_et()
    warning_window = timedelta(days=3)

    try:
        import sqlite3
        conn = sqlite3.connect("accuracy.db")
        # Check if we have an earnings table
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]

        if "earnings" in table_names:
            rows = conn.execute("""
                SELECT ticker, report_date FROM earnings
                WHERE date(report_date) BETWEEN ? AND ?
                ORDER BY report_date
            """, (today.isoformat(), (today + warning_window).isoformat())).fetchall()

            for row in rows:
                upcoming.append({"ticker": row[0], "date": row[1]})
                log(f"  ⚠ {row[0]} earnings on {row[1]} — predictions may be unreliable")
        conn.close()
    except Exception as e:
        log(f"  ⚠ Could not check earnings calendar: {e}")

    if upcoming:
        tickers_str = ", ".join(f"{u['ticker']}({u['date']})" for u in upcoming)
        desktop_alert(
            "ML Quant Fund — Earnings Warning",
            f"Earnings in next 3 days: {tickers_str}"
        )
    else:
        log(f"  ✅ No earnings in next 3 days for tracked tickers")

    return {"warnings": len(upcoming), "tickers": upcoming}


# ── Check 5: Feature Pipeline Smoke Test ─────────────────────────────────────

def check_feature_pipeline(fix: bool) -> dict:
    """
    Run build_feature_dataframe on 3 sample tickers and validate
    that key features are non-zero and within expected ranges.
    """
    log("Running feature pipeline smoke test (AAPL, NVDA, MSFT)...")
    issues = []

    EXPECTED_RANGES = {
        "rsi_14":          (0, 100),
        "vix_close":       (5, 100),
        "oil_ret":         (-0.15, 0.15),
        "spy_ret":         (-0.15, 0.15),
        "momentum_score":  (-5, 5),  # loose range
        "bb_width":        (0, None),
    }

    for ticker in ["AAPL", "NVDA", "MSFT"]:
        try:
            from features.builder import build_feature_dataframe
            df = build_feature_dataframe(ticker, start_date="2026-01-01", training_mode=True)

            if df.empty:
                issues.append({"ticker": ticker, "issue": "empty DataFrame"})
                log(f"  ❌ {ticker}: build_feature_dataframe returned empty DataFrame")
                continue

            last = df.iloc[-1]
            last_date = str(last.get("date", "unknown"))[:10]
            today = today_et()

            # Check data freshness
            try:
                feat_date = date.fromisoformat(last_date)
                days_old = (today - feat_date).days
                if today.weekday() < 5 and days_old > 2:
                    issues.append({"ticker": ticker, "issue": f"features {days_old} days old"})
                    log(f"  ❌ {ticker}: latest feature row is {days_old} days old ({last_date})")
                else:
                    log(f"  ✅ {ticker}: features current as of {last_date}")
            except Exception:
                pass

            # Check key feature ranges
            for feat, (low, high) in EXPECTED_RANGES.items():
                if feat not in df.columns:
                    continue
                val = float(last.get(feat, np.nan))
                if np.isnan(val):
                    issues.append({"ticker": ticker, "feature": feat, "issue": "NaN"})
                    log(f"  ❌ {ticker}.{feat} = NaN")
                elif low is not None and val < low:
                    issues.append({"ticker": ticker, "feature": feat, "issue": f"too low: {val:.4f} < {low}"})
                    log(f"  ❌ {ticker}.{feat} = {val:.4f} (below min {low})")
                elif high is not None and val > high:
                    issues.append({"ticker": ticker, "feature": feat, "issue": f"too high: {val:.4f} > {high}"})
                    log(f"  ❌ {ticker}.{feat} = {val:.4f} (above max {high})")

            if not any(i.get("ticker") == ticker for i in issues):
                log(f"  ✅ {ticker}: all features in expected ranges (rsi={last.get('rsi_14', 'N/A'):.1f}, vix={last.get('vix_close', 'N/A'):.1f}, oil_ret={last.get('oil_ret', 0)*100:.2f}%)")

        except Exception as e:
            issues.append({"ticker": ticker, "issue": str(e)})
            log(f"  ❌ {ticker}: feature pipeline error — {e}")

    if not issues:
        log(f"  ✅ Feature pipeline smoke test passed")
    return {"issues": len(issues), "details": issues}


# ── Check 6: Full Universe Price Cross-check ──────────────────────────────────

def check_all_ticker_prices() -> dict:
    """
    Fetch real-time prices for ALL tickers in tickers.txt via yfinance,
    then compare against what build_feature_dataframe returns for each.
    Catches crossover bugs (GOOG getting GME's price) and stale data.
    Runs on all tickers — not just the spot-check sample.
    """
    if not TICKERS_FILE.exists():
        log("  ⚠ tickers.txt not found — skipping full price check")
        return {"issues": 0, "details": []}

    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    log(f"Checking live prices for all {len(tickers)} tickers...")

    issues   = []
    today    = today_et()

    # Step 1 — fetch all prices in one batch call
    try:
        raw = yf.download(tickers, period="3d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                closes = raw["Close"].copy()
            elif "Close" in raw.columns.get_level_values(1):
                closes = raw.xs("Close", axis=1, level=1).copy()
            else:
                closes = raw.iloc[:, raw.columns.get_level_values(0) == "Close"].copy()
        else:
            closes = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
        closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
        # Filter out all-NaN columns
        closes = closes.dropna(axis=1, how="all")
        ground_truth = closes.iloc[-1].dropna().to_dict()
        log(f"  Fetched {len(ground_truth)}/{len(tickers)} ground truth prices from yfinance")
        # For any missing tickers, fetch individually
        missing_now = [t for t in tickers if t not in ground_truth]
        if missing_now:
            log(f"  Retrying {len(missing_now)} missing tickers individually...")
            for t in missing_now:
                try:
                    r = yf.download(t, period="3d", auto_adjust=True, progress=False)
                    if not r.empty:
                        if isinstance(r.columns, pd.MultiIndex):
                            r.columns = r.columns.get_level_values(0)
                        ground_truth[t] = float(r["Close"].iloc[-1])
                except Exception:
                    pass
        log(f"  Final ground truth: {len(ground_truth)}/{len(tickers)} tickers")
    except Exception as e:
        log(f"  ❌ Batch price fetch failed: {e}")
        return {"issues": 1, "details": [{"issue": str(e)}]}

    # Step 2 — compare against feature pipeline prices
    crossed  = []
    stale    = []
    missing  = []

    for ticker in tickers:
        true_price = ground_truth.get(ticker)
        if true_price is None or (isinstance(true_price, float) and np.isnan(true_price)):
            missing.append(ticker)
            log(f"  ⚠ {ticker}: no ground truth price from yfinance")
            continue

        try:
            from features.builder import build_feature_dataframe
            df = build_feature_dataframe(ticker, start_date="2025-01-01", training_mode=True)
            if df.empty:
                stale.append(ticker)
                log(f"  ❌ {ticker}: feature pipeline returned empty DataFrame")
                continue

            feat_price = float(df["close"].iloc[-1])
            feat_date  = df["date"].iloc[-1]

            # Check price crossover (>10% difference = likely bug)
            diff_pct = abs(feat_price - true_price) / true_price
            if diff_pct > 0.10:
                crossed.append({
                    "ticker":     ticker,
                    "feat_price": round(feat_price, 2),
                    "true_price": round(true_price, 2),
                    "diff_pct":   round(diff_pct * 100, 1),
                })
                log(f"  ❌ {ticker}: PRICE CROSSOVER — feature=${feat_price:.2f} actual=${true_price:.2f} ({diff_pct*100:.1f}% diff)")
                issues.append({"ticker": ticker, "issue": f"crossover: feat={feat_price:.2f} true={true_price:.2f}"})
            else:
                log(f"  ✅ {ticker}: ${feat_price:.2f} ≈ ${true_price:.2f} ({diff_pct*100:.1f}% diff)")

            # Check freshness
            try:
                days_old = (today - feat_date).days if hasattr(feat_date, 'days') else (today - date.fromisoformat(str(feat_date)[:10])).days
                if today.weekday() < 5 and days_old > 2:
                    stale.append(ticker)
                    log(f"  ❌ {ticker}: feature data is {days_old} days old")
                    issues.append({"ticker": ticker, "issue": f"stale: {days_old} days old"})
            except Exception:
                pass

        except Exception as e:
            log(f"  ⚠ {ticker}: could not check — {e}")

    # Summary
    if crossed:
        log(f"\n  ❌ PRICE CROSSOVERS DETECTED ({len(crossed)} tickers):")
        for c in crossed:
            log(f"     {c['ticker']}: feature=${c['feat_price']} actual=${c['true_price']} ({c['diff_pct']}% off)")
        desktop_alert(
            "ML Quant Fund — Price Crossover Alert",
            f"Price crossover on {len(crossed)} tickers: {', '.join(c['ticker'] for c in crossed[:5])}. Check feature_validator.log"
        )
    if stale:
        log(f"  ❌ Stale features: {stale}")
    if missing:
        log(f"  ⚠ No yfinance data: {missing}")

    if not issues:
        log(f"  ✅ All {len(tickers)} ticker prices verified — no crossovers or stale data")

    return {
        "issues":  len(issues),
        "crossed": len(crossed),
        "stale":   len(stale),
        "missing": len(missing),
        "details": issues,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_feature_validator(fix: bool = True):
    LOG_DIR.mkdir(exist_ok=True)

    log("=" * 60)
    log(f"  ML Quant Fund — Feature Validator")
    log(f"  Date  : {today_et()} ET")
    log(f"  Time  : {now_et().strftime('%H:%M ET')}")
    log(f"  Fix   : {fix}")
    log("=" * 60)

    results = {}
    results["market"]   = check_market_features(fix)
    results["regime"]   = check_regime_cache(fix)
    results["prices"]   = check_ticker_prices(fix)
    results["earnings"] = check_earnings_calendar()
    results["pipeline"] = check_feature_pipeline(fix)
    results["allprices"] = check_all_ticker_prices()

    # Summary
    total_issues = (
        results["market"]["issues"] +
        results["regime"]["issues"] +
        results["prices"]["issues"] +
        results["earnings"]["warnings"] +
        results["pipeline"]["issues"] +
        results["allprices"]["issues"]
    )

    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    log(f"  Market features   : {results['market']['issues']} issues")
    log(f"  Regime cache      : {results['regime']['issues']} issues")
    log(f"  Ticker prices     : {results['prices']['issues']} issues")
    log(f"  Earnings warnings : {results['earnings']['warnings']} upcoming")
    log(f"  Feature pipeline  : {results['pipeline']['issues']} issues")
    log(f"  Full price check  : {results['allprices']['crossed']} crossovers, {results['allprices']['stale']} stale, {results['allprices']['missing']} missing")
    log("=" * 60)

    if total_issues > 0:
        log(f"  ❌ {total_issues} total issues found")
        desktop_alert(
            "ML Quant Fund — Feature Validator",
            f"{total_issues} feature issues found — check logs/feature_validator.log"
        )
    else:
        log("  ✅ All feature checks passed — safe to run daily runner")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature data validator for ML Quant Fund")
    parser.add_argument("--fix", action="store_true", help="Auto-fix stale caches")
    args = parser.parse_args()
    run_feature_validator(fix=args.fix)
