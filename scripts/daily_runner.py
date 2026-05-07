#!/usr/bin/env python3
# scripts/daily_runner.py
# ─────────────────────────────────────────────────────────────────────────────
# Automated daily signal runner.
# Runs every trading day after market close, generates signals for all tickers,
# logs predictions to accuracy.db, and sends email/desktop alerts on BUY signals.
#
# Setup (run once):
#   chmod +x scripts/daily_runner.py
#
# Run manually:
#   python scripts/daily_runner.py
#
# Schedule automatically (runs at 4:30pm ET every weekday):
#   crontab -e
#   30 16 * * 1-5 cd /Users/YOUR_NAME/Desktop/ML_Quant_Fund && python scripts/daily_runner.py >> logs/daily_runner.log 2>&1
#
# Or use launchd on macOS (see setup instructions at bottom of file)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# Set BEFORE any transformers/HuggingFace imports load.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import os
import sys
import time
import logging
from utils.timezone import now_et, today_et, ts_et
from datetime import datetime, date
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_runner.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS      = [1, 3, 5]
TRAIN_START   = "2022-01-01"
BUY_THRESHOLD = 0.70     # minimum prob_eff to alert
SLEEP_BETWEEN = 2.0      # seconds between tickers (avoid rate limits)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_trading_day() -> bool:
    """Return True if today is a weekday (basic check, ignores holidays)."""
    from datetime import date as _date
    t = today_et()
    if isinstance(t, str):
        return _date.fromisoformat(t).weekday() < 5
    return t.weekday() < 5


def load_tickers() -> list[str]:
    """Load tickers from tickers.txt."""
    p = ROOT / "tickers.txt"
    if p.exists():
        return [t.strip().upper() for t in p.read_text().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "TSLA", "AMD", "GOOG"]


def get_regime_info() -> dict:
    """Get current market regime."""
    try:
        from models.regime_classifier import get_current_regime
        r = get_current_regime(use_cache=False)  # force fresh on daily run
        return {
            "label":               r.label,
            "confidence":          r.confidence,
            "confidence_threshold": r.confidence_threshold,
            "signal_multiplier":   r.signal_multiplier,
            "vix":                 r.vix_level,
        }
    except Exception as e:
        log.warning(f"Regime fetch failed: {e}")
        return {"label": "NEUTRAL", "confidence_threshold": 0.55,
                "signal_multiplier": 1.0, "vix": 20.0}


def load_watchlist() -> list[str]:
    """Load watchlist tickers — predictions only, excluded from accuracy scoring."""
    p = ROOT / "tickers_watchlist.txt"
    if not p.exists():
        return []
    return [
        t.strip().upper() for t in p.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]


def log_prediction_to_db(
    ticker: str, horizon: int, signal: str,
    prob: float, prob_eff: float, run_date: str,
    is_watchlist: bool = False,
    tier: str = "tactical",
):
    """Log prediction to accuracy.db for later reconciliation."""
    try:
        from accuracy.sink import log_prediction
        log_prediction(
            ticker=ticker,
            prediction_date=run_date,
            horizon=horizon,
            prob_up=prob_eff,
            prob_raw=prob,
            signal=signal,
            confidence="HIGH" if prob_eff >= 0.70 else "MEDIUM" if prob_eff >= 0.55 else "LOW",
            is_watchlist=is_watchlist,
            tier=tier,
        )
    except Exception as e:
        log.warning(f"DB log failed for {ticker}: {e}")


def send_desktop_alert(title: str, message: str):
    """Send macOS desktop notification."""
    try:
        os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
    except Exception:
        pass


def send_email_alert(buy_signals: list[dict]):
    """Send email summary of BUY signals (requires SMTP config in secrets.toml)."""
    if not buy_signals:
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        import streamlit as st

        smtp_host = st.secrets.get("SMTP_HOST", os.getenv("SMTP_HOST", ""))
        smtp_port = int(st.secrets.get("SMTP_PORT", os.getenv("SMTP_PORT", 587)))
        smtp_user = st.secrets.get("SMTP_USER", os.getenv("SMTP_USER", ""))
        smtp_pass = st.secrets.get("SMTP_PASS", os.getenv("SMTP_PASS", ""))
        to_email  = st.secrets.get("ALERT_EMAIL", os.getenv("ALERT_EMAIL", smtp_user))

        if not smtp_user or not smtp_pass:
            return

        lines = [f"ML Quant Fund — Daily BUY Signals\n{now_et().strftime('%Y-%m-%d %H:%M ET')}\n"]
        for s in buy_signals:
            lines.append(
                f"  {s['ticker']:6s}  h={s['horizon']}d  "
                f"prob={s['prob']:.1%}  eff={s['prob_eff']:.1%}  "
                f"conf={s['confidence']}"
            )
        body = "\n".join(lines)

        msg = MIMEText(body)
        msg["Subject"] = f"🟢 {len(buy_signals)} BUY Signal(s) — ML Quant Fund"
        msg["From"]    = smtp_user
        msg["To"]      = to_email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())

        log.info(f"Email sent to {to_email} ({len(buy_signals)} BUY signals)")
    except Exception as e:
        log.warning(f"Email alert failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_daily(force: bool = False, start_from: str = None, end_at: str = None):
    run_date = today_et()
    log.info(f"{'='*60}")
    log.info(f"  Daily Runner — {run_date}")
    log.info(f"{'='*60}")

    now = now_et()


    if not is_trading_day():
        log.info("Weekend — skipping run")
        return

    tickers = load_tickers()
    log.info(f"Tickers: {len(tickers)}")

    # start_from support — May 5 2026 (resume after crash)
    if start_from is not None:
        if start_from in tickers:
            idx = tickers.index(start_from)
            tickers = tickers[idx:]
            log.info(f"Starting from {start_from} (skipping first {idx} tickers, {len(tickers)} remaining)")
        else:
            log.warning(f"start_from={start_from} not in tickers list — running all")

    # end_at support — May 6 2026 (subprocess batching)
    if end_at is not None:
        if end_at in tickers:
            idx_end = tickers.index(end_at)
            tickers = tickers[:idx_end + 1]  # inclusive
            log.info(f"Stopping at {end_at} (inclusive). Batch size: {len(tickers)}")
        else:
            log.warning(f"end_at={end_at} not in tickers list — running to end")

    # Get regime
    regime = get_regime_info()
    log.info(f"Regime: {regime['label']} (VIX={regime.get('vix', '?'):.1f}  "
             f"threshold={regime['confidence_threshold']:.0%})")

    buy_signals   = []
    results       = []
    failed        = []

    # Memory diagnostic added May 5 2026 — find secondary leak after
    # FinBERT MPS fix (commits 06bab4d + 25004eb). RSS was growing
    # ~17MB/ticker, OOM-killing process at ticker 59-65.
    # Set DAILY_RUNNER_FORCE_GC=1 to also try gc.collect() each iter.
    import os as _mem_os
    import gc as _mem_gc
    import subprocess as _mem_sp
    _MEM_FORCE_GC = _mem_os.environ.get("DAILY_RUNNER_FORCE_GC", "0") == "1"

    def _mem_rss_mb():
        try:
            r = _mem_sp.run(["ps", "-o", "rss=", "-p", str(_mem_os.getpid())],
                            capture_output=True, text=True, timeout=2)
            return int(r.stdout.strip()) // 1024
        except Exception:
            return -1

    _mem_start = _mem_rss_mb()
    log.info(f"[mem] LOOP START RSS={_mem_start}MB force_gc={_MEM_FORCE_GC}")

    for i, ticker in enumerate(tickers, 1):
        # Memory log AFTER previous ticker completed (skipped on i=1)
        if i > 1:
            _mem_curr = _mem_rss_mb()
            _mem_delta = _mem_curr - _mem_start
            _mem_per = _mem_delta / (i - 1)
            log.info(f"[mem] before-ticker-{i} RSS={_mem_curr}MB total_delta=+{_mem_delta}MB avg=+{_mem_per:.1f}MB/ticker")
            if _MEM_FORCE_GC:
                _mem_gc.collect()
                _mem_post_gc = _mem_rss_mb()
                _mem_freed = _mem_curr - _mem_post_gc
                log.info(f"[mem] post-gc RSS={_mem_post_gc}MB freed={_mem_freed}MB")

        log.info(f"[{i:2d}/{len(tickers)}] {ticker}")

        try:
            from features.builder import build_feature_dataframe
            from signals.generator import generate_signals
            from signals.position_sizer import get_position_size, get_portfolio_value, format_plan, get_portfolio_plan

            # Pre-check: skip tickers with no Massive data BEFORE triggering
            # any feature build. Otherwise the yfinance fallback chain runs,
            # exhausts curl_cffi thread pool, and crashes the process at the
            # next ^VIX download. Added May 5 2026 (USAR/XYZ killed runs at
            # ticker 86 silently via getaddrinfo() thread failure).
            try:
                from features import massive_client as _mc_check
                # Use recent 30-day window — was hardcoded 2024-12 which is stale,
                # caused USAR/XYZ false-skips since they came onto Polygon after that.
                from datetime import date as _date_chk, timedelta as _td_chk
                _chk_end = _date_chk.today().strftime("%Y-%m-%d")
                _chk_start = (_date_chk.today() - _td_chk(days=30)).strftime("%Y-%m-%d")
                _check_df = _mc_check.download(ticker, start=_chk_start, end=_chk_end,
                                                auto_adjust=True, progress=False)
                if _check_df.empty:
                    log.warning(f"  ⚠ {ticker} has no Massive data — skipping (avoid yfinance fallback chain)")
                    failed.append(ticker)
                    time.sleep(SLEEP_BETWEEN)
                    continue
            except Exception as _check_e:
                log.warning(f"  ⚠ {ticker} pre-check failed: {_check_e} — skipping")
                failed.append(ticker)
                time.sleep(SLEEP_BETWEEN)
                continue

            df = build_feature_dataframe(ticker, start_date=TRAIN_START)

            for horizon in HORIZONS:
                try:
                    sig = generate_signals(ticker, df, horizon=horizon, confidence_threshold=BUY_THRESHOLD)

                    result = {
                        "ticker":          ticker,
                        "horizon":         horizon,
                        "signal":          sig.today_signal,
                        "prob":            sig.today_prob,
                        "prob_eff":        sig.today_prob_eff,
                        "confidence":      "HIGH" if sig.today_prob_eff >= 0.70
                                           else "MEDIUM" if sig.today_prob_eff >= BUY_THRESHOLD
                                           else "LOW",
                        "run_date":        run_date,
                        "current_price":   sig.current_price,
                        "price_target_up": sig.price_target_up,
                        "price_target_dn": sig.price_target_dn,
                        "expected_return": sig.expected_return,
                        "atr":             sig.atr,
                        "sharpe":          round(sig.metrics.sharpe, 3)        if sig.metrics and not (sig.metrics.sharpe != sig.metrics.sharpe) else None,
                        "max_drawdown":    round(sig.metrics.max_drawdown, 4)  if sig.metrics else None,
                        "cagr":            round(sig.metrics.cagr, 4)          if sig.metrics else None,
                        "accuracy":        round(sig.metrics.accuracy, 4)      if sig.metrics else None,
                        "n_trades":        sig.metrics.n_trades                if sig.metrics else None,
                        "profit_factor":   round(sig.metrics.profit_factor, 3) if sig.metrics else None,
                    }
                    results.append(result)

                    # Log to accuracy DB
                    from signals.generator import _TICKER_METADATA
                    _tier = _TICKER_METADATA.get(ticker, {}).get("tier", "tactical")
                    log_prediction_to_db(
                        ticker=ticker, horizon=horizon,
                        signal=sig.today_signal,
                        prob=sig.today_prob,
                        prob_eff=sig.today_prob_eff,
                        run_date=run_date,
                        tier=_tier,
                    )

                    # Log features used for this prediction
                    try:
                        last = df.iloc[-1]
                        from accuracy.sink import _get_conn
                        with _get_conn() as _fc:
                            _fc.execute("""
                                INSERT OR REPLACE INTO prediction_features
                                (ticker, prediction_date, horizon,
                                 oil_ret, oil_spy_corr, spy_ret, xlk_ret,
                                 dxy_ret, yield_10y, vix_close, vix_ret,
                                 vix_term_structure, fear_greed,
                                 rsi_14, macd, bb_pct, atr, vol_surge_eod, obv_trend,
                                 return_1d, return_5d, return_20d,
                                 premarket_gap, intraday_momentum,
                                 iv_skew_snap, pc_ratio_snap,
                                 monday_sentiment, beta_60d, short_ratio,
                                 sector_rel_ret, created_at)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """, (
                                ticker, str(run_date), horizon,
                                float(last.get("oil_ret", 0)),
                                float(last.get("oil_spy_corr", 0)),
                                float(last.get("spy_ret", 0)),
                                float(last.get("xlk_ret", 0)),
                                float(last.get("dxy_ret", 0)),
                                float(last.get("yield_10y", 0)),
                                float(last.get("vix_close", 0)),
                                float(last.get("vix_ret", 0)),
                                float(last.get("vix_term_structure", 1)),
                                float(last.get("fear_greed", 0.5)),
                                float(last.get("rsi_14", 50)),
                                float(last.get("macd", 0)),
                                float(last.get("bb_pct", 0.5)),
                                float(last.get("atr", 0)),
                                float(last.get("vol_surge_eod", 0)),
                                float(last.get("obv_trend", 0)),
                                float(last.get("return_1d", 0)),
                                float(last.get("return_5d", 0)),
                                float(last.get("return_20d", 0)),
                                float(last.get("premarket_gap", 0)),
                                float(last.get("intraday_momentum", 0)),
                                float(last.get("iv_skew_snap", 0)),
                                float(last.get("pc_ratio_snap", 1)),
                                float(last.get("monday_sentiment", 0)),
                                float(last.get("beta_60d", 1)),
                                float(last.get("short_ratio", 0)),
                                float(last.get("sector_rel_ret", 0)),
                                str(run_date),
                            ))
                    except Exception as _fe:
                        pass  # feature logging is best-effort

                    if sig.today_signal == "BUY":
                        # Add position sizing
                        try:
                            from signals.position_sizer import get_position_size, get_portfolio_value
                            pv  = get_portfolio_value()
                            pos = get_position_size(
                                ticker=ticker,
                                prob_eff=sig.today_prob_eff,
                                confidence=result["confidence"],
                                portfolio_value=pv,
                                current_price=result.get("current_price"),
                            )
                            result["position_pct"]     = pos.final_pct
                            result["position_dollars"]  = pos.dollars
                            result["position_shares"]   = pos.shares
                            result["position_rationale"] = pos.rationale
                        except Exception as pe:
                            log.warning(f"  Position sizing failed: {pe}")
                        buy_signals.append(result)
                        log.info(f"  🟢 BUY  h={horizon}d  "
                                 f"prob={sig.today_prob:.1%}  "
                                 f"eff={sig.today_prob_eff:.1%}")
                    else:
                        log.info(f"  ⚪ HOLD h={horizon}d  "
                                 f"prob={sig.today_prob:.1%}  "
                                 f"eff={sig.today_prob_eff:.1%}")

                except Exception as e:
                    log.warning(f"  ✗ {ticker} h={horizon}d: {e}")

        except Exception as e:
            log.error(f"  ✗ {ticker} feature build failed: {e}")
            failed.append(ticker)

        time.sleep(SLEEP_BETWEEN)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"  DONE — {len(results)} signals, {len(buy_signals)} BUY, "
             f"{len(failed)} failed")

    if buy_signals:
        log.info(f"\n  🟢 BUY SIGNALS TODAY:")
        for s in buy_signals:
            pos_str = ""
            if s.get("position_dollars"):
                pos_str = f"  size={s['position_pct']*100:.1f}% (${s['position_dollars']:,.0f})"
                if s.get("position_shares"):
                    pos_str += f" ~{s['position_shares']} shares"
            log.info(f"    {s['ticker']:6s} h={s['horizon']}d  "
                     f"eff={s['prob_eff']:.1%}  conf={s['confidence']}{pos_str}")
        send_desktop_alert(
            "ML Quant Fund",
            f"{len(buy_signals)} BUY signal(s): "
            f"{', '.join(s['ticker'] for s in buy_signals[:5])}"
        )
        send_email_alert(buy_signals)
    else:
        log.info("  No BUY signals today (regime may be suppressing signals)")

    # Save daily summary JSON
    summary = {
        "date":        run_date,
        "regime":      regime,
        "n_signals":   len(results),
        "n_buy":       len(buy_signals),
        "n_failed":    len(failed),
        "buy_signals": buy_signals,
        "all_signals": results,
    }
    summary_path = LOG_DIR / f"signals_{run_date}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Summary saved → {summary_path}")

    # Save dashboard cache (data/signals_cache.json) — loaded by default in 1_Dashboard.py
    # MERGE-MODE May 5 2026: if cache already exists for the same date, merge new
    # results in instead of overwriting. Protects against partial runs (e.g. resume
    # via start_from=) wiping out earlier ticker data.
    cache_path = ROOT / "data" / "signals_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    merged_signals = list(results)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                existing = json.load(f)
            if existing.get("date") == run_date and isinstance(existing.get("signals"), list):
                new_keys = {(s.get("ticker"), s.get("horizon")) for s in results}
                preserved = [s for s in existing["signals"]
                             if (s.get("ticker"), s.get("horizon")) not in new_keys]
                merged_signals = preserved + list(results)
                log.info(f"  Cache merge: {len(preserved)} preserved + {len(results)} new = {len(merged_signals)} total")
        except Exception as e:
            log.warning(f"  Cache merge failed, will overwrite: {e}")

    dashboard_cache = {
        "generated_at": now_et().strftime("%Y-%m-%dT%H:%M:%S"),
        "date":         run_date,
        "signals":      merged_signals,
    }
    with open(cache_path, "w") as f:
        json.dump(dashboard_cache, f, indent=2)
    log.info(f"  Dashboard cache saved → {cache_path} ({len(merged_signals)} signals)")

    # ── Watchlist predictions (no accuracy logging) ────────────────
    watchlist = load_watchlist()
    if watchlist:
        log.info(f"Watchlist: {len(watchlist)} tickers")
        watchlist_results = []
        for ticker in watchlist:
            try:
                from features.builder import build_feature_dataframe
                from signals.generator import generate_signals
                df = build_feature_dataframe(ticker, start_date=TRAIN_START)
                for horizon in HORIZONS:
                    try:
                        sig = generate_signals(ticker, df, horizon=horizon,
                                               confidence_threshold=BUY_THRESHOLD)
                        watchlist_results.append({
                            "ticker":          ticker,
                            "horizon":         horizon,
                            "signal":          sig.today_signal,
                            "prob":            sig.today_prob,
                            "prob_eff":        sig.today_prob_eff,
                            "confidence":      "HIGH" if sig.today_prob_eff >= 0.70
                                               else "MEDIUM" if sig.today_prob_eff >= BUY_THRESHOLD
                                               else "LOW",
                            "run_date":        run_date,
                            "current_price":   sig.current_price,
                            "price_target_up": sig.price_target_up,
                            "price_target_dn": sig.price_target_dn,
                            "expected_return": sig.expected_return,
                            "atr":             sig.atr,
                            "sharpe":          round(sig.metrics.sharpe, 3) if sig.metrics and sig.metrics.sharpe == sig.metrics.sharpe else None,
                            "is_watchlist":    True,
                        })
                        log.info(f"  WATCHLIST {ticker} {horizon}d: {sig.today_signal} ({sig.today_prob_eff:.1%})")
                        from signals.generator import _TICKER_METADATA
                        _tier = _TICKER_METADATA.get(ticker, {}).get("tier", "tactical")
                        log_prediction_to_db(
                            ticker=ticker,
                            horizon=horizon,
                            signal=sig.today_signal,
                            prob=sig.today_prob,
                            prob_eff=sig.today_prob_eff,
                            run_date=run_date,
                            is_watchlist=True,
                            tier=_tier,
                        )
                    except Exception as e:
                        log.warning(f"  WATCHLIST {ticker} {horizon}d failed: {e}")
            except Exception as e:
                log.warning(f"  WATCHLIST {ticker} failed: {e}")

        if watchlist_results:
            wl_cache_path = ROOT / "data" / "watchlist_cache.json"
            wl_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wl_cache_path, "w") as f:
                json.dump({
                    "generated_at": now_et().strftime("%Y-%m-%dT%H:%M:%S"),
                    "date":         run_date,
                    "signals":      watchlist_results,
                }, f, indent=2, default=str)
            log.info(f"  Watchlist cache saved → {wl_cache_path}")

    log.info(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    # Guard: only run on US trading days (Mon-Fri ET, excluding holidays)
    from utils.timezone import now_et
    now = now_et()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6 in ET
        print(f"Skipping — not a trading day ({now.strftime('%A %Y-%m-%d ET')})")
        exit(0)
    run_daily()

    # Auto-reconcile EOD outcomes after predictions are logged
    print("\nAuto-reconciling EOD outcomes...")
    try:
        from accuracy.sink import reconcile_outcomes, update_accuracy_cache
        n = reconcile_outcomes()
        print(f"Reconciled: {n} new outcomes")
        df = update_accuracy_cache()
        print(f"Accuracy cache updated: {len(df)} rows")
    except Exception as e:
        print(f"Reconcile failed: {e}")

def log_intraday_snapshot():
    """Log intraday snapshot at market open (9:30am ET). Skip if market not open."""
    import json, sqlite3
    from pathlib import Path
    from features.intraday_builder import get_all_intraday_signals, minutes_since_open
    from utils.timezone import now_et as _now_et, today_et, ts_et
    now_et_dt = _now_et()
    # Guard: only run during market hours (9:30am - 4:00pm ET, Mon-Fri)
    if now_et_dt.weekday() >= 5:
        print(f"Skipping snapshot — weekend ({now_et_dt.strftime('%A ET')})")
        return
    market_open_hour = now_et_dt.hour * 60 + now_et_dt.minute
    if market_open_hour < 9*60+30 or market_open_hour > 16*60:
        print(f"Skipping snapshot — outside market hours ({now_et_dt.strftime('%H:%M ET')})")
        return
    today  = now_et_dt.strftime("%Y-%m-%d")
    ts     = now_et_dt.strftime("%Y-%m-%dT%H:%M:%S")
    tickers = [t.strip() for t in open("tickers.txt").readlines() if t.strip()]

    # Wait until at least 5 minutes after open so 5-min bars exist
    mins = minutes_since_open()
    if mins < 5:
        wait_secs = (5 - mins) * 60 + 30
        print(f"  Waiting {wait_secs}s for first 5-min bar to populate...")
        import time; time.sleep(wait_secs)

    # First attempt — batch download
    signals = get_all_intraday_signals(tickers)

    # Retry failed tickers individually
    failed = [s["ticker"] for s in signals if not s.get("current_price") or s.get("error") == "No intraday data"]
    if failed:
        print(f"  Retrying {len(failed)} tickers individually...")
        import time
        retry_signals = []
        for tkr in failed:
            try:
                s = get_all_intraday_signals([tkr])
                if s and s[0].get("current_price"):
                    retry_signals.append(s[0])
                time.sleep(2)
            except Exception:
                pass
        retry_map = {s["ticker"]: s for s in retry_signals}
        signals = [retry_map.get(s["ticker"], s) for s in signals]
        valid = sum(1 for s in signals if s.get("current_price"))
        print(f"  After retry: {valid}/{len(signals)} tickers with data")

    # Save to intraday history
    out = Path("data/intraday_history")
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{today}.json"
    with open(outfile, "w") as f:
        json.dump(signals, f)
    print(f"Intraday snapshot saved: {outfile} ({len(signals)} tickers)")

    # Log predictions to accuracy.db
    skipped_no_price = []
    skipped_error    = []
    skipped_missing_keys = []
    skipped_db_error = []
    try:
        conn = sqlite3.connect("accuracy.db")
        logged = 0
        for s in signals:
            if s.get("error"):
                skipped_error.append((s.get("ticker", "?"), s.get("error")))
                continue
            if not s.get("current_price"):
                skipped_no_price.append(s.get("ticker", "?"))
                continue
            for hr, sig_key, prob_key in [(1,"signal_1hr","prob_1hr"),
                                           (2,"signal_2hr","prob_2hr"),
                                           (4,"signal_4hr","prob_4hr")]:
                # Check the per-horizon keys actually exist before inserting
                if sig_key not in s or prob_key not in s:
                    skipped_missing_keys.append((s["ticker"], hr))
                    continue
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO intraday_predictions
                        (ticker, prediction_ts, prediction_date, price_at_pred,
                         horizon_hr, prob_up, signal, created_at)
                        VALUES (?,?,?,?,?,?,?,?)
                    """, (s["ticker"], ts, today, s["current_price"],
                          hr, s[prob_key], s[sig_key], ts))
                    logged += 1
                except Exception as e:
                    skipped_db_error.append((s["ticker"], hr, str(e)[:80]))
        conn.commit()
        conn.close()
        print(f"Logged {logged} intraday predictions to accuracy.db")

        # Surface failures so they're not silent
        if skipped_no_price:
            sample = ", ".join(skipped_no_price[:10])
            extra  = f" (+{len(skipped_no_price)-10} more)" if len(skipped_no_price) > 10 else ""
            print(f"  WARN  {len(skipped_no_price)} tickers skipped (no current_price): {sample}{extra}")
        if skipped_error:
            sample = "; ".join(f"{t}={e[:40]}" for t, e in skipped_error[:5])
            extra  = f" (+{len(skipped_error)-5} more)" if len(skipped_error) > 5 else ""
            print(f"  WARN  {len(skipped_error)} tickers skipped (snapshot error): {sample}{extra}")
        if skipped_missing_keys:
            counts = {}
            for t, hr in skipped_missing_keys:
                counts[t] = counts.get(t, 0) + 1
            sample_tkrs = list(counts.keys())[:10]
            print(f"  WARN  {len(skipped_missing_keys)} (ticker,horizon) pairs missing prediction keys; affected tickers: {sample_tkrs}")
        if skipped_db_error:
            sample = "; ".join(f"{t}/h{hr}={e[:40]}" for t, hr, e in skipped_db_error[:3])
            print(f"  WARN  {len(skipped_db_error)} DB insert failures: {sample}")
    except Exception as e:
        print(f"Failed to log intraday predictions: {e}")
