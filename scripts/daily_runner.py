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
BUY_THRESHOLD = 0.55     # minimum prob_eff to alert
SLEEP_BETWEEN = 2.0      # seconds between tickers (avoid rate limits)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_trading_day() -> bool:
    """Return True if today is a weekday (basic check, ignores holidays)."""
    return today_et().weekday() < 5


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


def log_prediction_to_db(
    ticker: str, horizon: int, signal: str,
    prob: float, prob_eff: float, run_date: str,
):
    """Log prediction to accuracy.db for later reconciliation."""
    try:
        from accuracy.sink import log_prediction
        log_prediction(
            ticker=ticker, horizon=horizon,
            signal=signal, prob=prob, prob_eff=prob_eff,
            run_date=run_date,
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

def run_daily():
    run_date = today_et()
    log.info(f"{'='*60}")
    log.info(f"  Daily Runner — {run_date}")
    log.info(f"{'='*60}")

    if not is_trading_day():
        log.info("Weekend — skipping run")
        return

    tickers = load_tickers()
    log.info(f"Tickers: {len(tickers)}")

    # Get regime
    regime = get_regime_info()
    log.info(f"Regime: {regime['label']} (VIX={regime.get('vix', '?'):.1f}  "
             f"threshold={regime['confidence_threshold']:.0%})")

    buy_signals   = []
    results       = []
    failed        = []

    for i, ticker in enumerate(tickers, 1):
        log.info(f"[{i:2d}/{len(tickers)}] {ticker}")

        try:
            from features.builder import build_feature_dataframe
            from signals.generator import generate_signals

            df = build_feature_dataframe(ticker, start_date=TRAIN_START)

            for horizon in HORIZONS:
                try:
                    sig = generate_signals(ticker, df, horizon=horizon)

                    result = {
                        "ticker":     ticker,
                        "horizon":    horizon,
                        "signal":     sig.today_signal,
                        "prob":       sig.today_prob,
                        "prob_eff":   sig.today_prob_eff,
                        "confidence": "HIGH" if sig.today_prob_eff >= 0.70
                                      else "MEDIUM" if sig.today_prob_eff >= BUY_THRESHOLD
                                      else "LOW",
                        "run_date":   run_date,
                    }
                    results.append(result)

                    # Log to accuracy DB
                    log_prediction_to_db(
                        ticker=ticker, horizon=horizon,
                        signal=sig.today_signal,
                        prob=sig.today_prob,
                        prob_eff=sig.today_prob_eff,
                        run_date=run_date,
                    )

                    if sig.today_signal == "BUY":
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
            log.info(f"    {s['ticker']:6s} h={s['horizon']}d  "
                     f"eff={s['prob_eff']:.1%}  conf={s['confidence']}")
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
    from features.intraday_builder import get_all_intraday_signals
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

    signals = get_all_intraday_signals(tickers)

    # Save to intraday history
    out = Path("data/intraday_history")
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{today}.json"
    with open(outfile, "w") as f:
        json.dump(signals, f)
    print(f"Intraday snapshot saved: {outfile} ({len(signals)} tickers)")

    # Log predictions to accuracy.db
    try:
        conn = sqlite3.connect("accuracy.db")
        logged = 0
        for s in signals:
            if not s.get("current_price") or s.get("error"):
                continue
            for hr, sig_key, prob_key in [(1,"signal_1hr","prob_1hr"),
                                           (2,"signal_2hr","prob_2hr"),
                                           (4,"signal_4hr","prob_4hr")]:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO intraday_predictions
                        (ticker, prediction_ts, prediction_date, price_at_pred,
                         horizon_hr, prob_up, signal, created_at)
                        VALUES (?,?,?,?,?,?,?,?)
                    """, (s["ticker"], ts, today, s["current_price"],
                          hr, s[prob_key], s[sig_key], ts))
                    logged += 1
                except Exception:
                    pass
        conn.commit()
        conn.close()
        print(f"Logged {logged} intraday predictions to accuracy.db")
    except Exception as e:
        print(f"Failed to log intraday predictions: {e}")
