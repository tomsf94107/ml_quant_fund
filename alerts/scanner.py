# alerts/scanner.py
# ─────────────────────────────────────────────────────────────────────────────
# Market alert scanner. Checks multiple trigger types every N minutes
# during market hours. Fires Mac desktop notifications + Gmail.
#
# Trigger types:
#   1. Geopolitical shock keywords in Google News RSS
#   2. Fed/macro surprise keywords
#   3. Single stock crash/surge >5% intraday
#   4. Sentiment score drops below threshold (-0.7 default)
#   5. Commodity price move (oil, gold, silver >2%)
#
# Zero Streamlit imports. Run as standalone process:
#   python -m alerts.scanner                    # runs continuously during market hours
#   python -m alerts.scanner --once             # single scan and exit
#   python -m alerts.scanner --interval 15      # scan every 15 minutes
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import smtplib
import sqlite3
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import feedparser
import pandas as pd
import requests
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_DB    = Path(os.getenv("ALERTS_DB_PATH", "alerts.db"))
ET_TZ        = ZoneInfo("America/New_York")
MARKET_OPEN  = 9 * 60 + 30   # 9:30 AM ET in minutes
MARKET_CLOSE = 16 * 60        # 4:00 PM ET in minutes

# Default thresholds
DEFAULT_PRICE_MOVE_PCT    = 5.0    # % intraday move to trigger stock alert
DEFAULT_SENTIMENT_THRESH  = -0.70  # sentiment below this triggers alert
DEFAULT_COMMODITY_PCT     = 2.0    # % move in oil/gold/silver to trigger alert
DEFAULT_INTERVAL_MIN      = 30     # scan every 30 minutes by default

# Commodity tickers
COMMODITY_TICKERS = {
    "CL=F":     "Crude Oil (WTI)",
    "GC=F":     "Gold (Futures)",
    "AAAU":     "Gold (ETF)",
    "SI=F":     "Silver (Futures)",
    "SLV":      "Silver (ETF)",
    "NG=F":     "Natural Gas",
    "ZW=F":     "Wheat",
    "DX-Y.NYB": "US Dollar Index",
}

# ── Keyword triggers ──────────────────────────────────────────────────────────
GEOPOLITICAL_KEYWORDS = [
    "war", "invasion", "missile", "airstrike", "sanctions", "embargo",
    "nuclear", "escalation", "conflict", "coup", "assassination",
    "iran", "russia", "ukraine", "china taiwan", "north korea",
    "oil surge", "energy crisis", "opec", "supply shock",
    "tariff", "trade war", "export ban", "chip ban",
]

FED_MACRO_KEYWORDS = [
    "federal reserve", "fed rate", "fomc", "interest rate",
    "inflation", "cpi", "pce", "jobs report", "nonfarm payroll",
    "recession", "gdp", "yield curve", "treasury", "bond sell",
    "rate hike", "rate cut", "quantitative", "tightening", "easing",
    "powell", "yellen", "waller", "barr",
]

CRASH_KEYWORDS = [
    "crash", "collapse", "plunge", "tumble", "selloff", "sell-off",
    "circuit breaker", "halted", "suspended", "bankruptcy", "default",
    "margin call", "liquidation", "flash crash",
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(ALERTS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            trigger_type TEXT   NOT NULL,
            severity    TEXT    NOT NULL DEFAULT 'MEDIUM',
            ticker      TEXT,
            headline    TEXT,
            detail      TEXT,
            value       REAL,
            notified    INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _already_alerted(conn: sqlite3.Connection, trigger_type: str,
                     ticker: Optional[str], window_minutes: int = 60) -> bool:
    """Avoid duplicate alerts — check if same trigger fired within window."""
    cutoff = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
    row = conn.execute("""
        SELECT id FROM alerts
        WHERE trigger_type = ? AND (ticker = ? OR ticker IS NULL)
        AND timestamp >= ? AND notified = 1
    """, (trigger_type, ticker, cutoff)).fetchone()
    return row is not None


def _save_alert(conn: sqlite3.Connection, trigger_type: str, severity: str,
                headline: str, detail: str,
                ticker: Optional[str] = None, value: float = 0.0) -> int:
    cur = conn.execute("""
        INSERT INTO alerts (timestamp, trigger_type, severity, ticker,
                            headline, detail, value, notified)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
    """, (datetime.utcnow().isoformat(), trigger_type, severity,
          ticker, headline, detail, value))
    conn.commit()
    return cur.lastrowid


# ══════════════════════════════════════════════════════════════════════════════
#  NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _mac_notify(title: str, message: str, severity: str = "MEDIUM") -> None:
    """Send Mac desktop notification via osascript."""
    emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(severity, "⚠️")
    full_title = f"{emoji} ML Quant Alert"
    try:
        script = (
            f'display notification "{message}" '
            f'with title "{full_title}" '
            f'subtitle "{title}" '
            f'sound name "Basso"'
        )
        subprocess.run(["osascript", "-e", script], check=True, timeout=5)
    except Exception as e:
        print(f"  ⚠ Mac notification failed: {e}")


def _gmail_alert(subject: str, body: str) -> None:
    """Send Gmail alert using credentials from environment/secrets."""
    sender   = os.getenv("EMAIL_SENDER", "")
    receiver = os.getenv("EMAIL_RECEIVER", "")
    password = os.getenv("EMAIL_PASSWORD", "")

    # Try streamlit secrets if env vars not set
    if not sender:
        try:
            import streamlit as st
            sender   = st.secrets.get("EMAIL_SENDER", "")
            receiver = st.secrets.get("EMAIL_RECEIVER", "")
            password = st.secrets.get("EMAIL_PASSWORD", "")
        except Exception:
            pass

    if not all([sender, receiver, password]):
        print("  ⚠ Gmail not configured — skipping email alert")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = receiver

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px;">
        <h2 style="color:#ff4444;">🚨 ML Quant Fund Alert</h2>
        <pre style="background:#1a1a1a;padding:15px;border-radius:8px;
                    border-left:4px solid #ff4444;white-space:pre-wrap;">{body}</pre>
        <p style="color:#888;font-size:12px;">Sent by ML Quant Fund Alert Scanner · {datetime.now():%Y-%m-%d %H:%M ET}</p>
        </body></html>
        """
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(sender, password)
            s.send_message(msg)
        print(f"  ✓ Email sent: {subject}")
    except Exception as e:
        print(f"  ⚠ Gmail failed: {e}")


def fire_alert(title: str, body: str, severity: str = "MEDIUM",
               send_email: bool = True) -> None:
    """Fire both Mac notification and email."""
    print(f"\n{'🚨' if severity=='HIGH' else '⚠️'} ALERT [{severity}]: {title}")
    print(f"   {body}")
    _mac_notify(title, body[:200], severity)
    if send_email and severity in ("HIGH", "MEDIUM"):
        _gmail_alert(f"[{severity}] {title}", f"{title}\n\n{body}\n\nTime: {datetime.now():%Y-%m-%d %H:%M ET}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRIGGER CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_news_keywords(conn: sqlite3.Connection) -> list[dict]:
    """Scan Google News RSS for geopolitical + macro + crash keywords."""
    alerts = []
    feeds  = [
        ("https://news.google.com/rss/search?q=stock+market+geopolitical&hl=en-US&gl=US&ceid=US:en", "geo"),
        ("https://news.google.com/rss/search?q=federal+reserve+interest+rate&hl=en-US&gl=US&ceid=US:en", "fed"),
        ("https://news.google.com/rss/search?q=oil+price+surge+war&hl=en-US&gl=US&ceid=US:en", "commodity"),
        ("https://news.google.com/rss/search?q=stock+market+crash+plunge&hl=en-US&gl=US&ceid=US:en", "crash"),
    ]

    seen_titles = set()

    for url, feed_type in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = (entry.get("title") or "").lower()
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                matched_geo   = [k for k in GEOPOLITICAL_KEYWORDS if k in title]
                matched_fed   = [k for k in FED_MACRO_KEYWORDS     if k in title]
                matched_crash = [k for k in CRASH_KEYWORDS          if k in title]

                for keywords, ttype, severity in [
                    (matched_geo,   "GEOPOLITICAL", "HIGH"),
                    (matched_fed,   "FED_MACRO",    "MEDIUM"),
                    (matched_crash, "MARKET_CRASH", "HIGH"),
                ]:
                    if keywords and not _already_alerted(conn, ttype, None, window_minutes=120):
                        headline = entry.get("title", "")[:200]
                        detail   = f"Keywords: {', '.join(keywords[:3])}\nSource: {entry.get('link','')}"
                        _save_alert(conn, ttype, severity, headline, detail)
                        alerts.append({
                            "type": ttype, "severity": severity,
                            "title": f"{ttype.replace('_',' ').title()} Alert",
                            "body":  f"{headline}\n\n{detail}",
                        })
                        break  # one alert per entry
        except Exception as e:
            print(f"  ⚠ News feed error ({feed_type}): {e}")

    return alerts


def check_stock_moves(conn: sqlite3.Connection, tickers: list[str],
                      threshold_pct: float = DEFAULT_PRICE_MOVE_PCT) -> list[dict]:
    """Check for intraday moves > threshold_pct for each ticker."""
    alerts = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1d", interval="5m",
                               progress=False, auto_adjust=True)
            if data.empty or len(data) < 2:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # Ensure Close column exists after MultiIndex fix
            if "Close" not in data.columns:
                close_cols = [c for c in data.columns if "close" in str(c).lower()]
                if not close_cols:
                    continue
                data = data.rename(columns={close_cols[0]: "Close"})

            open_price  = float(data["Close"].iloc[0])
            latest      = float(data["Close"].iloc[-1])
            pct_change  = (latest - open_price) / open_price * 100

            if abs(pct_change) >= threshold_pct:
                direction = "surged" if pct_change > 0 else "crashed"
                severity  = "HIGH" if abs(pct_change) >= 8 else "MEDIUM"
                ttype     = f"PRICE_MOVE_{ticker}"

                if not _already_alerted(conn, ttype, ticker, window_minutes=60):
                    headline = f"{ticker} {direction} {pct_change:+.1f}% intraday"
                    detail   = (
                        f"Ticker:  {ticker}\n"
                        f"Open:    ${open_price:.2f}\n"
                        f"Current: ${latest:.2f}\n"
                        f"Change:  {pct_change:+.1f}%\n"
                        f"Time:    {datetime.now(ET_TZ):%H:%M ET}"
                    )
                    _save_alert(conn, ttype, severity, headline, detail,
                                ticker=ticker, value=pct_change)
                    alerts.append({
                        "type": ttype, "severity": severity,
                        "title": headline, "body": detail,
                    })
        except Exception as e:
            print(f"  ⚠ Price check failed for {ticker}: {e}")

    return alerts


def check_commodities(conn: sqlite3.Connection,
                      threshold_pct: float = DEFAULT_COMMODITY_PCT) -> list[dict]:
    """Check oil, gold, silver, natural gas for big moves."""
    alerts = []

    for sym, name in COMMODITY_TICKERS.items():
        try:
            # Use daily data for reliable prev close vs current price
            data = yf.download(sym, period="5d", interval="1d",
                               progress=False, auto_adjust=True)
            if data.empty or len(data) < 2:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            prev_close = float(data["Close"].iloc[-2])  # yesterday close
            latest     = float(data["Close"].iloc[-1])  # today close
            pct_change = (latest - prev_close) / prev_close * 100

            if abs(pct_change) >= threshold_pct:
                direction = "UP" if pct_change > 0 else "DOWN"
                severity  = "HIGH" if abs(pct_change) >= 4 else "MEDIUM"
                ttype     = f"COMMODITY_{sym.replace('=','').replace('-','')}"

                if not _already_alerted(conn, ttype, sym, window_minutes=120):
                    headline = f"{name} {direction} {pct_change:+.1f}%"
                    detail   = (
                        f"Commodity: {name} ({sym})\n"
                        f"Previous:  ${prev_close:.2f}\n"
                        f"Current:   ${latest:.2f}\n"
                        f"Change:    {pct_change:+.1f}%\n"
                        f"Impact:    {'Energy stocks, transportation, inflation' if 'Oil' in name or 'Gas' in name else 'Risk-off sentiment, USD strength' if 'Gold' in name else 'Industrial demand signal'}"
                    )
                    _save_alert(conn, ttype, severity, headline, detail,
                                ticker=sym, value=pct_change)
                    alerts.append({
                        "type": ttype, "severity": severity,
                        "title": headline, "body": detail,
                    })
        except Exception as e:
            print(f"  ⚠ Commodity check failed for {sym}: {e}")

    return alerts


def check_sentiment_threshold(conn: sqlite3.Connection, tickers: list[str],
                               threshold: float = DEFAULT_SENTIMENT_THRESH) -> list[dict]:
    """Alert when sentiment drops below threshold for any ticker."""
    from data.etl_sentiment import get_sentiment_score

    alerts = []
    for ticker in tickers:
        try:
            score = get_sentiment_score(ticker)
            if score == 0.0:
                continue  # no data yet

            if score <= threshold:
                severity = "HIGH" if score <= -0.85 else "MEDIUM"
                ttype    = f"SENTIMENT_{ticker}"

                if not _already_alerted(conn, ttype, ticker, window_minutes=180):
                    headline = f"{ticker} sentiment extremely negative: {score:+.3f}"
                    detail   = (
                        f"Ticker:    {ticker}\n"
                        f"Score:     {score:+.3f} (threshold: {threshold:+.3f})\n"
                        f"Meaning:   Strong negative news flow — consider avoiding BUY\n"
                        f"Action:    Check news, review position size"
                    )
                    _save_alert(conn, ttype, severity, headline, detail,
                                ticker=ticker, value=score)
                    alerts.append({
                        "type": ttype, "severity": severity,
                        "title": headline, "body": detail,
                    })
        except Exception as e:
            print(f"  ⚠ Sentiment check failed for {ticker}: {e}")

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
#  MARKET HOURS CHECK
# ══════════════════════════════════════════════════════════════════════════════

def is_market_hours() -> bool:
    """Return True if current ET time is within market hours Mon-Fri."""
    now = datetime.now(ET_TZ)
    if now.weekday() >= 5:   # Saturday/Sunday
        return False
    minutes = now.hour * 60 + now.minute
    return MARKET_OPEN <= minutes <= MARKET_CLOSE


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SCAN
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(
    tickers:         list[str],
    price_threshold: float = DEFAULT_PRICE_MOVE_PCT,
    sent_threshold:  float = DEFAULT_SENTIMENT_THRESH,
    commodity_pct:   float = DEFAULT_COMMODITY_PCT,
    send_email:      bool  = True,
    verbose:         bool  = True,
) -> list[dict]:
    """
    Run all trigger checks once. Returns list of alerts fired.
    Called by the CLI loop and by the Streamlit page manual scan button.
    """
    conn   = _init_db()
    fired  = []
    now_et = datetime.now(ET_TZ).strftime("%H:%M ET")

    if verbose:
        print(f"\n[{now_et}] Running market alert scan...")

    # 1. News keywords
    news_alerts = check_news_keywords(conn)
    fired.extend(news_alerts)

    # 2. Stock price moves
    price_alerts = check_stock_moves(conn, tickers, threshold_pct=price_threshold)
    fired.extend(price_alerts)

    # 3. Commodity moves
    commodity_alerts = check_commodities(conn, threshold_pct=commodity_pct)
    fired.extend(commodity_alerts)

    # 4. Sentiment threshold
    sent_alerts = check_sentiment_threshold(conn, tickers, threshold=sent_threshold)
    fired.extend(sent_alerts)

    conn.close()

    # Fire notifications
    for alert in fired:
        fire_alert(
            title=alert["title"],
            body=alert["body"],
            severity=alert["severity"],
            send_email=send_email,
        )

    if verbose:
        if fired:
            print(f"  → {len(fired)} alert(s) fired")
        else:
            print(f"  → No alerts. Market normal.")

    return fired


def load_recent_alerts(hours: int = 24) -> pd.DataFrame:
    """Load recent alerts from DB for the dashboard page."""
    if not ALERTS_DB.exists():
        return pd.DataFrame()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    try:
        conn = sqlite3.connect(ALERTS_DB)
        df   = pd.read_sql("""
            SELECT timestamp, trigger_type, severity, ticker,
                   headline, detail, value
            FROM alerts
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff,))
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — continuous loop or single scan
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from pathlib import Path as P

    # Load tickers from file
    def _load_tickers():
        p = P("tickers.txt")
        if p.exists():
            return [t.strip().upper() for t in p.read_text().splitlines() if t.strip()]
        return ["AAPL", "NVDA", "TSLA", "AMD", "GOOG", "META"]

    parser = argparse.ArgumentParser(description="Market alert scanner")
    parser.add_argument("--once",       action="store_true",
                        help="Run once and exit (default: continuous loop)")
    parser.add_argument("--interval",   type=int, default=DEFAULT_INTERVAL_MIN,
                        help=f"Scan interval in minutes (default: {DEFAULT_INTERVAL_MIN})")
    parser.add_argument("--tickers",    nargs="+", default=None,
                        help="Tickers to monitor (default: all from tickers.txt)")
    parser.add_argument("--price-pct",  type=float, default=DEFAULT_PRICE_MOVE_PCT,
                        help=f"Price move threshold % (default: {DEFAULT_PRICE_MOVE_PCT})")
    parser.add_argument("--sent-thresh",type=float, default=DEFAULT_SENTIMENT_THRESH,
                        help=f"Sentiment threshold (default: {DEFAULT_SENTIMENT_THRESH})")
    parser.add_argument("--no-email",   action="store_true",
                        help="Disable email alerts (Mac notifications only)")
    parser.add_argument("--market-only",action="store_true", default=True,
                        help="Only scan during market hours (default: True)")
    args = parser.parse_args()

    tickers = args.tickers or _load_tickers()

    print(f"🚨 ML Quant Alert Scanner")
    print(f"   Tickers:    {len(tickers)} stocks + {len(COMMODITY_TICKERS)} commodities")
    print(f"   Interval:   every {args.interval} min")
    print(f"   Price alert: >{args.price_pct}% move")
    print(f"   Sentiment:  below {args.sent_thresh}")
    print(f"   Email:      {'disabled' if args.no_email else 'enabled'}")
    print(f"   Market hrs: {'yes' if args.market_only else 'no'}")

    if args.once:
        run_scan(tickers, send_email=not args.no_email)
    else:
        print(f"\nRunning continuously. Ctrl+C to stop.\n")
        while True:
            if args.market_only and not is_market_hours():
                now = datetime.now(ET_TZ)
                print(f"[{now:%H:%M ET}] Outside market hours — sleeping 5 min...")
                time.sleep(300)
                continue

            run_scan(tickers,
                     price_threshold=args.price_pct,
                     sent_threshold=args.sent_thresh,
                     send_email=not args.no_email)

            print(f"  Next scan in {args.interval} minutes...")
            time.sleep(args.interval * 60)
