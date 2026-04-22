#!/usr/bin/env python3
"""
scripts/intraday_kill_switch.py
Intraday kill switch — runs every 5 minutes during market hours.
Monitors 4 signals for early warning of big drops.
Alert triggers when 2+ conditions fire simultaneously.
"""
import os
import sys
import logging
import subprocess
from datetime import datetime, date
from pathlib import Path

import requests
import yfinance as yf
import pandas as pd
import pytz

ROOT   = Path(__file__).parent.parent
LOG    = ROOT / "logs" / "kill_switch_alerts.log"
LOG.parent.mkdir(exist_ok=True)

ET     = pytz.timezone("America/New_York")
UW_KEY = os.getenv("UW_API_KEY", "")
HDRS   = {"Authorization": f"Bearer {UW_KEY}"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("kill_switch")


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


def send_alert(title: str, message: str, level: str = "WARNING"):
    log.warning(f"ALERT [{level}] — {title}: {message}")
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "🚨 {title}" sound name "Basso"'
        ], timeout=5)
    except Exception:
        pass
    try:
        import sqlite3
        with sqlite3.connect(ROOT / "accuracy.db", timeout=30) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kill_switch_alerts (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    fired_at TEXT NOT NULL,
                    title    TEXT NOT NULL,
                    message  TEXT NOT NULL,
                    level    TEXT NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO kill_switch_alerts (fired_at, title, message, level) VALUES (?,?,?,?)",
                (datetime.now().isoformat(), title, message, level)
            )
            conn.commit()
    except Exception:
        pass


def check_vix_spike() -> tuple:
    try:
        vix = yf.download("^VIX", period="1d", interval="1m", progress=False, auto_adjust=True)
        if vix.empty or len(vix) < 5:
            return False, "VIX: no data"
        if hasattr(vix.columns, "get_level_values"):
            vix.columns = vix.columns.get_level_values(0)
        close   = vix["Close"].dropna()
        current = float(close.iloc[-1])
        prev_5m = float(close.iloc[-5])
        change  = (current - prev_5m) / prev_5m if prev_5m > 0 else 0
        fired   = change > 0.10
        return fired, f"VIX: {current:.1f} ({change:+.1%} in 5min)"
    except Exception as e:
        return False, f"VIX: error ({e})"


def check_market_tide() -> tuple:
    try:
        r = requests.get("https://api.unusualwhales.com/api/market/market-tide",
                         headers=HDRS, timeout=8)
        if r.status_code != 200:
            return False, f"Tide: API error {r.status_code}"
        data   = r.json().get("data", [])
        today  = str(date.today())
        recent = [d for d in data if d.get("date","")[:10] == today][-3:]
        if not recent:
            return False, "Tide: no data today"
        net_call = sum(float(d.get("net_call_premium", 0) or 0) for d in recent)
        net_put  = sum(float(d.get("net_put_premium",  0) or 0) for d in recent)
        total    = abs(net_call) + abs(net_put)
        score    = (net_call + net_put) / total if total > 0 else 0
        fired    = score < -0.30
        return fired, f"Tide: score={score:.2f} calls=${net_call/1e6:.1f}M puts=${abs(net_put)/1e6:.1f}M"
    except Exception as e:
        return False, f"Tide: error ({e})"


def check_put_sweeps() -> tuple:
    try:
        r = requests.get("https://api.unusualwhales.com/api/option-trades/flow-alerts",
                         headers=HDRS, timeout=8)
        if r.status_code != 200:
            return False, f"Sweeps: API error {r.status_code}"
        alerts = r.json().get("data", [])
        put_sweeps = [
            a for a in alerts
            if a.get("type","").lower() == "put"
            and a.get("has_sweep", False)
            and a.get("ticker","").upper() in ("SPY","QQQ","SPX")
            and float(a.get("total_premium", 0) or 0) > 500_000
        ]
        put_prem = sum(float(a.get("total_premium", 0) or 0) for a in put_sweeps)
        fired    = len(put_sweeps) >= 2
        return fired, f"Sweeps: {len(put_sweeps)} large put sweeps (${put_prem/1e6:.1f}M)"
    except Exception as e:
        return False, f"Sweeps: error ({e})"


def check_spy_vwap() -> tuple:
    try:
        spy = yf.download("SPY", period="1d", interval="5m", progress=False, auto_adjust=True)
        if spy.empty or len(spy) < 10:
            return False, "VWAP: no data"
        if hasattr(spy.columns, "get_level_values"):
            spy.columns = spy.columns.get_level_values(0)
        spy  = spy.dropna(subset=["Close"])
        vwap = (spy["Close"] * spy["Volume"]).cumsum() / spy["Volume"].cumsum()
        price  = float(spy["Close"].iloc[-1])
        vwap_v = float(vwap.iloc[-1])
        dev    = (price - vwap_v) / vwap_v
        ret15  = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-3] - 1) if len(spy) >= 3 else 0
        fired  = dev < -0.005 and ret15 < -0.003
        return fired, f"VWAP: SPY=${price:.2f} VWAP=${vwap_v:.2f} dev={dev:.2%} 15m={ret15:.2%}"
    except Exception as e:
        return False, f"VWAP: error ({e})"


def run():
    if not is_market_open():
        log.info("Market closed — skipping")
        return

    now = datetime.now(ET).strftime("%H:%M ET")
    log.info(f"Kill switch check — {now}")

    v_fired, v_detail = check_vix_spike()
    t_fired, t_detail = check_market_tide()
    s_fired, s_detail = check_put_sweeps()
    w_fired, w_detail = check_spy_vwap()

    fired_count = sum([v_fired, t_fired, s_fired, w_fired])

    log.info(f"  {'🔴' if v_fired else '🟢'} {v_detail}")
    log.info(f"  {'🔴' if t_fired else '🟢'} {t_detail}")
    log.info(f"  {'🔴' if s_fired else '🟢'} {s_detail}")
    log.info(f"  {'🔴' if w_fired else '🟢'} {w_detail}")
    log.info(f"  Signals fired: {fired_count}/4")

    if fired_count >= 3:
        send_alert("CRITICAL — Market Drop Warning",
                   f"{fired_count}/4 kill switch signals fired. Potential large drop imminent.",
                   "CRITICAL")
    elif fired_count >= 2:
        send_alert("WARNING — Bearish Signals",
                   f"{fired_count}/4 kill switch signals fired. Monitor closely.",
                   "WARNING")
    else:
        log.info("  OK No alert — market conditions normal")


if __name__ == "__main__":
    run()
