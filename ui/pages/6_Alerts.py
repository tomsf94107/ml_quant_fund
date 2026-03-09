# ui/6_Alerts.py
# Market Alert Center — live feed of triggered alerts, manual scan, settings.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import shutil, os
from datetime import datetime, date
import pytz

def _auto_clear_stale_cache():
    """Clear commodity/options cache if it's from a previous trading day."""
    ET = pytz.timezone("America/New_York")
    now_et = datetime.now(ET)
    cache_dirs = ["data/cache/options", "data/cache/commodities"]
    for cd in cache_dirs:
        if not os.path.exists(cd):
            continue
        for f in os.listdir(cd):
            fpath = os.path.join(cd, f)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=ET)
            # If file is from a previous calendar day, delete it
            if mtime.date() < now_et.date():
                os.remove(fpath)

_auto_clear_stale_cache()
from streamlit_autorefresh import st_autorefresh

from alerts.scanner import (
    run_scan, load_recent_alerts, is_market_hours,
    COMMODITY_TICKERS,
    DEFAULT_PRICE_MOVE_PCT, DEFAULT_SENTIMENT_THRESH, DEFAULT_COMMODITY_PCT,
    DEFAULT_INTERVAL_MIN,
)

st.set_page_config(page_title="Alert Center", page_icon="🚨", layout="wide")
st.title("🚨 Market Alert Center")
st.caption("Real-time monitoring: geopolitical shocks, Fed surprises, price crashes, sentiment collapse, commodity moves.")

# ── Load tickers ──────────────────────────────────────────────────────────────
def _load_tickers() -> list[str]:
    p = Path(_ROOT) / "tickers.txt"
    if p.exists():
        return [t.strip().upper() for t in p.read_text().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "TSLA", "AMD"]


all_tickers = _load_tickers()

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Alert Settings")

    price_threshold = st.slider(
        "Stock price move alert (%)",
        min_value=2.0, max_value=15.0,
        value=DEFAULT_PRICE_MOVE_PCT, step=0.5,
        help="Alert when any ticker moves this % intraday"
    )
    sent_threshold = st.slider(
        "Sentiment alert threshold",
        min_value=-1.0, max_value=-0.3,
        value=DEFAULT_SENTIMENT_THRESH, step=0.05,
        help="Alert when sentiment drops below this level"
    )
    commodity_threshold = st.slider(
        "Commodity move alert (%)",
        min_value=1.0, max_value=8.0,
        value=DEFAULT_COMMODITY_PCT, step=0.5,
        help="Alert on oil/gold/silver moves"
    )

    send_email = st.toggle("📧 Send email alerts", value=True)
    market_only = st.toggle("🕐 Market hours only", value=True,
                             help="Only auto-scan 9:30am–4:00pm ET")

    st.markdown("---")
    st.markdown("## 📋 Monitor These Tickers")
    watch_tickers = st.multiselect(
        "Tickers to watch",
        options=all_tickers,
        default=all_tickers,
    )

    st.markdown("---")
    st.markdown("## 🔄 Auto-refresh")
    auto_interval = st.selectbox(
        "Auto-scan interval",
        options=[0, 15, 30, 60],
        format_func=lambda x: "Off" if x == 0 else f"Every {x} min",
        index=2,
    )

    if auto_interval > 0:
        st_autorefresh(interval=auto_interval * 60 * 1000, key="alert_refresh")


# ── Market status banner ──────────────────────────────────────────────────────
col_status, col_time = st.columns([3, 1])
with col_status:
    if is_market_hours():
        st.success("🟢 Market OPEN — Scanner active")
    else:
        st.warning("🔴 Market CLOSED — Manual scan available anytime")
with col_time:
    st.caption(f"Last checked: {datetime.now():%H:%M:%S}")


# ── Manual scan button ────────────────────────────────────────────────────────
st.subheader("🔍 Manual Scan")
st.caption("Use this when you see breaking news — don't wait for the auto-scan.")

col1, col2, col3 = st.columns(3)

with col1:
    scan_news      = st.checkbox("📰 News keywords", value=True)
with col2:
    scan_prices    = st.checkbox("📈 Stock prices", value=True)
with col3:
    scan_commodity = st.checkbox("🛢️ Commodities", value=True)

if st.button("🚨 Scan Now", type="primary", use_container_width=True):
    with st.spinner("Scanning markets..."):
        try:
            alerts = run_scan(
                tickers=watch_tickers,
                price_threshold=price_threshold,
                sent_threshold=sent_threshold,
                commodity_pct=commodity_threshold,
                send_email=send_email,
                verbose=False,
            )

            if alerts:
                for alert in alerts:
                    sev = alert["severity"]
                    if sev == "HIGH":
                        st.error(f"🚨 **{alert['title']}**\n\n{alert['body']}")
                    else:
                        st.warning(f"⚠️ **{alert['title']}**\n\n{alert['body']}")
            else:
                st.success("✅ No alerts — markets look normal right now.")
        except Exception as e:
            st.error(f"Scan failed: {e}")

st.divider()


# ── Live commodity prices ─────────────────────────────────────────────────────
from datetime import datetime
import pytz
ET = pytz.timezone("America/New_York")
_now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
st.subheader("🛢️ Live Commodity Prices")
st.caption(f"Last fetched: {_now}")
st.caption("🟢 RISK ON = investors confident, buying stocks & commodities → bullish  ·  🔴 RISK OFF = investors scared, fleeing to safety → bearish  ·  ⚪ Neutral = move below threshold")

try:
    import yfinance as yf
    syms = list(COMMODITY_TICKERS.keys())
    rows = []
    for sym in syms:
        try:
            data = yf.download(sym, period="5d", interval="1d",
                               progress=False, auto_adjust=True)
            if data.empty:
                continue
            if isinstance(data.columns, __import__('pandas').MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    data.columns = data.columns.get_level_values(0)
                elif 'Close' in data.columns.get_level_values(1):
                    data.columns = data.columns.get_level_values(1)
                else:
                    # Flatten and pick Close column explicitly
                    data.columns = ['_'.join([str(c) for c in col]).strip('_') for col in data.columns]
                    close_cols = [c for c in data.columns if 'close' in c.lower()]
                    if close_cols:
                        data = data.rename(columns={close_cols[0]: 'Close'})
            prev  = float(data["Close"].iloc[-2])  # last business day
            curr  = float(data["Close"].iloc[-1])   # today
            chg   = (curr - prev) / prev * 100
            rows.append({
                "Commodity": COMMODITY_TICKERS[sym],
                "Price":     f"${curr:.2f}",
                "Change":    f"{chg:+.2f}%",
                "Signal":    "🔴 RISK OFF" if chg <= -commodity_threshold
                             else "🟢 RISK ON" if chg >= commodity_threshold
                             else "⚪ Neutral",
            })
        except Exception:
            pass

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Commodity data unavailable.")
except Exception as e:
    st.warning(f"Commodity feed error: {e}")

st.divider()


# ── Recent alert history ──────────────────────────────────────────────────────
st.subheader("📋 Alert History")

hours_back = st.selectbox("Show alerts from last", [6, 12, 24, 48, 72],
                           format_func=lambda x: f"{x} hours", index=2)

alerts_df = load_recent_alerts(hours=hours_back)

if alerts_df.empty:
    st.info("No alerts in the selected time window. Run a scan to start monitoring.")
else:
    # Severity color coding
    def _sev_badge(sev: str) -> str:
        return {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}.get(sev, sev)

    alerts_df["severity"] = alerts_df["severity"].apply(_sev_badge)
    alerts_df["timestamp"] = alerts_df["timestamp"].dt.strftime("%m-%d %H:%M")

    st.dataframe(
        alerts_df[["timestamp", "severity", "trigger_type", "ticker",
                   "headline", "value"]].rename(columns={
            "trigger_type": "type",
            "headline":     "description",
            "value":        "magnitude",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Expandable detail view
    for _, row in alerts_df.head(5).iterrows():
        with st.expander(f"{row['timestamp']} — {row['headline'][:80]}"):
            st.text(row["detail"])

st.divider()


# ── Setup guide ───────────────────────────────────────────────────────────────
with st.expander("🛠️ Setup: Run scanner as background process"):
    st.markdown("""
**Option 1 — Run in a separate terminal (recommended):**
```bash
cd ~/Desktop/ML_Quant_Fund
python -m alerts.scanner --interval 30
```
This runs continuously during market hours, firing Mac notifications + email.

**Option 2 — Single scan (for testing):**
```bash
python -m alerts.scanner --once
```

**Option 3 — Schedule with launchd (Mac, runs automatically on login):**
```bash
# Create a launchd plist to auto-start the scanner
cat > ~/Library/LaunchAgents/com.mlquant.scanner.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlquant.scanner</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python</string>
        <string>-m</string>
        <string>alerts.scanner</string>
        <string>--interval</string>
        <string>30</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/atomnguyen/Desktop/ML_Quant_Fund</string>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/mlquant_scanner.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/mlquant_scanner_err.log</string>
</dict>
</plist>
EOF
launchctl load ~/Library/LaunchAgents/com.mlquant.scanner.plist
```

**Gmail setup** (if not already done):
1. Go to myaccount.google.com/apppasswords
2. Generate an App Password for "Mail"
3. Add to `.streamlit/secrets.toml`:
```toml
EMAIL_SENDER   = "your@gmail.com"
EMAIL_RECEIVER = "your@gmail.com"
EMAIL_PASSWORD = "your-app-password"
```
""")
