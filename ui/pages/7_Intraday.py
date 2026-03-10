"""
Page 7 — Intraday Signals
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from features.intraday_builder import get_intraday_signal, is_market_open, minutes_since_open
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

st.set_page_config(page_title="Intraday Signals", page_icon="⚡", layout="wide")
st.title("⚡ Intraday Signals")
st.caption("1hr / 2hr / 4hr direction signals based on momentum, VWAP, RSI, volume surge.")

now_et = datetime.now(ET)
market_open = is_market_open()
mso = minutes_since_open()

if market_open:
    st.success(f"Market OPEN — {now_et:%H:%M ET} — {mso} min since open")
    st_autorefresh(interval=15 * 60 * 1000, key="intraday_refresh")
else:
    st.warning(f"Market CLOSED — Showing last trading day data · {now_et:%H:%M ET}")

col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    DEFAULT_TICKERS = ["NVDA","AAPL","TSLA","META","MSFT","AMD","PLTR","TSM","NVO","CRWD","SHOP","NFLX"]
    tickers = st.multiselect("Tickers to scan", options=DEFAULT_TICKERS, default=DEFAULT_TICKERS[:8])
with col_ctrl2:
    st.write("")
    st.button("Refresh Now", type="primary", use_container_width=True)

if not tickers:
    st.info("Select at least one ticker.")
    st.stop()

with st.spinner("Fetching intraday data..."):
    signals = [get_intraday_signal(t) for t in tickers]

def fmt_signal(s, p):
    if s == "UP":   return f"UP ({p:.0%})"
    if s == "DOWN": return f"DOWN ({p:.0%})"
    return f"NEUTRAL ({p:.0%})"

st.subheader("Signal Summary")
st.caption("UP = bullish momentum · DOWN = bearish · NEUTRAL = no clear direction")

rows = []
for sig in signals:
    if not sig.get("current_price"):
        continue
    rows.append({
        "Ticker":    sig["ticker"],
        "Price":     f"${sig['current_price']:.2f}",
        "RSI":       f"{sig['rsi_14']:.1f}"       if sig["rsi_14"]     else "—",
        "VWAP Dev":  f"{sig['vwap_dev']:+.2f}%"   if sig["vwap_dev"] is not None else "—",
        "Vol Surge": f"{sig['vol_surge']:.2f}x"   if sig["vol_surge"] else "—",
        "1hr":       fmt_signal(sig["signal_1hr"], sig["prob_1hr"]),
        "2hr":       fmt_signal(sig["signal_2hr"], sig["prob_2hr"]),
        "4hr":       fmt_signal(sig["signal_4hr"], sig["prob_4hr"]),
    })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.warning("No intraday data available.")

st.divider()
st.subheader("Strong Signals (2+ horizons agree)")

strong = [s for s in signals if
    sum(1 for h in ["signal_1hr","signal_2hr","signal_4hr"] if s[h] in ("UP","DOWN")) >= 2
    and s.get("current_price")
]

if strong:
    for s in strong:
        horizons_up   = sum(1 for h in ["signal_1hr","signal_2hr","signal_4hr"] if s[h] == "UP")
        horizons_down = sum(1 for h in ["signal_1hr","signal_2hr","signal_4hr"] if s[h] == "DOWN")
        direction     = "UP" if horizons_up >= horizons_down else "DOWN"
        avg_prob      = sum(s[f"prob_{h}"] for h in ["1hr","2hr","4hr"]) / 3
        icon          = "🟢" if direction == "UP" else "🔴"

        with st.expander(f"{icon} {s['ticker']} — {direction} ({avg_prob:.0%} avg)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price",     f"${s['current_price']:.2f}")
            c2.metric("RSI",       f"{s['rsi_14']:.1f}" if s["rsi_14"] else "—")
            c3.metric("VWAP Dev",  f"{s['vwap_dev']:+.2f}%" if s["vwap_dev"] is not None else "—")
            c4.metric("Vol Surge", f"{s['vol_surge']:.2f}x"  if s["vol_surge"] else "—")
            st.markdown(f"""
| Horizon | Signal | Confidence |
|---------|--------|------------|
| 1 hour  | {fmt_signal(s["signal_1hr"], s["prob_1hr"])} | {"█" * int(s["prob_1hr"]*10)}{"░" * (10-int(s["prob_1hr"]*10))} |
| 2 hours | {fmt_signal(s["signal_2hr"], s["prob_2hr"])} | {"█" * int(s["prob_2hr"]*10)}{"░" * (10-int(s["prob_2hr"]*10))} |
| 4 hours | {fmt_signal(s["signal_4hr"], s["prob_4hr"])} | {"█" * int(s["prob_4hr"]*10)}{"░" * (10-int(s["prob_4hr"]*10))} |
""")
else:
    st.info("No strong directional signals right now.")

st.divider()
with st.expander("How to read Intraday Signals"):
    st.markdown("""
**1hr** — Next hour direction. Best for entry/exit in first 2hrs of trading.

**2hr** — Mid-session direction. Use to decide hold vs exit a morning position.

**4hr** — Rest-of-day direction. Accounts for late-day mean reversion toward VWAP.

**RSI** — Above 70 = overbought (may pull back). Below 30 = oversold (may bounce).

**VWAP Dev** — Positive = price above VWAP = institutions buying = bullish. Negative = bearish.

**Vol Surge** — Above 2x = strong conviction. Below 0.5x = weak, move may not hold.

**When to act:**
- UP on all 3 + Vol Surge > 1.5x + VWAP Dev positive = strong buy
- DOWN on all 3 + RSI > 70 = strong sell/short
- Mixed signals = wait or fade the move
""")

st.caption(f"Last updated: {datetime.now(ET):%Y-%m-%d %H:%M:%S ET} · Auto-refresh every 15min when market open")
