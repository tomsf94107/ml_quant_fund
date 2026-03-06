# ui/pages/7_Broker.py
# Broker Integration Dashboard — connects Alpaca + Robinhood, shows ML signal alerts.
# Read-only: no orders are placed.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Broker Dashboard", page_icon="🏦", layout="wide")

# ── Lazy imports ──────────────────────────────────────────────────────────────
@st.cache_resource
def _get_alpaca():
    from brokers.alpaca_client import AlpacaClient
    return AlpacaClient()

@st.cache_resource
def _get_robinhood():
    from brokers.robinhood_client import RobinhoodClient
    return RobinhoodClient()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🏦 Broker Dashboard")
st.caption("Read-only view: ML signals vs your actual holdings. No orders are placed.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon              = st.selectbox("Signal horizon", [1, 3, 5],
                                         format_func=lambda x: f"{x}d")
    confidence_threshold = st.slider("Min confidence threshold", 0.50, 0.95, 0.55, 0.01)
    high_conf_threshold  = st.slider("High-confidence BUY threshold", 0.60, 0.95, 0.70, 0.01)
    include_hold_alerts  = st.toggle("Alert on HOLD signals you hold", value=True)

    st.divider()
    st.markdown("## 🔑 Broker Status")

    # Alpaca status
    alpaca = _get_alpaca()
    if alpaca.is_configured():
        st.success("✓ Alpaca configured")
    else:
        st.warning("⚠ Alpaca not configured")
        st.caption("Add to `.streamlit/secrets.toml`:\n```\nALPACA_API_KEY = '...'\nALPACA_SECRET_KEY = '...'\nALPACA_PAPER = true\n```")

    # Robinhood status
    rh = _get_robinhood()
    if rh.is_configured():
        st.success("✓ Robinhood configured")
    else:
        st.warning("⚠ Robinhood not configured")
        st.caption("Add to `.streamlit/secrets.toml`:\n```\nROBINHOOD_USERNAME = 'email'\nROBINHOOD_PASSWORD = '...'\nROBINHOOD_MFA_SECRET = '...'\n```")

    st.divider()
    refresh = st.button("🔄 Refresh All", type="primary")

# ── Load signals ──────────────────────────────────────────────────────────────
st.subheader("📡 Step 1 — Load Latest ML Signals")

signals_df = st.session_state.get("broker_signals_df", pd.DataFrame())

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("🚀 Run Strategy Now", help="Runs the ML model on all tickers"):
        with st.spinner("Generating signals..."):
            try:
                from signals.generator import SignalGenerator
                gen = SignalGenerator()

                # Load tickers
                tickers_path = os.path.join(_ROOT, "tickers.txt")
                if os.path.exists(tickers_path):
                    tickers = [t.strip().upper() for t in
                                open(tickers_path).read().splitlines() if t.strip()]
                else:
                    tickers = ["AAPL", "NVDA", "TSLA", "AMD"]

                rows = []
                for ticker in tickers:
                    try:
                        sig = gen.generate(ticker, horizon=horizon)
                        if sig:
                            rows.append({
                                "ticker":     ticker,
                                "signal":     sig.get("signal", "HOLD"),
                                "prob_up":    sig.get("prob_up", 0.5),
                                "confidence": sig.get("confidence", "LOW"),
                                "blocked":    sig.get("blocked", False),
                            })
                    except Exception:
                        continue

                signals_df = pd.DataFrame(rows)
                st.session_state["broker_signals_df"] = signals_df
                buy_count = (signals_df["signal"] == "BUY").sum() if not signals_df.empty else 0
                st.success(f"✓ {len(signals_df)} signals generated — {buy_count} BUY")
            except Exception as e:
                st.error(f"Signal generation failed: {e}")

with col2:
    if not signals_df.empty:
        buy_n  = (signals_df["signal"] == "BUY").sum()
        hold_n = (signals_df["signal"] == "HOLD").sum()
        st.metric("BUY signals",  buy_n)
        st.metric("HOLD signals", hold_n)

if not signals_df.empty:
    with st.expander("📋 All signals"):
        display = signals_df.copy()
        display["prob_up"] = display["prob_up"].map("{:.1%}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# ── Load broker positions ─────────────────────────────────────────────────────
st.subheader("💼 Step 2 — Broker Positions")

col_alp, col_rh = st.columns(2)

alpaca_positions_df    = pd.DataFrame()
robinhood_positions_df = pd.DataFrame()
alpaca_held    = set()
robinhood_held = set()

# Alpaca
with col_alp:
    st.markdown("### 🦙 Alpaca")
    if alpaca.is_configured():
        if st.button("Fetch Alpaca positions") or refresh:
            with st.spinner("Connecting to Alpaca..."):
                acct = alpaca.get_account()
                alpaca_positions_df = alpaca.get_positions_df()
                alpaca_held = alpaca.get_held_tickers()
                if acct:
                    st.metric("Portfolio value", f"${acct.portfolio_value:,.2f}")
                    st.metric("Cash",            f"${acct.cash:,.2f}")
                    mode = "📄 Paper" if acct.is_paper else "💰 Live"
                    st.caption(mode)

        if not alpaca_positions_df.empty:
            display = alpaca_positions_df.copy()
            display["unrealized_plpc"] = display["unrealized_plpc"].map("{:+.1f}%".format)
            display["market_value"]    = display["market_value"].map("${:,.0f}".format)
            display["unrealized_pl"]   = display["unrealized_pl"].map("${:+,.0f}".format)
            st.dataframe(display[["symbol","qty","avg_entry","current_price",
                                   "market_value","unrealized_pl","unrealized_plpc"]],
                         use_container_width=True, hide_index=True)
        elif alpaca.is_configured():
            st.info("No open positions in Alpaca.")
    else:
        st.info("Configure Alpaca credentials in secrets.toml to connect.")

# Robinhood
with col_rh:
    st.markdown("### 🐦 Robinhood")
    if rh.is_configured():
        if st.button("Fetch Robinhood positions") or refresh:
            with st.spinner("Connecting to Robinhood..."):
                acct = rh.get_account()
                robinhood_positions_df = rh.get_positions_df()
                robinhood_held = rh.get_held_tickers()
                if acct:
                    st.metric("Portfolio value", f"${acct.portfolio_value:,.2f}")
                    st.metric("Cash",            f"${acct.cash:,.2f}")

        if not robinhood_positions_df.empty:
            display = robinhood_positions_df.copy()
            display["unrealized_plpc"] = display["unrealized_plpc"].map("{:+.1f}%".format)
            display["market_value"]    = display["market_value"].map("${:,.0f}".format)
            display["unrealized_pl"]   = display["unrealized_pl"].map("${:+,.0f}".format)
            st.dataframe(display[["symbol","qty","avg_entry","current_price",
                                   "market_value","unrealized_pl","unrealized_plpc"]],
                         use_container_width=True, hide_index=True)
        elif rh.is_configured():
            st.info("No open positions in Robinhood.")
    else:
        st.info("Configure Robinhood credentials in secrets.toml to connect.")

st.divider()

# ── Generate alerts ───────────────────────────────────────────────────────────
st.subheader("🚨 Step 3 — Signal Alerts")

if signals_df.empty:
    st.info("Run the strategy first (Step 1) to generate alerts.")
else:
    from brokers.signal_alerter import generate_alerts, alerts_to_df

    alerts = generate_alerts(
        signals_df            = signals_df,
        alpaca_held           = alpaca_held,
        robinhood_held        = robinhood_held,
        high_conf_threshold   = high_conf_threshold,
        include_hold_alerts   = include_hold_alerts,
        alpaca_positions      = alpaca_positions_df if not alpaca_positions_df.empty else None,
        robinhood_positions   = robinhood_positions_df if not robinhood_positions_df.empty else None,
    )

    if not alerts:
        st.success("✅ No alerts — signals and holdings are aligned.")
    else:
        # Summary metrics
        from collections import Counter
        type_counts = Counter(a.alert_type for a in alerts)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🔥 High-conf BUY",    type_counts.get("HIGH_CONF_BUY", 0))
        m2.metric("🟢 BUY not held",     type_counts.get("BUY_SIGNAL_NOT_HELD", 0))
        m3.metric("🔴 HOLD but held",    type_counts.get("HOLD_SIGNAL_HELD", 0))
        m4.metric("⛔ Risk blocked",     type_counts.get("RISK_BLOCK", 0))
        m5.metric("⚪ Untracked holds",  type_counts.get("POSITION_NO_SIGNAL", 0))

        # Color-coded alert table
        alerts_df = alerts_to_df(alerts)

        def _color_row(row):
            colors = {
                "HIGH_CONF_BUY":       "background-color: #1a3a1a; color: #7fff7f",
                "BUY_SIGNAL_NOT_HELD": "background-color: #1a2a1a; color: #aaffaa",
                "HOLD_SIGNAL_HELD":    "background-color: #3a1a1a; color: #ffaaaa",
                "RISK_BLOCK":          "background-color: #2a2a1a; color: #ffff99",
                "POSITION_NO_SIGNAL":  "background-color: #2a2a2a; color: #cccccc",
            }
            color = colors.get(row["type"], "")
            return [color] * len(row)

        styled = alerts_df.style.apply(_color_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download
        csv = alerts_df.to_csv(index=False)
        st.download_button("⬇ Download alerts CSV", csv,
                           file_name="broker_alerts.csv", mime="text/csv")

st.divider()

# ── Setup guide ───────────────────────────────────────────────────────────────
with st.expander("📖 Setup Guide"):
    st.markdown("""
### Adding credentials to `.streamlit/secrets.toml`

```toml
# Alpaca (get free API keys at alpaca.markets)
ALPACA_API_KEY    = "PKxxxxxxxxxxxxxxxx"
ALPACA_SECRET_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ALPACA_PAPER      = true   # set false for live account

# Robinhood
ROBINHOOD_USERNAME   = "your@email.com"
ROBINHOOD_PASSWORD   = "yourpassword"
ROBINHOOD_MFA_SECRET = "JBSWY3DPEHPK3PXP"  # base32 from Robinhood 2FA setup
```

### Getting your Robinhood MFA secret
1. Go to Robinhood → Account → Security → Two-Factor Authentication
2. Choose "Authenticator App"
3. When shown the QR code, click "Can't scan?" to get the text secret
4. Paste that text secret as `ROBINHOOD_MFA_SECRET`

### Installing dependencies
```bash
pip install alpaca-trade-api robin_stocks pyotp
```

### Important
- This dashboard is **read-only** — no orders are placed
- Robinhood's API is unofficial and may break with updates
- Alpaca paper trading is recommended for testing
    """)
