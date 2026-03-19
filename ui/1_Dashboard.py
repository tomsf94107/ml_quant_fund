# ui/1_Dashboard.py
# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard. This file contains ZERO business logic.
# Every computation is delegated to the backend modules.
#
# What was removed from v19.3:
#   ✗ All compat wrappers (build_features_compat, today_move_compat, etc.)
#   ✗ MAE/MSE/R² metrics — replaced by accuracy/ROC-AUC/Brier from sink.py
#   ✗ Inline signal computation inside button handler
#   ✗ Uncalibrated sigmoid confidence proxy
#   ✗ Rolling accuracy via compute_rolling_accuracy (now from sink.py)
#   ✗ Duplicate gspread auth
#
# What we kept:
#   ✓ Password auth
#   ✓ BUY/HOLD signal cards with confidence badges
#   ✓ Sharpe/MaxDD/CAGR/profit_factor backtest metrics
#   ✓ Equity curve chart
#   ✓ Email alerts on high-confidence BUY signals
#   ✓ Event risk badge from calendar page session state
#   ✓ block_tau and confidence threshold sliders
#   ✓ ZIP download of all signal CSVs
#   ✓ Insider signals section
# ─────────────────────────────────────────────────────────────────────────────

import os, sys
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timezone import now_et
from datetime import date, datetime, timedelta

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io
import zipfile
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Backend imports ───────────────────────────────────────────────────────────
from features.builder import build_feature_dataframe
from signals.generator import (
    generate_signals, signals_to_dataframe,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BLOCK_TAU,
)
from data.etl_sentiment import run_sentiment_etl, get_sentiment_score, _current_time_slot
from accuracy.sink import (
    log_predictions_batch, load_accuracy,
    load_prediction_history, reconcile_outcomes,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Startup: train models if missing (Streamlit Cloud first deploy) ──────────
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
try:
    from startup import models_are_trained, run_startup
    if not models_are_trained():
        with st.spinner("⚙️ First launch — training models (10-15 min)... Please wait."):
            run_startup(verbose=True)
except Exception as _e:
    st.warning(f"Startup check failed: {_e}")

st.set_page_config(
    page_title="ML Quant Fund",
    page_icon="📈",
    layout="wide",
)
st_autorefresh(interval=5 * 60 * 1000, key="auto-refresh")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _check_login():
    if st.session_state.get("auth_ok"):
        return
    pwd = st.text_input("Password:", type="password", key="login_pwd")
    if not pwd:
        st.stop()
    if pwd != st.secrets.get("app_password", "MlQ@nt@072025"):
        st.error("❌ Wrong password")
        st.stop()
    st.session_state["auth_ok"] = True

_check_login()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _send_alert(ticker: str, prob: float, horizon: int):
    try:
        msg = MIMEText(
            f"High-confidence BUY signal\n"
            f"Ticker  : {ticker}\n"
            f"Horizon : {horizon}d\n"
            f"Prob(up): {prob:.1%}\n"
            f"Time    : {now_et().strftime('%Y-%m-%d %H:%M ET')}"
        )
        msg["Subject"] = f"🟢 BUY signal · {ticker}"
        msg["From"]    = os.getenv("EMAIL_SENDER", "")
        msg["To"]      = os.getenv("EMAIL_RECEIVER", "")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(os.getenv("EMAIL_SENDER", ""), os.getenv("EMAIL_PASSWORD", ""))
            s.send_message(msg)
    except Exception as e:
        st.warning(f"Email failed: {e}")


def _confidence_badge(confidence: str) -> str:
    return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")


def _load_tickers() -> list[str]:
    path = os.path.join(_ROOT, "tickers.txt")
    if os.path.exists(path):
        return [t.strip().upper() for t in open(path).read().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "MSFT", "TSLA", "AMD", "META", "GOOG", "AMZN"]


def _save_tickers(lst: list[str]):
    with open(os.path.join(_ROOT, "tickers.txt"), "w") as f:
        f.write("\n".join(lst))


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📆 Date Range")
    start_date = st.date_input("Start", value=date(2022, 1, 1))
    end_date   = st.date_input("End",   value=date.today())

    st.markdown("## 🎯 Signal Settings")
    horizon              = st.selectbox("Horizon", [1, 3, 5], index=0,
                                        format_func=lambda x: f"{x}d")
    confidence_threshold = st.slider("Confidence threshold",
                                     0.50, 0.95, DEFAULT_CONFIDENCE_THRESHOLD, 0.01)
    block_tau            = st.slider("Block when risk_next_3d ≥",
                                     0, 6, DEFAULT_BLOCK_TAU, 1)

    st.markdown("## 💼 Portfolio")
    portfolio_value = st.number_input(
        "Portfolio value ($)",
        min_value=10000,
        max_value=10000000,
        value=300000,
        step=10000,
        help="Your Fidelity account value — used for position sizing"
    )
    st.caption(f"Max position: ${portfolio_value * 0.30:,.0f} (30%) | Min: ${portfolio_value * 0.05:,.0f} (5%)")

    st.markdown("## 📧 Alerts")
    enable_email = st.toggle("Email alerts on BUY", value=True)

    st.markdown("## 📦 Export")
    enable_zip = st.toggle("ZIP download", value=True)

    st.markdown("## 🗂 Tickers")
    all_tickers = _load_tickers()

    col_a, col_b = st.columns(2)
    if col_a.button("✅ Select All"):
        st.session_state["selected_tickers"] = all_tickers
    if col_b.button("❌ Clear"):
        st.session_state["selected_tickers"] = []

    # Filter saved tickers to only those still in tickers.txt
    saved = st.session_state.get("selected_tickers", all_tickers)
    valid_saved = [t for t in saved if t in all_tickers]
    tickers = st.multiselect(
        "Select tickers to run",
        options=all_tickers,
        default=valid_saved,
        key="selected_tickers",
    )

    with st.expander("✏️ Edit master list"):
        raw = st.text_area(
            "One per line",
            "\n".join(all_tickers),
            height=200,
        )
        if st.button("💾 Save list"):
            _save_tickers([t.strip().upper() for t in raw.splitlines() if t.strip()])
            st.success("Saved.")
            st.rerun()

    if st.button("🔄 Refresh accuracy cache"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("## 📰 Sentiment")
    sent_tickers = st.multiselect(
        "Tickers to refresh",
        options=all_tickers,
        default=[],
        placeholder="Select tickers...",
        key="sent_refresh_tickers",
    )
    if st.button("🔁 Refresh Sentiment Now",
                 disabled=not sent_tickers,
                 help="Run FinBERT on selected tickers. ~4s each."):
        with st.spinner(f"Running sentiment on {len(sent_tickers)} ticker(s)..."):
            try:
                run_sentiment_etl(
                    tickers=sent_tickers,
                    time_slot="intraday",
                    force=True,
                    verbose=False,
                )
                st.success(f"✓ Sentiment updated for {', '.join(sent_tickers)}")
            except Exception as e:
                st.error(f"Sentiment failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("📈 ML Quant Fund")
st.caption(f"🕒 {now_et().strftime('%Y-%m-%d %H:%M:%S ET')}")

# ── Event risk badge (set by Page 7 — calendar) ──────────────────────────────
risk_info  = st.session_state.get("event_risk_next72")
risk_label = risk_info["label"] if risk_info else None
risk_mult  = {"Low": 1.00, "Medium": 0.92, "High": 0.85}.get(risk_label, 1.00)

if risk_info:
    col1, col2 = st.columns([1, 5])
    col1.metric("Next 72h Risk", f"{risk_info['label']} ({risk_info['score']})")

# ── Macro regime badge ────────────────────────────────────────────────────────
try:
    import sys, os
    _ui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _ui_root not in sys.path:
        sys.path.insert(0, _ui_root)
    from ui.components.regime_widget import render_regime_widget
    render_regime_widget()
except Exception as e:
    st.caption(f"Regime widget unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RUN STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

use_cache = st.toggle("⚡ Use cached signals (faster)", value=False,
    help="Load last run results instead of re-fetching all data. Updates every hour automatically.")

if st.button("🚀 Run Strategy", type="primary"):

    csv_buffers   = []
    pred_log_rows = []
    signal_summary = []

    @st.cache_data(ttl=3600)
    def _cached_features(t, s, e):
        return build_feature_dataframe(t, start_date=s, end_date=e)

    progress = st.progress(0, text="Building features...")

    for i, tkr in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Processing {tkr}...")

        # ── 1. Build features ─────────────────────────────────────────────────
        try:
            df = _cached_features(
                tkr,
                start_date.isoformat(),
                end_date.isoformat(),
            )
        except Exception as e:
            st.warning(f"⚠️ {tkr}: feature build failed — {e}")
            continue

        if df.empty:
            st.warning(f"⚠️ {tkr}: no data returned")
            continue

        # ── 2. Generate signals ───────────────────────────────────────────────
        result = generate_signals(
            ticker=tkr,
            df=df,
            horizon=horizon,
            confidence_threshold=confidence_threshold,
            block_tau=block_tau,
            risk_label=risk_label,
        )

        if result.error:
            st.warning(f"⚠️ {tkr}: {result.error}")
            continue

        signal_summary.append(result)

        # ── 3. Log prediction ─────────────────────────────────────────────────
        confidence_str = (
            "HIGH"   if result.today_prob_eff >= 0.65 else
            "MEDIUM" if result.today_prob_eff >= 0.55 else
            "LOW"
        )
        pred_log_rows.append({
            "ticker":          tkr,
            "prediction_date": date.today().isoformat(),
            "horizon":         horizon,
            "prob_up":         result.today_prob_eff,
            "signal":          result.today_signal,
            "confidence":      confidence_str,
        })

    progress.progress(1.0, text="Done.")

    # ── Log all predictions to DB ─────────────────────────────────────────────
    if pred_log_rows:
        try:
            log_predictions_batch(pred_log_rows)
        except Exception as e:
            st.warning(f"Prediction logging failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────────────────────────────────────

    # ── Price sanity check ───────────────────────────────────────────────────
    # Fetch ground truth prices and flag any crossovers/stale prices
    try:
        import yfinance as yf
        all_syms = [r.ticker for r in signal_summary]
        raw_px = yf.download(all_syms, period="2d", auto_adjust=True, progress=False)
        if hasattr(raw_px.columns, "levels"):
            raw_px = raw_px["Close"]
        latest_px = raw_px.iloc[-1].to_dict() if not raw_px.empty else {}
        for r in signal_summary:
            true_price = latest_px.get(r.ticker)
            if true_price and r.current_price:
                diff_pct = abs(true_price - r.current_price) / true_price
                if diff_pct > 0.10:  # >10% off = price crossover
                    st.warning(f"⚠️ {r.ticker}: price mismatch — model used ${r.current_price:.2f}, market says ${true_price:.2f}. Refreshing data.")
                    # Force refresh by clearing cache for this ticker
                    _cached_features.clear()
    except Exception as _pe:
        pass  # price check is best-effort, never crash the dashboard

    if not signal_summary:
        st.error("No signals generated. Check tickers and date range.")
        st.stop()

    # ── Signal cards ──────────────────────────────────────────────────────────
    st.subheader("📡 Live Signals")
    cols = st.columns(min(len(signal_summary), 4))
    for i, r in enumerate(signal_summary):
        col = cols[i % 4]
        badge = _confidence_badge(confidence_str)
        signal_color = "🟢" if r.today_signal == "BUY" else "🔴"
        col.metric(
            label=r.ticker,
            value=f"{signal_color} {r.today_signal}",
            delta=f"p={r.today_prob_eff:.1%}  {badge}",
        )
        if enable_email and r.today_signal == "BUY" and r.today_prob_eff >= 0.65:
            _send_alert(r.ticker, r.today_prob_eff, horizon)

    # ── Forecast table ────────────────────────────────────────────────────────
    st.subheader("🎯 Price Forecast Table")

    import pandas as pd
    forecast_rows = []
    for r in signal_summary:
        exp_ret = r.expected_return or 0.0
        # Lean: direction implied by prob_up regardless of BUY/HOLD/SELL
        prob = r.today_prob_eff
        if prob >= 0.65:   lean = "⬆️ Strong UP"
        elif prob >= 0.55: lean = "⬆️ Weak UP"
        elif prob >= 0.45: lean = "⬇️ Weak DOWN"
        else:              lean = "⬇️ Strong DOWN"

        forecast_rows.append({
            "Ticker":       r.ticker,
            "Signal":       r.today_signal,
            "Lean":         lean,
            "Price":        f"${r.current_price:.2f}"    if r.current_price   else "—",
            "Prob Eff":     f"{r.today_prob_eff:.1%}",
            "Target ▲":     f"${r.price_target_up:.2f}"  if r.price_target_up else "—",
            "Target ▼":     f"${r.price_target_dn:.2f}"  if r.price_target_dn else "—",
            "Exp Return":   f"{exp_ret:+.2%}"             if r.expected_return is not None else "—",
            "ATR":          f"${r.atr:.2f}"               if r.atr             else "—",
            "Sharpe":       f"{r.metrics.sharpe:.2f}"     if not np.isnan(r.metrics.sharpe) else "—",
        })

    fdf = pd.DataFrame(forecast_rows)

    def _color_signal(val):
        if val == "BUY":  return "color: #22c55e; font-weight: bold"
        return "color: #94a3b8"

    def _color_exp(val):
        if val == "—": return ""
        try:
            n = float(val.replace("%","").replace("+",""))
            return "color: #22c55e" if n >= 0 else "color: #ef4444"
        except: return ""

    # ── Styled forecast table via HTML component ─────────────────────────────
    import json
    signals_json = json.dumps(forecast_rows)
    html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
      *{{box-sizing:border-box;margin:0;padding:0;}}
      .ft{{font-family:'IBM Plex Mono',monospace;background:#0a0a0f;border:1px solid #1e1e2e;border-radius:8px;overflow:hidden;}}
      .ft-head{{display:grid;grid-template-columns:10% 8% 11% 15% 12% 12% 12% 10% 10%;padding:8px 14px;background:#0d0d18;font-size:10px;color:#4a5568;letter-spacing:.08em;border-bottom:1px solid #1e1e2e;}}
      .ft-head span{{text-align:right;}} .ft-head span:first-child,.ft-head span:nth-child(2){{text-align:left;}}
      .ft-row{{display:grid;grid-template-columns:10% 8% 11% 15% 12% 12% 12% 10% 10%;padding:11px 14px;border-bottom:1px solid #0f0f1a;transition:background .12s;}}
      .ft-row:hover{{background:#13131f;}}
      .ft-row span{{font-size:12px;color:#cbd5e1;display:flex;align-items:center;justify-content:flex-end;}}
      .ft-row span:first-child{{font-weight:600;color:#f8fafc;font-size:13px;justify-content:flex-start;}}
      .ft-row span:nth-child(2){{justify-content:flex-start;}}
      .badge{{font-size:10px;font-weight:600;padding:2px 7px;border-radius:3px;}}
      .buy{{color:#22c55e;background:#052e16;border:1px solid #166534;}}
      .hold{{color:#f59e0b;background:#1a1008;border:1px solid #7c4a00;}}
      .sell{{color:#ef4444;background:#1c0a0a;border:1px solid #7f1d1d;}}
      .up{{color:#22c55e !important;}}
      .dn{{color:#ef4444 !important;}}
      .dim{{color:#64748b !important;}}
      .prob-col{{display:flex;flex-direction:column;gap:3px;align-items:flex-end;}}
      .bar-bg{{width:70px;height:3px;background:#1e1e2e;border-radius:2px;overflow:hidden;}}
      .bar-fill{{height:100%;border-radius:2px;}}
      .legend{{margin-top:10px;padding:10px 14px;background:#0d0d18;border:1px solid #1e1e2e;border-radius:6px;font-size:11px;color:#4a5568;line-height:1.8;}}
    </style>
    <div class="ft">
      <div class="ft-head">
        <span>TICKER</span><span>SIGNAL</span><span>PRICE</span>
        <span>PROB EFF</span><span>TARGET ▲</span><span>TARGET ▼</span>
        <span>EXP RETURN</span><span>ATR</span><span>SHARPE</span>
      </div>
      <div id="tbody"></div>
    </div>
    <div class="legend">
      📖 &nbsp;
      <span style="color:#22c55e">▲ Target = price + ATR</span> &nbsp;·&nbsp;
      <span style="color:#ef4444">▼ Target = price − ATR</span> &nbsp;·&nbsp;
      <span style="color:#94a3b8">Exp Return = prob-weighted gain/loss</span> &nbsp;·&nbsp;
      <span style="color:#3b82f6">Bar = prob vs threshold</span>
    </div>
    <script>
      const data = {signals_json};
      const tbody = document.getElementById('tbody');
      data.forEach(r => {{
        const prob = parseFloat(r['Prob Eff']);
        const exp  = parseFloat(r['Exp Return']);
        const sh   = parseFloat(r['Sharpe']);
        const sig  = r['Signal'];
        const bc   = prob >= 65 ? '#22c55e' : prob >= 55 ? '#f59e0b' : '#3b82f6';
        const sc   = sh >= 2 ? '#22c55e' : sh >= 1 ? '#f59e0b' : '#ef4444';
        const badgeClass = sig==='BUY' ? 'buy' : sig==='SELL' ? 'sell' : 'hold';
        const row = document.createElement('div');
        row.className = 'ft-row';
        row.innerHTML = `
          <span>${{r.Ticker}}</span>
          <span><span class="badge ${{badgeClass}}">${{sig}}</span></span>
          <span>${{r.Price}}</span>
          <span>
            <div class="prob-col">
              <span style="color:#94a3b8;font-size:12px">${{r['Prob Eff']}}</span>
              <div class="bar-bg"><div class="bar-fill" style="width:${{Math.min(prob,100)}}%;background:${{bc}}"></div></div>
              <span style="font-size:9px;color:#2d3748">threshold: 65%</span>
            </div>
          </span>
          <span class="up">${{r['Target ▲']}}</span>
          <span class="dn">${{r['Target ▼']}}</span>
          <span style="color:${{exp>=0?'#22c55e':'#ef4444'}};font-weight:500">${{r['Exp Return']}}</span>
          <span class="dim">${{r.ATR}}</span>
          <span style="color:${{sc}};font-weight:500">${{r.Sharpe}}</span>
        `;
        tbody.appendChild(row);
      }});
    </script>
    """
    st.components.v1.html(html, height=min(80 + len(forecast_rows) * 44, 800), scrolling=True)

    # ── Intraday Signals + Alignment Table ───────────────────────────────────
    st.subheader("⚡ Intraday Signals & EOD Alignment")
    st.caption("Compares EOD model signal with intraday 1hr/2hr/4hr momentum · tickers from your strategy run")

    try:
        from features.intraday_builder import get_all_intraday_signals, is_market_open
        intraday_tickers = [r.ticker for r in signal_summary]

        with st.spinner("Loading intraday signals..."):
            intra_sigs = get_all_intraday_signals(intraday_tickers)

        sig_lkp  = {s["ticker"]: s for s in intra_sigs}
        eod_lkp  = {r.ticker: r for r in signal_summary}

        def _isig_fmt(s, p):
            if s == "UP":   return f"🟢 UP ({p:.0%})"
            if s == "DOWN": return f"🔴 DOWN ({p:.0%})"
            return f"⚪ NTRL ({p:.0%})"

        def _alignment(eod_sig, i1, i2, i4):
            up = [i1,i2,i4].count("UP")
            dn = [i1,i2,i4].count("DOWN")
            if eod_sig == "BUY"  and up >= 2: return "🔥 BOTH BULLISH"
            if eod_sig == "SELL" and dn >= 2: return "🔥 BOTH BEARISH"
            if eod_sig == "BUY"  and dn >= 2: return "⚠️ CONFLICT"
            if eod_sig == "SELL" and up >= 2: return "⚠️ CONFLICT"
            if up >= 2: return "📈 INTRA BULL"
            if dn >= 2: return "📉 INTRA BEAR"
            return "➖ NEUTRAL"

        # Intraday price sanity check
        intra_price_issues = []
        for t in intraday_tickers:
            s = sig_lkp.get(t)
            e = eod_lkp.get(t)
            if s and e and s.get("current_price") and e.current_price:
                diff = abs(s["current_price"] - e.current_price) / e.current_price
                if diff > 0.10:
                    intra_price_issues.append(f"{t} (intraday ${s['current_price']:.2f} vs EOD ${e.current_price:.2f})")
        if intra_price_issues:
            st.warning(f"⚠️ Intraday price mismatch on: {', '.join(intra_price_issues)}")

        align_rows = []
        for t in intraday_tickers:
            s = sig_lkp.get(t)
            e = eod_lkp.get(t)
            if not s or not e or not s.get("current_price"):
                continue
            i1 = s["signal_1hr"]; i2 = s["signal_2hr"]; i4 = s["signal_4hr"]
            align_rows.append({
                "Ticker":    t,
                "Price":     f"${s['current_price']:.2f}",
                "EOD Signal": e.today_signal,
                "EOD Prob":  f"{e.today_prob_eff:.0%}",
                "1hr":       _isig_fmt(i1, s["prob_1hr"]),
                "2hr":       _isig_fmt(i2, s["prob_2hr"]),
                "4hr":       _isig_fmt(i4, s["prob_4hr"]),
                "Alignment": _alignment(e.today_signal, i1, i2, i4),
            })

        if align_rows:
            adf = pd.DataFrame(align_rows)

            # Sort: BOTH BULLISH first, then INTRA BULL, NEUTRAL, CONFLICT, INTRA BEAR
            sort_order = {"🔥 BOTH BULLISH": 0, "📈 INTRA BULL": 1, "➖ NEUTRAL": 2,
                          "⚠️ CONFLICT": 3, "📉 INTRA BEAR": 4, "🔥 BOTH BEARISH": 5}
            adf["_sort"] = adf["Alignment"].map(sort_order).fillna(9)
            adf = adf.sort_values("_sort").drop(columns=["_sort"])

            st.dataframe(adf, use_container_width=True, hide_index=True)

            # Summary counts
            counts = adf["Alignment"].value_counts()
            summary_parts = [f"{v}× {k}" for k,v in counts.items()]
            st.caption("  ·  ".join(summary_parts))

            with st.expander("📖 How to read Alignment", expanded=False):
                st.markdown("""
| Alignment | Meaning | Action |
|-----------|---------|--------|
| 🔥 BOTH BULLISH | EOD=BUY + Intraday UP | Highest conviction entry |
| 🔥 BOTH BEARISH | EOD=SELL + Intraday DOWN | Highest conviction avoid |
| 📈 INTRA BULL | EOD=HOLD but intraday momentum UP | Watch — may break out |
| 📉 INTRA BEAR | EOD=HOLD but intraday momentum DOWN | Avoid short-term |
| ⚠️ CONFLICT | EOD and intraday disagree | Wait for clarity |
| ➖ NEUTRAL | No strong signal in either direction | Hold current position |
""")

            # ── Live interpretation ───────────────────────────────────────────
            st.subheader("🧠 Live Interpretation")

            both_bull = [r["Ticker"] for r in align_rows if r["Alignment"] == "🔥 BOTH BULLISH"]
            both_bear = [r["Ticker"] for r in align_rows if r["Alignment"] == "🔥 BOTH BEARISH"]
            intra_bull = [r["Ticker"] for r in align_rows if r["Alignment"] == "📈 INTRA BULL"]
            intra_bear = [r["Ticker"] for r in align_rows if r["Alignment"] == "📉 INTRA BEAR"]
            conflict   = [r["Ticker"] for r in align_rows if r["Alignment"] == "⚠️ CONFLICT"]
            eod_buys   = [r for r in align_rows if r["EOD Signal"] == "BUY"]

            if both_bull:
                tickers_str = ", ".join(both_bull)
                st.success(f"🔥 **Highest conviction BUY:** {tickers_str} — EOD model AND intraday both bullish. Strongest entry signal.")
                try:
                    from signals.position_sizer import get_position_size
                    for ticker_b in both_bull:
                        row = next((r for r in results if r.get("ticker") == ticker_b), None)
                        if row:
                            pos = get_position_size(
                                ticker=ticker_b,
                                prob_eff=float(row.get("prob_eff", 0.7)),
                                confidence=row.get("confidence", "HIGH"),
                                portfolio_value=portfolio_value,
                                current_price=row.get("current_price"),
                            )
                            if pos.final_pct > 0:
                                shares_str = f" (~{pos.shares} shares)" if pos.shares else ""
                                st.info(f"📐 **{ticker_b} suggested size:** {pos.final_pct*100:.1f}% = ${pos.dollars:,.0f}{shares_str}")
                except Exception:
                    pass

            if both_bear:
                tickers_str = ", ".join(both_bear)
                st.error(f"🔥 **Highest conviction AVOID:** {tickers_str} — EOD model AND intraday both bearish.")

            if eod_buys and not both_bull:
                for r in eod_buys:
                    t = r["Ticker"]
                    p = r["EOD Prob"]
                    al = r["Alignment"]
                    if al == "➖ NEUTRAL":
                        st.info(f"✅ **{t} EOD BUY ({p})** — Model confident but intraday neutral. Valid entry, no intraday confirmation yet. Consider waiting for intraday to turn UP.")
                    elif al == "⚠️ CONFLICT":
                        st.warning(f"⚠️ **{t} EOD BUY ({p}) but intraday bearish** — Conflicting signals. Reduce position size or wait.")

            if intra_bull and not both_bull:
                tickers_str = ", ".join(intra_bull)
                st.info(f"📈 **Watch list:** {tickers_str} — Intraday momentum bullish but EOD model cautious. If EOD prob rises above threshold tomorrow, these become BUY candidates.")

            if intra_bear:
                tickers_str = ", ".join(intra_bear)
                st.warning(f"📉 **Avoid short-term:** {tickers_str} — Intraday momentum bearish. Even if EOD signal fires BUY, day traders are selling. Wait for intraday to recover.")

            if conflict:
                tickers_str = ", ".join(conflict)
                st.warning(f"⚠️ **Conflicting signals:** {tickers_str} — EOD and intraday disagree. Hold off until signals align.")

            if not both_bull and not eod_buys and not intra_bull:
                st.info("No strong directional signals right now. Market is in a wait-and-see mode.")
        else:
            st.info("No intraday data available.")

    except Exception as e:
        st.warning(f"Intraday signals unavailable: {e}")

    # ── How to read this table ────────────────────────────────────────────────
    with st.expander("📖 How to read the Forecast Table", expanded=False):
        st.markdown("""
**SIGNAL** — BUY or HOLD.
- **BUY** fires when `Prob Eff` exceeds the confidence threshold (default 65% in VOLATILE regime, 55% in NEUTRAL).
- **HOLD** means wait — either probability is too low or the regime is suppressing the signal.

**PRICE** — Today's closing price. This is the price the model used to generate the forecast.

**PROB EFF** — The model's confidence that this stock goes UP over the next `horizon` days, after all signal adjustments:
```
Raw ML prob × regime multiplier × sentiment × options flow × short interest
```
- Above threshold → BUY
- Below threshold → HOLD
- Currently in VOLATILE regime, threshold = 65%

**TARGET ▲** — Where the stock is likely to go if it moves UP.
Calculated as: `current price + ATR × √horizon`
Example: NVDA at $121 with ATR $5.35 → Target ▲ = $126.35

**TARGET ▼** — Where it goes if it moves DOWN.
Calculated as: `current price − ATR × √horizon`
This is your downside risk if the signal is wrong.

**EXP RETURN** — Probability-weighted expected return. The single most actionable number:
```
= (Prob_eff × upside%) − ((1 − Prob_eff) × downside%)
```
- **Positive (green)** = model expects to make money → worth watching
- **Negative (red)** = model expects to lose money → stay out
- Right now all negative because VOLATILE regime is suppressing prob_eff below 50%

**ATR** — Average True Range. How much this stock moves on a typical day in dollars.
- Low ATR (e.g. NVO $3.73) = stable, lower risk
- High ATR (e.g. TSLA $13.09) = volatile, bigger swings both ways

**SHARPE** — Historical risk-adjusted return from backtesting.
- Above 2.0 = excellent (green)
- 1.0–2.0 = good (yellow)
- Below 1.0 = poor (red)

---
**When to act:**
1. Exp Return turns **green** on a ticker
2. Prob Eff crosses the threshold → **BUY fires**
3. Regime shifts from VOLATILE → NEUTRAL/BULL (threshold drops from 65% → 55%)
""")

    # ── Per-ticker detail ─────────────────────────────────────────────────────
    for result in signal_summary:
        with st.expander(f"📊 {result.ticker} — Detail", expanded=False):
            m = result.metrics

            # Backtest KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe",   f"{m.sharpe:.2f}"       if not np.isnan(m.sharpe)       else "—")
            c2.metric("Max DD",   f"{m.max_drawdown:.1%}"  if not np.isnan(m.max_drawdown) else "—")
            c3.metric("CAGR",     f"{m.cagr:.1%}"          if not np.isnan(m.cagr)         else "—")
            c4.metric("Accuracy", f"{m.accuracy:.1%}"      if not np.isnan(m.accuracy)     else "—")

            st.caption(
                f"Trades: {m.n_trades} · "
                f"Exposure: {m.exposure:.1%} · "
                f"Profit factor: {m.profit_factor:.2f}"
                if not np.isnan(m.profit_factor) else
                f"Trades: {m.n_trades} · Exposure: {m.exposure:.1%}"
            )

            # Equity curve
            sdf = result.signal_df.copy()
            sdf["date"] = pd.to_datetime(sdf["date"])
            sdf = sdf.sort_values("date")

            ret_strat = (sdf["signal"] * sdf["return_1d"]).fillna(0)
            ret_mkt   = sdf["return_1d"].fillna(0)
            eq = pd.DataFrame({
                "Strategy": (1 + ret_strat).cumprod(),
                "Market":   (1 + ret_mkt).cumprod(),
            }, index=sdf["date"]).dropna()

            st.line_chart(eq)

            # Signal table (last 20 rows)
            show_cols = [c for c in
                ["date", "close", "prob", "prob_eff", "signal_raw", "gate_block"]
                if c in sdf.columns]
            st.dataframe(
                sdf[show_cols].tail(20).style.format({
                    "close":    "{:.2f}",
                    "prob":     "{:.1%}",
                    "prob_eff": "{:.1%}",
                }),
                use_container_width=True,
            )

            # CSV buffer for ZIP
            csv_bytes = sdf.to_csv(index=False).encode()
            st.download_button(
                f"⬇️ CSV — {result.ticker}",
                csv_bytes,
                file_name=f"{result.ticker}_signals.csv",
                mime="text/csv",
                key=f"csv_{result.ticker}",
            )
            csv_buffers.append((f"{result.ticker}_signals.csv", csv_bytes))

    # ── ZIP download ──────────────────────────────────────────────────────────
    if enable_zip and csv_buffers:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in csv_buffers:
                zf.writestr(fname, data)
        st.download_button(
            "📦 Download ALL as ZIP",
            zbuf.getvalue(),
            file_name="signals_export.zip",
            mime="application/zip",
            key="zip_all",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ACCURACY SECTION
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📊 Live Prediction Accuracy")

# ── EOD + Intraday Accuracy Tables ───────────────────────────────────────────
acc_tab1, acc_tab2 = st.tabs(["📅 EOD Model Accuracy", "⚡ Intraday Accuracy"])

with acc_tab1:
    try:
        from accuracy.sink import get_eod_accuracy_summary, get_spy_relative_accuracy
        
        # BUY/SELL accuracy
        eod_acc = get_eod_accuracy_summary()
        if eod_acc:
            edf = pd.DataFrame(eod_acc)
            edf["accuracy"]   = edf["accuracy"].apply(lambda x: f"{x:.1%}" if x is not None else "N/A")
            edf["avg_return"] = edf["avg_return"].apply(lambda x: f"{x:+.2%}" if x is not None else "N/A")
            edf.columns = ["Ticker", "# Outcomes", "Accuracy (BUY/SELL)", "Avg Return"]
            st.dataframe(edf, use_container_width=True, hide_index=True)
            valid = [r for r in eod_acc if r["accuracy"] is not None]
            if valid:
                avg = sum(r["accuracy"] for r in valid) / len(valid)
                st.caption(f"Overall BUY/SELL accuracy: {avg:.1%} · Only {sum(r['n'] for r in eod_acc if r['accuracy'] is not None)} BUY/SELL signals so far — need 60+ for statistical significance")
        else:
            st.info("No EOD accuracy data yet.")

        # SPY-relative accuracy
        st.markdown("---")
        st.markdown("**📊 Daily Performance vs SPY**")
        st.caption("Are our tickers outperforming the market each day? More meaningful than BUY accuracy with small sample.")
        spy_acc = get_spy_relative_accuracy()
        if spy_acc:
            sdf = pd.DataFrame(spy_acc)
            sdf["spy_ret"]      = sdf["spy_ret"].apply(lambda x: f"{x:+.2%}")
            sdf["avg_ret"]      = sdf["avg_ret"].apply(lambda x: f"{x:+.2%}")
            sdf["avg_vs_spy"]   = sdf["avg_vs_spy"].apply(lambda x: f"{x:+.2%}")
            sdf["pct_beat_spy"] = sdf["pct_beat_spy"].apply(lambda x: f"{x:.0%}")
            sdf["buy_acc"]      = sdf["buy_acc"].apply(lambda x: f"{x:.0%}" if x is not None else "—")
            sdf.columns = ["Date","SPY Return","Avg Return","Avg vs SPY","% Beat SPY","# BUYs","BUY Acc"]
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            avg_vs_spy = sum(r["avg_vs_spy"] for r in spy_acc) / len(spy_acc)
            avg_beat   = sum(r["pct_beat_spy"] for r in spy_acc) / len(spy_acc)
            if avg_vs_spy > 0:
                st.success(f"✅ On average our tickers beat SPY by {avg_vs_spy:+.2%} per day · {avg_beat:.0%} of tickers beat SPY")
            else:
                st.warning(f"⚠️ On average our tickers underperform SPY by {avg_vs_spy:+.2%} per day")
        else:
            st.info("No SPY comparison data yet.")
    except Exception as e:
        st.warning(f"EOD accuracy unavailable: {e}")

with acc_tab2:
    try:
        from accuracy.sink import get_intraday_accuracy_summary, reconcile_intraday_outcomes
        reconcile_intraday_outcomes()
        intra_acc = get_intraday_accuracy_summary()
        if intra_acc:
            idf = pd.DataFrame(intra_acc)
            idf["accuracy"] = idf["accuracy"].apply(lambda x: f"{x:.1%}" if x is not None else "N/A")
            idf["horizon_hr"] = idf["horizon_hr"].apply(lambda x: f"{x}hr")
            idf = idf.drop(columns=["computed_at"])
            idf.columns = ["Ticker", "Horizon", "Accuracy", "# Predictions"]
            st.dataframe(idf, use_container_width=True, hide_index=True)
            valid = [r for r in intra_acc if r["accuracy"] is not None]
            if valid:
                avg = sum(r["accuracy"] for r in valid) / len(valid)
                st.caption(f"Overall intraday accuracy: {avg:.1%} · Needs 5+ outcomes per ticker to be meaningful")
        else:
            st.info("No intraday accuracy data yet — check back after market hours once outcomes are reconciled.")
    except Exception as e:
        st.warning(f"Intraday accuracy unavailable: {e}")

@st.cache_data(ttl=300)
def _load_accuracy(horizon_filter):
    try:
        return load_accuracy(horizon=horizon_filter, window_days=90)
    except Exception:
        return pd.DataFrame()

acc_horizon = st.selectbox("Accuracy horizon", [1, 3, 5],
                            format_func=lambda x: f"{x}d", key="acc_hz")
acc_df = _load_accuracy(acc_horizon)

if acc_df.empty:
    st.info(
        "No accuracy data yet. Predictions are logged each time you run the strategy. "
        "Accuracy is computed after the horizon passes and outcomes can be verified."
    )
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Accuracy",  f"{acc_df['accuracy'].mean():.1%}")
    c2.metric("Avg ROC-AUC",   f"{acc_df['roc_auc'].mean():.3f}")
    c3.metric("Avg Brier",     f"{acc_df['brier_score'].mean():.3f}")

    st.dataframe(
        acc_df[["ticker", "horizon", "accuracy", "roc_auc",
                "brier_score", "n_predictions"]]
        .style.format({
            "accuracy":    lambda x: f"{x:.1%}" if x is not None and str(x) != 'nan' else "N/A",
            "roc_auc":     lambda x: f"{x:.3f}" if x is not None and str(x) != 'nan' else "N/A",
            "brier_score": lambda x: f"{x:.3f}" if x is not None and str(x) != 'nan' else "N/A",
        }),
        use_container_width=True,
    )

    # Accuracy trend for selected ticker
    acc_tickers = acc_df["ticker"].tolist()
    if acc_tickers:
        sel = st.selectbox("Trend for ticker", acc_tickers, key="acc_trend_tkr")
        hist = load_prediction_history(sel, horizon=acc_horizon, days=90)
        if not hist.empty:
            hist["rolling_acc"] = hist["correct"].rolling(20, min_periods=5).mean()
            chart = (
                alt.Chart(hist)
                .mark_line()
                .encode(
                    x=alt.X("prediction_date:T", title="Date"),
                    y=alt.Y("rolling_acc:Q", title="Rolling Accuracy (20d)",
                            scale=alt.Scale(domain=[0, 1])),
                    tooltip=["prediction_date:T", "rolling_acc:Q",
                             "signal:N", "actual_up:Q"],
                )
                .properties(title=f"{sel} — Rolling 20-day Accuracy")
            )
            st.altair_chart(chart, use_container_width=True)

    # Reconcile button (run manually or set on a schedule)
    if st.button("🔁 Reconcile outcomes now"):
        with st.spinner("Fetching actual outcomes..."):
            try:
                n = reconcile_outcomes()
                st.success(f"✓ {n} new outcomes recorded")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Reconciliation failed: {e}")
