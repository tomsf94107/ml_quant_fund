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
            f"Time    : {datetime.now():%Y-%m-%d %H:%M}"
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

    tickers = st.multiselect(
        "Select tickers to run",
        options=all_tickers,
        default=st.session_state.get("selected_tickers", all_tickers),
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
st.caption(f"🕒 {datetime.now():%Y-%m-%d %H:%M:%S}")

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

if st.button("🚀 Run Strategy", type="primary"):

    csv_buffers   = []
    pred_log_rows = []
    signal_summary = []

    progress = st.progress(0, text="Building features...")

    for i, tkr in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Processing {tkr}...")

        # ── 1. Build features ─────────────────────────────────────────────────
        try:
            df = build_feature_dataframe(
                tkr,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
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
        forecast_rows.append({
            "Ticker":       r.ticker,
            "Signal":       r.today_signal,
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
      .ft-head{{display:grid;grid-template-columns:80px 65px 95px 130px 105px 105px 105px 75px 75px;padding:8px 14px;background:#0d0d18;font-size:10px;color:#4a5568;letter-spacing:.08em;border-bottom:1px solid #1e1e2e;}}
      .ft-head span{{text-align:right;}} .ft-head span:first-child,.ft-head span:nth-child(2){{text-align:left;}}
      .ft-row{{display:grid;grid-template-columns:80px 65px 95px 130px 105px 105px 105px 75px 75px;padding:11px 14px;border-bottom:1px solid #0f0f1a;transition:background .12s;}}
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
