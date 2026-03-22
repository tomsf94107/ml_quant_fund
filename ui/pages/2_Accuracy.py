# ui/2_Accuracy.py
# Forecast Accuracy Dashboard — reads from accuracy/sink.py
# Shows real prediction vs outcome accuracy, not training metrics.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, timedelta

from accuracy.sink import (
    load_accuracy, load_prediction_history,
    reconcile_outcomes, update_accuracy_cache,
    get_spy_relative_accuracy, get_eod_accuracy_summary,
)

st.set_page_config(page_title="Forecast Accuracy", page_icon="🎯", layout="wide")
st.title("🎯 Forecast Accuracy Dashboard")
st.caption("Live prediction accuracy — how often the model was actually right after the fact.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon     = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}d")
    window_days = st.selectbox("Rolling window", [30, 60, 90, 180], index=2,
                               format_func=lambda x: f"{x}d")

    st.markdown("---")
    if st.button("🔁 Reconcile outcomes"):
        with st.spinner("Fetching actual outcomes from yfinance..."):
            try:
                n = reconcile_outcomes()
                st.success(f"✓ {n} new outcomes recorded")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("♻️ Recompute accuracy cache"):
        with st.spinner("Computing metrics..."):
            try:
                df = update_accuracy_cache(window_days=window_days)
                st.success(f"✓ Cache updated — {len(df)} ticker×horizon rows")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load(horizon, window_days):
    return load_accuracy(horizon=horizon, window_days=window_days)

# Auto-reconcile on page load — only after market close (4 PM ET)
import pytz as _pytz
from datetime import datetime as _dt
_now_et = _dt.now(_pytz.timezone("America/New_York"))
_after_close = _now_et.hour >= 16 or (_now_et.weekday() >= 5)
if _after_close:
    with st.spinner("Reconciling outcomes..."):
        try:
            n = reconcile_outcomes()
            if n > 0:
                st.success(f"✓ {n} new outcomes reconciled")
        except Exception:
            pass

acc_df = _load(horizon, window_days)

# ── SPY-relative accuracy ─────────────────────────────────────────────────────
st.subheader("📊 Daily Performance vs SPY")
st.caption("Green = our tickers beat the market that day. Focus on the Alpha column — it shows edge over SPY.")
try:
    spy_results = get_spy_relative_accuracy()
    if spy_results:
        sdf_raw = pd.DataFrame(spy_results)
        valid = [r for r in spy_results if r["avg_vs_spy"] == r["avg_vs_spy"]]
        if valid:
            avg_vs_spy = sum(r["avg_vs_spy"] for r in valid) / len(valid)
            avg_beat   = sum(r["pct_beat_spy"] for r in valid) / len(valid)
            days_beating = sum(1 for r in valid if r["pct_beat_spy"] >= 0.5)
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg daily alpha vs SPY", f"{avg_vs_spy:+.2%}", f"across {len(valid)} trading days")
            m2.metric("Days beating SPY", f"{days_beating}/{len(valid)}", f"{days_beating/len(valid):.0%} of days")
            m3.metric("Avg tickers beating SPY", f"{avg_beat:.0%}")

        st.markdown("---")

        # Build card HTML
        show_all_spy = st.session_state.get("show_all_spy", False)
        display_rows = spy_results if show_all_spy else spy_results[:10]

        CARD_CSS = """<style>
.spy-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);
border-radius:12px;padding:12px 18px;margin-bottom:7px;display:grid;
grid-template-columns:90px 1fr 1fr 1fr 90px;align-items:center;gap:12px}
.spy-hdr{background:var(--color-background-secondary);border-radius:8px;padding:8px 18px;margin-bottom:7px;
display:grid;grid-template-columns:90px 1fr 1fr 1fr 90px;gap:12px}
.spy-lbl{font-size:11px;color:var(--color-text-secondary)}
.spy-date{font-size:13px;font-weight:500}
.spy-val{font-size:14px}
.spy-alpha-up{font-size:15px;font-weight:500;color:var(--color-text-success)}
.spy-alpha-dn{font-size:15px;font-weight:500;color:var(--color-text-danger)}
.beat-good{padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;background:#EAF3DE;color:#3B6D11}
.beat-bad{padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;background:#FCEBEB;color:#A32D2D}
.beat-mid{padding:2px 10px;border-radius:20px;font-size:11px;background:var(--color-background-secondary);color:var(--color-text-secondary)}
</style>"""

        html = CARD_CSS
        html += '<div class="spy-hdr"><span class="spy-lbl">Date</span><span class="spy-lbl">SPY return</span><span class="spy-lbl">Our avg return</span><span class="spy-lbl">Alpha vs SPY</span><span class="spy-lbl">Beat SPY?</span></div>'

        for r in display_rows:
            alpha = r.get("avg_vs_spy")
            beat  = r.get("pct_beat_spy", 0)
            spy_r = r.get("spy_ret", 0)
            avg_r = r.get("avg_ret", 0)
            if alpha != alpha: continue

            border = "#3B6D11" if alpha > 0 else "#A32D2D"
            alpha_cls = "spy-alpha-up" if alpha > 0 else "spy-alpha-dn"
            spy_col = "color:var(--color-text-success)" if spy_r > 0 else "color:var(--color-text-danger)"
            avg_col = "color:var(--color-text-success)" if avg_r > 0 else "color:var(--color-text-danger)"

            if beat >= 0.60:   beat_cls, beat_txt = "beat-good", f"{beat:.0%} beat"
            elif beat >= 0.40: beat_cls, beat_txt = "beat-mid",  f"{beat:.0%} beat"
            else:              beat_cls, beat_txt = "beat-bad",  f"{beat:.0%} beat"

            html += f"""<div class="spy-card" style="border-left:3px solid {border}">
  <div class="spy-date">{r['date']}</div>
  <div class="spy-val" style="{spy_col}">{spy_r:+.2%}</div>
  <div class="spy-val" style="{avg_col}">{avg_r:+.2%}</div>
  <div class="{alpha_cls}">{alpha:+.2%}</div>
  <div><span class="{beat_cls}">{beat_txt}</span></div>
</div>"""

        st.html(html)

        col_exp, col_filt = st.columns([1,1])
        if len(spy_results) > 10:
            label = f"Show all {len(spy_results)} days" if not show_all_spy else "Show top 10 only"
            if col_exp.button(label, key="spy_expand"):
                st.session_state["show_all_spy"] = not show_all_spy
                st.rerun()
except Exception as e:
    st.warning(f"SPY comparison unavailable: {e}")

st.markdown("---")

# ── BUY/SELL signal accuracy ──────────────────────────────────────────────────
st.subheader("🎯 BUY signal accuracy by ticker")
st.caption("Only tickers with 2+ signals shown. Accuracy needs 30+ signals per ticker to be meaningful — treat these as early data only.")
try:
    eod_acc = get_eod_accuracy_summary()
    if eod_acc:
        valid = [r for r in eod_acc if r["accuracy"] is not None]
        total_signals = sum(r["n"] for r in valid)
        overall_acc   = sum(r["accuracy"] for r in valid) / len(valid) if valid else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall BUY accuracy", f"{overall_acc:.1%}", f"{total_signals} total signals")
        m2.metric("Tickers with signals", len(valid))
        m3.metric("Statistical significance", "Not yet" if total_signals < 200 else "Borderline",
                  f"Need 60+ per ticker")

        st.markdown("---")

        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        min_signals = fc1.selectbox("Min signals", [1, 2, 5, 10], index=1, key="min_sig")
        sort_by     = fc2.selectbox("Sort by", ["Accuracy ↓", "Accuracy ↑", "Signals ↓", "Avg return ↓"], key="sort_by")
        show_all_buy = st.session_state.get("show_all_buy", False)

        filtered = [r for r in valid if r["n"] >= min_signals]
        if sort_by == "Accuracy ↓":   filtered.sort(key=lambda x: x["accuracy"], reverse=True)
        elif sort_by == "Accuracy ↑": filtered.sort(key=lambda x: x["accuracy"])
        elif sort_by == "Signals ↓":  filtered.sort(key=lambda x: x["n"], reverse=True)
        elif sort_by == "Avg return ↓": filtered.sort(key=lambda x: x.get("avg_return",0) or 0, reverse=True)

        display_buy = filtered if show_all_buy else filtered[:10]

        BUY_CSS = """<style>
.buy-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);
border-radius:12px;padding:12px 16px;margin-bottom:7px;display:grid;
grid-template-columns:80px 55px 1fr 100px 110px;align-items:center;gap:12px}
.buy-hdr{background:var(--color-background-secondary);border-radius:8px;padding:8px 16px;margin-bottom:7px;
display:grid;grid-template-columns:80px 55px 1fr 100px 110px;gap:12px}
.buy-ticker{font-size:17px;font-weight:500}
.acc-bar{height:6px;border-radius:3px;background:var(--color-border-tertiary);overflow:hidden;margin-top:4px}
.acc-fill{height:100%;border-radius:3px}
.rel-good{padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;background:#EAF3DE;color:#3B6D11}
.rel-warn{padding:2px 10px;border-radius:20px;font-size:11px;background:#FAEEDA;color:#854F0B}
.rel-bad{padding:2px 10px;border-radius:20px;font-size:11px;background:#FCEBEB;color:#A32D2D}
.rel-neu{padding:2px 10px;border-radius:20px;font-size:11px;background:var(--color-background-secondary);color:var(--color-text-secondary)}
</style>"""

        html2 = BUY_CSS
        html2 += '<div class="buy-hdr"><span class="spy-lbl">Ticker</span><span class="spy-lbl">Signals</span><span class="spy-lbl">Accuracy</span><span class="spy-lbl">Avg return</span><span class="spy-lbl">Reliability</span></div>'

        for r in display_buy:
            acc = r["accuracy"]
            n   = r["n"]
            ret = r.get("avg_return") or 0
            acc_pct = acc * 100
            bar_color = "#639922" if acc >= 0.60 else "#EF9F27" if acc >= 0.50 else "#E24B4A"
            ret_col = "color:var(--color-text-success)" if ret > 0 else "color:var(--color-text-danger)"
            border  = "#3B6D11" if acc >= 0.60 else "#EF9F27" if acc >= 0.50 else "#A32D2D"

            if n >= 30:    rel_cls, rel_txt = "rel-good", "Reliable"
            elif n >= 10:  rel_cls, rel_txt = "rel-warn", "Early data"
            else:          rel_cls, rel_txt = "rel-neu",  "Too few"

            html2 += f"""<div class="buy-card" style="border-left:3px solid {border}">
  <div class="buy-ticker">{r['ticker']}</div>
  <div style="font-size:13px;color:var(--color-text-secondary)">{n}</div>
  <div>
    <div style="font-size:14px;font-weight:500;color:{bar_color}">{acc:.1%}</div>
    <div class="acc-bar"><div class="acc-fill" style="width:{min(acc_pct,100):.0f}%;background:{bar_color}"></div></div>
  </div>
  <div style="font-size:14px;{ret_col}">{ret:+.2%}</div>
  <div><span class="{rel_cls}">{rel_txt}</span></div>
</div>"""

        st.html(html2)

        col_exp2, _ = st.columns([1,2])
        if len(filtered) > 10:
            label2 = f"Show all {len(filtered)} tickers" if not show_all_buy else "Show top 10 only"
            if col_exp2.button(label2, key="buy_expand"):
                st.session_state["show_all_buy"] = not show_all_buy
                st.rerun()

        st.caption(f"Overall: {overall_acc:.1%} across {total_signals} BUY signals · Need 60+ per ticker for statistical significance · Check back May 1 2026")
    else:
        st.info("No BUY signals recorded yet.")
except Exception as e:
    st.warning(f"BUY accuracy unavailable: {e}")

st.markdown("---")
st.subheader("📈 Model Accuracy Cache")
if acc_df.empty:
    st.info("Run **Reconcile outcomes** then **Recompute accuracy cache** in the sidebar to populate this section.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers tracked",  len(acc_df))
c2.metric("Avg Accuracy",     f"{acc_df['accuracy'].mean():.1%}")
c3.metric("Avg ROC-AUC",      f"{acc_df['roc_auc'].mean():.3f}")
c4.metric("Avg Brier Score",  f"{acc_df['brier_score'].mean():.3f}")

st.caption(
    "**Brier score**: lower is better (0 = perfect, 0.25 = random). "
    "**ROC-AUC**: higher is better (0.5 = random, 1.0 = perfect)."
)

# ── Leaderboard table ─────────────────────────────────────────────────────────
st.subheader("📋 Accuracy by Ticker")

def _color_accuracy(val):
    if pd.isna(val): return ""
    if val >= 0.60:  return "color: #00c853"
    if val >= 0.55:  return "color: #ffab00"
    return "color: #ff1744"

styled = (
    acc_df[["ticker", "accuracy", "roc_auc", "brier_score", "n_predictions"]]
    .style
    .format({
        "accuracy":    "{:.1%}",
        "roc_auc":     "{:.3f}",
        "brier_score": "{:.3f}",
    })
    .applymap(_color_accuracy, subset=["accuracy"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Bar chart ─────────────────────────────────────────────────────────────────
st.subheader("📊 Accuracy Comparison")
metric_choice = st.selectbox("Metric", ["accuracy", "roc_auc", "brier_score"])

chart = (
    alt.Chart(acc_df.sort_values(metric_choice, ascending=(metric_choice == "brier_score")))
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort=None, title="Ticker"),
        y=alt.Y(f"{metric_choice}:Q", title=metric_choice.replace("_", " ").title()),
        color=alt.Color(
            f"{metric_choice}:Q",
            scale=alt.Scale(
                scheme="redyellowgreen" if metric_choice != "brier_score" else "redyellowgreen",
                reverse=(metric_choice == "brier_score"),
            ),
            legend=None,
        ),
        tooltip=["ticker", "accuracy", "roc_auc", "brier_score", "n_predictions"],
    )
    .properties(height=350)
)
st.altair_chart(chart, use_container_width=True)

# ── Trend chart for selected ticker ───────────────────────────────────────────
st.subheader("📈 Accuracy Over Time")
st.caption("One bar per trading day — green = correct prediction, red = wrong. Use filters to zoom in.")

tc1, tc2, tc3 = st.columns([2, 1, 1])
sel_ticker = tc1.selectbox("Ticker", acc_df["ticker"].tolist(), key="trend_ticker")
days_back  = tc2.selectbox("Period", [14, 30, 60, 90, 180], index=1,
                            format_func=lambda x: f"Last {x} days", key="trend_days")
show_expanded = tc3.selectbox("View", ["Compact", "Expanded"], key="trend_view")

hist = load_prediction_history(sel_ticker, horizon=horizon, days=days_back)

if hist.empty:
    st.info(f"No prediction history for {sel_ticker} yet.")
else:
    hist["prediction_date"] = hist["prediction_date"].astype(str).str[:10]
    hist["correct_label"]   = hist["correct"].apply(lambda x: "Correct" if x == 1 else "Wrong")
    hist["return_pct"]      = hist["actual_return"].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A")
    hist["prob_pct"]        = hist["prob_up"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    total    = len(hist)
    correct  = hist["correct"].sum()
    acc_live = correct / total if total > 0 else 0

    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Predictions", total)
    sm2.metric("Correct", f"{int(correct)}")
    sm3.metric("Live accuracy", f"{acc_live:.1%}")
    sm4.metric("Avg return", f"{hist['actual_return'].mean():+.2%}" if hist['actual_return'].notna().any() else "N/A")

    chart_height = 200 if show_expanded == "Compact" else 350

    hist["rolling_acc"] = hist["correct"].rolling(10, min_periods=3).mean()

    # Simple bar chart
    bars = (
        alt.Chart(hist)
        .mark_bar(size=max(8, min(30, 300 // max(len(hist), 1))))
        .encode(
            x=alt.X("prediction_date:N", title="Date",
                    sort=None,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=11)),
            y=alt.Y("prob_up:Q", title="Predicted confidence",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            color=alt.condition(
                "datum.correct == 1",
                alt.value("#3B6D11"),
                alt.value("#A32D2D"),
            ),
            tooltip=[
                alt.Tooltip("prediction_date:N", title="Date"),
                alt.Tooltip("prob_up:Q", format=".1%", title="Predicted prob"),
                alt.Tooltip("correct_label:N", title="Result"),
                alt.Tooltip("return_pct:N", title="Actual return"),
                alt.Tooltip("signal:N", title="Signal"),
            ],
        )
    )

    # 50% reference line
    ref_df = pd.DataFrame({"y": [0.5]})
    ref = alt.Chart(ref_df).mark_rule(
        color="#888780", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])))

    # Rolling accuracy line
    hist2 = hist.dropna(subset=["rolling_acc"])
    if len(hist2) >= 3:
        line = (
            alt.Chart(hist2)
            .mark_line(color="#378ADD", strokeWidth=2, interpolate="monotone")
            .encode(
                x=alt.X("prediction_date:N", sort=None),
                y=alt.Y("rolling_acc:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("rolling_acc:Q", format=".1%", title="10d rolling acc")],
            )
        )
        final_chart = alt.layer(bars, ref, line)
    else:
        final_chart = alt.layer(bars, ref)

    st.altair_chart(
        final_chart.properties(
            title=f"{sel_ticker} — {horizon}d · bar height=confidence · green=correct · red=wrong",
            height=chart_height,
        ),
        use_container_width=True,
    )
    st.caption("Dashed line = 50% baseline. Blue line = 10-day rolling accuracy (shows after 3+ predictions).")

    # Raw table
    with st.expander("🧾 Raw prediction history"):
        display = hist[["prediction_date", "prob_up", "signal",
                        "confidence", "actual_up", "actual_return", "correct"]].copy()
        display["predicted"] = display["prob_up"].apply(lambda x: "⬆️ UP" if x > 0.5 else "⬇️ DOWN")
        display["actual"]    = display["actual_up"].apply(lambda x: "⬆️ UP" if x == 1 else "⬇️ DOWN")
        display["correct"]   = display["correct"].apply(lambda x: "✅" if x == 1 else "❌")
        display = display[["prediction_date", "prob_up", "predicted", "actual",
                           "actual_return", "correct", "signal", "confidence"]]
        display.columns = ["Date", "Prob Up", "Predicted", "Actual",
                          "Return", "Correct", "Signal", "Confidence"]
        st.dataframe(
            display.style.format({
                "Prob Up": "{:.1%}",
                "Return":  "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    csv = hist.to_csv(index=False).encode()
    st.download_button(
        f"⬇️ Download {sel_ticker} history",
        csv,
        file_name=f"{sel_ticker}_accuracy_{horizon}d.csv",
        mime="text/csv",
        key="dl_acc",
    )


# ── Calibration Analysis ──────────────────────────────────────────────────────
st.divider()
st.subheader("📐 Model Calibration")
st.caption("A well-calibrated model's predicted probability matches actual win rate. If we say 70% confidence, it should win 70% of the time.")

try:
    import sqlite3
    import numpy as np

    conn = sqlite3.connect("accuracy.db")
    cal_df = pd.read_sql("""
        SELECT
            ROUND(p.prob_up, 1) as prob_bucket,
            COUNT(*) as n,
            ROUND(AVG(CASE WHEN o.actual_return > 0 THEN 1.0 ELSE 0.0 END), 4) as actual_win_rate
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
            AND p.prediction_date=o.prediction_date
            AND p.horizon=o.horizon
        WHERE p.prob_up IS NOT NULL
        GROUP BY ROUND(p.prob_up, 1)
        ORDER BY prob_bucket
    """, conn)
    conn.close()

    if not cal_df.empty:
        # Metrics
        total = cal_df["n"].sum()
        weighted_win = (cal_df["actual_win_rate"] * cal_df["n"]).sum() / total
        high_conf = cal_df[cal_df["prob_bucket"] >= 0.70]
        high_conf_win = (high_conf["actual_win_rate"] * high_conf["n"]).sum() / high_conf["n"].sum() if not high_conf.empty else None
        high_conf_n = high_conf["n"].sum() if not high_conf.empty else 0

        # Calibration error (mean absolute deviation from diagonal)
        cal_df["error"] = abs(cal_df["prob_bucket"] - cal_df["actual_win_rate"])
        mean_cal_error = (cal_df["error"] * cal_df["n"]).sum() / total
        cal_label = "Good" if mean_cal_error < 0.05 else "Fair" if mean_cal_error < 0.10 else "Poor"

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Total predictions", f"{total:,}")
        cc2.metric("Overall win rate", f"{weighted_win:.1%}")
        if high_conf_win is not None:
            cc3.metric(f"High conf win rate (n={high_conf_n})",
                      f"{high_conf_win:.1%}",
                      delta=f"{'needs more data' if high_conf_n < 30 else ''}")
        cc4.metric("Calibration", cal_label,
                  delta=f"error={mean_cal_error:.1%}")

        st.markdown("---")

        # Chart
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            # Build calibration chart data
            chart_data = cal_df.copy()
            chart_data["perfect"] = chart_data["prob_bucket"]
            chart_data["bucket_label"] = chart_data["prob_bucket"].apply(lambda x: f"{x:.0%}")

            base = alt.Chart(chart_data)

            # Perfect calibration line
            perfect_line = base.mark_line(
                color="#E24B4A", strokeDash=[5, 5], strokeWidth=1.5
            ).encode(
                x=alt.X("prob_bucket:Q", title="Predicted probability",
                        scale=alt.Scale(domain=[0.1, 1.0]),
                        axis=alt.Axis(format=".0%")),
                y=alt.Y("perfect:Q", title="Actual win rate",
                        scale=alt.Scale(domain=[0.0, 1.0]),
                        axis=alt.Axis(format=".0%")),
            )

            # Actual points sized by n
            actual_points = base.mark_circle(color="#378ADD", opacity=0.7).encode(
                x=alt.X("prob_bucket:Q"),
                y=alt.Y("actual_win_rate:Q"),
                size=alt.Size("n:Q", scale=alt.Scale(range=[50, 800]),
                              legend=alt.Legend(title="Sample count")),
                tooltip=[
                    alt.Tooltip("prob_bucket:Q", format=".0%", title="Predicted"),
                    alt.Tooltip("actual_win_rate:Q", format=".1%", title="Actual win rate"),
                    alt.Tooltip("n:Q", title="Sample count"),
                    alt.Tooltip("error:Q", format=".1%", title="Calibration error"),
                ]
            )

            chart = alt.layer(perfect_line, actual_points).properties(
                title="Calibration plot — predicted vs actual win rate (bubble size = sample count)",
                height=350,
            )
            st.altair_chart(chart, use_container_width=True)

        with col_table:
            st.markdown("**Calibration by bucket**")
            display_cal = cal_df[["prob_bucket", "n", "actual_win_rate", "error"]].copy()
            display_cal.columns = ["Predicted", "Count", "Actual", "Error"]

            def color_error(val):
                if val < 0.05: return "color: #3B6D11"
                if val < 0.10: return "color: #854F0B"
                return "color: #A32D2D"

            st.dataframe(
                display_cal.style
                    .format({"Predicted": "{:.0%}", "Actual": "{:.1%}", "Error": "{:.1%}"})
                    .applymap(color_error, subset=["Error"]),
                use_container_width=True,
                hide_index=True,
            )

        # Key finding
        st.markdown("---")
        st.markdown("**Key finding**")

        min_n_bucket = cal_df["n"].min()
        max_n_bucket = cal_df["n"].max()
        dominant_bucket = cal_df.loc[cal_df["n"].idxmax(), "prob_bucket"]
        dominant_win = cal_df.loc[cal_df["n"].idxmax(), "actual_win_rate"]

        if high_conf_n < 30:
            finding = f"""
The model has **{total} total predictions** graded so far, but high-confidence buckets (≥70%) have only **{high_conf_n} samples** —
far too few to draw conclusions. The bulk of predictions ({cal_df.loc[cal_df['n'].idxmax(), 'n']} samples) sit in the
{dominant_bucket:.0%} confidence bucket with a {dominant_win:.1%} actual win rate, close to the 50% baseline.

**What this means:** We cannot yet distinguish model skill from random chance. Calibration requires 50+ samples per
bucket to be statistically meaningful. Current mean calibration error is {mean_cal_error:.1%} ({cal_label}).

**Action:** Continue collecting live predictions. Revisit calibration on **Apr 26 2026** when intraday data
and on **May 1 2026** when EOD SELL signal data reaches minimum sample thresholds.
            """
        else:
            if mean_cal_error < 0.05:
                finding = f"""
The model is **well-calibrated** with a mean error of {mean_cal_error:.1%}. Predicted probabilities closely
match actual win rates across all buckets. High-confidence signals (≥70%) win {high_conf_win:.1%} of the time
on {high_conf_n} samples — this is statistically meaningful and supports trusting the BUY threshold.
            """
            else:
                finding = f"""
The model shows **poor calibration** (mean error {mean_cal_error:.1%}). High-confidence signals (≥70%) are
winning only {high_conf_win:.1%} — below their predicted rate. Consider lowering the BUY threshold or
recalibrating the model's output probabilities using Platt scaling.
            """

        st.info(finding.strip())

except Exception as e:
    st.warning(f"Calibration data unavailable: {e}")
