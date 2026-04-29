# ui/2_Accuracy.py
# Forecast Accuracy Dashboard — reads from accuracy/sink.py
# Shows real prediction vs outcome accuracy, not training metrics.
#
# Organized in tabs:
#   Overview      — SPY-relative alpha + BUY-by-ticker cards + KPIs
#   By ticker     — leaderboard, bar chart, per-ticker trend chart
#   Calibration   — predicted prob vs actual win rate
#   Explorer      — BI-tool filterable view: multi-dim grouping, Wilson CIs,
#                   Bayesian shrinkage, sector breakdowns

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import sqlite3
from pathlib import Path

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (shared across all tabs)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Auto-reconcile + load shared data
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_ticker, tab_calib, tab_explore = st.tabs([
    "📊 Overview",
    "📈 By ticker",
    "📐 Calibration",
    "🔍 Explorer",
])


# =============================================================================
# TAB 1 — Overview
# =============================================================================

with tab_overview:

    # ── SPY-relative accuracy ─────────────────────────────────────────────────
    st.subheader("Daily Performance vs SPY")
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

    # ── BUY/SELL signal accuracy ──────────────────────────────────────────────
    st.subheader("BUY signal accuracy by ticker")
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
    st.subheader("Model Accuracy Cache")
    if acc_df.empty:
        st.info("Run **Reconcile outcomes** then **Recompute accuracy cache** in the sidebar to populate this section.")
    else:
        # ── KPI row ────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers tracked",  len(acc_df))
        c2.metric("Avg Accuracy",     f"{acc_df['accuracy'].mean():.1%}")
        c3.metric("Avg ROC-AUC",      f"{acc_df['roc_auc'].mean():.3f}")
        c4.metric("Avg Brier Score",  f"{acc_df['brier_score'].mean():.3f}")

        st.caption(
            "**Brier score**: lower is better (0 = perfect, 0.25 = random). "
            "**ROC-AUC**: higher is better (0.5 = random, 1.0 = perfect)."
        )


# =============================================================================
# TAB 2 — By ticker (leaderboard, bar chart, trend)
# =============================================================================

with tab_ticker:
    if acc_df.empty:
        st.info("Run **Reconcile outcomes** then **Recompute accuracy cache** in the sidebar to populate this tab.")
    else:
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

        # ── Bar chart ──────────────────────────────────────────────────────────
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

        # ── Trend chart for selected ticker ────────────────────────────────────
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

            # Each bar = one trading day. Full height = correct (green), short = wrong (red)
            hist["bar_val"] = hist["correct"].apply(lambda x: 1.0 if x == 1 else 0.3)

            bars = (
                alt.Chart(hist)
                .mark_bar()
                .encode(
                    x=alt.X("prediction_date:N", title="Date", sort=None,
                            axis=alt.Axis(labelAngle=0, labelFontSize=20, labelPadding=8)),
                    y=alt.Y("bar_val:Q", title="",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=None),
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

            # Rolling accuracy line
            hist2 = hist.dropna(subset=["rolling_acc"])
            layers = [bars]
            if len(hist2) >= 3:
                line = (
                    alt.Chart(hist2)
                    .mark_line(color="#378ADD", strokeWidth=2.5, interpolate="monotone")
                    .encode(
                        x=alt.X("prediction_date:N", sort=None,
                                axis=alt.Axis(labelAngle=0, labelFontSize=20, labelPadding=8)),
                        y=alt.Y("rolling_acc:Q",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format=".0%", title="Rolling accuracy")),
                        tooltip=[alt.Tooltip("rolling_acc:Q", format=".1%", title="10d rolling acc")],
                    )
                )
                layers.append(line)

            st.altair_chart(
                alt.layer(*layers)
                .resolve_scale(y="independent")
                .properties(
                    title=f"{sel_ticker} — {horizon}d predictions · green=correct · red=wrong",
                    height=chart_height,
                ),
                use_container_width=True,
            )
            st.caption("Green bar = correct prediction. Red bar = wrong. Blue line = 10-day rolling accuracy (shows after 3+ predictions).")

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


# =============================================================================
# TAB 3 — Calibration
# =============================================================================

with tab_calib:
    st.subheader("📐 Model Calibration")
    st.caption("A well-calibrated model's predicted probability matches actual win rate. If we say 70% confidence, it should win 70% of the time.")

    try:
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


# =============================================================================
# TAB 4 — Explorer (BI-tool view)
# =============================================================================

with tab_explore:
    st.subheader("🔍 Multi-dimensional accuracy explorer")
    st.caption(
        "Filter on any column · group by any combination of dimensions · "
        "Wilson 95% CIs and Bayesian shrinkage to handle small samples honestly."
    )

    DB_PATH = Path("accuracy.db")
    METADATA_CSV = Path("tickers_metadata.csv")

    # ── Statistics helpers ────────────────────────────────────────────────────
    def wilson_ci(wins: int, n: int, z: float = 1.96):
        if n == 0:
            return (0.0, 1.0)
        p = wins / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - spread), min(1.0, center + spread))

    def shrink(wins: int, n: int, prior_p: float, k: int = 50) -> float:
        a = prior_p * k
        b = (1 - prior_p) * k
        return (a + wins) / (a + b + n)

    @st.cache_data(ttl=300, show_spinner="Loading predictions…")
    def _load_explorer_data():
        if not DB_PATH.exists():
            return pd.DataFrame()
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql("""
            SELECT p.ticker, p.prediction_date, p.horizon, p.prob_up,
                   o.actual_return, o.actual_up
            FROM predictions p
            JOIN outcomes o
              ON p.ticker = o.ticker
             AND p.prediction_date = o.prediction_date
             AND p.horizon = o.horizon
            WHERE o.actual_return IS NOT NULL
        """, conn)
        conn.close()

        if df.empty:
            return df

        df["prediction_date"] = pd.to_datetime(df["prediction_date"])
        df["signal"] = np.where(df["prob_up"] > 0.5, "BUY", "HOLD")
        df["correct"] = (df["actual_up"] == 1).astype(int)
        df["year"] = df["prediction_date"].dt.year.astype(str)
        df["quarter"] = df["prediction_date"].dt.to_period("Q").astype(str)
        df["month"] = df["prediction_date"].dt.to_period("M").astype(str)
        df["weekday"] = df["prediction_date"].dt.day_name()

        if METADATA_CSV.exists():
            meta = pd.read_csv(METADATA_CSV)
            meta["ticker"] = meta["ticker"].str.upper().str.strip()
            df["ticker"] = df["ticker"].str.upper().str.strip()
            if "bucket" in meta.columns:
                df = df.merge(meta[["ticker", "bucket"]], on="ticker", how="left")
                df["bucket"] = df["bucket"].fillna("UNKNOWN")
            else:
                df["bucket"] = "UNKNOWN"
        else:
            df["bucket"] = "UNKNOWN"

        return df

    def aggregate(df, group_cols, prior_p, prior_k):
        if df.empty:
            return pd.DataFrame()
        if not group_cols:
            wins = int(df["correct"].sum())
            n = len(df)
            ci = wilson_ci(wins, n)
            return pd.DataFrame([{
                "scope": "OVERALL", "n": n, "wins": wins,
                "accuracy": wins / n if n else 0.0,
                "acc_ci_lo": ci[0], "acc_ci_hi": ci[1],
                "acc_shrunk": shrink(wins, n, prior_p, prior_k),
                "avg_return": df["actual_return"].mean(),
                "return_z": (df["actual_return"].mean() / df["actual_return"].std()
                             if df["actual_return"].std() else 0.0),
                "sig_above_50": ci[0] > 0.50,
                "sig_below_50": ci[1] < 0.50,
            }])

        grouped = (df.groupby(group_cols)
                     .agg(n=("correct", "size"), wins=("correct", "sum"),
                          avg_return=("actual_return", "mean"),
                          return_std=("actual_return", "std"))
                     .reset_index())
        grouped["accuracy"] = grouped["wins"] / grouped["n"]
        cis = grouped.apply(lambda r: wilson_ci(int(r["wins"]), int(r["n"])), axis=1)
        grouped["acc_ci_lo"] = [c[0] for c in cis]
        grouped["acc_ci_hi"] = [c[1] for c in cis]
        grouped["acc_shrunk"] = grouped.apply(
            lambda r: shrink(int(r["wins"]), int(r["n"]), prior_p, prior_k), axis=1)
        grouped["return_z"] = grouped["avg_return"] / grouped["return_std"].replace(0, np.nan)
        grouped["sig_above_50"] = grouped["acc_ci_lo"] > 0.50
        grouped["sig_below_50"] = grouped["acc_ci_hi"] < 0.50
        return grouped.sort_values("acc_shrunk", ascending=False)

    df_raw = _load_explorer_data()
    if df_raw.empty:
        st.warning(
            f"No data found at {DB_PATH}. Run **Reconcile outcomes** in the sidebar first."
        )
        st.stop()

    if not METADATA_CSV.exists():
        st.info(
            f"⚠️ {METADATA_CSV} not found — sector grouping will show 'UNKNOWN' for all tickers. "
            f"Build it via your sector_accuracy work to enable Sector grouping."
        )

    # ── Filters (in expander, page-local — sidebar is for global settings) ────
    with st.expander("🔧 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)

        # Date range
        min_d = df_raw["prediction_date"].min().date()
        max_d = df_raw["prediction_date"].max().date()
        date_range = f1.date_input(
            "Date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="explorer_dates",
        )

        # Signal type
        signals_avail = sorted(df_raw["signal"].unique().tolist())
        selected_signals = f2.multiselect(
            "Signal", options=signals_avail, default=["BUY"],
            key="explorer_signals",
        )

        # Horizon
        horizons_avail = sorted(df_raw["horizon"].unique().tolist())
        selected_horizons = f3.multiselect(
            "Horizon (days)", options=horizons_avail, default=horizons_avail,
            key="explorer_horizons",
        )

        f4, f5 = st.columns(2)

        # Sector
        sectors_avail = sorted(df_raw["bucket"].unique().tolist())
        selected_sectors = f4.multiselect(
            "Sector", options=sectors_avail, default=sectors_avail,
            key="explorer_sectors",
        )

        # Ticker (empty = all)
        tickers_avail = sorted(df_raw["ticker"].unique().tolist())
        selected_tickers = f5.multiselect(
            "Ticker (empty = all)", options=tickers_avail, default=[],
            key="explorer_tickers",
        )

        f6, f7, f8 = st.columns(3)
        prob_lo, prob_hi = f6.slider(
            "prob_up range", 0.0, 1.0, (0.0, 1.0), 0.05,
            key="explorer_prob",
        )
        min_n = f7.number_input(
            "Min sample size per group", min_value=1, value=5, step=1,
            key="explorer_min_n",
        )
        prior_k = f8.slider(
            "Bayesian prior strength", 0, 200, 50, 10,
            help="Pseudo-count for Beta-Binomial shrinkage. Higher = more shrinkage. "
                 "Set to 0 for raw accuracy.",
            key="explorer_prior_k",
        )

    # Apply filters
    df = df_raw.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df = df[(df["prediction_date"].dt.date >= date_range[0]) &
                (df["prediction_date"].dt.date <= date_range[1])]
    if selected_signals:
        df = df[df["signal"].isin(selected_signals)]
    if selected_horizons:
        df = df[df["horizon"].isin(selected_horizons)]
    if selected_sectors:
        df = df[df["bucket"].isin(selected_sectors)]
    if selected_tickers:
        df = df[df["ticker"].isin(selected_tickers)]
    df = df[(df["prob_up"] >= prob_lo) & (df["prob_up"] <= prob_hi)]

    if df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    prior_p = df["correct"].mean()

    # ── Headline metrics ──────────────────────────────────────────────────────
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Predictions", f"{len(df):,}")
    e2.metric("Overall accuracy", f"{prior_p * 100:.1f}%")
    ci_overall = wilson_ci(int(df["correct"].sum()), len(df))
    e3.metric("95% CI",
              f"[{ci_overall[0] * 100:.1f}, {ci_overall[1] * 100:.1f}]%")
    e4.metric("Avg return", f"{df['actual_return'].mean() * 100:.3f}%")
    e5.metric("Sectors / Tickers",
              f"{df['bucket'].nunique()} / {df['ticker'].nunique()}")

    # ── Group-by selector ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Group by**")
    GROUP_OPTIONS = {
        "(none)": None,
        "Sector": "bucket",
        "Ticker": "ticker",
        "Horizon": "horizon",
        "Signal": "signal",
        "Year": "year",
        "Quarter": "quarter",
        "Month": "month",
        "Weekday": "weekday",
    }
    gb1, gb2, gb3 = st.columns(3)
    g1 = gb1.selectbox("Primary", list(GROUP_OPTIONS.keys()), index=1, key="explorer_g1")
    g2 = gb2.selectbox("Secondary", list(GROUP_OPTIONS.keys()), index=0, key="explorer_g2")
    g3 = gb3.selectbox("Tertiary", list(GROUP_OPTIONS.keys()), index=0, key="explorer_g3")

    group_cols = []
    for sel in [g1, g2, g3]:
        col = GROUP_OPTIONS[sel]
        if col and col not in group_cols:
            group_cols.append(col)

    # ── Aggregate + table ─────────────────────────────────────────────────────
    result = aggregate(df, group_cols, prior_p, prior_k)
    if "n" in result.columns:
        result = result[result["n"] >= min_n]

    st.markdown("---")
    st.markdown(f"**Results** — {len(result)} groups (min_n = {min_n})")

    if result.empty:
        st.info("No groups meet the minimum sample size threshold. Lower min_n above.")
    else:
        display_cols = group_cols + [
            "n", "accuracy", "acc_ci_lo", "acc_ci_hi", "acc_shrunk",
            "sig_above_50", "sig_below_50", "avg_return", "return_z",
        ]
        display_cols = [c for c in display_cols if c in result.columns]

        column_config = {
            "n": st.column_config.NumberColumn("n", format="%d"),
            "accuracy": st.column_config.ProgressColumn(
                "Accuracy", format="%.3f", min_value=0.0, max_value=1.0),
            "acc_ci_lo": st.column_config.NumberColumn("CI lo", format="%.3f"),
            "acc_ci_hi": st.column_config.NumberColumn("CI hi", format="%.3f"),
            "acc_shrunk": st.column_config.ProgressColumn(
                "Shrunk", format="%.3f", min_value=0.0, max_value=1.0,
                help="Bayesian-shrunk accuracy. Use this for ranking — handles small samples."),
            "sig_above_50": st.column_config.CheckboxColumn(
                "Edge?", help="Wilson CI lower bound > 50%. Real edge if checked."),
            "sig_below_50": st.column_config.CheckboxColumn(
                "Anti-edge?", help="Wilson CI upper bound < 50%. Real anti-edge if checked."),
            "avg_return": st.column_config.NumberColumn("Avg ret", format="%.4f"),
            "return_z": st.column_config.NumberColumn("Ret Z", format="%.2f"),
        }

        st.dataframe(
            result[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            height=min(600, 50 + 35 * len(result)),
        )

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV", csv, "accuracy_explorer.csv", "text/csv",
            key="explorer_download",
        )

    # ── Charts ────────────────────────────────────────────────────────────────
    if not result.empty and group_cols and len(result) > 1:
        st.markdown("---")
        st.markdown("**Visualization**")

        chart_tab_bar, chart_tab_scatter, chart_tab_time = st.tabs([
            "Top groups (bar)",
            "Sample size vs accuracy",
            "Rolling accuracy",
        ])

        primary_col = group_cols[0]

        with chart_tab_bar:
            cb1, cb2 = st.columns([3, 1])
            with cb2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["acc_shrunk", "accuracy", "avg_return", "n", "return_z"],
                    index=0, key="explorer_sort",
                )
                top_n = st.slider("Show top", 5, min(50, len(result)),
                                  min(20, len(result)), key="explorer_topn")
            with cb1:
                chart_data = result.nlargest(top_n, sort_by).copy()
                chart_data["label"] = chart_data[group_cols].astype(str).agg(" / ".join, axis=1)
                bar_chart = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", sort=None, title="",
                                axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y(f"{sort_by}:Q", title=sort_by),
                        color=alt.Color("acc_shrunk:Q",
                                        scale=alt.Scale(scheme="redyellowgreen",
                                                        domain=[0.30, 0.50, 0.70]),
                                        legend=None),
                        tooltip=["label:N", "n:Q", "accuracy:Q", "acc_ci_lo:Q",
                                 "acc_ci_hi:Q", "acc_shrunk:Q", "avg_return:Q"],
                    )
                    .properties(height=440)
                )
                if sort_by in ("accuracy", "acc_shrunk", "acc_ci_lo", "acc_ci_hi"):
                    rule = alt.Chart(pd.DataFrame({"y": [0.50]})).mark_rule(
                        strokeDash=[4, 4], color="gray"
                    ).encode(y="y:Q")
                    st.altair_chart(bar_chart + rule, use_container_width=True)
                else:
                    st.altair_chart(bar_chart, use_container_width=True)

        with chart_tab_scatter:
            scatter_data = result.copy()
            scatter_data["label"] = scatter_data[group_cols].astype(str).agg(" / ".join, axis=1)
            scatter_chart = (
                alt.Chart(scatter_data)
                .mark_circle(opacity=0.7)
                .encode(
                    x=alt.X("n:Q", scale=alt.Scale(type="log"),
                            title="Sample size (log scale)"),
                    y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.0, 1.0]),
                            axis=alt.Axis(format=".0%")),
                    size=alt.Size("n:Q", scale=alt.Scale(range=[50, 800]),
                                  legend=None),
                    color=alt.Color("acc_shrunk:Q",
                                    scale=alt.Scale(scheme="redyellowgreen",
                                                    domain=[0.30, 0.50, 0.70]),
                                    legend=alt.Legend(title="Shrunk acc")),
                    tooltip=["label:N", "n:Q", "accuracy:Q", "acc_ci_lo:Q",
                             "acc_ci_hi:Q", "avg_return:Q"],
                )
                .properties(height=440)
            )
            rule = alt.Chart(pd.DataFrame({"y": [0.50]})).mark_rule(
                strokeDash=[4, 4], color="gray"
            ).encode(y="y:Q")
            st.altair_chart(scatter_chart + rule, use_container_width=True)
            st.caption(
                "Top-right = real edge (high accuracy + many samples). "
                "Top-left = looks good but unreliable (high accuracy, few samples). "
                "Bottom = anti-edge or coin flip."
            )

        with chart_tab_time:
            rw = st.slider("Rolling window (predictions)", 20, 500, 100, 10,
                           key="explorer_rolling")
            ts = df.sort_values("prediction_date").copy()
            ts["rolling_acc"] = (ts["correct"]
                                  .rolling(window=rw,
                                           min_periods=max(10, rw // 4))
                                  .mean())
            ts["cumulative_acc"] = ts["correct"].expanding(min_periods=10).mean()
            time_long = ts[["prediction_date", "rolling_acc", "cumulative_acc"]].melt(
                id_vars="prediction_date", var_name="metric", value_name="accuracy"
            ).dropna()
            time_chart = (
                alt.Chart(time_long)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("prediction_date:T", title="Date"),
                    y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.0, 1.0]),
                            axis=alt.Axis(format=".0%")),
                    color=alt.Color("metric:N", legend=alt.Legend(title="")),
                    tooltip=["prediction_date:T", "metric:N",
                             alt.Tooltip("accuracy:Q", format=".1%")],
                )
                .properties(height=380)
            )
            rule = alt.Chart(pd.DataFrame({"y": [0.50]})).mark_rule(
                strokeDash=[4, 4], color="gray"
            ).encode(y="y:Q")
            st.altair_chart(time_chart + rule, use_container_width=True)
            st.caption(
                "Rolling accuracy declining = edge is decaying. "
                "Stable rolling line = stable signal. Spike events = regime shifts."
            )

    # ── How to read ───────────────────────────────────────────────────────────
    with st.expander("How to read this tab"):
        st.markdown("""
        - **Use `Shrunk` for ranking.** Bayesian shrinkage pulls small-sample groups toward the overall mean.
          Healthcare 42%/n=50 stays slightly below average; Infrastructure 91%/n=11 shrinks to ~60%.
        - **Use `Edge?` / `Anti-edge?` for go/no-go.** They flip true only when the Wilson 95% CI excludes
          50% — i.e. when the data actually rules out "this group is a coin flip" at standard confidence.
        - **Most groups at small n won't be statistically significant.** That's the feature, not a bug.
          A blocklist on noise is worse than no blocklist.
        - **The 5-click AUC sanity check:** set `prob_up range` to [0.80, 1.00], group by `Horizon`,
          look at accuracy. If high-confidence BUYs aren't meaningfully more accurate than low-confidence
          ones, the ensemble's headline AUC is a fiction.
        - **Out-of-sample test:** set date range to "all data minus last 30 days," note top sectors by
          shrunk accuracy. Then date range to "last 30 days only" — do the same sectors still outperform?
          If yes, ship a multiplier. If no, the pattern was data-mining noise.
        """)
