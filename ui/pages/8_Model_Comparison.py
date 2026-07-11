"""
Model Comparison — per-ticker vs GLOBAL ensemble.
Read-only. GLOBAL is NOT in the live signal (prob_eff = per-ticker x multipliers).
Each model's accuracy uses its OWN null filter (ranker has far fewer non-null rows).
"""
import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison — per-ticker vs GLOBAL")
st.caption("Read-only. GLOBAL is NOT in the live signal. Ensemble ~46-49%; ranker ~54% at h3/h5 on a small sample.")

DB = Path(__file__).parent.parent.parent / "accuracy.db"
if not DB.exists():
    DB = Path.home() / "Desktop" / "ML_Quant_Fund" / "accuracy.db"

def q(sql):
    con = sqlite3.connect(str(DB))
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()

win = st.selectbox("Accuracy window", ["All time", "Last 30 days", "Last 7 days"], index=0)
wd_o = {"All time":"", "Last 30 days":"AND o.prediction_date >= date('now','-30 days')",
        "Last 7 days":"AND o.prediction_date >= date('now','-7 days')"}[win]

st.subheader("Directional accuracy (each model, its own non-null rows)")
st.caption("50% = coin flip. Below 50% = worse than random. Note the differing n per model.")
pt = q(f"SELECT o.horizon AS h, COUNT(*) AS n, ROUND(100.0*SUM(((p.prob_up>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS acc_pct FROM predictions p JOIN outcomes o ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon WHERE p.prob_up IS NOT NULL {wd_o} GROUP BY o.horizon;")
ge = q(f"SELECT o.horizon AS h, COUNT(*) AS n, ROUND(100.0*SUM(((p.prob_up_global>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS acc_pct FROM predictions p JOIN outcomes o ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon WHERE p.prob_up_global IS NOT NULL {wd_o} GROUP BY o.horizon;")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Per-ticker** (live workhorse)"); st.dataframe(pt, use_container_width=True, hide_index=True)
with c2:
    st.markdown("**GLOBAL ensemble**"); st.dataframe(ge, use_container_width=True, hide_index=True)

st.subheader("GLOBAL ensemble: before vs after freeze (2026-05-26)")
split = q("SELECT o.horizon AS h, CASE WHEN o.prediction_date < '2026-05-26' THEN 'pre-freeze' ELSE 'post (stale)' END AS era, COUNT(*) AS n, ROUND(100.0*SUM(((p.prob_up_global>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS global_pct FROM predictions p JOIN outcomes o ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon WHERE p.prob_up_global IS NOT NULL GROUP BY o.horizon, era ORDER BY o.horizon, era;")
st.dataframe(split, use_container_width=True, hide_index=True)

st.subheader("Agreement rate — per-ticker vs GLOBAL ensemble")
agr = q(f"SELECT horizon AS h, COUNT(*) AS n, ROUND(100.0*SUM(CASE WHEN (prob_up>=0.5)=(prob_up_global>=0.5) THEN 1 ELSE 0 END)/COUNT(*),1) AS agree_pct, ROUND(AVG(ABS(prob_up - prob_up_global)),3) AS mean_divergence FROM predictions WHERE prob_up_global IS NOT NULL {wd_o.replace('o.prediction_date','prediction_date')} GROUP BY horizon;")
st.dataframe(agr, use_container_width=True, hide_index=True)

st.subheader("Latest predictions — biggest disagreements")
h_sel = st.radio("Horizon", [1,3,5], horizontal=True)
dis = q(f"SELECT ticker, ROUND(prob_up,3) AS perticker, ROUND(prob_up_global,3) AS global, ROUND(prob_up - prob_up_global,3) AS delta, CASE WHEN (prob_up>=0.5)=(prob_up_global>=0.5) THEN 'agree' ELSE 'DISAGREE' END AS direction FROM predictions WHERE prediction_date=(SELECT MAX(prediction_date) FROM predictions) AND prob_up_global IS NOT NULL AND horizon={h_sel} ORDER BY ABS(prob_up - prob_up_global) DESC LIMIT 25;")
def _dir(v):
    return "background-color:#7f1d1d;color:#fff" if v == "DISAGREE" else ""
if not dis.empty:
    st.dataframe(dis.style.applymap(_dir, subset=["direction"]), use_container_width=True, hide_index=True)
else:
    st.info("No predictions with GLOBAL for the latest date/horizon.")

st.divider()
st.caption("Historical GLOBAL accuracy reflects the model version at prediction time. Fresh needs ~1 week to mature. Ranker's ~54% is small-sample.")

# ── Horizon Health (shared panel) ────────────────────────────────────────────
st.divider()
try:
    from ui._horizon_health_panel import render_horizon_health
    render_horizon_health()
except Exception as _hh_e:
    st.caption(f"(horizon health panel unavailable: {_hh_e})")
