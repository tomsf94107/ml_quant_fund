"""
Shared horizon-health panel — reads the horizon_health table (written daily by
analysis.horizon_health_compute in Pipeline B Stage 7). Compute-once-display-many:
this only READS the stored snapshot + history. Import render_horizon_health()
anywhere that wants to show it.
"""
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

def _db():
    p = Path(__file__).parent.parent / "accuracy.db"
    return p if p.exists() else Path.home() / "Desktop" / "ML_Quant_Fund" / "accuracy.db"

def _q(sql):
    con = sqlite3.connect(str(_db()))
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()

def render_horizon_health():
    st.subheader("Horizon Health — high-confidence accuracy per horizon")
    st.caption("prob_up≥0.70 signals. h1 broke ~week 2026-24 (mid-June); h3/h5 are the working edge. "
               "Written daily by Pipeline B. h1 back above 55% for 2-3 straight weeks = regime passed.")

    # table may not exist yet (first run before B Stage 7)
    try:
        latest_date = _q("SELECT MAX(run_date) d FROM horizon_health").iloc[0]["d"]
    except Exception:
        st.info("No horizon_health data yet — runs with the next Pipeline B (Stage 7).")
        return
    if latest_date is None:
        st.info("No horizon_health data yet — runs with the next Pipeline B (Stage 7).")
        return

    # ── latest snapshot ──
    snap = _q(f"""
        SELECT band, window_days, horizon, n, acc_pct, avg_ret_pct
        FROM horizon_health WHERE run_date='{latest_date}'
        ORDER BY band, window_days, horizon;
    """)
    st.markdown(f"**Latest snapshot** ({latest_date})")
    # pivot to h1/h3/h5 columns per band+window
    for (band, win), grp in snap.groupby(["band", "window_days"]):
        cols = st.columns([1.4, 1, 1, 1])
        cols[0].markdown(f"**{band} · {win}d**")
        for i, h in enumerate([1, 3, 5]):
            row = grp[grp["horizon"] == h]
            if row.empty:
                cols[i+1].metric(f"h{h}", "—")
                continue
            acc = row.iloc[0]["acc_pct"]; ret = row.iloc[0]["avg_ret_pct"]; n = int(row.iloc[0]["n"])
            # color hint via delta sign (green up / red down vs 50)
            delta = None
            if acc is not None:
                delta = f"{acc-50:+.1f} vs 50"
            cols[i+1].metric(f"h{h} ({n})", f"{acc}%" if acc is not None else "—",
                             delta=delta, delta_color="normal")

    # ── h1 status banner ──
    h1 = snap[(snap["band"]=="highconf") & (snap["window_days"]==30) & (snap["horizon"]==1)]
    if not h1.empty and h1.iloc[0]["acc_pct"] is not None:
        a = h1.iloc[0]["acc_pct"]
        if a >= 55:
            st.success(f"h1 high-conf 30d = {a}% → RECOVERING (watch for 2-3 straight weeks)")
        elif a >= 50:
            st.warning(f"h1 high-conf 30d = {a}% → WEAK (near coin-flip)")
        else:
            st.error(f"h1 high-conf 30d = {a}% → BROKEN (below random; use h3/h5 for BUYs)")

    # ── trend over time (as history accumulates) ──
    hist = _q("""
        SELECT run_date, horizon, acc_pct
        FROM horizon_health
        WHERE band='highconf' AND window_days=30
        ORDER BY run_date, horizon;
    """)
    if hist["run_date"].nunique() >= 2:
        st.markdown("**Trend — high-conf 30d accuracy over time**")
        pivot = hist.pivot(index="run_date", columns="horizon", values="acc_pct")
        pivot.columns = [f"h{c}" for c in pivot.columns]
        st.line_chart(pivot)
    else:
        st.caption("Trend chart appears once ≥2 daily snapshots accumulate.")
