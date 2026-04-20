# ui/10_AB_Test.py
# A/B Test Results Dashboard
# Compares Run A (no UW) vs Run B (UW enhanced) and sentiment pre/post

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import sqlite3
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date
from pathlib import Path

DB_PATH    = Path(_ROOT) / "accuracy.db"
CACHE_A_UW = Path(_ROOT) / "data" / "ab_polygon_A.json"
CACHE_B_UW = Path(_ROOT) / "data" / "ab_polygon_B.json"
CACHE_A_ST = Path(_ROOT) / "data" / "ab_cache_A.json"
CACHE_B_ST = Path(_ROOT) / "data" / "ab_cache_B.json"

st.set_page_config(page_title="A/B Test Results", page_icon="🧪", layout="wide")
st.title("🧪 A/B Test Results")
st.caption("Run A = baseline &nbsp;·&nbsp; Run B = enhanced &nbsp;·&nbsp; Compare after market close.")


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _score_signals(sigs: dict, date_str: str) -> tuple:
    try:
        conn = sqlite3.connect(DB_PATH)
        correct = total = buy_correct = buy_total = 0
        for t, s in sigs.items():
            row = conn.execute(
                "SELECT actual_return FROM outcomes WHERE ticker=? AND prediction_date=? AND horizon=1",
                (t, date_str)
            ).fetchone()
            if row:
                prob      = s.get("prob_eff", s.get("prob", 0.5))
                pred_up   = prob > 0.5
                actual_up = row[0] > 0
                correct  += int(pred_up == actual_up)
                total    += 1
                if s.get("signal") == "BUY":
                    buy_correct += int(actual_up)
                    buy_total   += 1
        conn.close()
        return correct, total, buy_correct, buy_total
    except Exception:
        return 0, 0, 0, 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    test_type = st.radio("Test type", ["Polygon / UW", "Sentiment"])
    st.markdown("---")
    st.markdown("**How to run:**")
    if test_type == "Polygon / UW":
        st.code("abpolygon --save A\nabpolygon --save B\n# next day:\nabpolygon --compare", language="bash")
    else:
        st.code("# Auto: pipeline saves A\n# After sentiment + runfund:\nabtest --save B\n# next day:\nabtest --compare", language="bash")
    if st.button("🔄 Refresh"):
        st.rerun()


# ── Polygon A/B ───────────────────────────────────────────────────────────────
if test_type == "Polygon / UW":
    st.subheader("🌑 Polygon / Unusual Whales A/B test")

    da = _load_json(CACHE_A_UW)
    db = _load_json(CACHE_B_UW)

    if not da or not db:
        st.info("No snapshots yet. Run: abpolygon --save A && abpolygon --save B")
        st.stop()

    sigs_a = {s["ticker"]: s for s in da.get("signals", []) if s.get("horizon") == 1}
    sigs_b = {s["ticker"]: s for s in db.get("signals", []) if s.get("horizon") == 1}
    date_str = da.get("date", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run A signals",  len(sigs_a))
    c2.metric("Run B signals",  len(sigs_b))
    c3.metric("Date",           date_str)
    pa = db.get("polygon_active", {})
    c4.metric("Polygon active", f"dp={pa.get('dark_pool',False)} opts={pa.get('options',False)}")

    changed = [
        (t, sigs_a[t].get("prob_eff", 0.5), sigs_b[t].get("prob_eff", 0.5),
         sigs_b[t].get("polygon_combined", 1.0))
        for t in sigs_a if t in sigs_b
        and abs(sigs_a[t].get("prob_eff", 0.5) - sigs_b[t].get("prob_eff", 0.5)) > 0.005
    ]

    st.metric("Signals changed by Polygon", len(changed))

    if changed:
        st.subheader("Top changes A → B")
        rows = sorted(changed, key=lambda x: abs(x[2]-x[1]), reverse=True)[:15]
        cdf  = pd.DataFrame(rows, columns=["Ticker","Prob A","Prob B","Combined mult"])
        cdf["Prob A"] = cdf["Prob A"].apply(lambda x: f"{x:.3f}")
        cdf["Prob B"] = cdf["Prob B"].apply(lambda x: f"{x:.3f}")
        cdf["Combined mult"] = cdf["Combined mult"].apply(lambda x: f"{x:.3f}")
        st.dataframe(cdf, use_container_width=True, hide_index=True)

    st.markdown("---")
    ca, ta, ca_b, ta_b = _score_signals(sigs_a, date_str)
    cb, tb, cb_b, tb_b = _score_signals(sigs_b, date_str)

    if ta > 0 and tb > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Run A accuracy",   f"{100*ca/ta:.1f}%", f"{ca}/{ta}")
        col2.metric("Run B accuracy",   f"{100*cb/tb:.1f}%", f"{cb}/{tb}",
                    delta=f"{100*(cb/tb - ca/ta):+.1f}%")
        diff = (cb/tb - ca/ta) * 100
        col3.metric("Polygon impact",   f"{diff:+.1f}%",
                    "B better" if diff > 0.5 else "A better" if diff < -0.5 else "No difference")

        if ta_b > 0 and tb_b > 0:
            st.markdown("**BUY signal accuracy**")
            bc1, bc2 = st.columns(2)
            bc1.metric("Run A BUY accuracy", f"{100*ca_b/ta_b:.1f}%", f"{ca_b}/{ta_b}")
            bc2.metric("Run B BUY accuracy", f"{100*cb_b/tb_b:.1f}%", f"{cb_b}/{tb_b}")
    else:
        st.info(f"Outcomes not yet available for {date_str} — check after market close.")

    if not pa.get("dark_pool") and not pa.get("options"):
        st.warning("Polygon returned 403 — upgrade to Stocks Developer + Options Starter to activate.")


# ── Sentiment A/B ─────────────────────────────────────────────────────────────
else:
    st.subheader("📰 Sentiment A/B test — pre vs post sentiment")

    da = _load_json(CACHE_A_ST)
    db = _load_json(CACHE_B_ST)

    if not da or not db:
        st.info("No snapshots yet.\n1. Pipeline saves A automatically at 7 AM\n2. After runfund post-sentiment: abtest --save B")
        st.stop()

    sigs_a   = {s["ticker"]: s for s in da.get("signals", []) if s.get("horizon") == 1}
    sigs_b   = {s["ticker"]: s for s in db.get("signals", []) if s.get("horizon") == 1}
    date_str = da.get("date", "")

    c1, c2, c3 = st.columns(3)
    c1.metric("Run A (pre-sentiment)",  da.get("generated_at", "")[:19])
    c2.metric("Run B (post-sentiment)", db.get("generated_at", "")[:19])
    c3.metric("Date", date_str)

    changed = [
        (t, sigs_a[t].get("prob_eff", 0.5), sigs_b[t].get("prob_eff", 0.5))
        for t in sigs_a if t in sigs_b
        and abs(sigs_a[t].get("prob_eff", 0.5) - sigs_b[t].get("prob_eff", 0.5)) > 0.01
    ]
    st.metric("Signals changed by sentiment", len(changed))

    if changed:
        rows = sorted(changed, key=lambda x: abs(x[2]-x[1]), reverse=True)[:15]
        cdf  = pd.DataFrame(rows, columns=["Ticker","Prob A","Prob B"])
        cdf["Prob A"] = cdf["Prob A"].apply(lambda x: f"{x:.3f}")
        cdf["Prob B"] = cdf["Prob B"].apply(lambda x: f"{x:.3f}")
        st.dataframe(cdf, use_container_width=True, hide_index=True)

    st.markdown("---")
    ca, ta, _, _ = _score_signals(sigs_a, date_str)
    cb, tb, _, _ = _score_signals(sigs_b, date_str)

    if ta > 0 and tb > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Pre-sentiment accuracy",  f"{100*ca/ta:.1f}%")
        col2.metric("Post-sentiment accuracy", f"{100*cb/tb:.1f}%",
                    delta=f"{100*(cb/tb - ca/ta):+.1f}%")
        diff = (cb/tb - ca/ta) * 100
        col3.metric("Sentiment impact", f"{diff:+.1f}%",
                    "B better" if diff > 0.5 else "A better" if diff < -0.5 else "No difference")
    else:
        st.info(f"Outcomes not yet available for {date_str} — check after 4 AM Vietnam.")
