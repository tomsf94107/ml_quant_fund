# ui/pages/4_Events.py
# Market Events Calendar — DB-backed (UW data refreshed nightly).
# Sources:
#   - Economic calendar: accuracy.db.economic_calendar
#     (refreshed M/W/F by scripts/refresh_economic_calendar.py)
#   - Earnings: accuracy.db.earnings_calendar JOIN earnings_cache
#     (refreshed daily by scripts/refresh_earnings_calendar.py)
#   - Risk score: passed to Dashboard signal gate via session state

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Market Events Calendar",
    page_icon="📅",
    layout="wide",
)

st.title("📅 Market Events Calendar")
st.caption("Forward-looking calendar — earnings and economic events. Feeds risk gate into Dashboard.")

UW_KEY   = os.getenv("UW_API_KEY", "")
HDRS     = {"Authorization": f"Bearer {UW_KEY}"}
BASE_URL = "https://api.unusualwhales.com"
DB_PATH  = Path(_ROOT) / "accuracy.db"

BIOTECH_TICKERS = {"ORIC", "VKTX", "QURE", "INSM", "MRNA", "BRKR"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_tickers() -> list[str]:
    p = Path(_ROOT) / "tickers.txt"
    tickers = [t.strip().upper() for t in p.read_text().splitlines()
               if t.strip() and not t.startswith("#")] if p.exists() else []
    wl = Path(_ROOT) / "tickers_watchlist.txt"
    if wl.exists():
        tickers += [t.strip().upper() for t in wl.read_text().splitlines()
                    if t.strip() and not t.startswith("#")]
    return list(dict.fromkeys(tickers))


def _load_meta() -> dict:
    try:
        mp = Path(_ROOT) / "tickers_metadata.csv"
        if mp.exists():
            df = pd.read_csv(mp)
            return df.set_index("ticker").to_dict("index")
    except Exception:
        pass
    return {}


_META    = _load_meta()
_TICKERS = _load_tickers()


# ── UW Data fetchers ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uw_economic_calendar(start: str, end: str) -> pd.DataFrame:
    """
    Read economic events from accuracy.db.economic_calendar.
    Populated M/W/F by scripts/refresh_economic_calendar.py via cron.
    
    May 13 2026: Switched from live UW fetch to DB read to eliminate
    per-page-view UW API calls. DB refreshes 3x/week (M/W/F 15:55 ET).
    Worst-case staleness: 2 days (acceptable for risk_gate's 3-day window).
    """
    import sqlite3
    from pathlib import Path
    db_path = Path(__file__).parent.parent.parent / "accuracy.db"
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT event_date, event_time, title, impact, country, forecast, previous
                FROM economic_calendar
                WHERE event_date >= ? AND event_date <= ?
                ORDER BY event_date, event_time
            """, (start, end)).fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            "title":    r["title"],
            "category": "Economic",
            "date":     r["event_date"],
            "time":     r["event_time"] or "",
            "impact":   r["impact"],
            "source":   "DB (UW)",
            "tickers":  "SPY, QQQ, TLT",
            "notes":    f"country={r['country']} forecast={r['forecast']} prev={r['previous']}",
        } for r in rows])
    except Exception as ex:
        import streamlit as _st_err
        _st_err.warning(f"economic_calendar DB read failed: {ex}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uw_earnings_all(tickers: tuple, days_forward: int = 60) -> pd.DataFrame:
    """
    Read earnings from accuracy.db.earnings_calendar (joined with
    earnings_cache for est_eps / drift fields).

    Populated daily by scripts/refresh_earnings_calendar.py via cron.

    May 14 2026: Switched from per-ticker UW API calls (128 calls/refresh)
    to a single DB read. earnings_calendar is materialized from
    earnings_cache, which is itself maintained by daily_uw_snapshot.py.
    Worst-case staleness: 1 day (acceptable for forward calendar).

    Returns same schema as the prior live fetcher so downstream code
    (Tab "Earnings Calendar" filters, suppress_buy alerts) is unchanged.
    """
    import sqlite3
    from pathlib import Path
    db_path = Path(__file__).parent.parent.parent / "accuracy.db"

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            df_raw = pd.read_sql("""
                SELECT
                    cal.ticker,
                    cal.next_date     AS report_date,
                    cal.next_time     AS report_time,
                    cal.expected_move,
                    cal.days_until    AS days_away,
                    cache.actual_eps,
                    cache.est_eps,
                    cache.pre_drift_3d  AS pre_drift_raw,
                    cache.post_drift_3d AS post_drift_raw
                FROM earnings_calendar cal
                LEFT JOIN earnings_cache cache
                  ON cal.ticker = cache.ticker
                 AND cal.next_date = cache.report_date
                WHERE cal.days_until <= ?
                  AND cal.ticker IN ({})
                ORDER BY cal.days_until
            """.format(",".join(["?"] * len(tickers))),
                conn,
                params=(days_forward, *tickers))
    except Exception as ex:
        import streamlit as _st_err
        _st_err.warning(f"earnings_calendar DB read failed: {ex}")
        return pd.DataFrame()

    if df_raw.empty:
        return pd.DataFrame()

    rows = []
    mega_caps = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"}
    for _, r in df_raw.iterrows():
        ticker      = r["ticker"]
        exp_move    = float(r["expected_move"] or 0)
        days_away   = int(r["days_away"])
        pre_raw     = r["pre_drift_raw"]
        post_raw    = r["post_drift_raw"]
        pre_drift   = float(pre_raw)  if pre_raw  is not None else None
        post_drift  = float(post_raw) if post_raw is not None else None
        actual_eps  = r["actual_eps"]
        est_eps     = r["est_eps"]

        bucket = _META.get(ticker, {}).get("bucket", "—")
        tier   = _META.get(ticker, {}).get("tier", "—")

        # BUY suppression: same logic as live version
        suppress = False
        suppress_reason = ""
        if days_away <= 2 and post_drift is not None and post_drift < 0:
            suppress = True
            suppress_reason = f"Neg post-drift ({post_drift:.1%})"
        elif days_away <= 2 and exp_move > 0.08:
            suppress = True
            suppress_reason = f"High uncertainty (±{exp_move:.1%})"

        rows.append({
            "ticker":           ticker,
            "bucket":           bucket,
            "tier":             tier,
            "report_date":      r["report_date"],
            "report_time":      r["report_time"] or "",
            "days_away":        days_away,
            "expected_move":    f"±{exp_move:.1%}" if exp_move else "—",
            "pre_drift":        f"{pre_drift:+.1%}" if pre_drift is not None else "—",
            "post_drift":       f"{post_drift:+.1%}" if post_drift is not None else "—",
            "suppress_buy":     suppress,
            "suppress_reason":  suppress_reason,
            "actual_eps":       actual_eps,
            "est_eps":          est_eps,
            "is_earnings_week": days_away <= 5,
            "impact":           "High" if ticker in mega_caps else "Medium",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
# ── Risk score ────────────────────────────────────────────────────────────────
def compute_risk_score(econ_df: pd.DataFrame, earn_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not econ_df.empty:
        w = {"Low": 1, "Medium": 2, "High": 3}
        for _, r in econ_df.iterrows():
            rows.append({"date": r["date"], "score": w.get(r["impact"], 1)})
    if not earn_df.empty:
        for _, r in earn_df.iterrows():
            rows.append({"date": r["report_date"], "score": 2})
    if not rows:
        return pd.DataFrame(columns=["date", "score"])
    df = pd.DataFrame(rows)
    return df.groupby("date")["score"].sum().reset_index()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    days_forward = st.slider("Days forward", 7, 90, 45)
    show_econ    = st.checkbox("Economic events", value=True)
    show_earn    = st.checkbox("Earnings", value=True)
    show_suppress= st.checkbox("Show suppressed tickers only", value=False)
    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── Date range ────────────────────────────────────────────────────────────────
today    = date.today()
end_date = today + timedelta(days=days_forward)
start_s  = today.isoformat()
end_s    = end_date.isoformat()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading from Unusual Whales..."):
    econ_df = fetch_uw_economic_calendar(start_s, end_s) if show_econ else pd.DataFrame()
    earn_df = fetch_uw_earnings_all(tuple(_TICKERS), days_forward) if show_earn else pd.DataFrame()


# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Earnings upcoming", len(earn_df) if not earn_df.empty else 0, f"next {days_forward} days")
c2.metric("Earnings this week", len(earn_df[earn_df["days_away"] <= 5]) if not earn_df.empty else 0)
c3.metric("BUY suppressed", len(earn_df[earn_df["suppress_buy"]]) if not earn_df.empty else 0, "near-term earnings")
c4.metric("Economic events", len(econ_df) if not econ_df.empty else 0, f"next {days_forward} days")

st.markdown("---")


# ── Risk score + session state ────────────────────────────────────────────────
risk_df = compute_risk_score(econ_df, earn_df)
if not risk_df.empty:
    horizon72 = (today + timedelta(days=3)).isoformat()
    risk72    = int(risk_df[risk_df["date"] <= horizon72]["score"].sum())
    label     = "Low" if risk72 < 3 else ("Medium" if risk72 < 6 else "High")
    color     = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
    st.session_state["event_risk_next72"] = {"score": risk72, "label": label}
    st.info(f"**Next 72h event risk:** {color} {label} (score={risk72}) — fed into Dashboard signal gate")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Earnings", "📉 Economic", "🌡️ Risk heatmap"])


# ════════════════════════════════════════
# TAB 1 — EARNINGS
# ════════════════════════════════════════
with tab1:
    st.subheader("Earnings calendar — your tickers")
    st.caption("Sorted by days until report. BUY suppression fires when post-drift is negative or expected move >8%.")

    if earn_df.empty:
        st.info("No earnings data. Check UW API key or try refreshing.")
    else:
        display = earn_df.copy()
        if show_suppress:
            display = display[display["suppress_buy"]]

        display["time"] = display["report_time"].apply(
            lambda x: "🌅 pre" if "pre" in str(x).lower() else "🌙 post"
        )
        display["suppress"] = display.apply(
            lambda r: f"❌ {r['suppress_reason']}" if r["suppress_buy"] else "✅ OK", axis=1
        )

        show_cols = ["ticker", "bucket", "tier", "report_date", "time",
                     "days_away", "expected_move", "pre_drift", "post_drift", "suppress"]
        show_df = display[show_cols].copy()
        show_df.columns = ["Ticker", "Bucket", "Tier", "Date", "Time",
                           "Days away", "Exp move", "Pre drift", "Post drift", "Signal"]

        def _color_signal(val):
            if "❌" in str(val): return "color: #E24B4A; font-weight: 500"
            if "✅" in str(val): return "color: #1D9E75"
            return ""

        def _color_drift(val):
            if val == "—": return ""
            try:
                n = float(str(val).replace("%","").replace("+",""))
                return "color: #1D9E75" if n > 0 else "color: #E24B4A"
            except Exception:
                return ""

        st.dataframe(
            show_df.style
            .applymap(_color_signal, subset=["Signal"])
            .applymap(_color_drift,  subset=["Pre drift", "Post drift"]),
            use_container_width=True, hide_index=True,
        )

        # Suppress summary
        suppressed = earn_df[earn_df["suppress_buy"]]
        if not suppressed.empty:
            st.markdown("---")
            st.warning(f"⚠️ {len(suppressed)} tickers have BUY suppressed due to upcoming earnings:")
            st.write(", ".join(suppressed["ticker"].tolist()))

        # Week view
        st.markdown("---")
        st.subheader("This week")
        this_week = earn_df[earn_df["days_away"] <= 5].sort_values("days_away")
        if not this_week.empty:
            cols = st.columns(min(len(this_week), 5))
            for i, (_, row) in enumerate(this_week.iterrows()):
                with cols[i % 5]:
                    color = "#FCEBEB" if row["suppress_buy"] else "#EAF3DE"
                    tcolor = "#A32D2D" if row["suppress_buy"] else "#3B6D11"
                    st.markdown(f"""
                    <div style="background:{color};border-radius:8px;padding:10px;margin-bottom:8px;">
                        <b style="color:{tcolor};font-size:15px">{row['ticker']}</b><br>
                        <span style="font-size:12px;color:#5F5E5A">{row['report_date']} · {row['report_time']}</span><br>
                        <span style="font-size:12px">Exp: {row['expected_move']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No earnings this week.")


# ════════════════════════════════════════
# TAB 2 — ECONOMIC
# ════════════════════════════════════════
with tab2:
    st.subheader("Economic events — Unusual Whales calendar")
    st.caption("FOMC, CPI, NFP, GDP and other market-moving macro events.")

    if econ_df.empty:
        st.info("No economic events data. Check UW API key.")
    else:
        def _color_impact(val):
            if val == "High":   return "color: #E24B4A; font-weight: 500"
            if val == "Medium": return "color: #BA7517; font-weight: 500"
            return "color: #3B6D11"

        display = econ_df[["date","time","title","impact","tickers","notes"]].copy()
        display.columns = ["Date","Time","Event","Impact","Tickers","Notes"]
        st.dataframe(
            display.style.applymap(_color_impact, subset=["Impact"]),
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════
# TAB 4 — RISK HEATMAP
# ════════════════════════════════════════
with tab3:
    st.subheader("Daily risk score heatmap")
    st.caption("Higher = more market-moving events that day. Feeds into Dashboard signal gate.")

    if not risk_df.empty:
        risk_df["date"] = pd.to_datetime(risk_df["date"])
        chart = (
            alt.Chart(risk_df)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("score:Q", title="Risk score"),
                color=alt.Color(
                    "score:Q",
                    scale=alt.Scale(scheme="redyellowgreen", reverse=True, domain=[0, 9]),
                    legend=None,
                ),
                tooltip=["date:T", "score:Q"],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("ICS export")
        if not earn_df.empty:
            lines = ["BEGIN:VCALENDAR", "VERSION:2.0"]
            for _, r in earn_df.iterrows():
                dt = r["report_date"].replace("-","")
                lines += [
                    "BEGIN:VEVENT",
                    f"DTSTART:{dt}",
                    f"DTEND:{dt}",
                    f"SUMMARY:Earnings: {r['ticker']} ({r['report_time']})",
                    f"DESCRIPTION:Exp move {r['expected_move']} | post drift {r['post_drift']}",
                    "END:VEVENT",
                ]
            if not econ_df.empty:
                for _, r in econ_df.iterrows():
                    dt = r["date"].replace("-","")
                    lines += [
                        "BEGIN:VEVENT",
                        f"DTSTART:{dt}",
                        f"DTEND:{dt}",
                        f"SUMMARY:{r['title']} [{r['impact']}]",
                        "END:VEVENT",
                    ]
            lines.append("END:VCALENDAR")
            st.download_button(
                "📥 Export .ics",
                data="\n".join(lines),
                file_name="market_events.ics",
                mime="text/calendar",
            )
    else:
        st.info("No risk data available.")
