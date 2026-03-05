# ui/4_Events.py
# Market Events Calendar — forward-looking risk calendar.
# Mostly preserved from 7_Market_Events_Calendar_v1_1.py.
# Changes: removed set_page_config conflict, cleaned imports, kept all good logic.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="Market Events Calendar",
    page_icon="🗓️",
    layout="wide",
)

st.title("🗓️ Market Events Calendar")
st.caption(
    "Forward-looking calendar of market-moving events. "
    "Feeds risk score into the Dashboard's signal gate."
)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Fed/Rate", "Economic", "Earnings",
    "IPO/Lockup", "OPEC/Energy", "Geopolitics",
    "Options/Holiday", "Company-Specific",
]
IMPACT_LEVELS = ["Low", "Medium", "High"]
Event = Dict[str, object]

# ── Secret helper ─────────────────────────────────────────────────────────────
def _get_secret(*keys: str) -> Optional[str]:
    try:
        cur = st.secrets
        for k in keys:
            cur = cur[k]
        return cur if isinstance(cur, str) else None
    except Exception:
        if len(keys) == 1:
            return st.secrets.get(keys[0])
    return None

# ── Live data fetchers ────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_economic_fmp(start: date, end: date, api_key: str) -> pd.DataFrame:
    url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    try:
        r = requests.get(url, params={
            "from": start.isoformat(), "to": end.isoformat(), "apikey": api_key
        }, timeout=15)
        r.raise_for_status()
        data = r.json() or []
    except Exception as e:
        st.warning(f"FMP error: {e}")
        return pd.DataFrame()

    rows = []
    for it in data:
        try:
            when = datetime.fromisoformat(it["date"].replace("Z", "+00:00"))
        except Exception:
            continue
        imp = it.get("impact", "")
        impact = "High" if "High" in imp else ("Medium" if "Medium" in imp else "Low")
        notes = "; ".join(
            f"{k.title()}: {it[k]}" for k in ("country", "actual", "estimate", "previous")
            if it.get(k) not in (None, "")
        )
        rows.append({
            "title": it.get("event", "Economic Event"),
            "category": "Economic",
            "start": when,
            "end": when + timedelta(minutes=30),
            "impact": impact,
            "source": "FMP",
            "tickers": "SPY, QQQ",
            "notes": notes,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_earnings_finnhub(start: date, end: date, token: str) -> pd.DataFrame:
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": start.isoformat(), "to": end.isoformat(), "token": token},
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("earningsCalendar") or []
    except Exception as e:
        st.warning(f"Finnhub earnings error: {e}")
        return pd.DataFrame()

    rows = []
    for it in items:
        d = it.get("date")
        if not d:
            continue
        try:
            base = datetime.fromisoformat(d)
        except Exception:
            continue
        hint = (it.get("time") or "").lower()
        when = base.replace(hour=16 if hint == "amc" else 8 if hint == "bmo" else 13)
        sym  = it.get("symbol", "?")
        rows.append({
            "title":    f"Earnings: {sym}",
            "category": "Earnings",
            "start":    when,
            "end":      when + timedelta(hours=1),
            "impact":   "High" if sym in ("AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA") else "Medium",
            "source":   "Finnhub",
            "tickers":  sym,
            "notes":    "; ".join(f"{k}={it[k]}" for k in
                        ("epsEstimate","epsActual","revenueEstimate","revenueActual")
                        if it.get(k) not in (None, "")),
        })
    return pd.DataFrame(rows)


# ── Seed events (always shown, no API needed) ─────────────────────────────────
def seed_sample_events(start: date, end: date) -> List[Event]:
    span = (end - start).days
    base = datetime.combine(start, datetime.min.time())
    samples = [
        {
            "title": "FOMC Rate Decision",
            "category": "Fed/Rate",
            "start": base + timedelta(days=min(14, max(1, span // 3))),
            "end":   base + timedelta(days=min(14, max(1, span // 3)), hours=2),
            "impact": "High",
            "source": "Fed",
            "tickers": "SPY, TLT, GLD",
            "notes": "Rate decision + press conference. High volatility expected.",
        },
        {
            "title": "CPI Release",
            "category": "Economic",
            "start": base + timedelta(days=min(7, max(1, span // 5)), hours=8, minutes=30),
            "end":   base + timedelta(days=min(7, max(1, span // 5)), hours=9),
            "impact": "High",
            "source": "BLS",
            "tickers": "SPY, TLT, XLF",
            "notes": "Consumer Price Index — key inflation signal.",
        },
        {
            "title": "Mega-Cap Earnings",
            "category": "Earnings",
            "start": base + timedelta(days=min(10, max(1, span // 4)), hours=16),
            "end":   base + timedelta(days=min(10, max(1, span // 4)), hours=17),
            "impact": "High",
            "source": "Company",
            "tickers": "AAPL, MSFT, NVDA",
            "notes": "Earnings call 30–60min after close.",
        },
        {
            "title": "Options Expiration",
            "category": "Options/Holiday",
            "start": base + timedelta(days=min(18, max(1, span // 2))),
            "end":   base + timedelta(days=min(18, max(1, span // 2)), hours=1),
            "impact": "Medium",
            "source": "CBOE",
            "tickers": "SPY, QQQ, IWM",
            "notes": "Monthly options expiry — elevated vol possible.",
        },
    ]
    return [e for e in samples if start <= e["start"].date() <= end]


# ── Risk score ────────────────────────────────────────────────────────────────
def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "risk"])
    w = {"Low": 1, "Medium": 2, "High": 3}
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["start"]).dt.date
    tmp["w"]    = tmp["impact"].map(lambda x: w.get(str(x), 1))
    return tmp.groupby("date")["w"].sum().reset_index().rename(columns={"w": "risk"})


# ── ICS export ────────────────────────────────────────────────────────────────
def export_ics(df: pd.DataFrame) -> str:
    fmt = lambda dt: dt.strftime("%Y%m%dT%H%M%S")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MLQuant//Market Events//EN"]
    for _, r in df.iterrows():
        lines += [
            "BEGIN:VEVENT",
            f"DTSTAMP:{fmt(datetime.utcnow())}",
            f"DTSTART:{fmt(r['start'])}",
            f"DTEND:{fmt(r['end'])}",
            f"SUMMARY:{r['title']}",
            f"CATEGORIES:{r['category']}",
            f"DESCRIPTION:Impact={r['impact']} | Tickers={r['tickers']} | {r['notes']}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    window_days  = st.slider("Days forward", 7, 90, 45)
    start_date   = st.date_input("Start", value=date.today())
    end_date     = start_date + timedelta(days=window_days)
    category_sel = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
    impact_sel   = st.multiselect("Impact",     IMPACT_LEVELS, default=IMPACT_LEVELS)

    st.markdown("---")
    st.subheader("🔌 Live data (optional)")
    fmp_key     = _get_secret("FMP_API_KEY")     or _get_secret("providers", "fmp_api_key")
    finnhub_key = _get_secret("FINNHUB_TOKEN")   or _get_secret("providers", "finnhub_token")
    use_fmp     = st.checkbox("FinancialModelingPrep (Economic)", value=bool(fmp_key))
    use_finnhub = st.checkbox("Finnhub (Earnings)",               value=bool(finnhub_key))

    if not fmp_key and not finnhub_key:
        st.caption(
            "Add to secrets.toml to enable live feeds:\n"
            "```\nFMP_API_KEY = '...'\nFINNHUB_TOKEN = '...'\n```"
        )

    st.markdown("---")
    st.subheader("➕ Add custom event")
    with st.form("add_event"):
        e_title    = st.text_input("Title")
        e_cat      = st.selectbox("Category", CATEGORIES)
        e_date     = st.date_input("Date", value=start_date)
        e_time     = st.time_input("Time")
        e_impact   = st.selectbox("Impact", IMPACT_LEVELS, index=1)
        e_tickers  = st.text_input("Tickers")
        e_notes    = st.text_area("Notes", height=60)
        submitted  = st.form_submit_button("Add")

if "manual_events" not in st.session_state:
    st.session_state["manual_events"] = []

if submitted and e_title:
    dt = datetime.combine(e_date, e_time)
    st.session_state["manual_events"].append({
        "title": e_title, "category": e_cat,
        "start": dt, "end": dt + timedelta(hours=1),
        "impact": e_impact, "source": "Manual",
        "tickers": e_tickers, "notes": e_notes,
    })
    st.success(f"Added: {e_title}")

# ── Assemble events ───────────────────────────────────────────────────────────
frames = [pd.DataFrame(seed_sample_events(start_date, end_date))]

if use_fmp and fmp_key:
    frames.append(fetch_economic_fmp(start_date, end_date, fmp_key))
elif use_fmp:
    st.warning("FMP enabled but no API key found in secrets.")

if use_finnhub and finnhub_key:
    frames.append(fetch_earnings_finnhub(start_date, end_date, finnhub_key))
elif use_finnhub:
    st.warning("Finnhub enabled but no token found in secrets.")

if st.session_state["manual_events"]:
    frames.append(pd.DataFrame(st.session_state["manual_events"]))

valid  = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
all_df = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

if not all_df.empty:
    all_df = all_df[
        all_df["category"].isin(category_sel) &
        all_df["impact"].isin(impact_sel)
    ].sort_values("start").reset_index(drop=True)

# ── Compute + store 72h risk in session state (read by Dashboard) ─────────────
risk_df = compute_risk_score(all_df) if not all_df.empty else pd.DataFrame(columns=["date","risk"])
if not risk_df.empty:
    horizon72 = date.today() + timedelta(days=3)
    risk72    = int(risk_df[risk_df["date"] <= horizon72]["risk"].sum())
    label     = "Low" if risk72 < 3 else ("Medium" if risk72 < 6 else "High")
    st.session_state["event_risk_next72"] = {"score": risk72, "label": label}
    color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
    st.metric("Next 72h Event Risk", f"{color} {label}", delta=f"score = {risk72}")
else:
    st.session_state.pop("event_risk_next72", None)

st.caption("This risk score is passed to the Dashboard to down-weight signals on high-risk days.")
st.divider()

# ── Timeline ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📅 Timeline")
    if all_df.empty:
        st.info("No events match current filters.")
    elif not HAS_PLOTLY:
        st.warning("Install plotly for timeline: `pip install plotly`")
    else:
        fig = px.timeline(
            all_df,
            x_start="start", x_end="end", y="category",
            color="impact",
            color_discrete_map={"High": "#ff1744", "Medium": "#ffab00", "Low": "#00c853"},
            hover_data=["title", "tickers", "notes", "source"],
            title="Upcoming Market Events",
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=480, legend_title_text="Impact")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Upcoming events")
    if all_df.empty:
        st.info("No events.")
    else:
        display = all_df.copy()
        display["start"] = pd.to_datetime(display["start"]).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            display[["start", "title", "category", "impact", "tickers"]],
            use_container_width=True,
            hide_index=True,
        )

        ics = export_ics(all_df)
        st.download_button(
            "📥 Export .ics",
            data=ics,
            file_name="market_events.ics",
            mime="text/calendar",
            key="dl_ics",
        )

# ── Risk heatmap ──────────────────────────────────────────────────────────────
if not risk_df.empty:
    st.subheader("🌡️ Daily Risk Score")
    risk_df["date"] = pd.to_datetime(risk_df["date"])
    risk_chart = (
        alt.Chart(risk_df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("risk:Q", title="Risk Score"),
            color=alt.Color(
                "risk:Q",
                scale=alt.Scale(scheme="redyellowgreen", reverse=True, domain=[0, 9]),
                legend=None,
            ),
            tooltip=["date:T", "risk:Q"],
        )
        .properties(height=200, title="Higher = more market-moving events that day")
    )
    import altair as alt
    st.altair_chart(risk_chart, use_container_width=True)
