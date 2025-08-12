import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import plotly.express as px
from typing import List, Dict, Optional

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Market-Moving Events Calendar",
    page_icon="🗓️",
    layout="wide",
)

st.title("🗓️ Market-Moving Events Calendar")
st.caption(
    "Forward-looking calendar of scheduled events with optional live data feeds (FMP + Finnhub). "
    "Use filters on the left to focus on what matters."
)

# ------------------------------------------------------------
# Helpers & constants
# ------------------------------------------------------------
CATEGORIES = [
    "Fed/Rate",
    "Economic",
    "Earnings",
    "IPO/Lockup",
    "OPEC/Energy",
    "Geopolitics",
    "Options/Holiday",
    "Company-Specific",
]

IMPACT_LEVELS = ["Low", "Medium", "High"]

Event = Dict[str, object]


def _get_secret(*keys: str) -> Optional[str]:
    """Fetch a secret, checking both nested and flat styles.
    Example: _get_secret('providers', 'fmp_api_key') or _get_secret('FMP_API_KEY')
    """
    try:
        cur = st.secrets
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                raise KeyError
        return cur if isinstance(cur, str) else None
    except Exception:
        # flat lookup fallback
        if len(keys) == 1 and keys[0] in st.secrets:
            val = st.secrets.get(keys[0])
            return val if isinstance(val, str) else None
    return None


# ------------------------------------------------------------
# Live provider fetchers (optional)
# ------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_economic_fmp(start: date, end: date, api_key: str) -> pd.DataFrame:
    """FinancialModelingPrep Economic Calendar.
    Docs: https://site.financialmodelingprep.com/developer/docs/economic-calendar-api/
    """
    url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or []
    except Exception as e:
        st.warning(f"FMP Economic Calendar error: {e}")
        return pd.DataFrame()

    # Map to unified schema
    rows: List[Event] = []
    for it in data:
        # it keys include: 'event', 'country', 'impact', 'date', 'actual', 'previous', 'estimate'
        try:
            when = datetime.fromisoformat(it.get("date").replace("Z", "+00:00")) if it.get("date") else None
        except Exception:
            when = None
        if not when:
            continue
        imp_raw = (it.get("impact") or "").title()
        impact = "High" if "High" in imp_raw else ("Medium" if "Medium" in imp_raw else "Low")
        title = it.get("event") or "Economic Event"
        notes = []
        if it.get("country"): notes.append(f"Country: {it['country']}")
        for k in ("actual", "estimate", "previous"):
            if it.get(k) not in (None, ""):
                notes.append(f"{k.title()}: {it[k]}")
        rows.append({
            "title": title,
            "category": "Economic",
            "start": when,
            "end": when + timedelta(minutes=30),
            "impact": impact,
            "source": "FMP",
            "tickers": "SPY, QQQ, IWM",
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_earnings_finnhub(start: date, end: date, token: str) -> pd.DataFrame:
    """Finnhub earnings calendar.
    Docs: https://finnhub.io/docs/api/earnings-calendar
    """
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {"from": start.isoformat(), "to": end.isoformat(), "token": token}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        items = data.get("earningsCalendar") or []
    except Exception as e:
        st.warning(f"Finnhub Earnings error: {e}")
        return pd.DataFrame()

    rows: List[Event] = []
    for it in items:
        # keys: date, symbol, time (bmo/amc/...), epsEstimate, epsActual, revenueEstimate, revenueActual
        d = it.get("date")
        if not d:
            continue
        try:
            # Treat 16:00 local as default if 'amc' (after market close), 08:00 if 'bmo'
            base_dt = datetime.fromisoformat(d)
        except Exception:
            continue
        time_hint = (it.get("time") or "").lower()
        if time_hint == "amc":
            when = base_dt.replace(hour=16, minute=0)
        elif time_hint == "bmo":
            when = base_dt.replace(hour=8, minute=0)
        else:
            when = base_dt.replace(hour=13, minute=0)
        title = f"Earnings: {it.get('symbol','?')}"
        notes = []
        for k in ("epsEstimate", "epsActual", "revenueEstimate", "revenueActual"):
            if it.get(k) not in (None, ""):
                notes.append(f"{k}={it[k]}")
        rows.append({
            "title": title,
            "category": "Earnings",
            "start": when,
            "end": when + timedelta(hours=1),
            "impact": "High" if any(s in title for s in ("AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA")) else "Medium",
            "source": "Finnhub",
            "tickers": it.get("symbol",""),
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ipo_finnhub(start: date, end: date, token: str) -> pd.DataFrame:
    """Finnhub IPO calendar.
    Docs: https://finnhub.io/docs/api/calendar-ipo
    """
    url = "https://finnhub.io/api/v1/calendar/ipo"
    params = {"from": start.isoformat(), "to": end.isoformat(), "token": token}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        items = data.get("ipoCalendar") or []
    except Exception as e:
        st.warning(f"Finnhub IPO error: {e}")
        return pd.DataFrame()

    rows: List[Event] = []
    for it in items:
        d = it.get("date") or it.get("ipoDate")
        if not d:
            continue
        try:
            when = datetime.fromisoformat(d)
        except Exception:
            continue
        symbol = it.get("symbol") or it.get("ticker") or "IPO"
        title = f"IPO: {symbol}"
        price = it.get("price") or it.get("priceRange")
        notes = []
        if price: notes.append(f"Price: {price}")
        if it.get("exchange"): notes.append(f"Exch: {it['exchange']}")
        rows.append({
            "title": title,
            "category": "IPO/Lockup",
            "start": when.replace(hour=18, minute=0),
            "end": when.replace(hour=20, minute=0),
            "impact": "Low",
            "source": "Finnhub",
            "tickers": symbol,
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Sample events (fallback if no providers configured)
# ------------------------------------------------------------
def seed_sample_events(start: date, end: date) -> List[Event]:
    span = (end - start).days
    base = datetime.combine(start, datetime.min.time())
    samples: List[Event] = [
        {
            "title": "FOMC Rate Decision",
            "category": "Fed/Rate",
            "start": base + timedelta(days=min(10, max(1, span // 3))),
            "end": base + timedelta(days=min(10, max(1, span // 3)), hours=2),
            "impact": "High",
            "source": "Fed",
            "tickers": "SPY, QQQ, TLT",
            "notes": "Press conference 30min after statement.",
        },
        {
            "title": "US CPI (Monthly)",
            "category": "Economic",
            "start": base + timedelta(days=min(5, max(1, span // 4)), hours=8, minutes=30),
            "end": base + timedelta(days=min(5, max(1, span // 4)), hours=9),
            "impact": "High",
            "source": "BLS",
            "tickers": "SPY, XLF, IWM",
            "notes": "Consensus: 0.2% m/m",
        },
        {
            "title": "Quadruple Witching (Options Expiration)",
            "category": "Options/Holiday",
            "start": base + timedelta(days=min(20, max(2, span - 1)), hours=9, minutes=30),
            "end": base + timedelta(days=min(20, max(2, span - 1)), hours=16),
            "impact": "High",
            "source": "Exchanges",
            "tickers": "SPY, QQQ, IWM",
            "notes": "Expect volume/volatility spikes.",
        },
        {
            "title": "MegaCap Earnings (AAPL)",
            "category": "Earnings",
            "start": base + timedelta(days=min(8, max(1, span // 3)), hours=16),
            "end": base + timedelta(days=min(8, max(1, span // 3)), hours=17),
            "impact": "High",
            "source": "Company",
            "tickers": "AAPL, QQQ",
            "notes": "Call 30–60min later.",
        },
    ]
    return [ev for ev in samples if start <= ev["start"].date() <= end]


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def to_dataframe(events: List[Event]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(by=["start", "impact"], ascending=[True, False]).reset_index(drop=True)
    return df


def make_timeline(df: pd.DataFrame):
    if df.empty:
        st.info("No events in the selected filters/date window.")
        return
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="category",
        color="impact",
        hover_data=["title", "source", "tickers", "notes"],
        title="Timeline of Events",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=520, legend_title_text="Impact")
    st.plotly_chart(fig, use_container_width=True)


def export_ics(df: pd.DataFrame) -> str:
    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M%S")
    ics_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AMF//Market Events//EN",
    ]
    for _, r in df.iterrows():
        ics_lines.extend([
            "BEGIN:VEVENT",
            f"DTSTAMP:{fmt(datetime.utcnow())}",
            f"DTSTART:{fmt(r['start'])}",
            f"DTEND:{fmt(r['end'])}",
            f"SUMMARY:{r['title']}",
            f"CATEGORIES:{r['category']}",
            f"DESCRIPTION:Impact={r['impact']} | Source={r['source']} | Tickers={r['tickers']} | Notes={r['notes']}",
            "END:VEVENT",
        ])
    ics_lines.append("END:VCALENDAR")
    return "
".join(ics_lines)


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "risk"])
    w = {"Low": 1, "Medium": 2, "High": 3}
    tmp = df.copy()
    tmp["date"] = tmp["start"].dt.date
    tmp["w"] = tmp["impact"].map(lambda x: w.get(str(x), 1))
    out = tmp.groupby("date")["w"].sum().reset_index().rename(columns={"w": "risk"})
    return out.sort_values("date")


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Filters")
window_days = st.sidebar.slider("Window (days forward)", 7, 90, 45, 1)
start_date = st.sidebar.date_input("Start date", value=date.today())
end_date = start_date + timedelta(days=window_days)

category_sel = st.sidebar.multiselect("Categories", options=CATEGORIES, default=CATEGORIES)
impact_sel = st.sidebar.multiselect("Impact levels", options=IMPACT_LEVELS, default=IMPACT_LEVELS)

st.sidebar.markdown("---")
st.sidebar.subheader("Live data providers (optional)")
use_fmp = st.sidebar.checkbox("Use FinancialModelingPrep (Economic)")
use_finnhub = st.sidebar.checkbox("Use Finnhub (Earnings & IPO)")

fmp_key = _get_secret("providers", "fmp_api_key") or _get_secret("FMP_API_KEY")
finnhub_key = _get_secret("providers", "finnhub_token") or _get_secret("FINNHUB_TOKEN")

st.sidebar.caption("Set secrets in .streamlit/secrets.toml, e.g.

[providers]
fmp_api_key='YOUR_FMP_KEY'
finnhub_token='YOUR_FINNHUB_TOKEN'")

st.sidebar.markdown("---")
st.sidebar.subheader("Add a custom event")
with st.sidebar.form("add_event_form"):
    e_title = st.text_input("Title")
    e_category = st.selectbox("Category", CATEGORIES)
    e_date = st.date_input("Date", value=start_date)
    e_start_time = st.time_input("Start time", value=datetime.now().time().replace(second=0, microsecond=0))
    e_end_time = st.time_input("End time", value=(datetime.now() + timedelta(hours=1)).time().replace(second=0, microsecond=0))
    e_impact = st.selectbox("Impact", IMPACT_LEVELS, index=1)
    e_source = st.text_input("Source", value="Manual")
    e_tickers = st.text_input("Tickers (comma-separated)", value="")
    e_notes = st.text_area("Notes", value="")
    submitted = st.form_submit_button("➕ Add Event")

# Session state to hold manual events
if "manual_events" not in st.session_state:
    st.session_state.manual_events: List[Event] = []

if submitted and e_title:
    start_dt = datetime.combine(e_date, e_start_time)
    end_dt = datetime.combine(e_date, e_end_time)
    st.session_state.manual_events.append({
        "title": e_title,
        "category": e_category,
        "start": start_dt,
        "end": end_dt,
        "impact": e_impact,
        "source": e_source,
        "tickers": e_tickers,
        "notes": e_notes,
    })
    st.success(f"Added event: {e_title}")

# ------------------------------------------------------------
# Data assembly (seed + live + manual)
# ------------------------------------------------------------
seeded_df = pd.DataFrame(seed_sample_events(start_date, end_date))

frames = [seeded_df]

if use_fmp:
    if not fmp_key:
        st.warning("Enable FMP but no API key found. Add FMP key to secrets.")
    else:
        econ_df = fetch_economic_fmp(start_date, end_date, fmp_key)
        frames.append(econ_df)

if use_finnhub:
    if not finnhub_key:
        st.warning("Enable Finnhub but no token found. Add Finnhub token to secrets.")
    else:
        earn_df = fetch_earnings_finnhub(start_date, end_date, finnhub_key)
        ipo_df = fetch_ipo_finnhub(start_date, end_date, finnhub_key)
        frames.extend([earn_df, ipo_df])

manual_df = pd.DataFrame(st.session_state.manual_events) if st.session_state.manual_events else pd.DataFrame()
frames.append(manual_df)

all_df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if any(frames) else pd.DataFrame()

# Filter
if not all_df.empty:
    all_df = all_df[all_df["category"].isin(category_sel) & all_df["impact"].isin(impact_sel)]

# ------------------------------------------------------------
# Layout: Risk overview + Timeline + Upcoming table
# ------------------------------------------------------------
col0, = st.columns([1])
with col0:
    risk_df = compute_risk_score(all_df) if not all_df.empty else pd.DataFrame(columns=["date","risk"])
    if not risk_df.empty:
        st.subheader("Event Risk (next days)")
        fig_risk = px.bar(risk_df, x="date", y="risk", title=None, labels={"risk":"Weighted events"})
        st.plotly_chart(fig_risk, use_container_width=True)

col1, col2 = st.columns([2, 1], gap="large")
with col1:
    make_timeline(all_df)

with col2:
    st.subheader("Upcoming events")
    if all_df.empty:
        st.info("No upcoming events match your filters.")
    else:
        nice_df = all_df.copy()
        nice_df["start"] = pd.to_datetime(nice_df["start"]).dt.strftime("%Y-%m-%d %H:%M")
        nice_df["end"] = pd.to_datetime(nice_df["end"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            nice_df[["start", "end", "title", "category", "impact", "tickers", "source"]],
            use_container_width=True,
            hide_index=True,
        )
        ics_text = export_ics(all_df)
        st.download_button(
            label="📥 Export to .ics (Calendar)",
            file_name="market_events.ics",
            mime="text/calendar",
            data=ics_text,
        )

with st.expander("⚙️ Integration hooks"):
    st.markdown(
        "- **Risk adjustments**: `compute_risk_score()` can feed a 48–72h risk weight into your strategy.
"
        "- **Live feeds**: Toggle providers in the sidebar; add keys in `secrets.toml`.
"
        "- **Extendable**: Add more providers (Polygon, Trading Economics) by mapping to the same schema."
    )
