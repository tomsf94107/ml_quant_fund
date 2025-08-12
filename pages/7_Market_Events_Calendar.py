import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.express as px
from typing import List, Dict

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
    "Forward-looking calendar of scheduled events + a place for your own notes. "
    "Use filters on the left to focus on what matters."
)

# ------------------------------------------------------------
# Helpers
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


def seed_sample_events(start: date, end: date) -> List[Event]:
    """Generate sample events within the selected date window.
    Replace this with real feeds later (e.g., Economic API, earnings calendar, custom CSV/Sheet).
    """
    span = (end - start).days
    base = datetime.combine(start, datetime.min.time())

    # Some plausible placeholders; adjust dates if outside window
    samples: List[Event] = [
        {
            "title": "FOMC Rate Decision",
            "category": "Fed/Rate",
            "start": base + timedelta(days=min(10, span // 3)),
            "end": base + timedelta(days=min(10, span // 3), hours=2),
            "impact": "High",
            "source": "Fed",
            "tickers": "SPY, QQQ, TLT",
            "notes": "Press conference 30min after statement.",
        },
        {
            "title": "US CPI (Monthly)",
            "category": "Economic",
            "start": base + timedelta(days=min(5, span // 4), hours=8, minutes=30),
            "end": base + timedelta(days=min(5, span // 4), hours=9),
            "impact": "High",
            "source": "BLS",
            "tickers": "SPY, XLF, IWM",
            "notes": "Consensus: 0.2% m/m",
        },
        {
            "title": "US PPI",
            "category": "Economic",
            "start": base + timedelta(days=min(12, span // 2), hours=8, minutes=30),
            "end": base + timedelta(days=min(12, span // 2), hours=9),
            "impact": "Medium",
            "source": "BLS",
            "tickers": "XLI, SPY",
            "notes": "Watch core read-through.",
        },
        {
            "title": "Quadruple Witching (Options Expiration)",
            "category": "Options/Holiday",
            "start": base + timedelta(days=min(20, span - 1), hours=9, minutes=30),
            "end": base + timedelta(days=min(20, span - 1), hours=16),
            "impact": "High",
            "source": "Exchanges",
            "tickers": "SPY, QQQ, IWM",
            "notes": "Expect volume/volatility spikes.",
        },
        {
            "title": "OPEC+ Meeting",
            "category": "OPEC/Energy",
            "start": base + timedelta(days=min(16, span - 3), hours=7),
            "end": base + timedelta(days=min(16, span - 3), hours=10),
            "impact": "Medium",
            "source": "OPEC",
            "tickers": "XLE, CL=F",
            "notes": "Production guidance key.",
        },
        {
            "title": "MegaCap Earnings (AAPL)",
            "category": "Earnings",
            "start": base + timedelta(days=min(8, span // 3), hours=16),
            "end": base + timedelta(days=min(8, span // 3), hours=17),
            "impact": "High",
            "source": "Company",
            "tickers": "AAPL, QQQ",
            "notes": "Call 30–60min later.",
        },
        {
            "title": "IPO: ACME Corp (Pricing)",
            "category": "IPO/Lockup",
            "start": base + timedelta(days=min(14, span // 2), hours=18),
            "end": base + timedelta(days=min(14, span // 2), hours=20),
            "impact": "Low",
            "source": "Underwriters",
            "tickers": "",
            "notes": "First trade next day.",
        },
        {
            "title": "G20 Summit (Leaders' Session)",
            "category": "Geopolitics",
            "start": base + timedelta(days=min(18, span - 2), hours=9),
            "end": base + timedelta(days=min(18, span - 2), hours=17),
            "impact": "Medium",
            "source": "G20",
            "tickers": "DXY, EEM",
            "notes": "Trade language watch.",
        },
        {
            "title": "Product Launch (MSFT)",
            "category": "Company-Specific",
            "start": base + timedelta(days=min(22, span - 2), hours=13),
            "end": base + timedelta(days=min(22, span - 2), hours=15),
            "impact": "Medium",
            "source": "Company",
            "tickers": "MSFT",
            "notes": "Guidance tone matters.",
        },
    ]

    # Keep only within window
    pruned = []
    for ev in samples:
        if start <= ev["start"].date() <= end:
            pruned.append(ev)
    return pruned


def to_dataframe(events: List[Event]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(by=["start", "impact"], ascending=[True, False]).reset_index(drop=True)
    return df


def make_timeline(df: pd.DataFrame):
    if df.empty:
        st.info("No events in the selected filters/date window.")
        return
    # Plotly timeline needs start/end columns
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="category",
        color="impact",
        hover_data=["title", "source", "tickers", "notes"],
        title="Timeline of Events",
    )
    fig.update_yaxes(autorange="reversed")  # most recent categories at top
    fig.update_layout(height=520, legend_title_text="Impact")
    st.plotly_chart(fig, use_container_width=True)


def export_ics(df: pd.DataFrame) -> str:
    """Create a simple ICS string for download."""
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
    return "\n".join(ics_lines)


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Filters")
window_days = st.sidebar.slider("Window (days forward)", 7, 90, 45, 1)
start_date = st.sidebar.date_input("Start date", value=date.today())
end_date = start_date + timedelta(days=window_days)

category_sel = st.sidebar.multiselect(
    "Categories",
    options=CATEGORIES,
    default=CATEGORIES,
)
impact_sel = st.sidebar.multiselect(
    "Impact levels",
    options=IMPACT_LEVELS,
    default=IMPACT_LEVELS,
)

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
    st.session_state.manual_events.append(
        {
            "title": e_title,
            "category": e_category,
            "start": start_dt,
            "end": end_dt,
            "impact": e_impact,
            "source": e_source,
            "tickers": e_tickers,
            "notes": e_notes,
        }
    )
    st.success(f"Added event: {e_title}")

# ------------------------------------------------------------
# Data assembly
# ------------------------------------------------------------
seeded = seed_sample_events(start_date, end_date)
manual = st.session_state.manual_events

all_events = seeded + manual

# Filter
events_df = to_dataframe(all_events)
if not events_df.empty:
    events_df = events_df[events_df["category"].isin(category_sel) & events_df["impact"].isin(impact_sel)]

# ------------------------------------------------------------
# Layout: Timeline + Upcoming table
# ------------------------------------------------------------
col1, col2 = st.columns([2, 1], gap="large")
with col1:
    make_timeline(events_df)

with col2:
    st.subheader("Upcoming events")
    if events_df.empty:
        st.info("No upcoming events match your filters.")
    else:
        nice_df = events_df.copy()
        nice_df["start"] = nice_df["start"].dt.strftime("%Y-%m-%d %H:%M")
        nice_df["end"] = nice_df["end"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            nice_df[["start", "end", "title", "category", "impact", "tickers", "source"]],
            use_container_width=True,
            hide_index=True,
        )

        ics_text = export_ics(events_df)
        st.download_button(
            label="📥 Export to .ics (Calendar)",
            file_name="market_events.ics",
            mime="text/calendar",
            data=ics_text,
        )

# ------------------------------------------------------------
# Future hooks (non-functional stubs for now)
# ------------------------------------------------------------
with st.expander("⚙️ Integration hooks (planned)"):
    st.markdown(
        "- **Risk adjustments**: Provide a function to compute an 'event risk score' for the next 48–72 hours and feed it into the strategy.\n"
        "- **Real-time news flags**: Ingest breaking headlines and tag them with the same categories to surface on this page.\n"
        "- **External feeds**: Wire to economic calendar API, earnings calendar, and your Google Sheet for custom lists."
    )
