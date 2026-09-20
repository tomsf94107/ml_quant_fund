# ui/pages/19_Institutional_Detail.py
# Drill into ONE manager or ONE ticker. The summary page answers "what is
# happening"; this answers "what is this manager doing" and "who owns this
# name".
#
# WHY A SEPARATE PAGE
#     The summary page aggregates across 127 managers and 51 quarters. Every
#     aggregate hides the thing that makes 13F worth reading at all: the
#     TRAJECTORY of one book. Thiel Macro closed NVDA and VST in 2025-Q3, went
#     to zero reportable long equity in Q4, and rebuilt in 2026-Q2 with eight
#     new positions, 71% energy and power. No aggregate shows that. A manager
#     view does.
#
# WHAT "FLAT" MEANS HERE
#     13F covers LONG US EQUITY and options only. A manager showing zero
#     positions holds nothing REPORTABLE -- not nothing. Thiel Macro's two
#     empty quarters could have been futures, FX, bonds, cash or private
#     positions, none of which appear. Read a gap as "no reportable long
#     equity", never as "out of the market".
#
# THE 500-ROW CAP
#     Managers whose book exceeds 500 positions are excluded from the roster,
#     so anything shown here is a COMPLETE book. If a manager ever appears with
#     exactly 500 rows in a quarter, that quarter is truncated and its position
#     count, sector weights and turnover are artifacts of the cap.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(_ROOT)
DB_INST, DB_DARK = R / "institutions.db", R / "institutional_trades.db"
DB_PX = R / "prices.db"
META = R / "tickers_metadata.csv"

st.set_page_config(page_title="Institutional detail", page_icon="🔎", layout="wide")
st.title("🔎 Institutional detail")


def ro(p):
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=60)


@st.cache_data(ttl=1800)
def meta():
    if not META.exists():
        return pd.DataFrame(columns=["ticker", "bucket", "tier"])
    m = pd.read_csv(META)
    return m[[c for c in ("ticker", "bucket", "tier") if c in m.columns]]


@st.cache_data(ttl=1800)
def cols_available():
    c = ro(DB_INST)
    s = {r[1] for r in c.execute("PRAGMA table_info(inst_holdings)")}
    c.close()
    return s


@st.cache_data(ttl=1800)
def manager_book(name):
    cols = cols_available()
    pc = "put_call" if "put_call" in cols else "NULL AS put_call"
    uc = "units_change" if "units_change" in cols else "NULL AS units_change"
    c = ro(DB_INST)
    df = pd.read_sql(f"""
        SELECT report_date, ticker, units, value, sector, security_type,
               {pc} AS put_call, {uc} AS units_change
        FROM inst_holdings WHERE name = ? ORDER BY report_date DESC, value DESC
    """, c, params=(name,))
    c.close()
    if df.empty:
        return df
    sh = df.security_type == "Share"
    df.loc[sh, "_book"] = df[sh].groupby("report_date")["value"].transform("sum")
    df["pct_book"] = (df["value"] / df["_book"] * 100).round(2)
    df["filed"] = (pd.to_datetime(df["report_date"])
                   + pd.Timedelta(days=45)).dt.date.astype(str)
    return df.merge(meta(), on="ticker", how="left")


@st.cache_data(ttl=1800)
def ticker_holders(tk):
    cols = cols_available()
    pc = "put_call" if "put_call" in cols else "NULL AS put_call"
    uc = "units_change" if "units_change" in cols else "NULL AS units_change"
    c = ro(DB_INST)
    df = pd.read_sql(f"""
        SELECT h.report_date, h.name, h.units, h.value, h.security_type,
               {pc} AS put_call, {uc} AS units_change, r.total_value AS mgr_aum
        FROM inst_holdings h LEFT JOIN inst_roster r ON r.name = h.name
        WHERE h.ticker = ? ORDER BY h.report_date DESC, h.value DESC
    """, c, params=(tk,))
    c.close()
    return df


@st.cache_data(ttl=1800)
def names():
    c = ro(DB_INST)
    m = [r[0] for r in c.execute(
        "SELECT name FROM inst_holdings GROUP BY name ORDER BY COUNT(*) DESC")]
    t = [r[0] for r in c.execute(
        "SELECT ticker FROM inst_holdings WHERE ticker IS NOT NULL "
        "GROUP BY ticker ORDER BY SUM(value) DESC LIMIT 3000")]
    c.close()
    return m, t


if not DB_INST.exists():
    st.error("institutions.db not found — run the ingest first.")
    st.stop()
MGRS, TICKS = names()
if not MGRS:
    st.error("No holdings yet — the backfill is still running.")
    st.stop()

mode = st.radio("Drill into", ["Manager", "Ticker"], horizontal=True)

# ══════════════════════════════════════════════════════════ MANAGER
if mode == "Manager":
    who = st.selectbox("Manager", MGRS)
    B = manager_book(who)
    if B.empty:
        st.warning("No rows for this manager."); st.stop()

    qs = sorted(B.report_date.unique())
    latest = qs[-1]
    cur = B[B.report_date == latest]
    shares = cur[cur.security_type == "Share"]

    k = st.columns(5)
    k[0].metric("Quarters on file", len(qs))
    k[1].metric("Positions, latest", len(shares))
    k[2].metric("Book value", f"${shares.value.sum()/1e9:.2f}B")
    k[3].metric("Top position", f"{shares.pct_book.max():.1f}%"
                if len(shares) else "—")
    k[4].metric("Latest quarter", latest)
    st.caption(f"Filed {cur.filed.max()} — 45 days after quarter end by rule. "
               f"History {qs[0]} to {qs[-1]}. A quarter missing from that span "
               f"is archive coverage, not necessarily a liquidation.")

    # POSITION COUNT OVER TIME. The clearest read on a book: a collapse to zero
    # and a rebuild is a thesis change, and it is invisible in any aggregate.
    st.subheader("Book size by quarter")
    hist = (B[B.security_type == "Share"].groupby("report_date")
            .agg(positions=("ticker", "nunique"),
                 value_b=("value", lambda s: s.sum() / 1e9)))
    c1, c2 = st.columns(2)
    c1.markdown("**Positions**"); c1.line_chart(hist["positions"])
    c2.markdown("**Book value $B**"); c2.line_chart(hist["value_b"])
    if (hist["positions"] == 0).any() or hist["positions"].min() <= 2:
        st.info("A quarter at or near zero means no REPORTABLE long equity. "
                "13F does not cover shorts, futures, FX, bonds, cash or "
                "private positions, so this is not 'out of the market'.")

    st.subheader(f"Current book — {latest}")
    show = shares[[c for c in ("ticker", "units", "value", "pct_book", "units_change",
                               "sector", "bucket", "tier") if c in shares]]
    st.dataframe(show.rename(columns={"pct_book": "% book", "units_change": "Units Δ"}),
                 use_container_width=True, hide_index=True, height=420)

    opt = cur[cur.security_type == "Option"]
    if len(opt):
        st.subheader("Options — long positions only")
        st.dataframe(opt[[c for c in ("ticker", "put_call", "units", "value")
                          if c in opt]],
                     use_container_width=True, hide_index=True)
        st.caption("13F reports LONG options only. Short calls and short puts "
                   "never appear, so a put line may hedge something unreported.")

    st.subheader("Every position, every quarter")
    st.dataframe(B[[c for c in ("filed", "report_date", "ticker", "security_type",
                                "put_call", "units", "units_change", "value",
                                "pct_book", "sector", "bucket", "tier")
                    if c in B]],
                 use_container_width=True, hide_index=True, height=460)
    st.download_button("Download CSV", B.to_csv(index=False).encode(),
                       file_name=f"{who.replace(' ','_')}_book.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════ TICKER
else:
    tk = st.selectbox("Ticker", TICKS)
    T = ticker_holders(tk)
    if T.empty:
        st.warning("No holders on file."); st.stop()

    qs = sorted(T.report_date.unique())
    latest = qs[-1]
    cur = T[T.report_date == latest]
    sh = cur[cur.security_type == "Share"]

    k = st.columns(5)
    k[0].metric("Holders, latest", sh.name.nunique())
    k[1].metric("Held value", f"${sh.value.sum()/1e9:.2f}B")
    k[2].metric("Quarters on file", len(qs))
    if "units_change" in cur and cur.units_change.notna().any():
        k[3].metric("Added", int((sh.units_change > 0).sum()))
        k[4].metric("Trimmed", int((sh.units_change < 0).sum()))

    st.subheader("Holder count by quarter")
    hc = (T[T.security_type == "Share"].groupby("report_date")
          .agg(holders=("name", "nunique"),
               value_b=("value", lambda s: s.sum() / 1e9)))
    c1, c2 = st.columns(2)
    c1.markdown("**Holders**"); c1.line_chart(hc["holders"])
    c2.markdown("**Held value $B**"); c2.line_chart(hc["value_b"])
    st.caption("A rising holder count is crowding. Whether crowding predicts "
               "anything is T2, pre-registered and not yet measured — a count "
               "is a description, not evidence.")

    st.subheader(f"Who holds it — {latest}")
    st.dataframe(sh[[c for c in ("name", "units", "value", "units_change", "mgr_aum")
                     if c in sh]].sort_values("value", ascending=False),
                 use_container_width=True, hide_index=True, height=400)

    opt = cur[cur.security_type == "Option"]
    if len(opt):
        st.subheader("Option holders")
        st.dataframe(opt[[c for c in ("name", "put_call", "units", "value")
                          if c in opt]],
                     use_container_width=True, hide_index=True)
        st.caption("Calls and puts are shown separately and never netted — one "
                   "manager can hold both, as a view and a hedge.")

    st.subheader("Every holder, every quarter")
    st.dataframe(T, use_container_width=True, hide_index=True, height=440)
    st.download_button("Download CSV", T.to_csv(index=False).encode(),
                       file_name=f"{tk}_holders.csv", mime="text/csv")
