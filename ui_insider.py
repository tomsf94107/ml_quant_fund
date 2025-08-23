# ui_insider.py
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

def _last_valid(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return s.iloc[-1] if len(s) else None

def render_insider_summary_card(feats: pd.DataFrame):
    """
    Show small KPI cards for insider signals using the latest available row.

    Expected columns (auto-handles missing):
      - Calculated features: ins_net_shares_7d, ins_net_shares_30d, ins_pressure_30d_z, ins_large_or_exec_7d
      - DB rollups (optional): ins_net_shares_7d_db, ins_net_shares_21d_db
    """
    if feats is None or feats.empty:
        st.info("No insider data for this range.")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    v1 = _last_valid(feats, "ins_net_shares_7d")
    v2 = _last_valid(feats, "ins_net_shares_30d")
    v3 = _last_valid(feats, "ins_pressure_30d_z")
    v4 = _last_valid(feats, "ins_large_or_exec_7d")
    v5 = _last_valid(feats, "ins_net_shares_7d_db")
    v6 = _last_valid(feats, "ins_net_shares_21d_db")

    c1.metric("Net Shares (7d, calc)",  f"{v1:,.0f}" if v1 is not None else "–")
    c2.metric("Net Shares (30d, calc)", f"{v2:,.0f}" if v2 is not None else "–")
    c3.metric("Insider Pressure z (30d)", f"{v3:,.2f}" if v3 is not None else "–",
              help="Z-scored 30d: net_shares + 0.5*holdings_delta + 0.25*buy-minus-sell")
    c4.metric("Exec/Large (7d)", f"{int(v4)}" if v4 is not None else "–",
              help="Days in last 7 with exec trades or ≥$1M transactions")
    c5.metric("Net Shares (7d, DB)",  f"{v5:,.0f}" if v5 is not None else "–",
              help="Direct from SQLite insider_flows.insider_7d")
    c6.metric("Net Shares (21d, DB)", f"{v6:,.0f}" if v6 is not None else "–",
              help="Direct from SQLite insider_flows.insider_21d")

def insider_highlight_toggle(default=True):
    return st.checkbox(
        "Highlight days with exec/large insider activity",
        value=default,
        help="Uses the 7-day rolling count `ins_large_or_exec_7d` > 0"
    )

def chart_price_with_insider(feats: pd.DataFrame, price_col: str = "close", highlight: bool = True):
    """
    Altair chart: price line + optional highlighted points for days where
    ins_large_or_exec_7d > 0.

    Expects columns: ['date', price_col] and optionally:
      - 'ins_large_or_exec_7d', 'ins_pressure_30d_z'
      - 'ins_net_shares_7d' (calc)
      - 'ins_net_shares_7d_db', 'ins_net_shares_21d_db' (DB rollups)
    """
    if feats is None or feats.empty or price_col not in feats.columns or "date" not in feats.columns:
        return None

    d = feats.copy()
    d = d.dropna(subset=["date"]).copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    # Build tooltip list dynamically based on available columns
    tooltip_base = ["date:T", alt.Tooltip(f"{price_col}:Q", format=",.2f")]
    if "ins_pressure_30d_z" in d.columns:
        tooltip_base.append(alt.Tooltip("ins_pressure_30d_z:Q", format=",.2f", title="Insider z (30d)"))
    if "ins_net_shares_7d_db" in d.columns:
        tooltip_base.append(alt.Tooltip("ins_net_shares_7d_db:Q", format=",.0f", title="Net Shares 7d (DB)"))
    if "ins_net_shares_21d_db" in d.columns:
        tooltip_base.append(alt.Tooltip("ins_net_shares_21d_db:Q", format=",.0f", title="Net Shares 21d (DB)"))

    base = alt.Chart(d).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y(f"{price_col}:Q", title="Price"),
        tooltip=tooltip_base
    )

    if highlight and "ins_large_or_exec_7d" in d.columns:
        d["_flag"] = (d["ins_large_or_exec_7d"].fillna(0) > 0).astype(int)
        # Build point tooltip, include DB rollups if present
        point_tooltip = [
            "date:T",
            alt.Tooltip(f"{price_col}:Q", format=",.2f"),
            alt.Tooltip("ins_large_or_exec_7d:Q", title="Exec/Large (7d)")
        ]
        if "ins_net_shares_7d" in d.columns:
            point_tooltip.append(alt.Tooltip("ins_net_shares_7d:Q", title="Net Shares 7d (calc)", format=",.0f"))
        if "ins_pressure_30d_z" in d.columns:
            point_tooltip.append(alt.Tooltip("ins_pressure_30d_z:Q", title="Insider z (30d)", format=",.2f"))
        if "ins_net_shares_7d_db" in d.columns:
            point_tooltip.append(alt.Tooltip("ins_net_shares_7d_db:Q", title="Net Shares 7d (DB)", format=",.0f"))
        if "ins_net_shares_21d_db" in d.columns:
            point_tooltip.append(alt.Tooltip("ins_net_shares_21d_db:Q", title="Net Shares 21d (DB)", format=",.0f"))

        pts = alt.Chart(d[d["_flag"] == 1]).mark_point(size=60).encode(
            x="date:T",
            y=f"{price_col}:Q",
            tooltip=point_tooltip
        )
        return (base + pts).interactive()

    return base.interactive()
