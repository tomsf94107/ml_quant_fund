# ui/components/regime_widget.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit component that shows the current macro regime in the Dashboard.
# Import and call render_regime_widget() from 1_Dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st


def render_regime_widget():
    """Render the macro regime banner in the Dashboard."""
    try:
        from models.regime_classifier import get_current_regime, get_regime_history
        import pandas as pd

        with st.spinner("Checking market regime..."):
            regime = get_current_regime(use_cache=True)

        # ── Regime banner ─────────────────────────────────────────────────────
        color_map = {
            "BULL":     ("#0a3d0a", "#4cff4c", "🐂"),
            "BEAR":     ("#3d0a0a", "#ff4c4c", "🐻"),
            "VOLATILE": ("#3d2e0a", "#ffc04c", "⚡"),
            "NEUTRAL":  ("#1a1a2e", "#8888cc", "〰️"),
        }
        bg, fg, icon = color_map.get(regime.label, color_map["NEUTRAL"])

        st.markdown(f"""
        <div style="
            background: {bg};
            border: 1px solid {fg};
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 12px;
        ">
            <span style="color:{fg}; font-size:1.1em; font-weight:700;">
                {icon} Market Regime: {regime.label}
            </span>
            <span style="color:#aaa; font-size:0.85em; margin-left:12px;">
                confidence {regime.confidence:.0%}
            </span>
            <br>
            <span style="color:#ccc; font-size:0.85em;">
                {regime.description}
            </span>
            <br>
            <span style="color:#888; font-size:0.80em;">
                Threshold: {regime.confidence_threshold:.0%} &nbsp;|&nbsp;
                Multiplier: {regime.signal_multiplier:.2f}x &nbsp;|&nbsp;
                VIX: {regime.vix_level:.1f} ({regime.vix_percentile:.0f}th pct) &nbsp;|&nbsp;
                SPY 20d: {regime.spy_trend_20d:+.1%} &nbsp;|&nbsp;
                Bonds: {regime.bond_signal}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Regime history chart ──────────────────────────────────────────────
        with st.expander("📊 Regime history (1 year)"):
            with st.spinner("Computing regime history..."):
                hist = get_regime_history(lookback_days=252)

            if not hist.empty:
                import altair as alt

                color_scale = alt.Scale(
                    domain=["BULL", "NEUTRAL", "VOLATILE", "BEAR"],
                    range=["#4cff4c", "#8888cc", "#ffc04c", "#ff4c4c"]
                )

                chart = alt.Chart(hist).mark_rect().encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("regime:N", title="Regime",
                             sort=["BULL", "NEUTRAL", "VOLATILE", "BEAR"]),
                    color=alt.Color("regime:N", scale=color_scale, legend=None),
                    tooltip=["date:T", "regime:N",
                              alt.Tooltip("vix:Q", format=".1f", title="VIX"),
                              alt.Tooltip("spy_ret_20d:Q", format=".1%", title="SPY 20d")]
                ).properties(height=120)

                vix_chart = alt.Chart(hist).mark_line(color="#ffc04c", strokeWidth=1.5).encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("vix:Q", title="VIX"),
                    tooltip=["date:T", alt.Tooltip("vix:Q", format=".1f")]
                ).properties(height=100)

                st.altair_chart(chart, use_container_width=True)
                st.altair_chart(vix_chart, use_container_width=True)
            else:
                st.info("Regime history unavailable.")

    except Exception as e:
        st.warning(f"Regime classifier unavailable: {e}")
