#!/usr/bin/env python3
# Run this from ~/Desktop/ML_Quant_Fund:
#   python3 patch_dashboard.py

from pathlib import Path

path = Path("ui/1_Dashboard.py")
src  = path.read_text()

# ── 1. Replace signal cards with enriched cards + forecast table ─────────────
old_cards = '''    # ── Signal cards ──────────────────────────────────────────────────────────
    st.subheader("📡 Live Signals")
    cols = st.columns(min(len(signal_summary), 4))
    for i, r in enumerate(signal_summary):
        col = cols[i % 4]
        badge = _confidence_badge(confidence_str)
        signal_color = "🟢" if r.today_signal == "BUY" else "🔴"
        col.metric(
            label=r.ticker,
            value=f"{signal_color} {r.today_signal}",
            delta=f"p={r.today_prob_eff:.1%}  {badge}",
        )
        if enable_email and r.today_signal == "BUY" and r.today_prob_eff >= 0.65:
            _send_alert(r.ticker, r.today_prob_eff, horizon)'''

new_cards = '''    # ── Signal cards ──────────────────────────────────────────────────────────
    st.subheader("📡 Live Signals")
    cols = st.columns(min(len(signal_summary), 4))
    for i, r in enumerate(signal_summary):
        col = cols[i % 4]
        badge = _confidence_badge(confidence_str)
        signal_color = "🟢" if r.today_signal == "BUY" else "🔴"
        col.metric(
            label=r.ticker,
            value=f"{signal_color} {r.today_signal}",
            delta=f"p={r.today_prob_eff:.1%}  {badge}",
        )
        if enable_email and r.today_signal == "BUY" and r.today_prob_eff >= 0.65:
            _send_alert(r.ticker, r.today_prob_eff, horizon)

    # ── Forecast table ────────────────────────────────────────────────────────
    st.subheader("🎯 Price Forecast Table")

    import pandas as pd
    forecast_rows = []
    for r in signal_summary:
        exp_ret = r.expected_return or 0.0
        forecast_rows.append({
            "Ticker":       r.ticker,
            "Signal":       r.today_signal,
            "Price":        f"${r.current_price:.2f}"    if r.current_price   else "—",
            "Prob Eff":     f"{r.today_prob_eff:.1%}",
            "Target ▲":     f"${r.price_target_up:.2f}"  if r.price_target_up else "—",
            "Target ▼":     f"${r.price_target_dn:.2f}"  if r.price_target_dn else "—",
            "Exp Return":   f"{exp_ret:+.2%}"             if r.expected_return is not None else "—",
            "ATR":          f"${r.atr:.2f}"               if r.atr             else "—",
            "Sharpe":       f"{r.metrics.sharpe:.2f}"     if not np.isnan(r.metrics.sharpe) else "—",
        })

    fdf = pd.DataFrame(forecast_rows)

    def _color_signal(val):
        if val == "BUY":  return "color: #22c55e; font-weight: bold"
        return "color: #94a3b8"

    def _color_exp(val):
        if val == "—": return ""
        try:
            n = float(val.replace("%","").replace("+",""))
            return "color: #22c55e" if n >= 0 else "color: #ef4444"
        except: return ""

    styled = (
        fdf.style
        .map(_color_signal, subset=["Signal"])
        .map(_color_exp,    subset=["Exp Return"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── How to read this table ────────────────────────────────────────────────
    with st.expander("📖 How to read the Forecast Table", expanded=False):
        st.markdown("""
**SIGNAL** — BUY or HOLD.
- **BUY** fires when `Prob Eff` exceeds the confidence threshold (default 65% in VOLATILE regime, 55% in NEUTRAL).
- **HOLD** means wait — either probability is too low or the regime is suppressing the signal.

**PRICE** — Today's closing price. This is the price the model used to generate the forecast.

**PROB EFF** — The model's confidence that this stock goes UP over the next `horizon` days, after all signal adjustments:
```
Raw ML prob × regime multiplier × sentiment × options flow × short interest
```
- Above threshold → BUY
- Below threshold → HOLD
- Currently in VOLATILE regime, threshold = 65%

**TARGET ▲** — Where the stock is likely to go if it moves UP.
Calculated as: `current price + ATR × √horizon`
Example: NVDA at $121 with ATR $5.35 → Target ▲ = $126.35

**TARGET ▼** — Where it goes if it moves DOWN.
Calculated as: `current price − ATR × √horizon`
This is your downside risk if the signal is wrong.

**EXP RETURN** — Probability-weighted expected return. The single most actionable number:
```
= (Prob_eff × upside%) − ((1 − Prob_eff) × downside%)
```
- **Positive (green)** = model expects to make money → worth watching
- **Negative (red)** = model expects to lose money → stay out
- Right now all negative because VOLATILE regime is suppressing prob_eff below 50%

**ATR** — Average True Range. How much this stock moves on a typical day in dollars.
- Low ATR (e.g. NVO $3.73) = stable, lower risk
- High ATR (e.g. TSLA $13.09) = volatile, bigger swings both ways

**SHARPE** — Historical risk-adjusted return from backtesting.
- Above 2.0 = excellent (green)
- 1.0–2.0 = good (yellow)
- Below 1.0 = poor (red)

---
**When to act:**
1. Exp Return turns **green** on a ticker
2. Prob Eff crosses the threshold → **BUY fires**
3. Regime shifts from VOLATILE → NEUTRAL/BULL (threshold drops from 65% → 55%)
""")'''

assert old_cards in src, "Could not find signal cards section"
src = src.replace(old_cards, new_cards)
path.write_text(src)
print("✓ Forecast table + how-to-read guide added to dashboard")
