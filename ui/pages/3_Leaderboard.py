# ui/3_Leaderboard.py
# Signal Leaderboard — ranks all tickers by backtest performance metrics.
# Replaces v2.5 pg3 Signal Leaderboard (which used MAE/MSE/R² and Google Sheets).

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from features.builder import build_feature_dataframe
from signals.generator import generate_signals, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BLOCK_TAU

st.set_page_config(page_title="Signal Leaderboard", page_icon="🏆", layout="wide")
st.title("🏆 Signal Leaderboard")
st.caption("Ranks all tickers by backtest Sharpe ratio. Run Strategy to refresh.")

with st.expander("📖 How to read this leaderboard", expanded=False):
    st.markdown("""
### How to decide what to Buy / Hold / Avoid

Think of each metric like rating a restaurant:

---

#### 📊 The Metrics Explained

| Metric | What it means | What to look for |
|--------|--------------|-----------------|
| **Sharpe** | How *consistently* the strategy wins — like a reliable chef | ✅ >2 good, >3 excellent |
| **CAGR** | How much money you'd make per year if you followed every signal | ✅ Higher is better — but only if Sharpe is also high |
| **MaxDD** | Worst-case loss before recovery — your worst nightmare scenario | ✅ Closer to 0% is safer |
| **Accuracy** | How often the model correctly called the direction | ✅ >55% is a real edge |
| **n_trades** | How many BUY signals fired over the backtest period | ✅ More trades = more data to trust the stats |
| **profit_factor** | Gross profit ÷ gross loss. 2.0 = made $2 for every $1 lost | ✅ >1.5 is solid |

---

#### 🎯 The Simple Decision Rule

**Best picks = HIGH Sharpe + HIGH Accuracy + LOW MaxDD**

- **Sharpe >3, Accuracy >60%, MaxDD >-20%** → Strong conviction ticker, watch for BUY signal
- **Sharpe 1-3, Accuracy 55-60%** → Decent — only buy if regime is BULL/NEUTRAL
- **MaxDD worse than -40%** → Too risky regardless of other metrics
- **CAGR looks extreme (>300%)** → Likely a data artifact, treat with skepticism

---

#### ⚡ Why everything shows HOLD right now

The **VOLATILE regime** (VIX at 98th percentile) has raised the BUY threshold to **65%**.
Most tickers are scoring 30-55% probability — not enough to trigger.
This is the model protecting you from entering in a choppy market.

**When regime shifts to NEUTRAL or BULL** → watch your top Sharpe tickers first.
Those are your highest-conviction entries when the market cooperates.

---

#### 🏆 Quick Reference: Your Best Tickers

| Ticker | Why it's good | Watch for |
|--------|--------------|-----------|
| **NVO** | Highest Sharpe, lowest drawdown, 66% accuracy | First BUY in BULL regime |
| **TSM** | High CAGR + Sharpe, slightly higher drawdown | Strong momentum plays |
| **CNC** | Best profit factor (8×), stable drawdown | Defensive position |
| **SLV** | Consistent, commodity hedge | Good in inflation regimes |
| **OPEN** | Extreme CAGR but -50% MaxDD | Avoid — too risky |
    """)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_tickers() -> list[str]:
    for p in [Path(_ROOT) / "tickers.txt", Path(_ROOT).parent / "tickers.txt"]:
        if p.exists():
            return [t.strip().upper() for t in p.read_text().splitlines() if t.strip()]
    return ["AAPL", "NVDA", "TSLA", "AMD"]

def _save_tickers(lst: list[str]):
    for p in [Path(_ROOT) / "tickers.txt", Path(_ROOT).parent / "tickers.txt"]:
        if p.exists():
            p.write_text("\n".join(lst))
            return

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    horizon              = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}d")
    confidence_threshold = st.slider("Confidence threshold",
                                     0.50, 0.95, DEFAULT_CONFIDENCE_THRESHOLD, 0.01)
    block_tau            = st.slider("Block when risk_next_3d ≥", 0, 6, DEFAULT_BLOCK_TAU, 1)
    start_date           = st.date_input("Start", value=date(2022, 1, 1))
    end_date             = st.date_input("End",   value=date.today())

    st.markdown("### 🗂️ Tickers")
    all_tickers = _load_tickers()
    col_a, col_b = st.columns(2)
    if col_a.button("✅ Select All"):
        st.session_state["lb_selected_tickers"] = all_tickers
    if col_b.button("❌ Clear"):
        st.session_state["lb_selected_tickers"] = []

    tickers = st.multiselect(
        "Select tickers to run",
        options=all_tickers,
        default=st.session_state.get("lb_selected_tickers", all_tickers),
        key="lb_selected_tickers",
    )

    with st.expander("✏️ Edit master list"):
        raw = st.text_area("One per line", "\n".join(all_tickers), height=200)
        if st.button("💾 Save list"):
            _save_tickers([t.strip().upper() for t in raw.splitlines() if t.strip()])
            st.success("Saved — reload to apply")

    run = st.button("🚀 Run Leaderboard", type="primary")

# ── Run ───────────────────────────────────────────────────────────────────────
if run:
    rows = []
    progress = st.progress(0, text="Building leaderboard...")

    for i, tkr in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Processing {tkr}...")
        try:
            df = build_feature_dataframe(
                tkr,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            if df.empty:
                continue

            result = generate_signals(
                ticker=tkr,
                df=df,
                horizon=horizon,
                confidence_threshold=confidence_threshold,
                block_tau=block_tau,
            )

            if result.error:
                continue

            m = result.metrics
            rows.append({
                "ticker":         tkr,
                "signal":         result.today_signal,
                "prob":           result.today_prob_eff,
                "sharpe":         m.sharpe,
                "cagr":           m.cagr,
                "max_drawdown":   m.max_drawdown,
                "accuracy":       m.accuracy,
                "n_trades":       m.n_trades,
                "exposure":       m.exposure,
                "profit_factor":  m.profit_factor,
            })
        except Exception as e:
            st.warning(f"⚠️ {tkr}: {e}")

    progress.progress(1.0, text="Done.")

    if not rows:
        st.error("No results. Check tickers and date range.")
        st.stop()

    lb = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    lb.index += 1   # rank starts at 1
    st.session_state["leaderboard"] = lb

# ── Display ───────────────────────────────────────────────────────────────────
lb = st.session_state.get("leaderboard")

if lb is None:
    st.info("Click **Run Leaderboard** in the sidebar to generate rankings.")
    st.stop()

# ── Top 3 podium ──────────────────────────────────────────────────────────────
st.subheader("🥇 Top Performers")
top3 = lb.head(3)
cols = st.columns(3)
medals = ["🥇", "🥈", "🥉"]
for col, (_, row), medal in zip(cols, top3.iterrows(), medals):
    signal_color = "🟢" if row["signal"] == "BUY" else "🔴"
    col.metric(
        label=f"{medal} {row['ticker']}",
        value=f"Sharpe {row['sharpe']:.2f}",
        delta=f"{signal_color} {row['signal']}  p={row['prob']:.1%}",
    )
    col.caption(
        f"CAGR {row['cagr']:.1%} · "
        f"MaxDD {row['max_drawdown']:.1%} · "
        f"Acc {row['accuracy']:.1%}"
    )

st.divider()

# ── Full leaderboard table ────────────────────────────────────────────────────
st.subheader("📋 Full Rankings")

with st.expander("📖 How to read this table"):
    st.markdown("""
**Think of each column like a restaurant rating:**

| Column | What it means | What to look for |
|--------|--------------|-----------------|
| **Sharpe** | How *consistently* the strategy wins — like a reliable chef | **> 2.0** is excellent, > 3.0 is exceptional |
| **CAGR** | Annualized return if you followed every signal | **Higher = better**, but only trust it if Sharpe is also high |
| **MaxDD** | Worst loss before recovering — your worst nightmare scenario | **Closer to 0%** is safer |
| **Accuracy** | How often the model called direction correctly | **> 55%** is a real edge |
| **n_trades** | How many BUY signals fired in the backtest period | More trades = more data to trust the stats |
| **profit_factor** | $ made for every $ lost — > 1 means profitable | **> 2.0** is strong |

---
**How to pick what to buy:**
Look for tickers that score well on ALL four: high Sharpe + high CAGR + low MaxDD + high Accuracy.

**Right now everything shows HOLD** because the market regime is VOLATILE (VIX at 98th percentile).
The model is saying *"these are great tickers — but wait for a better entry point."*

→ When regime shifts to **NEUTRAL or BULL**, watch the top-ranked tickers first.
    """)

def _color_signal(val):
    return "color: #00c853" if val == "BUY" else "color: #ff1744"

def _color_sharpe(val):
    if pd.isna(val): return ""
    if val >= 1.5:  return "color: #00c853"
    if val >= 0.5:  return "color: #ffab00"
    return "color: #ff1744"

styled = (
    lb.reset_index()[["index", "ticker", "signal", "prob", "sharpe",
                       "cagr", "max_drawdown", "accuracy",
                       "n_trades", "profit_factor"]]
    .rename(columns={"index": "rank"})
    .style
    .format({
        "prob":          "{:.1%}",
        "sharpe":        "{:.2f}",
        "cagr":          "{:.1%}",
        "max_drawdown":  "{:.1%}",
        "accuracy":      "{:.1%}",
        "profit_factor": "{:.2f}",
    })
    .applymap(_color_signal,  subset=["signal"])
    .applymap(_color_sharpe,  subset=["sharpe"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Scatter: Sharpe vs CAGR ───────────────────────────────────────────────────
st.subheader("📊 Sharpe vs CAGR")

scatter = (
    alt.Chart(lb.reset_index())
    .mark_circle(size=120)
    .encode(
        x=alt.X("sharpe:Q", title="Sharpe Ratio"),
        y=alt.Y("cagr:Q",   title="CAGR", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "signal:N",
            scale=alt.Scale(
                domain=["BUY", "HOLD"],
                range=["#00c853", "#ff1744"],
            ),
        ),
        size=alt.Size("n_trades:Q", legend=None),
        tooltip=[
            "ticker",
            alt.Tooltip("sharpe:Q",       format=".2f"),
            alt.Tooltip("cagr:Q",         format=".1%"),
            alt.Tooltip("max_drawdown:Q", format=".1%"),
            alt.Tooltip("accuracy:Q",     format=".1%"),
            "signal",
            alt.Tooltip("prob:Q",         format=".1%"),
        ],
        text="ticker:N",
    )
    .properties(height=400, title="Green = BUY signal today · Size = number of trades")
)

labels = scatter.mark_text(align="left", dx=8, fontSize=11).encode(text="ticker:N")
st.altair_chart((scatter + labels).interactive(), use_container_width=True)

# ── Bar: Max Drawdown ─────────────────────────────────────────────────────────
st.subheader("🛡️ Max Drawdown by Ticker")
dd_chart = (
    alt.Chart(lb.reset_index().sort_values("max_drawdown"))
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort=None),
        y=alt.Y("max_drawdown:Q", title="Max Drawdown",
                axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "max_drawdown:Q",
            scale=alt.Scale(scheme="redyellowgreen", reverse=True),
            legend=None,
        ),
        tooltip=["ticker", alt.Tooltip("max_drawdown:Q", format=".1%")],
    )
    .properties(height=300)
)
st.altair_chart(dd_chart, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
csv = lb.reset_index().to_csv(index=False).encode()
st.download_button(
    "⬇️ Download leaderboard CSV",
    csv,
    file_name="leaderboard.csv",
    mime="text/csv",
    key="dl_lb",
)

# ── Dynamic analyst explanation ───────────────────────────────────────────────
st.divider()
st.subheader("📖 How to Read This — Explained Simply")

df_exp = lb.reset_index().copy()

# Filter out extreme CAGR outliers for examples
df_clean = df_exp[df_exp["cagr"] < 5.0].copy()

# Pick dynamic examples from actual data
best_sharpe  = df_clean.loc[df_clean["sharpe"].idxmax()]
best_cagr    = df_clean.loc[df_clean["cagr"].idxmax()]
worst_dd     = df_clean.loc[df_clean["max_drawdown"].idxmin()]
best_dd      = df_clean.loc[df_clean["max_drawdown"].idxmax()]
best_acc     = df_clean.loc[df_clean["accuracy"].idxmax()]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
**📊 Sharpe = How reliable is the chef**
- **{best_sharpe['ticker']} Sharpe {best_sharpe['sharpe']:.2f}** = every time you go, the food is consistently great
- Low Sharpe = sometimes amazing, sometimes terrible — unpredictable
- ✅ You want **HIGH Sharpe** — means the strategy wins consistently, not just occasionally

---

**📈 CAGR = How much money you'd make per year**
- **{best_cagr['ticker']} CAGR {best_cagr['cagr']:.0%}** = a $10,000 investment would grow to **${10000 * (1 + best_cagr['cagr']):,.0f}** in one year
- ✅ You want **HIGH CAGR** — but only if Sharpe is also high (otherwise too risky)
""")

with col2:
    st.markdown(f"""
**🛡️ MaxDD = Worst nightmare scenario**
- **{best_dd['ticker']} MaxDD {best_dd['max_drawdown']:.1%}** = worst case you ever lost {abs(best_dd['max_drawdown']):.1%} before recovering
- **{worst_dd['ticker']} MaxDD {worst_dd['max_drawdown']:.1%}** = at one point you lost {abs(worst_dd['max_drawdown']):.0%} of your money
- ✅ You want **LOW MaxDD** (close to 0%)

---

**🎯 Accuracy = How often the model was right**
- **{best_acc['ticker']} Accuracy {best_acc['accuracy']:.1%}** = model correctly called direction {best_acc['accuracy']:.0%} of the time
- ✅ You want **>55%** — anything above 55% is a real edge
""")

# ── Dynamic verdict table ─────────────────────────────────────────────────────
st.markdown("**The simple buy decision — look for tickers that score well on ALL four:**")

def _verdict(row):
    score = sum([
        row["sharpe"]       >= 2.0,
        row["cagr"]         >= 0.5,
        row["max_drawdown"] >= -0.25,
        row["accuracy"]     >= 0.58,
    ])
    if score == 4: return "⭐ Best overall"
    if score == 3: return "✅ Strong"
    if score == 2: return "⚠️ Selective"
    return "🚫 Avoid"

def _flag(val, metric):
    if metric == "sharpe":
        return "✅" if val >= 2.0 else "⚠️" if val >= 1.0 else "🚫"
    if metric == "cagr":
        return "✅" if val >= 0.5 else "⚠️" if val >= 0.2 else "🚫"
    if metric == "maxdd":
        return "✅" if val >= -0.20 else "⚠️" if val >= -0.35 else "🚫"
    if metric == "accuracy":
        return "✅" if val >= 0.58 else "⚠️" if val >= 0.54 else "🚫"
    return ""

# Show top 8 by Sharpe, skip extreme CAGR outliers
top8 = df_clean.nlargest(8, "sharpe")
verdict_rows = []
for _, row in top8.iterrows():
    verdict_rows.append({
        "Ticker":   row["ticker"],
        "Sharpe":   f"{_flag(row['sharpe'], 'sharpe')} {row['sharpe']:.2f}",
        "CAGR":     f"{_flag(row['cagr'], 'cagr')} {row['cagr']:.0%}",
        "MaxDD":    f"{_flag(row['max_drawdown'], 'maxdd')} {row['max_drawdown']:.1%}",
        "Accuracy": f"{_flag(row['accuracy'], 'accuracy')} {row['accuracy']:.1%}",
        "Verdict":  _verdict(row),
    })

st.dataframe(
    verdict_rows,
    use_container_width=True,
    hide_index=True,
)

# ── Bottom line ───────────────────────────────────────────────────────────────
try:
    from models.regime_classifier import get_current_regime
    regime = get_current_regime(use_cache=True)
    buy_tickers = df_exp[df_exp["signal"] == "BUY"]["ticker"].tolist()
    top3 = df_clean.nlargest(3, "sharpe")["ticker"].tolist()

    if buy_tickers:
        bottom_line = (
            f"🟢 **{len(buy_tickers)} BUY signal(s) today:** {', '.join(buy_tickers)}. "
            f"Regime is **{regime.label}** — signals are {'strong' if regime.label == 'BULL' else 'selective'}."
        )
    else:
        bottom_line = (
            f"⚡ **Everything is HOLD** because the market is **{regime.label}** "
            f"(threshold raised to {regime.confidence_threshold:.0%}). "
            f"When regime shifts to NEUTRAL or BULL → watch **{', '.join(top3)}** first. "
            f"Those are your highest-conviction tickers when the market cooperates."
        )
    st.info(bottom_line)
except Exception:
    pass
