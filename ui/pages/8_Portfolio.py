# ui/pages/8_Portfolio.py
# ─────────────────────────────────────────────────────────────────────────────
# Private portfolio tracker — positions never stored in git or DB.
# Encrypted local file only, password protected.
# Other pages read from st.session_state["portfolio"] seamlessly.
# ─────────────────────────────────────────────────────────────────────────────

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import base64
import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.timezone import now_et

PORTFOLIO_FILE = Path(".portfolio.enc")
SALT           = b"mlquantfund_salt_v1"

st.set_page_config(page_title="Portfolio — ML Quant Fund", page_icon="💼", layout="wide")

# ── Encryption ────────────────────────────────────────────────────────────────

def _derive_key(password: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), SALT, iterations=100_000)

def _xor_encrypt(data: bytes, key: bytes) -> bytes:
    key_stream = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, key_stream))

def save_portfolio(data: dict, password: str):
    key = _derive_key(password)
    raw = json.dumps(data, indent=2).encode()
    PORTFOLIO_FILE.write_bytes(base64.b64encode(_xor_encrypt(raw, key)))

def load_portfolio(password: str) -> dict | None:
    if not PORTFOLIO_FILE.exists():
        return None
    try:
        key = _derive_key(password)
        raw = _xor_encrypt(base64.b64decode(PORTFOLIO_FILE.read_bytes()), key)
        return json.loads(raw.decode())
    except Exception:
        return None

# ── Session helpers ───────────────────────────────────────────────────────────

def get_portfolio() -> dict:
    return st.session_state.get("portfolio", {
        "portfolio_value": 300000, "cash": 300000,
        "positions": [], "last_updated": str(date.today()),
    })

def save_sync(data: dict):
    data["last_updated"] = str(date.today())
    st.session_state["portfolio"] = data
    pwd = st.session_state.get("portfolio_password", "")
    if pwd:
        save_portfolio(data, pwd)

# ── Signal fetch ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_signals(tickers: tuple) -> pd.DataFrame:
    rows = []
    try:
        from features.builder import build_feature_dataframe
        from signals.generator import generate_signals
        for ticker in tickers:
            try:
                df = build_feature_dataframe(ticker, start_date="2024-01-01")
                price = float(df["close"].iloc[-1]) if "close" in df.columns else None
                for h in [1, 3, 5]:
                    sig = generate_signals(ticker, df, horizon=h)
                    rows.append({
                        "ticker": ticker, "horizon": h,
                        "signal": sig.today_signal,
                        "prob_up": round(sig.today_prob, 3),
                        "prob_eff": round(sig.today_prob_eff, 3),
                        "confidence": "HIGH" if sig.today_prob_eff >= 0.70
                                      else "MEDIUM" if sig.today_prob_eff >= 0.55
                                      else "LOW",
                        "price": price,
                    })
            except Exception:
                for h in [1, 3, 5]:
                    rows.append({"ticker": ticker, "horizon": h, "signal": "N/A",
                                 "prob_up": 0.5, "prob_eff": 0.5, "confidence": "LOW", "price": None})
    except Exception as e:
        st.warning(f"Signal fetch error: {e}")
    return pd.DataFrame(rows)

def sig_label(signal: str, prob: float) -> str:
    pct = f"{prob*100:.0f}%"
    if signal == "BUY":   return f"🟢 BUY {pct}"
    if signal == "SELL":  return f"🔴 SELL {pct}"
    if signal == "N/A":   return "⚪ N/A"
    arrow = "⬆️" if prob > 0.5 else "⬇️"
    return f"{arrow} HOLD {pct}"

def suggestion(ticker, signal, prob_eff, conf, port_val, price, owns):
    try:
        from signals.position_sizer import get_position_size
        pos = get_position_size(ticker=ticker, prob_eff=prob_eff, confidence=conf,
                                portfolio_value=port_val, current_price=price)
        if signal == "BUY" and conf in ("HIGH", "MEDIUM"):
            verb = "Add" if owns else "Buy"
            s = f" ~{pos.shares} shares" if pos.shares else ""
            return f"{verb} ${pos.dollars:,.0f}{s}"
        if owns and prob_eff < 0.30: return "Consider selling"
        if owns and prob_eff < 0.40: return "Consider trimming"
        return "Hold"
    except Exception:
        return "Hold"

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════

if "portfolio_unlocked" not in st.session_state:
    st.session_state["portfolio_unlocked"] = False

if not st.session_state["portfolio_unlocked"]:
    st.title("💼 Portfolio")
    st.caption("Private — encrypted on your Mac only. Never sent to GitHub or any server.")
    st.markdown("### Unlock")

    if PORTFOLIO_FILE.exists():
        pwd = st.text_input("Password", type="password")
        if st.button("Unlock"):
            data = load_portfolio(pwd)
            if data is not None:
                st.session_state["portfolio"] = data
                st.session_state["portfolio_unlocked"] = True
                st.session_state["portfolio_password"] = pwd
                st.rerun()
            else:
                st.error("Wrong password")
    else:
        st.info("No portfolio file found — create one below.")
        p1 = st.text_input("Set password", type="password")
        p2 = st.text_input("Confirm password", type="password")
        if st.button("Create portfolio"):
            if not p1:
                st.error("Password cannot be empty")
            elif p1 != p2:
                st.error("Passwords don't match")
            else:
                empty = {"portfolio_value": 300000, "cash": 300000,
                         "positions": [], "last_updated": str(date.today())}
                save_portfolio(empty, p1)
                st.session_state["portfolio"] = empty
                st.session_state["portfolio_unlocked"] = True
                st.session_state["portfolio_password"] = p1
                st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.title("💼 Portfolio")
st.caption("Private — encrypted on your Mac only. Never sent to GitHub or any server.")

portfolio  = get_portfolio()
positions  = portfolio.get("positions", [])
port_val   = float(portfolio.get("portfolio_value", 300000))
cash       = float(portfolio.get("cash", port_val))
invested   = port_val - cash

total_pnl  = sum(float(p.get("shares",0)) * (float(p.get("current_price", p.get("avg_cost",0))) - float(p.get("avg_cost",0))) for p in positions)
total_cost = sum(float(p.get("shares",0)) * float(p.get("avg_cost",0)) for p in positions)

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio value",  f"${port_val:,.0f}")
c2.metric("Cash on hand",     f"${cash:,.0f}", f"{cash/port_val*100:.1f}%")
c3.metric("Invested",         f"${invested:,.0f}", f"{len(positions)} positions")
c4.metric("Unrealised P&L",   f"${total_pnl:+,.0f}",
          f"{total_pnl/total_cost*100:+.2f}%" if total_cost > 0 else "—")
c5.metric("Last updated",     portfolio.get("last_updated", "—"))

st.divider()

# ── Settings ──────────────────────────────────────────────────────────────────
with st.expander("⚙️ Security settings"):
    st.markdown("**Change password**")
    cc1, cc2, cc3 = st.columns(3)
    op = cc1.text_input("Current password", type="password", key="op")
    np1 = cc2.text_input("New password", type="password", key="np1")
    np2 = cc3.text_input("Confirm new", type="password", key="np2")
    if st.button("Change password"):
        if load_portfolio(op) is not None:
            if np1 == np2 and np1:
                save_portfolio(portfolio, np1)
                st.session_state["portfolio_password"] = np1
                st.success("Password changed")
            else:
                st.error("New passwords don't match")
        else:
            st.error("Wrong current password")

    st.divider()
    if st.button("🔒 Lock portfolio", type="secondary"):
        st.session_state["portfolio_unlocked"] = False
        st.session_state.pop("portfolio_password", None)
        st.rerun()

st.divider()

# ── Quick update portfolio value & cash ──────────────────────────────────────
with st.form("quick_update"):
    st.markdown("### Portfolio overview")
    qu1, qu2 = st.columns(2)
    qu_val  = qu1.number_input("Total portfolio value ($)", min_value=10000,
                                max_value=10_000_000, value=int(port_val), step=1000)
    qu_cash = qu2.number_input("Cash on hand ($)", min_value=0,
                                max_value=int(qu_val), value=int(cash), step=1000)
    if st.form_submit_button("Update"):
        portfolio["portfolio_value"] = qu_val
        portfolio["cash"]            = qu_cash
        save_sync(portfolio)
        st.success("Updated")
        st.rerun()

st.divider()

# ── Add position ──────────────────────────────────────────────────────────────
st.markdown("### Add / update position")
with st.form("add_pos"):
    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([2, 1, 1, 2, 2, 2])
    new_t        = fc1.text_input("Ticker", placeholder="AAPL").upper().strip()
    new_s        = fc2.number_input("Shares", min_value=0.0, step=1.0)
    new_c        = fc3.number_input("Avg cost ($)", min_value=0.0, step=0.01)
    new_acct     = fc4.selectbox("Account", ["Traditional", "Roth IRA", "Other"])
    new_platform = fc5.selectbox("Platform", ["Fidelity", "Robinhood", "Other"])
    new_n        = fc6.text_input("Note", placeholder="optional")
    if st.form_submit_button("Add position") and new_t and new_s > 0 and new_c > 0:
        existing = next((p for p in positions if p["ticker"] == new_t
                         and p.get("account") == new_acct
                         and p.get("platform") == new_platform), None)
        if existing:
            os_ = float(existing["shares"]); oc = float(existing["avg_cost"])
            ts  = os_ + new_s
            existing["shares"]   = ts
            existing["avg_cost"] = round((os_*oc + new_s*new_c) / ts, 4)
            if new_n: existing["note"] = new_n
            st.success(f"Updated {new_t} ({new_acct} / {new_platform})")
        else:
            positions.append({"ticker": new_t, "shares": new_s, "avg_cost": new_c,
                               "account": new_acct, "platform": new_platform,
                               "added": str(date.today()), "note": new_n,
                               "current_price": new_c})
            st.success(f"Added {new_t} ({new_acct} / {new_platform})")
        portfolio["positions"] = positions
        save_sync(portfolio)
        st.rerun()

st.divider()

# ── Current positions ─────────────────────────────────────────────────────────
if not positions:
    st.info("No positions yet — add one above.")
else:
    st.markdown("### Current positions — signals & suggestions")
    owned = [p["ticker"] for p in positions]

    with st.spinner("Loading signals..."):
        sigs = fetch_signals(tuple(owned))

    rows = []
    for p in positions:
        t     = p["ticker"]
        sh    = float(p.get("shares", 0))
        ac    = float(p.get("avg_cost", 0))
        ts    = sigs[sigs["ticker"] == t] if not sigs.empty else pd.DataFrame()

        cp = ac
        if not ts.empty and ts.iloc[0]["price"]:
            cp = float(ts.iloc[0]["price"])
            p["current_price"] = cp

        def gs(h):
            if ts.empty: return "N/A", 0.5, 0.5, "LOW"
            r = ts[ts["horizon"] == h]
            if r.empty: return "N/A", 0.5, 0.5, "LOW"
            row = r.iloc[0]
            return row["signal"], float(row["prob_up"]), float(row["prob_eff"]), row["confidence"]

        s1,p1,pe1,c1 = gs(1); s3,p3,pe3,c3 = gs(3); s5,p5,pe5,c5 = gs(5)
        bc = c1 if c1=="HIGH" else (c3 if c3=="HIGH" else c5)
        bs = s1 if c1=="HIGH" else (s3 if c3=="HIGH" else s5)
        bp = pe1 if c1=="HIGH" else (pe3 if c3=="HIGH" else pe5)

        pnl = sh * (cp - ac)
        pnl_pct = pnl / (sh*ac) * 100 if sh*ac > 0 else 0

        rows.append({
            "Ticker":   t,
            "Account":  p.get("account", "—"),
            "Platform": p.get("platform", "—"),
            "Shares":   sh,
            "Avg $":    f"${ac:.2f}",
            "Price":    f"${cp:.2f}",
            "Value":    f"${sh*cp:,.0f}",
            "P&L":      f"${pnl:+,.0f} ({pnl_pct:+.1f}%)",
            "1d":       sig_label(s1, p1),
            "3d":       sig_label(s3, p3),
            "5d":       sig_label(s5, p5),
            "Conf":     bc,
            "Action":   suggestion(t, bs, bp, bc, port_val, cp, True),
            "Note":     p.get("note",""),
        })

    portfolio["positions"] = positions
    save_sync(portfolio)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    rm = st.selectbox("Remove position", [""] + owned)
    if rm and st.button(f"Remove {rm}", type="secondary"):
        portfolio["positions"] = [p for p in positions if p["ticker"] != rm]
        save_sync(portfolio)
        st.success(f"Removed {rm}")
        st.rerun()

    st.divider()

    # ── New opportunities ─────────────────────────────────────────────────────
    st.markdown("### New opportunities — BUY signals not in your portfolio")
    try:
        all_t   = [t.strip() for t in Path("tickers.txt").read_text().splitlines() if t.strip()]
        new_t   = [t for t in all_t if t not in owned]
        with st.spinner(f"Scanning {len(new_t)} tickers..."):
            ns = fetch_signals(tuple(new_t))
        if not ns.empty:
            buys = ns[(ns["signal"]=="BUY") & (ns["confidence"].isin(["HIGH","MEDIUM"]))]
            if not buys.empty:
                opp = []
                for tk in buys["ticker"].unique():
                    ts2 = buys[buys["ticker"]==tk]
                    best = ts2.sort_values("prob_eff", ascending=False).iloc[0]
                    def gs2(h):
                        r = ts2[ts2["horizon"]==h]
                        return (r.iloc[0]["signal"], r.iloc[0]["prob_up"]) if not r.empty else ("N/A", 0.5)
                    sg1,pp1=gs2(1); sg3,pp3=gs2(3); sg5,pp5=gs2(5)
                    opp.append({
                        "Ticker":     tk,
                        "Price":      f"${best['price']:.2f}" if best["price"] else "N/A",
                        "1d":         sig_label(sg1,pp1),
                        "3d":         sig_label(sg3,pp3),
                        "5d":         sig_label(sg5,pp5),
                        "Confidence": best["confidence"],
                        "Entry size": suggestion(tk, best["signal"], best["prob_eff"],
                                                  best["confidence"], port_val, best["price"], False),
                    })
                st.dataframe(pd.DataFrame(opp), use_container_width=True, hide_index=True)
            else:
                st.info("No new HIGH/MEDIUM BUY signals outside your current positions.")
    except Exception as e:
        st.warning(f"Could not load opportunities: {e}")

st.divider()
st.caption(f"Encrypted at `{PORTFOLIO_FILE.absolute()}` | Never in git | {now_et().strftime('%Y-%m-%d %H:%M ET')}")
