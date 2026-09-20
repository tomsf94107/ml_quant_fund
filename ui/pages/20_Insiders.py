# ui/pages/20_Insiders.py
# Form 4 -- the only ACTUAL transaction record among the three institutional
# sources, and the fastest.
#
# WHAT IT IS
#     Officers, directors and anyone holding more than 10% of a class must
#     report trades in their own company's stock within TWO BUSINESS DAYS. Each
#     row is a real transaction: date, shares, price paid, named person, title.
#     Compare that to 13F -- a quarter-end photograph filed 45 days later with
#     no transactions in it at all -- and to dark-pool prints, which are
#     same-day but anonymous.
#
#     Berkshire's Occidental purchases land here at $55.78 and $57.38, two days
#     after the trade, 43 days before the same purchase reaches a 13F.
#
# THE 2% THAT MEANS ANYTHING
#     383,355 rows by transaction_code:
#       S 124,678 sales (mostly scheduled 10b5-1) · A 106,390 grants ·
#       M 60,700 option exercises · F 59,412 withheld for tax · G 8,060 gifts ·
#       P 8,009 OPEN-MARKET PURCHASES · C 6,331 · J 5,860
#     Only P is someone choosing to spend their own money. A page pooling these
#     would show tax withholding as insider selling and option exercises as
#     accumulation. P is the default and changing it is an explicit choice.
#
# TESTED, AND IT DOES NOT PREDICT
#     7,816 clean P buys, 2019-2026, entry at the close AFTER filing_date,
#     benchmarked against SPY over identical bars:
#       all P        -0.10 / -0.64 / -2.76pp at h=20/60/120
#       C-suite      -1.32 / -2.49 / -4.11
#       cluster 2+   -2.07 / -2.83 / -6.92   t=-8.43, negative in BOTH halves
#       >=$250k      +0.67 / +2.24 / +1.01   sign flips across halves
#       10% owners      --  / +6.58 / +6.07  but +20.68 then -13.28
#       officers/dir    --  / -2.47 / -5.01  t=-8.54, consistent
#
#     The 9pp gap between 10% OWNERS and OFFICERS is the real finding -- the
#     first test pooled them and hid it. But the owner result lives entirely
#     before 2022 and fails the same-sign criterion.
#
#     The literature disagrees and the gap is explainable, not dismissable:
#     Jeng, Metrick & Zeckhauser measured what INSIDERS earn from the TRADE
#     date, not what a follower earns from the FILING date; Lakonishok & Lee
#     put the effect in SMALL caps, and this universe is large-cap AI and tech;
#     Cohen, Malloy & Pomorski separate opportunistic from routine filers with
#     a multi-year per-person classifier that does not exist here; and Rozeff &
#     Zaman show insiders are CONTRARIAN, so SPY does not control for the
#     drawdown these names were already in.
#
#     This page is research context. Nothing on it is evidence.

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
DB_INS, DB_PX = R / "insider_trades.db", R / "prices.db"
DB_INST, DB_DARK = R / "institutions.db", R / "institutional_trades.db"
META = R / "tickers_metadata.csv"

CODES = {"P": "P — open-market purchase", "S": "S — sale",
         "A": "A — grant", "M": "M — option exercise",
         "F": "F — withheld for tax", "G": "G — gift",
         "C": "C — conversion", "J": "J — other"}

st.set_page_config(page_title="Insiders", page_icon="👤", layout="wide")
st.title("👤 Insider transactions — Form 4")
st.caption(
    "Officers, directors and 10%+ owners, filed within two business days. Real "
    "transactions with dates and prices — the fastest and only transaction-level "
    "source here. Tested on this universe it does not predict returns."
)


def ro(p):
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=60)


def age(s):
    try:
        return (date.today() - datetime.strptime(str(s)[:10], "%Y-%m-%d").date()).days
    except Exception:
        return None


@st.cache_data(ttl=1800)
def load():
    """Filings joined to the day's close, your bucket map, and 13F holder count.

    PRICE SANITY IS NOT OPTIONAL. 103 of 8,009 P rows carry a price above
    10,000 and on 4 of them price EQUALS shares -- AIG shows 50,000,000 shares
    at "50,000,000 each" for a notional of 2.5e15. Every one sorts to the top
    of any dollar ranking. A further 931 sit more than 25% from that day's
    close, which no real fill does; ARES appears repeatedly at a flat 1000.00
    against a 29.10 close. Both are flagged rather than silently dropped, so
    the count is visible on the page.
    """
    c = ro(DB_INS)
    c.execute(f"ATTACH '{DB_PX}' AS p")
    df = pd.read_sql("""
        SELECT r.ticker, r.filing_date, r.trade_date, r.insider_name,
               r.insider_title, r.transaction_code, r.shares,
               r.price_per_share, r.acquired_disposed, r.is_csuite,
               r.role_weight, b.close
        FROM insider_filings_raw r
        LEFT JOIN p.raw_bars b
          ON b.ticker = r.ticker AND b.d = substr(r.trade_date, 1, 10)
        WHERE r.ticker IS NOT NULL AND r.trade_date >= '2019-01-01'
          AND r.filing_date >= r.trade_date
    """, c)
    c.close()
    if df.empty:
        return df

    df["lag"] = (pd.to_datetime(df.filing_date)
                 - pd.to_datetime(df.trade_date)).dt.days
    # NOTIONAL IS COMPUTED, NEVER READ. The stored notional_usd inherits the
    # corrupt price on those rows.
    df["usd"] = (df.shares * df.price_per_share).round(0)
    df["px_ok"] = df.price_per_share.between(0.01, 10000) & \
                  (df.price_per_share != df.shares)
    df["prem"] = ((df.price_per_share / df.close - 1) * 100).round(1)
    df["prem_ok"] = df.px_ok & df.close.notna() & df.prem.abs().le(25)
    # A P should always be "A" acquired and an S always "D". 99 rows
    # contradict -- 62 P marked D, 37 S marked A. Parse errors, flagged.
    df["ad_bad"] = ((df.transaction_code == "P") & (df.acquired_disposed == "D")) | \
                   ((df.transaction_code == "S") & (df.acquired_disposed == "A"))

    # FIRST-TIME BUYER. The nearest thing available to the opportunistic-vs-
    # routine split the literature says drives the effect: 1,452 of 8,009 P
    # buys are that person's first EVER purchase in that name.
    p = df[df.transaction_code == "P"]
    firsts = p.groupby(["insider_name", "ticker"])["trade_date"].transform("min")
    df["first_buy"] = False
    df.loc[p.index, "first_buy"] = (p.trade_date == firsts)

    if META.exists():
        m = pd.read_csv(META)
        df = df.merge(m[[c for c in ("ticker", "bucket", "tier")
                         if c in m.columns]], on="ticker", how="left")
    return df


@st.cache_data(ttl=1800)
def holders13f():
    if not DB_INST.exists():
        return pd.DataFrame(columns=["ticker", "holders_13f"])
    c = ro(DB_INST)
    try:
        q = c.execute("SELECT MAX(report_date) FROM inst_holdings").fetchone()[0]
        d = pd.read_sql("SELECT ticker, COUNT(DISTINCT name) holders_13f "
                        "FROM inst_holdings WHERE report_date=? "
                        "AND security_type='Share' GROUP BY 1", c, params=(q,))
    except Exception:
        d = pd.DataFrame(columns=["ticker", "holders_13f"])
    c.close()
    return d


@st.cache_data(ttl=1800)
def dark30():
    c = ro(DB_DARK)
    d = pd.read_sql("""
        SELECT ticker,
               SUM(CASE WHEN side='BUY' THEN notional_usd ELSE 0 END) b,
               SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END) s
        FROM institutional_trades WHERE side IN ('BUY','SELL')
          AND trade_date >= date((SELECT MAX(trade_date) FROM institutional_trades),
                                 '-30 days')
        GROUP BY 1""", c)
    c.close()
    t = d.b + d.s
    d["dp30"] = ((d.b - d.s) / t.where(t > 0) * 100).round(1)
    return d[["ticker", "dp30"]]


F = load()
if F.empty:
    st.error("insider_filings_raw is empty."); st.stop()
HD, DP = holders13f(), dark30()
P = F[F.transaction_code == "P"]

k = st.columns(6)
k[0].metric("Filings", f"{len(F):,}")
k[1].metric("Tickers", F.ticker.nunique())
k[2].metric("Newest", f"{age(F.filing_date.max())}d")
k[3].metric("P buys", f"{len(P):,}")
k[4].metric("Median lag", f"{int(P.lag.median())}d")
k[5].metric("First-time", f"{int(P.first_buy.sum()):,}")

st.caption(
    f"**{len(P):,} of {len(F):,} rows are a purchase.** The rest are sales "
    f"(mostly scheduled 10b5-1), grants, option exercises and tax withholding. "
    f"Pooling them would read tax withholding as insider selling."
)

# ───────────────────────────────────────────────────────── filters
st.subheader("Filters")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Who filed**")
    codes = st.multiselect("Code", list(CODES), default=["P"],
                           format_func=lambda x: CODES.get(x, x))
    f_type = st.selectbox("Filer type",
                          ["All", "10% Owner", "Director", "CEO / officer"])
    rw = sorted(F.role_weight.dropna().unique())
    f_rw = st.multiselect("role_weight", rw,
                          help="Already in the schema and used by nothing. "
                               "Level 1.0 has exactly 1,615 rows — the same "
                               "count as 10% Owner, so it encodes the filer "
                               "hierarchy.")
    f_who = st.multiselect("Insider", sorted(F.insider_name.dropna().unique())[:3000])
with c2:
    st.markdown("**What**")
    f_tick = st.multiselect("Ticker", sorted(F.ticker.unique()))
    f_bucket = st.multiselect("Sub-industry",
                              sorted(F.bucket.dropna().unique()) if "bucket" in F else [])
    f_tier = st.multiselect("Your tier",
                            sorted(F.tier.dropna().unique()) if "tier" in F else [])
    f_first = st.selectbox("First-time buy?", ["All", "First ever", "Repeat"])
    f_prem = st.selectbox("vs market close",
                          ["All", "Paid ABOVE", "Paid below"])
with c3:
    st.markdown("**When**")
    hi = F.trade_date.max()[:10]
    hi_d = datetime.strptime(hi, "%Y-%m-%d").date()
    rng = st.date_input("Trade date between",
                        (hi_d.replace(year=hi_d.year - 1), hi_d))
    min_usd = st.number_input("Min $ (computed, not stored)", 0, 50_000_000,
                              0, 25_000)
    max_lag = st.slider("Max filing lag (days)", 0, 60, 60)
    clean = st.checkbox("Clean prices only", value=True,
                        help="Excludes 103 rows with a price above 10,000 or "
                             "equal to the share count — AIG shows 50,000,000 "
                             "shares at 50,000,000 each.")

d = F[F.transaction_code.isin(codes)] if codes else F.copy()
if f_type == "10% Owner":
    d = d[d.insider_title.fillna("").str.contains("10%", case=False)]
elif f_type == "Director":
    d = d[d.insider_title.fillna("").str.contains("Director", case=False)]
elif f_type == "CEO / officer":
    d = d[d.is_csuite == 1]
if f_rw:     d = d[d.role_weight.isin(f_rw)]
if f_who:    d = d[d.insider_name.isin(f_who)]
if f_tick:   d = d[d.ticker.isin(f_tick)]
if f_bucket: d = d[d.bucket.isin(f_bucket)]
if f_tier:   d = d[d.tier.isin(f_tier)]
if f_first == "First ever": d = d[d.first_buy]
elif f_first == "Repeat":   d = d[~d.first_buy]
if f_prem == "Paid ABOVE":  d = d[d.prem_ok & (d.prem > 0)]
elif f_prem == "Paid below": d = d[d.prem_ok & (d.prem < 0)]
if clean:    d = d[d.px_ok]
if min_usd:  d = d[d.usd.fillna(0) >= min_usd]
d = d[d.lag <= max_lag]
if isinstance(rng, tuple) and len(rng) == 2:
    d = d[(d.trade_date >= str(rng[0])) & (d.trade_date <= str(rng[1]))]
st.caption(f"{len(F):,} → **{len(d):,}** filings")

# ─────────────────────────────────────── first-time / premium
g1, g2 = st.columns(2)
g1.markdown(f"**First-time buyers** — {int(P.first_buy.sum()):,} of {len(P):,}")
g1.caption("That person's first EVER purchase in that name")
ft = d[d.first_buy].sort_values("trade_date", ascending=False)
g1.dataframe(ft[["trade_date", "ticker", "insider_name", "insider_title", "usd"]]
             .head(12).rename(columns={"trade_date": "Traded", "ticker": "Tkr",
                                       "insider_name": "Insider",
                                       "insider_title": "Title", "usd": "$"}),
             use_container_width=True, hide_index=True, height=300)

n_ok = int(P.prem_ok.sum())
n_above = int((P.prem_ok & (P.prem > 0)).sum())
g2.markdown(f"**Paid ABOVE market** — {n_above:,} of {n_ok:,} "
            f"({100*n_above/max(n_ok,1):.0f}%)")
g2.caption("Price paid vs that day's close. Paying up is a different statement "
           "from buying a dip — no public tracker shows it.")
pr = d[d.prem_ok].sort_values("prem", ascending=False)
g2.dataframe(pr[["trade_date", "ticker", "insider_name", "price_per_share",
                 "close", "prem"]].head(12).rename(columns={
                     "trade_date": "Traded", "ticker": "Tkr",
                     "insider_name": "Insider", "price_per_share": "Paid",
                     "close": "Close", "prem": "Prem %"}),
             use_container_width=True, hide_index=True, height=300)
st.caption("Premium band capped at ±25% — 931 rows sit further out and are "
           "parse errors, e.g. ARES at a flat 1000.00 against a 29.10 close.")

# ─────────────────────────────────────── hierarchy / active
h1, h2 = st.columns(2)
h1.markdown("**Filer hierarchy**")
hier = (d.assign(kind=lambda x: x.insider_title.fillna("").apply(
            lambda t: "10% Owner" if "10%" in t else
                      "Director" if "Director" in t else
                      "Officer" if t else "unknown"))
        .groupby("kind").agg(n=("ticker", "size"),
                             tickers=("ticker", "nunique"),
                             people=("insider_name", "nunique"),
                             avg_usd=("usd", "mean")).reset_index())
h1.dataframe(hier, use_container_width=True, hide_index=True)
h1.caption("10% Owners are INSTITUTIONS filing at a two-day lag — the same "
           "information 13F shows 45 days later. Measured +6.58pp at h=60 but "
           "+20.68 before 2022 and −13.28 after, so it fails the halves test.")

h2.markdown("**Most active**")
act = (d.groupby(["insider_name", "insider_title"])
       .agg(n=("ticker", "size"), tickers=("ticker", "nunique"),
            usd=("usd", "sum")).reset_index())
h2.dataframe(act.nlargest(12, "n"), use_container_width=True,
             hide_index=True, height=300)
h2.caption("⚠ Needs entity dedup. STAHL MURRAY (1,114) and HORIZON KINETICS "
           "(466) are the same firm on the same single ticker — 1,580 filings, "
           "one position. `insider_holdings` carries an `insider_norm` column; "
           "this table does not.")

# ─────────────────────────────────────── clusters + three clocks
x1, x2 = st.columns(2)
cl = d.copy()
cl["mon"] = cl.filing_date.str[:7]
cg = (cl.groupby(["ticker", "mon"])
      .agg(people=("insider_name", "nunique"), usd=("usd", "sum")).reset_index())
cg = cg[cg.people >= 2].sort_values(["mon", "people"], ascending=[False, False])
x1.markdown("**Clusters — 2+ insiders, same month**")
x1.dataframe(cg.head(12), use_container_width=True, hide_index=True, height=300)
x1.caption("Measured −2.07pp at h=20 to −6.92pp at h=120, t=−8.43, negative in "
           "both halves. The cut most likely to work is the worst. Shown "
           "because it is interesting, not because it works.")

x2.markdown("**Three clocks on one name**")
tc = (d.groupby("ticker").agg(insiders=("insider_name", "nunique"),
                              usd=("usd", "sum")).reset_index()
      .merge(DP, on="ticker", how="left").merge(HD, on="ticker", how="left"))
x2.dataframe(tc.nlargest(12, "insiders")[
    ["ticker", "insiders", "dp30", "holders_13f"]].rename(columns={
        "insiders": "Insiders 2d", "dp30": "DP 1d %",
        "holders_13f": "13F 82d"}),
    use_container_width=True, hide_index=True, height=300)
x2.caption("Each column a different age: insider 2 days, dark pool 1 day, 13F "
           "up to 90. Agreement is interesting; none of it is tested.")

# ─────────────────────────────────────── data quality
st.subheader("Data quality")
q = st.columns(4)
q[0].metric("Contradictory A/D", int(F.ad_bad.sum()))
q[0].caption("62 P marked D, 37 S marked A")
q[1].metric("Suspect price", int((~F.px_ok & F.price_per_share.notna()).sum()))
q[1].caption("above 10,000 or equal to shares")
q[2].metric(">25% off close", int((F.px_ok & F.close.notna() &
                                   F.prem.abs().gt(25)).sum()))
q[2].caption("no real fill lands there")
q[3].metric("Tickers covered", F.ticker.nunique())
q[3].caption("of your 422-name book")

# ─────────────────────────────────────── table
st.subheader("All filings")
srch = st.text_input("Search ticker, insider, title or bucket", "")
t = d.copy()
if srch:
    s = srch.strip().lower()
    m = False
    for col in ("ticker", "insider_name", "insider_title", "bucket", "tier"):
        if col in t:
            m = m | t[col].fillna("").astype(str).str.lower().str.contains(s)
    t = t[m]
COLS = [c for c in ("filing_date", "trade_date", "lag", "ticker",
                    "insider_name", "insider_title", "transaction_code",
                    "shares", "price_per_share", "close", "prem", "usd",
                    "first_buy", "role_weight", "bucket", "tier") if c in t]
show = t[COLS].sort_values("filing_date", ascending=False)
st.dataframe(show.head(3000).rename(columns={
    "filing_date": "Filed", "trade_date": "Traded", "lag": "Lag",
    "ticker": "Tkr", "insider_name": "Insider", "insider_title": "Title",
    "transaction_code": "Cd", "shares": "Shares",
    "price_per_share": "Price", "close": "Close", "prem": "vs close %",
    "usd": "$", "first_buy": "1st?", "role_weight": "rw"}),
    use_container_width=True, hide_index=True, height=520)
b1, b2 = st.columns([3, 1])
b1.caption(f"{len(show):,} rows, first 3,000 shown. $ is computed as shares × "
           f"price — the stored notional column inherits the corrupt prices.")
b2.download_button("Download CSV", show.to_csv(index=False).encode(),
                   file_name=f"insiders_{len(show)}.csv", mime="text/csv")

with st.expander("Why this is context and not a signal"):
    st.markdown("""
**Measured 2026-09-20**, 7,816 clean P buys, 2019–2026, entry at the close
**after** `filing_date`, benchmarked against SPY over identical bars.

| cut | h=20 | h=60 | h=120 | halves |
|---|---|---|---|---|
| all P | −0.10 | −0.64 | **−2.76** | flips |
| C-suite | −1.32 | −2.49 | **−4.11** | flips |
| cluster 2+ | −2.07 | −2.83 | **−6.92** | consistent |
| ≥$250k | +0.67 | +2.24 | +1.01 | flips |
| 10% owners | — | **+6.58** | +6.07 | +20.7 → −13.3 |
| officers/directors | — | −2.47 | **−5.01** | consistent |

The **9pp gap between 10% owners and officers** is the finding — the first test
pooled them and hid it. But the owner result lives entirely before 2022.

**The literature disagrees and the gap is explainable.** Jeng, Metrick &
Zeckhauser measured what *insiders* earn from the *trade* date, not what a
follower earns from the *filing* date. Lakonishok & Lee put the effect in
**small caps**; this universe is large-cap AI and tech. Cohen, Malloy &
Pomorski separate *opportunistic* from *routine* filers with a multi-year
per-person classifier that does not exist here — `first_buy` on this page is
the closest approximation. And Rozeff & Zaman show insiders are **contrarian**,
buying what has already fallen, so SPY does not control for the drawdown.

A test meeting those conditions needs small-cap coverage this repo does not
have. Until then, nothing here is evidence.
""")
