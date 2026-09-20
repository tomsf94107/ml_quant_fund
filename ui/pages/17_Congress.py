# ui/pages/17_Congress.py
# Congressional and executive-branch disclosures — what was filed, by whom, and
# whether any of it beat SPY. Tested 2026-09-18 across eight angles: eight
# nulls. The page leads with that rather than ranking members by a statistic
# that does not discriminate.
#
# GRAIN. congress_returns is keyed per (ticker, filed_date, member, direction).
# congress_trades keeps `amounts` in its primary key, so one filing split across
# line items is several rows there and ONE in returns. A naive join fans out
# ~1.4x and UNEVENLY -- members who file many line items count many times. The
# first party split read +2.34pp vs -0.30pp fanned and +0.74pp vs -0.28pp at the
# return grain. Every load here goes through _returns_grain(), which applies
# DISTINCT on the join key. Do not write a second join.

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

DB = Path(os.getenv("CONGRESS_DB_PATH", Path(_ROOT) / "congress_trades.db"))

st.set_page_config(page_title="Congress", page_icon="🏛", layout="wide")
st.title("🏛 Congressional trading")
st.caption(
    "27,223 disclosures, 306 members. Returns start 2016-07-18 where SPY "
    "history begins; 398 earlier filings have no benchmark. Eight angles tested "
    "against SPY over identical bars — eight nulls. This is a research surface "
    "and an exposure map, not a source of tickers to buy."
)


# ---------------------------------------------------------------- data layer

@st.cache_data(ttl=900)
def load_trades() -> pd.DataFrame:
    """Every disclosure with roster context. One row per LINE ITEM.

    Correct for counts, dollar volume and trade-size analysis, where each line
    carries its own amount. NOT for returns -- see _returns_grain below.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    df = pd.read_sql("SELECT * FROM congress_360", con)
    con.close()
    df["filed_at_date"] = pd.to_datetime(df["filed_at_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    return df


@st.cache_data(ttl=900)
def load_returns(horizon: int) -> pd.DataFrame:
    """One row per (ticker, filed_date, member, direction) with roster context.

    THE DISTINCT IS LOAD-BEARING. Without it a member who splits one filing
    across forty line items contributes forty observations to a mean that
    should see one.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    df = pd.read_sql("""
        WITH d AS (SELECT DISTINCT ticker, filed_at_date, name, txn_type,
                          member_type
                   FROM congress_trades)
        SELECT r.ticker, r.filed_at_date, r.transaction_date, r.name,
               r.txn_type, r.ret_filed, r.spy_filed, r.excess_filed,
               r.excess_txn, r.matured,
               d.member_type, m.party, m.state, m.chamber, m.is_leadership,
               m.leadership,
               -- Strip the THOMAS codes. congress_members.committees holds a
               -- mix of ids and names; a reader wants "House Committee on
               -- Armed Services", not "HSAS28, HSAS35".
               m.committees, m.sectors AS oversees
        FROM congress_returns r
        JOIN d ON d.ticker = r.ticker AND d.filed_at_date = r.filed_at_date
              AND d.name = r.name AND d.txn_type = r.txn_type
        LEFT JOIN congress_members m ON m.match_key = LOWER(
            TRIM(SUBSTR(d.name, 1, INSTR(d.name || ' ', ' ') - 1)) || ' ' ||
            TRIM(REPLACE(d.name, RTRIM(d.name, REPLACE(d.name, ' ', '')), '')))
        WHERE r.horizon = ?
    """, con, params=(horizon,))
    con.close()
    df["filed_at_date"] = pd.to_datetime(df["filed_at_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["lag_days"] = (df["filed_at_date"] - df["transaction_date"]).dt.days
    df["committees"] = df["committees"].fillna("").apply(
        lambda s: ", ".join(p.strip() for p in s.split(",")
                            if p.strip() and not p.strip()[:2].isupper()
                            or " " in p.strip()) or None)
    df["branch"] = df["member_type"].apply(
        lambda x: "executive" if x == "executive" else "legislative")
    return df


def ci95(s: pd.Series):
    """(mean, half-width) in percentage points. Normal approximation.

    Shown on every estimate because the point estimate alone is what turns
    '+4.1pp on 63 trades' into a headline. That row's interval is +/-7.9.
    """
    s = s.dropna()
    n = len(s)
    if n < 2:
        return (float("nan"), float("nan"))
    return (s.mean() * 100, 1.96 * s.std(ddof=1) / math.sqrt(n) * 100)


def fmt_ci(mean, half):
    if mean != mean:
        return "—"
    return f"{mean:+.2f} ±{half:.2f}"


# ------------------------------------------------------------- measurement

st.sidebar.header("How it is measured")
st.sidebar.caption("Changes what the numbers mean, not which rows are shown.")
horizon = st.sidebar.selectbox(
    "Horizon (trading days)", [20, 60, 120, 252], index=1,
    help="252 is the window the published studies use, so the numbers here "
         "stay comparable to them. 20-120 are the tradeable ones.")
ret_basis = st.sidebar.radio(
    "Return", ["Excess vs SPY", "Raw", "Both"], index=0,
    help="Raw is uninterpretable on its own. In a rising market almost "
         "everything goes up: +4.16% is excellent against a flat tape and "
         "nothing against SPY's +4.18%.")
min_n = st.sidebar.slider(
    "Min trades to rank", 5, 500, 30, 5,
    help="30 is the conventional floor for a mean to be testable. Members "
         "below it are shown, greyed, never ranked.")

trades = load_trades()
rets = load_returns(horizon)

# ------------------------------------------------------------------ alert

latest = trades["filed_at_date"].max()
today_rows = trades[trades["filed_at_date"] == latest]
with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**Filed {latest:%Y-%m-%d} — {len(today_rows)} new**")
    c2.caption("ingest 06:00 VN daily")
    if len(today_rows):
        chips = []
        for _, r in today_rows.head(8).iterrows():
            lag = (r["filed_at_date"] - r["transaction_date"]).days
            late = " · late" if lag and lag > 45 else ""
            chips.append(f"`{r['ticker'] or '—'} · {str(r['txn_type']).lower()} "
                         f"· {str(r['name']).split()[-1]} · {lag}d{late}`")
        st.markdown(" ".join(chips))

# ----------------------------------------------------------------- filters

st.subheader("Filters")
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Who filed**")
    f_branch = st.multiselect("Branch", sorted(rets["branch"].dropna().unique()))
    f_cham = st.multiselect("Chamber", sorted(rets["member_type"].dropna().unique()))
    f_party = st.multiselect("Party", sorted(rets["party"].dropna().unique()))
    f_lead = st.selectbox("Leadership", ["All", "Leadership only", "Rank and file"])
    f_member = st.multiselect("Member", sorted(rets["name"].dropna().unique()))
with f2:
    st.markdown("**What was traded**")
    f_dir = st.multiselect("Direction", sorted(rets["txn_type"].dropna().unique()))
    f_tick = st.multiselect("Ticker", sorted(rets["ticker"].dropna().unique()))
    f_state = st.multiselect("State", sorted(rets["state"].dropna().unique()))
with f3:
    st.markdown("**When**")
    lo = rets["filed_at_date"].min().date()
    hi = rets["filed_at_date"].max().date()
    f_range = st.date_input("Filed between", (lo, hi), min_value=lo, max_value=hi)
    f_lag = st.slider("Filing lag (days)", 0, 400, (0, 400),
                      help="Statutory limit is 45. Measured mean is 45.5 and "
                           "14.6% are filed past it.")
    f_mat = st.selectbox("Maturity", ["Matured only", "All", "Still open"])

d = rets.copy()
if f_branch: d = d[d["branch"].isin(f_branch)]
if f_cham:   d = d[d["member_type"].isin(f_cham)]
if f_party:  d = d[d["party"].isin(f_party)]
if f_member: d = d[d["name"].isin(f_member)]
if f_dir:    d = d[d["txn_type"].isin(f_dir)]
if f_tick:   d = d[d["ticker"].isin(f_tick)]
if f_state:  d = d[d["state"].isin(f_state)]
if f_lead == "Leadership only":  d = d[d["is_leadership"] == 1]
elif f_lead == "Rank and file":  d = d[d["is_leadership"] == 0]
if f_mat == "Matured only": d = d[d["matured"] == 1]
elif f_mat == "Still open":  d = d[d["matured"] == 0]
if isinstance(f_range, tuple) and len(f_range) == 2:
    d = d[(d["filed_at_date"].dt.date >= f_range[0]) &
          (d["filed_at_date"].dt.date <= f_range[1])]
d = d[(d["lag_days"].fillna(0) >= f_lag[0]) & (d["lag_days"].fillna(0) <= f_lag[1])]

st.caption(f"{len(rets):,} → **{len(d):,}** rows at the return grain "
           f"(one per ticker, filing date, member and direction)")

# ---------------------------------------------------------------- headline

buys = d[(d["txn_type"] == "Buy") & (d["matured"] == 1)]
m, h = ci95(buys["excess_filed"])
raw, _ = ci95(buys["ret_filed"])
spy, _ = ci95(buys["spy_filed"])

with st.container(border=True):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Excess vs SPY, buys, h={horizon}", fmt_ci(m, h))
    k2.metric("Raw", f"{raw:+.2f}%" if raw == raw else "—")
    k3.metric("SPY, same bars", f"{spy:+.2f}%" if spy == spy else "—")
    k4.metric("Matured buys", f"{len(buys):,}")
    if m == m and abs(m) < h:
        st.caption("The interval straddles zero. This is a null, not a small "
                   "positive. The raw figure is the market — it is what "
                   "circulates as “Congress returns 4% a trade”.")

# ---------------------------------------------- subsets, bars pre-registered

st.subheader("Subsets")
st.caption("Bars registered before the numbers were seen: excess > 1pp, "
           "n ≥ 500, positive in ≥ 7 of 10 years. Slicing a null and keeping "
           "the positive slice is p-hacking, so the bar is fixed in code.")

def subset_rows(frame):
    out = []
    def add(label, sub):
        mm, hh = ci95(sub["excess_filed"])
        yrs = sub.groupby(sub["filed_at_date"].dt.year)["excess_filed"].mean()
        pos = int((yrs > 0).sum())
        passes = (mm == mm and mm > 1.0 and len(sub) >= 500
                  and len(yrs) and pos / len(yrs) >= 0.7)
        out.append({"Subset": label, "n": len(sub), "Excess ±95% CI": fmt_ci(mm, hh),
                    "Years +": f"{pos}/{len(yrs)}",
                    "Bar": "pass" if passes
                           else ("n low" if len(sub) < 500 else "fail")})
    b = frame[(frame["txn_type"] == "Buy") & (frame["matured"] == 1)]
    add("All buys", b)
    add("Leadership", b[b["is_leadership"] == 1])
    add("Rank and file", b[b["is_leadership"] == 0])
    for p in ("Democrat", "Republican"):
        add(f"Party — {p}", b[b["party"] == p])
        add(f"Party — {p}, ex-2020", b[(b["party"] == p) &
                                       (b["filed_at_date"].dt.year != 2020)])
    add("Senate", b[b["member_type"] == "senate"])
    add("House", b[b["member_type"] == "house"])
    add("Executive branch", b[b["branch"] == "executive"])
    add("Sells", frame[(frame["txn_type"] == "Sell") & (frame["matured"] == 1)])
    return pd.DataFrame(out)

st.dataframe(subset_rows(d), use_container_width=True, hide_index=True)
st.caption("2020 is the confound in every party comparison: Republican buys "
           "read +14.89pp that year on 460 rows — the COVID crash and recovery "
           "— and carry roughly two thirds of the raw gap. The ex-2020 rows are "
           "there so it cannot hide.")

# ------------------------------------------------------------ member table

st.subheader("Members")
b = d[(d["txn_type"] == "Buy") & (d["matured"] == 1)]
if len(b):
    g = b.groupby("name").agg(
        n=("excess_filed", "size"),
        excess=("excess_filed", "mean"),
        sd=("excess_filed", "std"),
        raw=("ret_filed", "mean"),
    ).reset_index()
    g["half"] = 1.96 * g["sd"] / g["n"].pow(0.5) * 100
    g["Excess ±95% CI"] = [fmt_ci(e * 100, hh) for e, hh in zip(g["excess"], g["half"])]
    g["Raw %"] = (g["raw"] * 100).round(2)
    meta = d.groupby("name")[["party", "member_type", "committees"]].first()
    g = g.join(meta, on="name")
    g["Ranked"] = g["n"] >= min_n
    # Default sort is COUNT, not excess. Eight angles came back null, so
    # ranking members by a statistic that does not discriminate is the exact
    # misreading this page exists to prevent. Excess stays sortable.
    g = g.sort_values("n", ascending=False)
    st.dataframe(
        g[["name", "party", "member_type", "committees", "n", "Raw %",
           "Excess ±95% CI", "Ranked"]]
        .rename(columns={"name": "Member", "party": "Party",
                         "member_type": "Chamber", "committees": "Oversees"}),
        use_container_width=True, hide_index=True)
    st.caption(f"Sorted by trade count, not excess. {int((~g['Ranked']).sum())} "
               f"members fall below the {min_n}-trade minimum — shown, never "
               f"ranked. A member with n=63 and ±7.9 is not a top performer.")

# ------------------------------------------------- pending, tickers, clusters

c1, c2 = st.columns(2)
with c1:
    st.subheader("Filed recently — cannot be scored yet")
    st.caption(f"A {horizon}-bar window has not closed. These are the only "
               f"forward-looking rows on the page.")
    # Built from `rets`, not `d`: the Maturity filter defaults to "Matured
    # only", which removes every row this panel exists to show. It still
    # honours the other filters by reusing their masks on the full frame.
    p = rets[rets["matured"] == 0].copy()
    if f_branch: p = p[p["branch"].isin(f_branch)]
    if f_party:  p = p[p["party"].isin(f_party)]
    if f_member: p = p[p["name"].isin(f_member)]
    if f_tick:   p = p[p["ticker"].isin(f_tick)]
    p = p.sort_values("filed_at_date", ascending=False)
    _p = p[["filed_at_date", "ticker", "name", "txn_type", "lag_days"]].head(25).copy()
    _p["filed_at_date"] = _p["filed_at_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(_p, use_container_width=True, hide_index=True,
                 column_config={"filed_at_date": "Filed", "ticker": "Ticker",
                                "name": "Member", "txn_type": "Dir",
                                "lag_days": "Lag"})
with c2:
    st.subheader("Most traded names")
    t = (d[d["matured"] == 1].groupby("ticker")
         .agg(n=("excess_filed", "size"),
              members=("name", "nunique"),
              excess=("excess_filed", "mean")).reset_index())
    t = t[t["n"] >= 20].sort_values("n", ascending=False)
    t["excess"] = (t["excess"] * 100).round(2)
    st.dataframe(t.head(25), use_container_width=True, hide_index=True)

# ------------------------------------------------- every transaction, raw

st.subheader("All transactions")
st.caption(
    "One row per disclosed line item — the level the filing is actually made "
    "at. A filing split across several amount bands appears as several rows. "
    "Click any column header to sort; use the search box for a ticker or name."
)

tx = load_trades()

# Same filters as above, applied to the line-item frame. Kept explicit rather
# than factored into a helper: the two frames have different columns and a
# shared helper would silently drop a filter when one of them lacks the field.
if f_branch: tx = tx[tx["branch"].isin(f_branch)]
if f_cham:   tx = tx[tx["chamber"].isin(f_cham)]
if f_party:  tx = tx[tx["party"].isin(f_party)]
if f_member: tx = tx[tx["name"].isin(f_member)]
if f_dir:    tx = tx[tx["txn_type"].isin(f_dir)]
if f_tick:   tx = tx[tx["ticker"].isin(f_tick)]
if f_state:  tx = tx[tx["state"].isin(f_state)]
if f_lead == "Leadership only":  tx = tx[tx["is_leadership"] == 1]
elif f_lead == "Rank and file":  tx = tx[tx["is_leadership"] == 0]
if isinstance(f_range, tuple) and len(f_range) == 2:
    tx = tx[(tx["filed_at_date"].dt.date >= f_range[0]) &
            (tx["filed_at_date"].dt.date <= f_range[1])]
tx = tx[(tx["lag_days"].fillna(0) >= f_lag[0]) &
        (tx["lag_days"].fillna(0) <= f_lag[1])]

# Attach the forward return. Repeats across line items of one filing by
# design -- one ticker, one filing date, one forward return.
_r = rets[["ticker", "filed_at_date", "name", "txn_type",
           "excess_filed", "ret_filed", "spy_filed", "matured"]]
tx = tx.merge(_r, on=["ticker", "filed_at_date", "name", "txn_type"], how="left")

q = st.text_input("Search ticker, member, issuer or committee", "")
if q:
    ql = q.strip().lower()
    mask = False
    for col in ("ticker", "name", "issuer", "committees", "state", "party"):
        if col in tx:
            mask = mask | tx[col].fillna("").astype(str).str.lower().str.contains(ql)
    tx = tx[mask]

show = tx.sort_values("filed_at_date", ascending=False).copy()
show["Filed"] = show["filed_at_date"].dt.strftime("%Y-%m-%d")
show["Traded"] = show["transaction_date"].dt.strftime("%Y-%m-%d")
show["Late"] = show["filed_late"].map({1: "yes", 0: ""})
show["Size $"] = show["amount_mid"].map(
    lambda v: f"{v:,.0f}" if v == v and v is not None else "—")
for c, src in (("Excess %", "excess_filed"), ("Raw %", "ret_filed"),
               ("SPY %", "spy_filed")):
    show[c] = (show[src] * 100).round(2) if src in show else None
show["Scored"] = show["matured"].map({1.0: "yes", 0.0: "not yet"})

COLS = ["Filed", "Traded", "lag_days", "Late", "ticker", "issuer", "txn_type",
        "amounts", "Size $", "name", "party", "branch", "chamber", "state",
        "leadership", "committees", "reporter",
        "Excess %", "Raw %", "SPY %", "Scored"]
COLS = [c for c in COLS if c in show.columns]

st.dataframe(
    show[COLS].rename(columns={
        "lag_days": "Lag d", "ticker": "Ticker", "issuer": "Issuer",
        "txn_type": "Type", "amounts": "Range", "name": "Member",
        "party": "Party", "branch": "Branch", "chamber": "Chamber",
        "state": "State", "leadership": "Title", "committees": "Committees",
        "reporter": "Filed as"}),
    use_container_width=True, hide_index=True, height=520)

c_a, c_b = st.columns([3, 1])
c_a.caption(
    f"**{len(show):,}** transactions of {len(load_trades()):,}. "
    f"Excess repeats across line items of one filing — one ticker, one filing "
    f"date, one forward return. Do not average this column; the panels above "
    f"deduplicate to the return grain before computing any mean."
)
c_b.download_button(
    "Download CSV", show[COLS].to_csv(index=False).encode(),
    file_name=f"congress_transactions_{len(show)}.csv", mime="text/csv")

with st.expander("What this page is for, given eight nulls"):
    st.markdown("""
Nothing measured here supports buying what Congress buys. Tested at the return
grain against SPY over identical bars: all buys **−0.03pp**, leadership
**−0.04pp** and *below* rank-and-file, party gap **+0.74pp** with two thirds of
it in 2020, Senate positive in 4 of 7 years, trade size non-monotone with the
largest band worst, filing lag contradictory between adjacent bands, clusters
non-monotone. Members' own entry timing was **worse** than the filing-date entry
by 0.27pp, so the 45-day lag hides no edge.

It reproduces the post-2012 literature. Ziobrowski's pre-STOCK-Act findings
(Senate +12%/yr, House +55bps/month, 1985–2001) do not carry forward.

**What it is good for**

- **Exposure mapping** — which of the 422 traded names are politically wired.
  A risk question, not an alpha one.
- **Crowding** — retail follows these filings and NANC/KRUZ trade them. The
  flow is real where the alpha is not.
- **A feature path** — `congress_net_shares` exists in `features/builder.py`
  and is not in `FEATURE_COLUMNS`. A −0.03pp standalone can still carry
  conditional information, untested.

See `CLOSED_AXIS_congress_trading_2026-09-18.md`.
""")
