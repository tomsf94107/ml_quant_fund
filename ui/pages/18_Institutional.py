# ui/pages/18_Institutional.py
# 13F holdings for concentrated managers, joined to daily dark-pool flow and
# insider Form 4, with every reading stamped by its own age.
#
# THREE CLOCKS, AND THEY MUST NOT BE CONFLATED
#     dark pool   daily, minutes of lag,  1,967 tickers, ANONYMOUS
#     insider     2 business days,          375 tickers
#     13F        45 days, quarterly,      3,587 held / 1,621 priced
#
# 13F is due Feb 17, May 15, Aug 14, Nov 16 -- 45 days after quarter end by
# rule -- and does not move in between, so a reading near the next quarter's
# close is 90 days stale. Dark-pool prints are same-day but ANONYMOUS:
# counterparties are never disclosed, so they cannot filter or attribute a 13F
# row. They appear as a strip and as a column, never as a row filter.
#
# WHY CONCENTRATED MANAGERS ONLY
#     The vendor caps holdings at 500 rows per page. Citadel exceeds 3,148
#     tickers over 12 pages and 68% of its rows are Options -- market makers
#     hold options as dealer INVENTORY, not as a view, and 13F reports LONG
#     options only, so a put line may hedge something unreported. The roster is
#     managers whose whole book fits under the cap: 127 of 500 probed. A
#     manager making 30 bets has a thesis to read; one holding 3,000 does not.
#
# WHAT IS TESTED
#     The dark-pool features (audit-validated 2026-05-17, n=458) and insider
#     rollups are already in FEATURE_COLUMNS -- prob_up contains them, so
#     agreement between them is not new information. The 13F legs are UNTESTED:
#     T1-T4 are pre-registered in PREREG_institutional_13f.md. Panels resting
#     on untested claims say so on screen.

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
DB_INS = R / "insider_trades.db"
META = R / "tickers_metadata.csv"

st.set_page_config(page_title="Institutional", page_icon="🏦", layout="wide")
st.title("🏦 Institutional")

MM = {"CITADEL ADVISORS LLC", "CTC LLC", "MILLENNIUM MANAGEMENT LLC",
      "SUSQUEHANNA INTERNATIONAL GROUP, LLP", "JANE STREET GROUP, LLC",
      "OPTIVER HOLDING B.V.", "IMC-CHICAGO, LLC", "MORGAN STANLEY"}


def ro(p):
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=60)


def age(s):
    try:
        return (date.today() - datetime.strptime(str(s)[:10], "%Y-%m-%d").date()).days
    except Exception:
        return None


@st.cache_data(ttl=1800)
def meta():
    """Your thematic map: ticker -> bucket, tier. 419 of 1,621 priced names.
    Granular where the book cares (Core Silicon, Neoclouds, Semiconductor
    Equipment), coarse elsewhere. Not GICS -- and better than GICS here, since
    it encodes WHY a name sits in a bucket. Anything outside it reads Unmapped
    rather than being forced into a taxonomy built for something else."""
    if not META.exists():
        return pd.DataFrame(columns=["ticker", "bucket", "tier"])
    m = pd.read_csv(META)
    return m[[c for c in ("ticker", "bucket", "tier") if c in m.columns]]


@st.cache_data(ttl=1800)
def holdings():
    if not DB_INST.exists():
        return pd.DataFrame()
    c = ro(DB_INST)
    cols = {r[1] for r in c.execute("PRAGMA table_info(inst_holdings)")}
    pc = "put_call" if "put_call" in cols else "NULL AS put_call"
    uc = "units_change" if "units_change" in cols else "NULL AS units_change"
    pt = "pct_total" if "pct_total" in cols else "NULL AS pct_total"
    df = pd.read_sql(f"""
        SELECT name, report_date, ticker, units, value, sector,
               security_type, {pc} AS put_call, {uc} AS units_change,
               {pt} AS pct_total
        FROM inst_holdings
    """, c)
    c.close()
    if df.empty:
        return df
    # % OF BOOK. Without it, every ranking is dominated by whoever is largest.
    # Citadel's AAPL put is $6.2B and 0.7% of its book; Thiel's VST is $59M and
    # 14.1% of his. The second is conviction, the first is inventory -- dollar
    # rank puts them the wrong way round.
    # pct_total IS THE VENDOR'S OWN FIELD and it is a FRACTION -- 0.1482
    # means 14.82% of the book. Scaled here rather than recomputed, because
    # the vendor denominates against the WHOLE book including options and
    # funds, while a local sum over Share rows would silently use a different
    # denominator and the two would disagree by manager.
    #
    # This column is why the page ranks the way it does. Citadel's 2026-06-30
    # AAPL book is a $9.0B call line and a $6.2B put line -- $15.3B, and
    # 0.17% of its book, which is dealer inventory. Thiel Macro's VST is $59M
    # and 14.12%, which is a conviction bet. Sorted by dollars the first
    # outranks the second by two orders of magnitude.
    if "pct_total" in df.columns and df["pct_total"].notna().any():
        df["pct_book"] = (df["pct_total"] * 100).round(2)
    else:
        sh = df.security_type == "Share"
        df.loc[sh, "_book"] = (df[sh].groupby(["name", "report_date"])["value"]
                               .transform("sum"))
        df["pct_book"] = (df["value"] / df["_book"] * 100).round(2)
    # 13F is FILED 45 days after quarter end. That is the tradeable date;
    # keying on quarter end embeds 45 days of look-ahead.
    df["filed"] = (pd.to_datetime(df["report_date"])
                   + pd.Timedelta(days=45)).dt.date.astype(str)
    return df.merge(meta(), on="ticker", how="left")


@st.cache_data(ttl=1800)
def dark(days):
    """Signed dark-pool notional: (buy - sell) / total. side is resolved by the
    ingest via Lee-Ready; UNKNOWN is 1.2% of 8.75M prints and is dropped, not
    split -- assigning it would invent direction."""
    c = ro(DB_DARK)
    df = pd.read_sql(f"""
        SELECT ticker, MAX(trade_date) last_trade,
               SUM(CASE WHEN side='BUY'  THEN notional_usd ELSE 0 END) b,
               SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END) s,
               COUNT(*) prints
        FROM institutional_trades
        WHERE side IN ('BUY','SELL')
          -- REGULAR PRINTS ONLY (2026-09-22). Average-price, prior-reference and
          -- contingent prints (52.5% of notional over 20 days) report a price
          -- not comparable to the NBBO at report time, and crosses/auctions have
          -- no aggressor -- Lee-Ready's side is noise for all of them. MU's
          -- +$6.1B 'net buying' was entirely such prints.
          AND (sale_cond_codes IS NULL OR sale_cond_codes IN ('', '[]'))
          AND COALESCE(is_cross, 0) = 0 AND COALESCE(is_closing_auction, 0) = 0
          AND trade_date >= date((SELECT MAX(trade_date) FROM institutional_trades),
                                 '-{days} days')
        GROUP BY ticker
    """, c)
    c.close()
    t = df.b + df.s
    df[f"dp{days}"] = ((df.b - df.s) / t.where(t > 0) * 100).round(1)
    return df[["ticker", "last_trade", f"dp{days}", "prints"]]


@st.cache_data(ttl=1800)
def insider(days=60):
    c = ro(DB_INS)
    df = pd.read_sql(f"""
        SELECT ticker, MAX(date) ins_last,
               SUM(net_shares * COALESCE(role_weight,1.0)) ins_net
        FROM insider_flows
        WHERE date >= date((SELECT MAX(date) FROM insider_flows), '-{days} days')
        GROUP BY ticker
    """, c)
    c.close()
    return df


H = holdings()
if H.empty:
    st.error("institutions.db has no holdings yet — the backfill is still "
             "running. Nothing here can render without it.")
    st.stop()

d5, d30, INS = dark(5), dark(30), insider()

# ────────────────────────────────────────────────────── daily strip
with st.container(border=True):
    a, b = st.columns([3, 1])
    a.markdown("**DAILY — separate clock, anonymous, cannot filter 13F rows**")
    b.caption(f"dark pool {age(d5.last_trade.max())}d · "
              f"insider {age(INS.ins_last.max()) if len(INS) else '—'}d")
    liq = d30[d30.prints >= 200][["ticker"]]
    j5 = d5.merge(liq, on="ticker")
    k = st.columns(4)
    k[0].caption("DP buying 5d")
    k[0].write(" · ".join(f"{r.ticker} {r.dp5:+.0f}"
                          for r in j5.nlargest(3, "dp5").itertuples()))
    k[1].caption("DP selling 5d")
    k[1].write(" · ".join(f"{r.ticker} {r.dp5:+.0f}"
                          for r in j5.nsmallest(3, "dp5").itertuples()))
    k[2].caption("Insider net buyers 60d")
    k[2].write(f"{int((INS.ins_net > 0).sum())} tickers" if len(INS) else "—")
    k[3].caption("13F latest quarter")
    k[3].write(f"{H.report_date.max()} · filed {H.filed.max()}")

# ────────────────────────────────────────────────────────── filters
st.subheader("Filters")
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Who filed**")
    f_mgr = st.multiselect("Manager", sorted(H.name.unique()))
    ex_mm = st.checkbox(
        "Exclude market makers", value=True,
        help="Dealer inventory is not a view. 13F reports LONG options only, "
             "so a put line may hedge something unreported.")
with f2:
    st.markdown("**What was traded**")
    f_sec = st.multiselect("Security", sorted(H.security_type.dropna().unique()))
    f_pc = st.multiselect("Call / put", ["call", "put"])
    f_sector = st.multiselect("Sector", sorted(H.sector.dropna().unique()))
    f_bucket = st.multiselect(
        "Sub-industry (your map)",
        sorted(H.bucket.dropna().unique()) if "bucket" in H else [])
    f_tier = st.multiselect(
        "Your tier", sorted(H.tier.dropna().unique()) if "tier" in H else [])
    f_tick = st.multiselect("Ticker", sorted(H.ticker.dropna().unique()))
with f3:
    st.markdown("**When**")
    qs = sorted(H.report_date.unique())
    f_q = st.select_slider("Quarter range", options=qs,
                           value=(qs[max(0, len(qs) - 8)], qs[-1]))
    min_pct = st.slider(
        "Min % of book", 0.0, 25.0, 0.0, 0.5,
        help="The conviction filter. Dollar rank puts dealer inventory above "
             "a 14% position.")
    rank_by = st.selectbox("Rank by", ["manager count", "$ value"])

d = H.copy()
if ex_mm:    d = d[~d.name.isin(MM)]
if f_mgr:    d = d[d.name.isin(f_mgr)]
if f_sec:    d = d[d.security_type.isin(f_sec)]
if f_pc:     d = d[d.put_call.isin(f_pc)]
if f_sector: d = d[d.sector.isin(f_sector)]
if f_bucket: d = d[d.bucket.isin(f_bucket)]
if f_tier:   d = d[d.tier.isin(f_tier)]
if f_tick:   d = d[d.ticker.isin(f_tick)]
d = d[(d.report_date >= f_q[0]) & (d.report_date <= f_q[1])]
if min_pct > 0:
    d = d[d.pct_book.fillna(0) >= min_pct]

latest = d.report_date.max() if len(d) else None
cur = d[d.report_date == latest] if latest else d
st.caption(f"{len(H):,} → **{len(d):,}** rows · latest quarter {latest} · "
           f"filed {cur.filed.max() if len(cur) else '—'} "
           f"({age(cur.filed.max()) if len(cur) else '—'} days ago)")

# ───────────────────────────────────────────────────────── trending
st.subheader(f"Trending — {latest}")
sh = cur[cur.security_type == "Share"]
key = "mgrs" if rank_by == "manager count" else "val"
if "units_change" in sh and sh.units_change.notna().any():
    bought = (sh[sh.units_change > 0].groupby("ticker")
              .agg(mgrs=("name", "nunique"), val=("value", "sum")).reset_index())
    sold = (sh[sh.units_change < 0].groupby("ticker")
            .agg(mgrs=("name", "nunique"), val=("value", "sum")).reset_index())
else:
    bought = sold = pd.DataFrame(columns=["ticker", "mgrs", "val"])
    st.info("units_change is NULL on rows collected before the schema carried "
            "it — the buy/sell lanes fill on the next ingest pass.")

t1, t2 = st.columns(2)
t1.markdown("**Most bought**")
t1.dataframe(bought.nlargest(12, key) if len(bought) else bought,
             use_container_width=True, hide_index=True, height=300)
t2.markdown("**Most sold**")
t2.dataframe(sold.nlargest(12, key) if len(sold) else sold,
             use_container_width=True, hide_index=True, height=300)

o1, o2 = st.columns(2)
opt = cur[cur.security_type == "Option"]
for col, side, lab in ((o1, "call", "Most calls"), (o2, "put", "Most puts")):
    x = (opt[opt.put_call == side].groupby("ticker")["value"].sum()
         .nlargest(8).reset_index())
    col.markdown(f"**{lab}** — $ value")
    col.dataframe(x, use_container_width=True, hide_index=True, height=230)
st.caption("Index names dominating the put lane is hedging, not a bearish call "
           "on those ETFs — 13F reports LONG options only.")

# ───────────────────────────────────────────────────────── rotation
st.subheader("Rotation")
allq = sorted(d.report_date.unique())
prev = allq[-2] if len(allq) > 1 else None
r1, r2 = st.columns(2)
for col, k2, lab, note in (
        (r1, "sector", "Sector", f"{cur.ticker.nunique():,} held tickers"),
        (r2, "bucket", "Sub-industry",
         f"your map — {int(cur.bucket.notna().sum()):,} of {len(cur):,} rows mapped"
         if "bucket" in cur else "no map")):
    if k2 not in cur or cur[k2].isna().all():
        continue
    now = cur[cur.security_type == "Share"].groupby(k2)["value"].sum()
    if prev:
        was = (d[(d.report_date == prev) & (d.security_type == "Share")]
               .groupby(k2)["value"].sum())
        delta = (now - was.reindex(now.index).fillna(0)) / 1e9
    else:
        delta = now / 1e9
    col.markdown(f"**{lab} — net $B**")
    col.caption(note)
    col.bar_chart(delta.sort_values(ascending=False).head(12))

# ───────────────────────────────────────────────────────────── tier
if "tier" in cur and cur.tier.notna().any():
    st.subheader("By your conviction tier")
    tv = (cur[cur.security_type == "Share"].groupby("tier")
          .agg(names=("ticker", "nunique"),
               value_b=("value", lambda s: round(s.sum() / 1e9, 2)),
               holders=("name", "nunique")).reset_index())
    st.dataframe(tv.sort_values("value_b", ascending=False),
                 use_container_width=True, hide_index=True)
    st.caption("Institutions piling into names you rate **lotto** is a crowding "
               "warning on your riskiest positions, not confirmation of them.")

# ──────────────────────────────────────────── consensus and crowding
c1, c2 = st.columns(2)
if len(bought) and len(sold):
    cons = (bought.merge(sold, on="ticker", how="outer", suffixes=("_b", "_s"))
            .fillna(0))
    cons["net"] = cons.mgrs_b - cons.mgrs_s
    cons["read"] = [
        "consensus buy" if b >= 3 * max(s, 1) else
        "exodus" if s >= 3 * max(b, 1) else "fight"
        for b, s in zip(cons.mgrs_b, cons.mgrs_s)]
    c1.markdown("**Consensus vs fight**")
    c1.caption("Net flow hides a fight — 41 buying and 29 selling is not agreement")
    c1.dataframe(cons.reindex(cons.net.abs().sort_values(ascending=False).index)
                 [["ticker", "mgrs_b", "mgrs_s", "read"]].head(12),
                 use_container_width=True, hide_index=True, height=300)

crowd = (cur[cur.security_type == "Share"].groupby("ticker")
         .agg(holders=("name", "nunique")).reset_index()
         .merge(d30[["ticker", "dp30", "prints"]], on="ticker", how="left"))
crowd = crowd[crowd.prints.fillna(0) >= 200]
if len(crowd):
    hi = crowd.holders.quantile(0.8)
    crowd["read"] = [
        "distributing" if h >= hi and (f or 0) < -10 else
        "accumulating" if (f or 0) > 10 else ""
        for h, f in zip(crowd.holders, crowd.dp30)]
    c2.markdown("**Crowding × daily flow — T4, UNTESTED**")
    c2.caption("The only legitimate join of the two clocks. Pre-registered in "
               "PREREG_institutional_13f.md, not yet measured.")
    c2.dataframe(crowd[crowd.read != ""].nlargest(12, "holders")
                 [["ticker", "holders", "dp30", "read"]],
                 use_container_width=True, hide_index=True, height=300)

# ──────────────────────────────────────────────────────── transactions
# ------------------------------------------------------------------ dark-pool trending
# DARK-POOL FLOW -- TRENDING (2026-09-22).
@st.cache_data(ttl=1800)
def dark_flow(days):
    """Buy and sell notional per ticker over the window. dark() returns only the
    signed %, which cannot rank by size. Same side rule as dark(): the ingest's
    Lee-Ready call; UNKNOWN is dropped, not split."""
    c = ro(DB_DARK)
    df = pd.read_sql(f"""
        SELECT ticker,
               SUM(CASE WHEN side='BUY'  THEN notional_usd ELSE 0 END) buy_usd,
               SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END) sell_usd,
               COUNT(*) prints, MAX(trade_date) last_trade
        FROM institutional_trades
        WHERE side IN ('BUY','SELL')
          -- REGULAR PRINTS ONLY (2026-09-22). Average-price, prior-reference and
          -- contingent prints (52.5% of notional over 20 days) report a price
          -- not comparable to the NBBO at report time, and crosses/auctions have
          -- no aggressor -- Lee-Ready's side is noise for all of them. MU's
          -- +$6.1B 'net buying' was entirely such prints.
          AND (sale_cond_codes IS NULL OR sale_cond_codes IN ('', '[]'))
          AND COALESCE(is_cross, 0) = 0 AND COALESCE(is_closing_auction, 0) = 0
          AND trade_date >= date((SELECT MAX(trade_date) FROM institutional_trades),
                                 '-{int(days)} days')
        GROUP BY ticker
    """, c)
    c.close()
    return df


st.subheader("Dark-pool flow — trending")
st.caption("Side is the ingest's Lee-Ready call on the NBBO carried with each print; "
           "REGULAR prints only: average-price, prior-reference and contingent prints "
           "(~52% of notional) and crosses/auctions are excluded, since their reported "
           "price isn't comparable to the NBBO and their side is noise. UNKNOWN (~2%) "
           "is dropped, not split. Dated by trade. Context, not "
           "a signal: whether this flow predicts returns is T4, untestable until "
           "2026-11-14, and crowding came out opposite to the usual story (T2).")
_dpa, _dpb, _dpc = st.columns(3)
_dp_win = _dpa.radio("Window (days)", [5, 20, 60], index=1, horizontal=True, key="dp_trend_win")
_dp_by = _dpb.radio("Rank by", ["net $", "buy %"], horizontal=True, key="dp_trend_by")
_dp_min = _dpc.number_input("Min total notional, $M (ticker tables)", min_value=0.0,
                            value=5.0, step=1.0, key="dp_trend_min")
_dpF = dark_flow(_dp_win)
if _dpF.empty:
    st.info("No dark-pool prints in the window.")
else:
    _dpF["total_usd"] = _dpF["buy_usd"] + _dpF["sell_usd"]
    _dpF["net_usd"] = _dpF["buy_usd"] - _dpF["sell_usd"]
    _dpF["buy_pct"] = (_dpF["buy_usd"] / _dpF["total_usd"].where(_dpF["total_usd"] > 0) * 100).round(1)
    _dpH = holdings()
    if not _dpH.empty:
        _dps = _dpH[_dpH["sector"].notna() & (_dpH["sector"].astype(str).str.strip() != "")]
        _dp_sec = (_dps.groupby(["ticker", "sector"]).size().reset_index(name="n")
                       .sort_values("n", ascending=False).drop_duplicates("ticker")[["ticker", "sector"]])
    else:
        _dp_sec = pd.DataFrame(columns=["ticker", "sector"])
    _dpF = (_dpF.merge(_dp_sec, on="ticker", how="left")
                .merge(meta()[["ticker", "bucket"]], on="ticker", how="left"))
    _dp_show = ["ticker", "sector", "bucket", "net_$M", "buy_pct", "total_$M", "prints", "last_trade"]
    _dpT = _dpF[_dpF["total_usd"] >= _dp_min * 1e6].copy()
    _dpT["net_$M"] = (_dpT["net_usd"] / 1e6).round(1)
    _dpT["total_$M"] = (_dpT["total_usd"] / 1e6).round(1)
    _dp_key = "net_$M" if _dp_by == "net $" else "buy_pct"
    _dpx, _dpy = st.columns(2)
    _dpx.markdown(f"**Net buying** — {len(_dpT):,} tickers at or above ${_dp_min:g}M")
    _dpx.dataframe(_dpT.sort_values(_dp_key, ascending=False).head(15)[_dp_show],
                   hide_index=True, use_container_width=True)
    _dpy.markdown("**Net selling**")
    _dpy.dataframe(_dpT.sort_values(_dp_key, ascending=True).head(15)[_dp_show],
                   hide_index=True, use_container_width=True)

    def _dp_group(col):
        f = _dpF.dropna(subset=[col])
        g = f.groupby(col).agg(buy_usd=("buy_usd", "sum"), sell_usd=("sell_usd", "sum"),
                               tickers=("ticker", "nunique"))
        g["buy_$M"] = (g["buy_usd"] / 1e6).round(1)
        g["sell_$M"] = (g["sell_usd"] / 1e6).round(1)
        g["net_$M"] = ((g["buy_usd"] - g["sell_usd"]) / 1e6).round(1)
        g["buy_pct"] = (g["buy_usd"] / (g["buy_usd"] + g["sell_usd"]) * 100).round(1)
        return (g.sort_values("net_$M", ascending=False).reset_index()
                 [[col, "buy_$M", "sell_$M", "net_$M", "buy_pct", "tickers"]])

    _dp_tot = _dpF["total_usd"].sum()
    _dp_cov = lambda col: (_dpF.dropna(subset=[col])["total_usd"].sum() / _dp_tot * 100) if _dp_tot else 0.0
    _dpu, _dpv = st.columns(2)
    _dpu.markdown(f"**By sector** — covers {_dp_cov('sector'):.0f}% of notional (13F sector field)")
    _dpu.dataframe(_dp_group("sector"), hide_index=True, use_container_width=True)
    _dpv.markdown(f"**By sub-sector** — covers {_dp_cov('bucket'):.0f}% (your buckets; the rest unmapped)")
    _dpv.dataframe(_dp_group("bucket"), hide_index=True, use_container_width=True)

st.subheader("All positions")
st.caption(
    "One row per manager × quarter × ticker × security type. **13F contains no "
    "transaction data** — it is a snapshot of what was held on the last day of "
    "the quarter, plus a derived quarter-over-quarter delta. A manager who "
    "bought 1M shares in April, sold 800k in May and bought 500k in June "
    "appears as one number, +700k: three trades, no dates, no prices. Shares "
    "and options are never pooled.")
q = st.text_input("Search ticker, manager, sector or bucket", "")
tx = d.copy()
if q:
    ql = q.strip().lower()
    m = False
    for c in ("ticker", "name", "sector", "bucket", "tier"):
        if c in tx:
            m = m | tx[c].fillna("").astype(str).str.lower().str.contains(ql)
    tx = tx[m]
COLS = [c for c in ("filed", "report_date", "name", "ticker", "security_type",
                    "put_call", "units", "units_change", "value", "pct_book",
                    "sector", "bucket", "tier") if c in tx]
show = (tx[COLS].sort_values(["report_date", "value"], ascending=[False, False])
        .rename(columns={"filed": "Filed", "report_date": "Quarter",
                         "name": "Manager", "ticker": "Ticker",
                         "security_type": "Security", "put_call": "P/C",
                         "units_change": "Units Δ", "value": "$",
                         "pct_book": "% book", "sector": "Sector",
                         "bucket": "Bucket", "tier": "Tier"}))
st.dataframe(show.head(3000), use_container_width=True, hide_index=True, height=520)
cA, cB = st.columns([3, 1])
cA.caption(f"{len(show):,} rows, first 3,000 shown. Ranked by $ value a market "
           f"maker's inventory outranks a 14% conviction position — use the "
           f"% of book filter.")
cB.download_button("Download CSV", show.to_csv(index=False).encode(),
                   file_name=f"institutional_{len(show)}.csv", mime="text/csv")

with st.expander("What is tested and what is not"):
    st.markdown("""
**Already model inputs, so not new information.** The dark-pool features
(`inst_block_buy_sell_7d`, `inst_signed_flow_30d`, `inst_auction_imbal_5d`,
`inst_signed_flow_5d` — audit-validated 2026-05-17, n=458, 4 of 8 candidates
survived) and the insider rollups (`insider_7d/21d/60d/90d`) are in
`FEATURE_COLUMNS`. `prob_up` already contains them, so agreement between them
is not fresh evidence.

**The 13F legs are untested.** T1 consensus buying, T2 crowding → drawdown,
T3 thesis shift, T4 crowding × flow — all pre-registered in
`PREREG_institutional_13f.md`, none run. The prior is not encouraging:
Fama-French find aggregate active managers have zero gross alpha and that the
top 3% can expect zero, and the copycat literature's edge is the FEE SAVING,
which does not apply to someone not paying the fee.

**Coverage**: 3,587 real tickers held, 1,621 priced, 419 in your bucket map.
A blank cell is missing data, not a zero reading.

**Survivorship**: the roster was built from the CURRENT institution list, so
managers who closed were never probed and are absent. Any backtest on this
roster is optimistic by an unknown amount.
""")
