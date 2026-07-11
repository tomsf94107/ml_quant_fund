#!/usr/bin/env python3
"""
darkpool_signed_flow.py
Reclassify PLTR dark-pool prints with quote-free signed-flow methods and
compare them against the VWAP heuristic your monitor uses.

WHY: the monitor tags prints ABOVE the daily VWAP as buys. On a stock grinding
down, intraday VWAP keeps falling, so prints get tagged "buy" just for clearing
a sinking average -> the +32.5% buy skew can be a measurement artifact.

Reads dark-pool prints straight from earnings_monitor.db (NO live UW calls) and
computes per-day + 7d signed flow under:
  1. VWAP heuristic   reproduces your monitor (sanity-check column)
  2. Tick test        the tick-rule half of Lee-Ready; quote-free
  3. BVC daily        Bulk Volume Classification (Easley/Lopez de Prado/O'Hara)
  4. Lee-Ready FULL   quote rule + tick tiebreak -- only if you wire NBBO quotes

True Lee-Ready needs the NBBO at each trade, which UW prints do NOT include.
Methods 2-3 are exactly what remove the VWAP bias and run from your DB out of
the box. Method 4 is stubbed for when you have a quote source.

Rule 1: no silent errors. Missing db / table / columns / rows -> raise loudly.
Run:  python darkpool_signed_flow.py        (env: ml_quant_310)
"""

import os
import sqlite3
import numpy as np
import pandas as pd

# scipy only needed for the Student-t CDF in BVC; fall back to normal if absent.
try:
    from scipy.stats import t as student_t, norm
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ----------------------------------------------------------------------------- CONFIG
DB_PATH    = os.environ.get(
    "DARKPOOL_DB",
    os.path.expanduser("~/Desktop/ML_Quant_Fund/earnings_monitor.db"),
)
TICKER     = "PLTR"
START_DATE = "2026-05-27"   # ET trading-day window start (match your monitor)
END_DATE   = None           # None = through latest in DB

# Leave None to auto-detect from schema; set explicitly if detection fails (it
# will tell you the table/column list when it can't map).
DARKPOOL_TABLE = None       # e.g. "darkpool_prints"
COL_TICKER     = None       # e.g. "ticker"
COL_PRICE      = None       # e.g. "price"
COL_SIZE       = None       # e.g. "size"  (shares)
COL_TS         = None       # e.g. "executed_at" (UTC timestamp)

# BVC params
BVC_DF    = 0.25            # Student-t deg. freedom (Easley et al. example)
BVC_USE_T = True            # False -> normal CDF

# Lee-Ready (full) — only runs if get_nbbo_for_trades() is implemented
RUN_FULL_LEE_READY = False
POLYGON_API_KEY    = os.environ.get("POLYGON_API_KEY")
# -----------------------------------------------------------------------------


def _fail(msg):
    raise RuntimeError(msg)


def _connect():
    if not os.path.exists(DB_PATH):
        _fail(f"DB not found: {DB_PATH}  (set DB_PATH or $DARKPOOL_DB)")
    return sqlite3.connect(DB_PATH)


def _detect_schema(con):
    tabs = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", con
    )["name"].tolist()
    if not tabs:
        _fail("No tables in DB.")

    table = DARKPOOL_TABLE
    if table is None:
        cand = [t for t in tabs if any(k in t.lower() for k in ("dark", "pool", "print"))]
        if len(cand) == 1:
            table = cand[0]
        else:
            _fail("Could not auto-pick dark-pool table. "
                  f"Set DARKPOOL_TABLE to one of: {tabs}")
    if table not in tabs:
        _fail(f"DARKPOOL_TABLE '{table}' not in DB. Tables: {tabs}")

    cols = pd.read_sql_query(f"PRAGMA table_info('{table}')", con)["name"].tolist()
    low = {c.lower(): c for c in cols}

    def pick(override, keys, what):
        if override is not None:
            if override not in cols:
                _fail(f"{what} column '{override}' not in {table}. Columns: {cols}")
            return override
        for k in keys:
            for lc, orig in low.items():
                if k in lc:
                    return orig
        _fail(f"Could not find {what} column in {table}. Columns: {cols} "
              f"(set the COL_* override).")

    c_tkr   = pick(COL_TICKER, ("ticker", "symbol", "sym"),               "ticker")
    c_price = pick(COL_PRICE,  ("price", "px", "prc"),                    "price")
    c_size  = pick(COL_SIZE,   ("size", "share", "qty", "volume", "quantity"), "size")
    c_ts    = pick(COL_TS,     ("executed", "timestamp", "datetime", "time", "ts", "date"), "timestamp")
    return table, c_tkr, c_price, c_size, c_ts


def load_prints(con):
    table, c_tkr, c_price, c_size, c_ts = _detect_schema(con)
    q = (f'SELECT "{c_ts}" AS ts, "{c_price}" AS price, "{c_size}" AS size '
         f'FROM "{table}" WHERE "{c_tkr}" = ?')
    df = pd.read_sql_query(q, con, params=(TICKER,))
    if df.empty:
        _fail(f"No {TICKER} rows in {table}. Check ticker spelling / table.")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"]  = pd.to_numeric(df["size"],  errors="coerce")
    df["ts"]    = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    bad = df[["price", "size", "ts"]].isna().any(axis=1)
    if bad.any():
        _fail(f"{int(bad.sum())} rows failed numeric/timestamp parse "
              f"(mapped price={c_price}, size={c_size}, ts={c_ts}). "
              f"Fix mapping; not dropping silently.")

    # ET trading-day bucket (UW timestamps are UTC)
    df["day"]   = df["ts"].dt.tz_convert("America/New_York").dt.date.astype(str)
    df["value"] = df["price"] * df["size"]

    df = df[df["day"] >= START_DATE]
    if END_DATE:
        df = df[df["day"] <= END_DATE]
    if df.empty:
        _fail(f"No {TICKER} prints in window >= {START_DATE}.")
    return df.sort_values("ts").reset_index(drop=True), (table, c_price, c_size, c_ts)


# ---- method 1: VWAP heuristic (reproduce monitor) ---------------------------
def classify_vwap(df):
    out = {}
    for day, g in df.groupby("day"):
        vwap = g["value"].sum() / g["size"].sum()
        buy  = g.loc[g["price"] > vwap, "value"].sum()
        sell = g.loc[g["price"] < vwap, "value"].sum()
        out[day] = (float(buy), float(sell))
    return out


# ---- method 2: tick test (Lee-Ready tick rule) ------------------------------
def classify_tick(df):
    out = {}
    for day, g in df.groupby("day"):
        g = g.sort_values("ts")
        p = g["price"].to_numpy()
        v = g["value"].to_numpy()
        sign = np.zeros(len(g), dtype=int)
        last = 0
        for i in range(len(g)):
            if i == 0:
                sign[i] = 0                 # first print of day: no prior -> neutral
            elif p[i] > p[i - 1]:
                sign[i] = 1;  last = 1
            elif p[i] < p[i - 1]:
                sign[i] = -1; last = -1
            else:
                sign[i] = last              # zero tick -> carry last sign
        out[day] = (float(v[sign == 1].sum()), float(v[sign == -1].sum()))
    return out


# ---- method 3: BVC daily ----------------------------------------------------
def _cdf(z):
    if BVC_USE_T and _HAVE_SCIPY:
        return float(student_t.cdf(z, df=BVC_DF))
    if _HAVE_SCIPY:
        return float(norm.cdf(z))
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def classify_bvc(df):
    days = sorted(df["day"].unique())
    rep, val = {}, {}
    for day, g in df.groupby("day"):
        rep[day] = g["value"].sum() / g["size"].sum()   # day VWAP as bucket price
        val[day] = float(g["value"].sum())
    dP = np.array([np.nan] + [rep[days[i]] - rep[days[i - 1]] for i in range(1, len(days))])
    sigma = np.nanstd(dP[1:], ddof=1) if len(days) > 2 else np.nan
    out = {}
    for i, day in enumerate(days):
        if i == 0 or not np.isfinite(sigma) or sigma == 0:
            bf = 0.5
        else:
            bf = _cdf(dP[i] / sigma)
        out[day] = (val[day] * bf, val[day] * (1.0 - bf))
    return out


# ---- method 4: full Lee-Ready (needs NBBO) ----------------------------------
def get_nbbo_for_trades(df):
    """
    Return a DataFrame aligned to df with columns ['bid','ask'] = prevailing
    NBBO at each trade ts. NOT implemented by default -- UW prints have no quotes.

    Polygon example (untested vs your account; needs POLYGON_API_KEY):
        GET /v3/quotes/PLTR?timestamp.lte=<ns>&order=desc&limit=1   per trade
    Databento / Alpaca historical quotes work too. Populate, then set
    RUN_FULL_LEE_READY=True.
    """
    raise NotImplementedError(
        "Wire a quote source (Polygon/Databento) into get_nbbo_for_trades() "
        "then set RUN_FULL_LEE_READY=True."
    )


def classify_lee_ready(df):
    q = get_nbbo_for_trades(df)
    df = df.copy()
    df["mid"] = (q["bid"].to_numpy() + q["ask"].to_numpy()) / 2.0
    out = {}
    for day, g in df.groupby("day"):
        g = g.sort_values("ts")
        p = g["price"].to_numpy(); m = g["mid"].to_numpy(); v = g["value"].to_numpy()
        sign = np.zeros(len(g), dtype=int); last = 0
        for i in range(len(g)):
            if p[i] > m[i]:
                sign[i] = 1;  last = 1
            elif p[i] < m[i]:
                sign[i] = -1; last = -1
            else:                                    # at mid -> tick test
                if i > 0 and p[i] > p[i - 1]:
                    sign[i] = 1;  last = 1
                elif i > 0 and p[i] < p[i - 1]:
                    sign[i] = -1; last = -1
                else:
                    sign[i] = last
        out[day] = (float(v[sign == 1].sum()), float(v[sign == -1].sum()))
    return out


# ---- reporting --------------------------------------------------------------
def skew(buy, sell):
    tot = buy + sell
    return (buy - sell) / tot if tot else 0.0


def fmt_money(x):
    return f"{x / 1e6:,.1f}M"


def report(name, by_day):
    days = sorted(by_day)
    tb = ts = 0.0
    print(f"\n=== {name} ===")
    print(f"{'Day':<12}{'Buy$':>12}{'Sell$':>12}{'Net$':>13}{'Skew':>9}")
    for d in days:
        b, s = by_day[d]
        tb += b; ts += s
        print(f"{d:<12}{fmt_money(b):>12}{fmt_money(s):>12}{fmt_money(b - s):>13}{skew(b, s) * 100:>8.1f}%")
    print(f"{'7d agg':<12}{fmt_money(tb):>12}{fmt_money(ts):>12}{fmt_money(tb - ts):>13}{skew(tb, ts) * 100:>8.1f}%")
    return skew(tb, ts) * 100


def main():
    con = _connect()
    try:
        df, meta = load_prints(con)
    finally:
        con.close()
    table, c_price, c_size, c_ts = meta
    print(f"Loaded {len(df):,} {TICKER} dark-pool prints from '{table}' "
          f"({df['day'].min()} .. {df['day'].max()})  "
          f"[cols: price={c_price}, size={c_size}, ts={c_ts}]")

    summary = {}
    summary["VWAP heuristic"] = report("VWAP heuristic (monitor baseline)", classify_vwap(df))
    summary["Tick test"]      = report("Tick test (Lee-Ready tick rule, quote-free)", classify_tick(df))
    bvc_label = f"BVC daily (t df={BVC_DF})" if (BVC_USE_T and _HAVE_SCIPY) else "BVC daily (normal)"
    if BVC_USE_T and not _HAVE_SCIPY:
        print("\n[note] scipy not found -> BVC using normal CDF instead of Student-t.")
    summary[bvc_label] = report(bvc_label, classify_bvc(df))

    if RUN_FULL_LEE_READY:
        summary["Lee-Ready (full)"] = report("Lee-Ready FULL (quote rule + tick)", classify_lee_ready(df))
    else:
        print("\n[skip] Lee-Ready FULL: no NBBO source wired "
              "(set RUN_FULL_LEE_READY=True after implementing get_nbbo_for_trades).")

    print("\n================ 7-DAY SKEW: METHOD COMPARISON ================")
    base = summary.get("VWAP heuristic")
    for k, v in summary.items():
        delta = "" if k == "VWAP heuristic" else f"   (vs VWAP {v - base:+.1f}pp)"
        print(f"  {k:<28}{v:+6.1f}%{delta}")
    print("\nIf tick-test / BVC skew collapses toward 0 vs VWAP's +32.5%, the")
    print("'buying' was largely a falling-VWAP artifact, not real accumulation.")


if __name__ == "__main__":
    main()
