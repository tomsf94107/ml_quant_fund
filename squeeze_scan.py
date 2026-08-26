#!/usr/bin/env python3
"""
squeeze_scan.py -- Multi-day short-squeeze pattern detector for BYND-class names.

WHY THIS EXISTS
  A >50%/day filter finds nothing on BYND. The real squeeze signature is a
  MULTI-DAY cluster: base -> catalyst -> accumulation ramp -> vertical ignition
  -> blow-off top on peak volume -> full fade. This script detects that from
  price+volume alone (Massive/Polygon OHLCV). That IS the trigger.

  Short interest / borrow / options skew / dark-pool flow are FUEL confirmation,
  not the trigger. Your `monitor` already pulls those correctly -- run it
  alongside this and paste both. Optional hooks below let you fold that data
  into the same panel, but they must be wired to YOUR real functions/schema
  first (see WIRING notes; do not trust them until verified -- Rule 1).

FLAGS
  RAMP     : >=3 consecutive up-closes with volume expanding   -> early warning (~5d pre-spike)
  IGNITION : 1-3d cumulative pop over threshold on >=4x rvol    -> the vertical move
  TOP      : local-max(20d) volume day that closes RED          -> blow-off / exit signal

OUTPUT
  - CSV + SQLite panel
  - scannable stdout report (paste this back)

RUN  (from repo root; uses features/massive_client -- no keys/host to export)
  python squeeze_scan.py BYND --start 2026-01-01 --si-db short_interest.db
"""

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


def _et_today():
    """Massive 403s on future windows. VN local date runs AHEAD of ET from
    00:00-11:00 VN (= 13:00-24:00 ET), so date.today() requests a bar that does
    not exist yet. Anchor every default to the US/Eastern calendar date."""
    return datetime.now(ZoneInfo("America/New_York")).date()

import pandas as pd

# --------------------------------------------------------------------------
# CONFIG -- thresholds derived from the Apr-2026 BYND episode. Tune freely.
# --------------------------------------------------------------------------
CFG = {
    "vol_win":   20,     # rolling window for avg-volume / local-max
    "ramp_days": 3,      # consecutive up-closes to flag a ramp
    "ig_ret1":   0.20,   # 1-day pop threshold  (Apr 20 was +41%)
    "ig_ret2":   0.25,   # 2-day cumulative pop threshold
    "ig_ret3":   0.30,   # 3-day cumulative pop threshold
    "ig_rvol":   4.0,    # relative-volume gate for ignition (Apr 20 ~6x, Apr 30 ~4.7x)
}

# --------------------------------------------------------------------------
# DATA PULL -- reuse features/massive_client.py (no keys/host to set here).
# auto_adjust=True keeps price continuity across a reverse split.
# --------------------------------------------------------------------------
def get_ohlcv(ticker, start, end):
    """Reuse the fund's Massive client (key/host/cache/yfinance-fallback already configured).
    massive_client.download() is yfinance-shaped: returns ['Open','High','Low','Close','Volume']
    on a DatetimeIndex."""
    import sys as _sys
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from features import massive_client as mc

    raw = mc.download(ticker, start=start, end=end, auto_adjust=True)
    if raw is None or len(raw) == 0:
        sys.exit(f"No OHLCV for {ticker} from massive_client. Check ticker / date range.")

    df = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})   # index name varies; first col is the date
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def detect_splits(df):
    """Infer split/reverse-split days from overnight price jumps that aren't ignitions.
    With auto_adjust=True the series is already split-continuous, so this is a
    heuristic flag, not authoritative. A real reverse split shows as an 8-K in your
    monitor -- treat that as the source of truth."""
    out = []
    g = df["open"] / df["close"].shift(1) - 1
    for i, gap in g.items():
        if pd.notna(gap) and abs(gap) >= 0.9:   # ~2x+ overnight move => probable split artifact
            out.append((df.loc[i, "date"].strftime("%Y-%m-%d"), f"gap {gap*100:+.0f}% (check for split/RS 8-K)"))
    return out or [("(none detected)", "")]


# --------------------------------------------------------------------------
# DETECTOR -- the actual signal
# --------------------------------------------------------------------------
def compute_signals(df, cfg=CFG):
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_2d"] = df["close"].pct_change(2)
    df["ret_3d"] = df["close"].pct_change(3)
    df["gap"]    = df["open"] / df["close"].shift(1) - 1
    mp = max(5, cfg["vol_win"] // 2)
    def _mech_clean_mean(w):
        med = w.median()
        kept = w[w <= 3.0 * med] if med > 0 else w
        return kept.mean() if len(kept) else float("nan")
    _prior = df["volume"].shift(1).rolling(cfg["vol_win"], min_periods=mp)
    df["_med20"] = _prior.median()
    df["vol20"]  = _prior.apply(_mech_clean_mean, raw=False)
    df["mech_day"] = (df["_med20"] > 0) & (df["volume"] > 3.0 * df["_med20"])
    df["rvol"]   = df["volume"] / df["vol20"]
    df["red"]    = df["close"] < df["open"]
    df["vol_max20"] = df["volume"] == df["volume"].rolling(cfg["vol_win"], min_periods=mp).max()
    df["below_1"]   = df["close"] < 1.0

    # consecutive up-closes
    up = df["close"] > df["close"].shift(1)
    df["up_streak"] = (up * (up.groupby((~up).cumsum()).cumcount() + 1)).where(up, 0).astype(int)
    vol_up = df["volume"] > df["volume"].shift(1)

    df["FLAG_ramp"] = (df["up_streak"] >= cfg["ramp_days"]) & vol_up
    df["FLAG_ignition"] = (
        (df["ret_1d"] >= cfg["ig_ret1"])
        | (df["ret_2d"] >= cfg["ig_ret2"])
        | (df["ret_3d"] >= cfg["ig_ret3"])
    ) & (df["rvol"] >= cfg["ig_rvol"]) & (df["ret_1d"] > 0) & ~df["mech_day"]   # ignition is an up-move
    df["FLAG_top"] = df["vol_max20"] & df["red"]
    return df


# --------------------------------------------------------------------------
# OPTIONAL ENRICHMENT -- FUEL context. WIRE TO YOUR REAL CODE BEFORE TRUSTING.
# --------------------------------------------------------------------------
def enrich_fuel(ticker, si_db=None):
    """
    Returns a dict of latest squeeze-fuel context, or notes on what to wire.
    Two honest gaps:
      - BORROW FEE / shares-available ARE in UW (/api/shorts/{ticker}/data):
        fee_rate + short_shares_available, daily/intraday. Verified live 2026-07.
        (Earlier note claiming ORTEX/Fintel-only was untested and is wrong.)
      - UW option-flow / dark-pool: reuse your monitor's working fetchers; do not
        re-implement endpoints here. Replace the import names below with yours
        (grep '^def ' scripts/monitor_ticker.py).
    """
    ctx = {}

    # ---- FINRA short interest from your existing short_interest.db ----
    if si_db and os.path.exists(si_db):
        try:
            con = sqlite3.connect(si_db)
            # WIRING: verify table/column names with `.schema` first (Rule 1).
            # Expected-ish: a table of (ticker, settlement_date, short_interest, days_to_cover).
            q = """
                SELECT * FROM short_interest
                WHERE ticker = ?
                ORDER BY settlement_date DESC
                LIMIT 4
            """
            si = pd.read_sql(q, con, params=[ticker])
            con.close()
            # clip the FINRA OTC junk (days_to_cover 999.99) per your rule
            if "days_to_cover" in si.columns:
                si.loc[si["days_to_cover"] > 50, "days_to_cover"] = pd.NA
            ctx["short_interest"] = si.to_dict("records")
        except Exception as e:
            ctx["short_interest"] = f"WIRING NEEDED (schema mismatch): {e}"
    else:
        ctx["short_interest"] = "pass --si-db /path/to/short_interest.db to include"

    # ---- UW options flow + dark pool ----
    # VERIFIED 2026-08-26: fetch_options_flow / fetch_darkpool DO NOT EXIST in
    # scripts/monitor_ticker.py. What exists there: assess_squeeze() (line 4283)
    # and section_squeeze() (line 4343). The previous bare `except Exception`
    # swallowed the ImportError and printed a WIRING note that READ LIKE DATA in
    # the FUEL CONTEXT panel on every run since this file was written.
    # section_squeeze() needs a live sqlite3 conn -- wiring it is its own task.
    ctx["options_flow"] = "NOT WIRED -- no such fetcher in monitor_ticker.py"
    ctx["darkpool"]     = "NOT WIRED -- no such fetcher in monitor_ticker.py"

    # Borrow fee IS available from UW; monitor's section_squeeze fetches it live.
    # This standalone scanner can call /api/shorts/{ticker}/data the same way.
    ctx["borrow_fee"] = "available via UW /api/shorts/{ticker}/data (see monitor section_squeeze)"
    return ctx


# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------
def render(df, ticker, splits, ctx, tail=60):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", tail + 5)

    show = df.tail(tail).copy()
    for c in ["ret_1d", "ret_2d", "ret_3d", "gap"]:
        show[c] = (show[c] * 100).round(1)
    show["rvol"] = show["rvol"].round(2)
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "close", "ret_1d", "ret_2d", "rvol", "up_streak",
            "FLAG_ramp", "FLAG_ignition", "FLAG_top", "below_1"]

    print("=" * 78)
    print(f"  SQUEEZE SCAN -- {ticker}   (last {tail} sessions)")
    print("=" * 78)
    print(show[cols].to_string(index=False))

    print("\n--- FLAGGED DAYS ---")
    for name, col in [("RAMP", "FLAG_ramp"), ("IGNITION", "FLAG_ignition"), ("TOP", "FLAG_top")]:
        hits = df[df[col]]
        dates = ", ".join(hits["date"].dt.strftime("%Y-%m-%d").tolist()) or "(none)"
        print(f"  {name:9s}: {dates}")

    print("\n--- SPLITS (reverse-split catalyst watch) ---")
    for d, ratio in (splits or [("(none)", "")]):
        print(f"  {d}  {ratio}")

    print("\n--- FUEL CONTEXT (confirmation, not trigger) ---")
    for k, v in ctx.items():
        print(f"  {k}: {v}")
    print("=" * 78)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--start", default=(_et_today() - timedelta(days=180)).isoformat())
    p.add_argument("--end", default=_et_today().isoformat())
    p.add_argument("--si-db", default=None, help="path to short_interest.db (optional)")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--tail", type=int, default=60)
    args = p.parse_args()
    # clamp explicit --end as well: a user-supplied future date 403s identically
    args.end = min(args.end, _et_today().isoformat())

    df = compute_signals(get_ohlcv(args.ticker, args.start, args.end))
    splits = detect_splits(df)
    ctx = enrich_fuel(args.ticker, args.si_db)

    base = os.path.join(args.out_dir, f"squeeze_{args.ticker}")
    df.to_csv(base + ".csv", index=False)
    con = sqlite3.connect(base + ".db")
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql("squeeze_panel", con, if_exists="replace", index=False)
    con.close()

    render(df, args.ticker, splits, ctx, tail=args.tail)
    print(f"\nWrote {base}.csv and {base}.db")


if __name__ == "__main__":
    main()
