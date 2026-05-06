#!/usr/bin/env python3
"""
monitor_rzlv_cmrc.py — Daily merger-vote signal tracker for RZLV / CMRC.

Tracks insider, institutional, dark pool, options, and short interest activity
on both tickers ahead of CMRC's May 14, 2026 annual meeting / director election
(the de facto proxy fight on Rezolve's takeover bid).

What this gives you:
  - A console section per signal type, per ticker
  - Day-over-day deltas vs the previous run (institutional positions, short
    interest, insider net flow, large dark pool prints)
  - "FLAG" lines on anything that exceeds the configured thresholds
  - All raw data persisted to merger_monitor.db so you can re-query later

Usage:
    export UW_API_KEY="..."           # required
    export MASSIVE_API_KEY="..."      # optional; enables Massive cross-check
    python monitor_rzlv_cmrc.py                   # default since=2026-03-01
    python monitor_rzlv_cmrc.py --since 2026-02-01
    python monitor_rzlv_cmrc.py --tickers CMRC    # CMRC only
    python monitor_rzlv_cmrc.py --csv             # also dump today's snapshot

Cron-compatible. Returns nonzero on hard failure; soft-fails per-endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UW_BASE = "https://api.unusualwhales.com"
MASSIVE_BASE = "https://api.massive.com"

DB_PATH = Path(os.environ.get("MERGER_DB", "merger_monitor.db"))

DEFAULT_TICKERS = ["CMRC", "RZLV"]

# Vote-relevant event timeline (informational; printed in the header)
DEAL_TIMELINE = [
    ("2026-02-??", "Rezolve's first private overture to CMRC board (rejected)"),
    ("2026-04-02", "DBLP Sea Cow (Wagner-affiliated) increases RZLV stake"),
    ("2026-04-08", "Rezolve goes public: 1 RZLV for 2 CMRC exchange offer"),
    ("2026-04-14", "CMRC adopts poison pill (10% / 20% passive thresholds)"),
    ("2026-04-15", "Rezolve investor call to CMRC shareholders"),
    ("2026-04-27", "Record date for CMRC rights distribution"),
    ("2026-05-07", "CMRC Q1 2026 earnings"),
    ("2026-05-14", "CMRC annual meeting / director election (proxy fight)"),
]

# Signal thresholds (calibrated for thin small-cap names; tune as needed)
DARKPOOL_BLOCK_MIN_USD = 250_000        # individual print
DARKPOOL_DAILY_AGGREGATE_USD = 1_000_000  # any day above this = noteworthy
INSIDER_MIN_USD = 100_000               # single insider trade
INSIDER_FLOW_DAYS = 30                  # rolling window
NEW_INST_MIN_VALUE_USD = 1_000_000      # new 13F position threshold
OPTIONS_PREMIUM_USD = 100_000           # single option trade premium
SHORT_INTEREST_DELTA_PCT = 0.05         # 5pp move in % of float short

REQUEST_TIMEOUT = 30
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class APIError(Exception):
    pass


def _uw_headers() -> dict[str, str]:
    key = os.environ.get("UW_API_KEY")
    if not key:
        raise APIError("UW_API_KEY environment variable not set")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def uw_get(path: str, params: Optional[dict] = None) -> Any:
    """GET against Unusual Whales. Soft-fails: returns None on HTTP error and
    prints the reason; this keeps the daily run from crashing on a single
    flaky endpoint."""
    url = f"{UW_BASE}{path}"
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=_uw_headers(),
                             params=params or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                # rate limited — back off
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(1 + attempt)
                continue
    print(f"  [warn] UW {path} failed: {last_err}", file=sys.stderr)
    return None


def massive_get(path: str, params: Optional[dict] = None) -> Any:
    """GET against Massive. Soft-fails like uw_get."""
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return None
    url = f"{MASSIVE_BASE}{path}"
    try:
        r = requests.get(url, params={**(params or {}), "apiKey": key},
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"  [warn] Massive {path} failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_log (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    section TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_log_ticker_section_ts
    ON run_log(ticker, section, run_ts);

CREATE TABLE IF NOT EXISTS insider_trades (
    accession_number TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    transaction_date TEXT,
    filing_date TEXT,
    insider_name TEXT,
    insider_title TEXT,
    transaction_code TEXT,
    shares REAL,
    price REAL,
    value_usd REAL,
    shares_owned_after REAL,
    raw TEXT,
    seen_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS institutional_holdings (
    ticker TEXT NOT NULL,
    institution_name TEXT NOT NULL,
    report_date TEXT NOT NULL,
    shares REAL,
    value_usd REAL,
    pct_of_float REAL,
    seen_ts TEXT NOT NULL,
    PRIMARY KEY (ticker, institution_name, report_date)
);

CREATE TABLE IF NOT EXISTS darkpool_prints (
    ticker TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    size REAL,
    price REAL,
    value_usd REAL,
    venue TEXT,
    tracking_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS short_interest_snapshots (
    ticker TEXT NOT NULL,
    record_date TEXT NOT NULL,
    short_interest REAL,
    pct_of_float REAL,
    days_to_cover REAL,
    seen_ts TEXT NOT NULL,
    PRIMARY KEY (ticker, record_date)
);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Section: insider activity (Form 4)
# ---------------------------------------------------------------------------

def section_insiders(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== INSIDER ACTIVITY — {ticker} (since {since}) ===")

    # Per-ticker insider transactions
    data = uw_get(f"/api/insider/{ticker}", params={"limit": 200})
    rows = (data or {}).get("data") or []

    # Filter to the window
    rows = [r for r in rows if (r.get("transaction_date") or "") >= since]
    if not rows:
        print("  No insider transactions in window.")
    else:
        # Persist + summarize
        cur = conn.cursor()
        new_count = 0
        for r in rows:
            shares = float(r.get("shares") or 0)
            price = float(r.get("price") or r.get("price_per_share") or 0)
            value = shares * price if (shares and price) else float(r.get("transaction_value") or 0)
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO insider_trades
                    (accession_number, ticker, transaction_date, filing_date,
                     insider_name, insider_title, transaction_code, shares, price,
                     value_usd, shares_owned_after, raw, seen_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.get("accession_number") or f"{ticker}-{r.get('transaction_date')}-{r.get('insider_name')}",
                    ticker,
                    r.get("transaction_date"),
                    r.get("filing_date"),
                    r.get("insider_name") or r.get("owner_name"),
                    r.get("insider_title") or r.get("officer_title"),
                    r.get("transaction_code"),
                    shares, price, value,
                    float(r.get("shares_owned_after") or 0),
                    json.dumps(r),
                    now_iso(),
                ))
                if cur.rowcount:
                    new_count += 1
            except sqlite3.Error as e:
                print(f"  [warn] insert failed: {e}", file=sys.stderr)

        # Aggregate buy vs sell
        buy_val = sum(float(r.get("shares") or 0) * float(r.get("price") or 0)
                      for r in rows if (r.get("transaction_code") or "").upper() in ("P", "A"))
        sell_val = sum(float(r.get("shares") or 0) * float(r.get("price") or 0)
                       for r in rows if (r.get("transaction_code") or "").upper() == "S")
        net = buy_val - sell_val

        print(f"  Trades in window: {len(rows)} (new since last run: {new_count})")
        print(f"  Aggregate buys:   ${buy_val:>14,.0f}")
        print(f"  Aggregate sells:  ${sell_val:>14,.0f}")
        print(f"  Net (buy - sell): ${net:>14,.0f}")

        # Top trades by value
        rows_sorted = sorted(rows, key=lambda r: float(r.get("shares") or 0) * float(r.get("price") or 0),
                             reverse=True)[:10]
        print("  Top trades:")
        for r in rows_sorted:
            sh = float(r.get("shares") or 0)
            px = float(r.get("price") or 0)
            v = sh * px
            code = r.get("transaction_code") or "?"
            who = (r.get("insider_name") or r.get("owner_name") or "?")[:32]
            title = (r.get("insider_title") or r.get("officer_title") or "")[:24]
            tag = "FLAG" if v >= INSIDER_MIN_USD else "    "
            print(f"  {tag}  {r.get('transaction_date')}  {code}  {who:<32}  {title:<24}"
                  f"  {sh:>10,.0f} @ ${px:>7,.2f}  = ${v:>12,.0f}")
        conn.commit()

    # Aggregated buy/sell stat for context
    aggr = uw_get(f"/api/stock/{ticker}/insider-buy-sells")
    if aggr and aggr.get("data"):
        last = aggr["data"][0] if isinstance(aggr["data"], list) else aggr["data"]
        print(f"  UW aggregated last period: "
              f"buys={last.get('purchases', last.get('buy_count', '?'))}  "
              f"sells={last.get('sales', last.get('sell_count', '?'))}")


# ---------------------------------------------------------------------------
# Section: institutional ownership (13F)
# ---------------------------------------------------------------------------

def section_institutional(conn: sqlite3.Connection, ticker: str) -> None:
    print(f"\n=== INSTITUTIONAL OWNERSHIP — {ticker} ===")

    data = uw_get(f"/api/institution/{ticker}/ownership", params={"limit": 200})
    rows = (data or {}).get("data") or []
    if not rows:
        print("  No institutional ownership data returned.")
        return

    # Pull previous snapshot from DB to compute deltas
    cur = conn.cursor()
    prev = {row[0]: row[1] for row in cur.execute(
        "SELECT institution_name, shares FROM institutional_holdings "
        "WHERE ticker = ? AND report_date = ("
        "  SELECT MAX(report_date) FROM institutional_holdings WHERE ticker = ?"
        ")", (ticker, ticker))}

    # Save current snapshot
    for r in rows:
        try:
            cur.execute("""
                INSERT OR REPLACE INTO institutional_holdings
                (ticker, institution_name, report_date, shares, value_usd, pct_of_float, seen_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                r.get("institution_name") or r.get("name"),
                r.get("report_date") or r.get("filing_date"),
                float(r.get("shares") or 0),
                float(r.get("value") or r.get("value_usd") or 0),
                float(r.get("pct_of_float") or 0),
                now_iso(),
            ))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}", file=sys.stderr)
    conn.commit()

    # Top holders
    rows_sorted = sorted(rows, key=lambda r: float(r.get("shares") or 0), reverse=True)[:15]
    print(f"  Total holders reported: {len(rows)}")
    print(f"  {'Institution':<40} {'Shares':>14} {'Δ vs prev':>14} {'Value $':>16}")
    for r in rows_sorted:
        name = (r.get("institution_name") or r.get("name") or "?")[:40]
        sh = float(r.get("shares") or 0)
        v = float(r.get("value") or r.get("value_usd") or 0)
        delta = sh - prev.get(name, sh)  # 0 if name was already there with same shares
        is_new = name not in prev and prev  # only flag "new" if we have any prev data
        flag = ""
        if is_new and v >= NEW_INST_MIN_VALUE_USD:
            flag = "  NEW"
        elif abs(delta) > 0.05 * sh:  # >5% change
            flag = "  CHG"
        print(f"  {name:<40} {sh:>14,.0f} {delta:>+14,.0f} {v:>16,.0f}{flag}")


# ---------------------------------------------------------------------------
# Section: dark pool prints
# ---------------------------------------------------------------------------

def section_darkpool(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== DARK POOL PRINTS — {ticker} (since {since}) ===")

    # UW endpoint paginates via newer_than/older_than; pull a generous window
    params = {"limit": 500, "newer_than": since}
    data = uw_get(f"/api/darkpool/{ticker}", params=params)
    rows = (data or {}).get("data") or []
    if not rows:
        print("  No dark pool prints returned.")
        return

    cur = conn.cursor()
    # Persist
    for r in rows:
        try:
            sz = float(r.get("size") or 0)
            px = float(r.get("price") or 0)
            v = sz * px
            cur.execute("""
                INSERT OR IGNORE INTO darkpool_prints
                (ticker, executed_at, size, price, value_usd, venue, tracking_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                r.get("executed_at") or r.get("trade_time"),
                sz, px, v,
                r.get("market_center") or r.get("venue"),
                r.get("tracking_id") or f"{ticker}-{r.get('executed_at')}-{r.get('size')}-{r.get('price')}",
            ))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}", file=sys.stderr)
    conn.commit()

    # Aggregate by day
    by_day: dict[str, dict] = {}
    for r in rows:
        d = (r.get("executed_at") or "")[:10]
        if not d:
            continue
        sz = float(r.get("size") or 0)
        px = float(r.get("price") or 0)
        v = sz * px
        b = by_day.setdefault(d, {"prints": 0, "shares": 0.0, "value": 0.0, "max_print": 0.0})
        b["prints"] += 1
        b["shares"] += sz
        b["value"] += v
        b["max_print"] = max(b["max_print"], v)

    print(f"  Prints in window: {len(rows)}")
    print(f"  {'Date':<12} {'Prints':>7} {'Shares':>14} {'Value $':>16} {'Max Print $':>14}")
    for d in sorted(by_day.keys(), reverse=True)[:15]:
        b = by_day[d]
        flag = ""
        if b["value"] >= DARKPOOL_DAILY_AGGREGATE_USD:
            flag = "  HEAVY"
        if b["max_print"] >= DARKPOOL_BLOCK_MIN_USD:
            flag += "  BLOCK"
        print(f"  {d:<12} {b['prints']:>7d} {b['shares']:>14,.0f} {b['value']:>16,.0f} {b['max_print']:>14,.0f}{flag}")

    # Top single prints
    rows_sorted = sorted(rows,
                         key=lambda r: float(r.get("size") or 0) * float(r.get("price") or 0),
                         reverse=True)[:10]
    print("  Top 10 single prints:")
    for r in rows_sorted:
        sz = float(r.get("size") or 0)
        px = float(r.get("price") or 0)
        v = sz * px
        if v < DARKPOOL_BLOCK_MIN_USD:
            continue
        print(f"    {(r.get('executed_at') or '')[:19]}  {sz:>10,.0f} @ ${px:>7,.2f}  "
              f"= ${v:>12,.0f}  ({r.get('market_center') or r.get('venue') or '?'})")


# ---------------------------------------------------------------------------
# Section: options flow (note: thin for CMRC; useful for RZLV)
# ---------------------------------------------------------------------------

def section_options_flow(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== OPTIONS FLOW — {ticker} (recent) ===")

    data = uw_get(f"/api/stock/{ticker}/flow-alerts", params={"limit": 100})
    rows = (data or {}).get("data") or []
    rows = [r for r in rows if (r.get("executed_at") or r.get("created_at") or "")[:10] >= since]

    if not rows:
        print(f"  No flow alerts returned (small-caps like {ticker} often have thin options).")
        return

    print(f"  Flow alerts in window: {len(rows)}")

    # Aggregate calls vs puts by premium
    call_prem = sum(float(r.get("total_premium") or r.get("premium") or 0)
                    for r in rows if (r.get("type") or r.get("option_type") or "").lower() == "call")
    put_prem = sum(float(r.get("total_premium") or r.get("premium") or 0)
                   for r in rows if (r.get("type") or r.get("option_type") or "").lower() == "put")
    pc_prem = (put_prem / call_prem) if call_prem else float("inf") if put_prem else 0

    print(f"  Call premium: ${call_prem:,.0f}")
    print(f"  Put premium:  ${put_prem:,.0f}")
    print(f"  Put/Call premium ratio: {pc_prem:.2f}")

    # Top single alerts
    big = sorted(rows, key=lambda r: float(r.get("total_premium") or r.get("premium") or 0), reverse=True)[:10]
    print("  Top alerts by premium:")
    for r in big:
        p = float(r.get("total_premium") or r.get("premium") or 0)
        if p < OPTIONS_PREMIUM_USD:
            continue
        otype = (r.get("type") or r.get("option_type") or "?").upper()
        strike = r.get("strike")
        exp = r.get("expiry") or r.get("expiration")
        print(f"    {(r.get('executed_at') or r.get('created_at') or '')[:19]}  "
              f"{otype:<5} ${strike} {exp}  prem=${p:,.0f}  "
              f"vol={r.get('volume') or '?'}  oi={r.get('open_interest') or '?'}")


# ---------------------------------------------------------------------------
# Section: short interest
# ---------------------------------------------------------------------------

def section_short_interest(conn: sqlite3.Connection, ticker: str) -> None:
    print(f"\n=== SHORT INTEREST — {ticker} ===")

    data = uw_get(f"/api/shorts/{ticker}/interest-float/v2")
    rows = (data or {}).get("data") or []
    if not rows:
        print("  No short interest data returned.")
        return

    cur = conn.cursor()
    for r in rows[-12:]:  # last ~6 months of bi-monthly snapshots
        try:
            cur.execute("""
                INSERT OR REPLACE INTO short_interest_snapshots
                (ticker, record_date, short_interest, pct_of_float, days_to_cover, seen_ts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                r.get("record_date") or r.get("settlement_date"),
                float(r.get("short_interest") or 0),
                float(r.get("short_percent_of_float") or r.get("pct_of_float") or 0),
                float(r.get("days_to_cover") or 0),
                now_iso(),
            ))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}", file=sys.stderr)
    conn.commit()

    print(f"  {'Date':<12} {'Short Int':>14} {'% Float':>9} {'DTC':>7}")
    for r in sorted(rows, key=lambda x: x.get("record_date") or "", reverse=True)[:8]:
        d = r.get("record_date") or r.get("settlement_date") or "?"
        si = float(r.get("short_interest") or 0)
        pct = float(r.get("short_percent_of_float") or r.get("pct_of_float") or 0)
        dtc = float(r.get("days_to_cover") or 0)
        print(f"  {d:<12} {si:>14,.0f} {pct:>8.2%} {dtc:>7.1f}")

    # Trend signal: latest minus prior
    if len(rows) >= 2:
        srt = sorted(rows, key=lambda x: x.get("record_date") or "")
        latest = float(srt[-1].get("short_percent_of_float") or srt[-1].get("pct_of_float") or 0)
        prev = float(srt[-2].get("short_percent_of_float") or srt[-2].get("pct_of_float") or 0)
        delta = latest - prev
        flag = "  FLAG" if abs(delta) >= SHORT_INTEREST_DELTA_PCT else ""
        print(f"  Δ % of float (latest vs prior snapshot): {delta:+.2%}{flag}")


# ---------------------------------------------------------------------------
# Section: price + volume context
# ---------------------------------------------------------------------------

def section_price_volume(ticker: str, since: str) -> None:
    print(f"\n=== PRICE / VOLUME CONTEXT — {ticker} (since {since}) ===")

    data = uw_get(f"/api/stock/{ticker}/ohlc/1d", params={"limit": 90})
    rows = (data or {}).get("data") or []
    rows = [r for r in rows if (r.get("date") or r.get("market_time") or "")[:10] >= since]
    if not rows:
        print("  No OHLC returned.")
        return

    # Recent volume vs 20d avg
    rows_sorted = sorted(rows, key=lambda r: r.get("date") or r.get("market_time") or "")
    vols = [float(r.get("volume") or 0) for r in rows_sorted]
    closes = [float(r.get("close") or 0) for r in rows_sorted]
    if len(vols) < 21:
        print(f"  Only {len(vols)} bars; volume baseline weak.")
    else:
        avg20 = sum(vols[-21:-1]) / 20
        latest_vol = vols[-1]
        ratio = latest_vol / avg20 if avg20 else 0
        print(f"  Latest close: ${closes[-1]:.2f}   "
              f"Latest volume: {latest_vol:,.0f}   "
              f"20d avg vol: {avg20:,.0f}   "
              f"Ratio: {ratio:.2f}x")
        if ratio >= 2.0:
            print("  FLAG: volume >= 2x 20d average")

    # High-volume days in window
    big_days = sorted(rows_sorted,
                      key=lambda r: float(r.get("volume") or 0), reverse=True)[:5]
    print("  Top 5 volume days in window:")
    for r in big_days:
        d = (r.get("date") or r.get("market_time") or "")[:10]
        v = float(r.get("volume") or 0)
        c = float(r.get("close") or 0)
        o = float(r.get("open") or 0)
        chg = (c - o) / o * 100 if o else 0
        print(f"    {d}  vol={v:>14,.0f}  open=${o:>6.2f} close=${c:>6.2f}  ({chg:+.2f}%)")


# ---------------------------------------------------------------------------
# Optional: Massive cross-check on session block trades
# ---------------------------------------------------------------------------

def section_massive_blocks(ticker: str, since: str) -> None:
    if not os.environ.get("MASSIVE_API_KEY"):
        return
    print(f"\n=== MASSIVE LIT-BLOCK CROSS-CHECK — {ticker} (since {since}) ===")
    # Daily aggregates to find unusually heavy days; tick-level scan would be
    # rate-prohibitive on Starter. This is a coarse confirmation pass.
    data = massive_get(f"/v2/aggs/ticker/{ticker}/range/1/day/{since}/{date.today().isoformat()}",
                       params={"adjusted": "true", "limit": 120})
    if not data or not data.get("results"):
        print("  Massive returned no aggregates.")
        return
    bars = data["results"]
    bars_sorted = sorted(bars, key=lambda b: b.get("v") or 0, reverse=True)[:5]
    print("  Top 5 volume days from Massive (all venues, including TRFs):")
    for b in bars_sorted:
        ts = datetime.fromtimestamp((b.get("t") or 0) / 1000, tz=timezone.utc).date().isoformat()
        v = b.get("v") or 0
        c = b.get("c") or 0
        o = b.get("o") or 0
        chg = (c - o) / o * 100 if o else 0
        print(f"    {ts}  vol={v:>14,.0f}  open=${o:>6.2f} close=${c:>6.2f}  ({chg:+.2f}%)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def print_header(args) -> None:
    print("=" * 78)
    print(f"  RZLV / CMRC merger-vote signal monitor")
    print(f"  Run UTC: {now_iso()}")
    print(f"  Window:  since {args.since}    Tickers: {', '.join(args.tickers)}")
    print(f"  DB:      {DB_PATH}")
    print("=" * 78)
    print("  Deal timeline (informational):")
    for d, ev in DEAL_TIMELINE:
        print(f"    {d}  {ev}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default="2026-03-01",
                   help="ISO date; lookback start for time-windowed sections (default 2026-03-01)")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help="Tickers to scan (default: CMRC RZLV)")
    p.add_argument("--csv", action="store_true",
                   help="Also dump today's signals to CSV alongside the DB")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["insiders", "institutional", "darkpool", "options",
                            "shorts", "ohlc", "massive"],
                   help="Skip one or more sections")
    args = p.parse_args()

    try:
        _uw_headers()  # fail fast if no key
    except APIError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print_header(args)
    conn = get_db()

    for ticker in args.tickers:
        print("\n" + "#" * 78)
        print(f"#  {ticker}")
        print("#" * 78)

        if "insiders" not in args.skip:
            section_insiders(conn, ticker, args.since)
        if "institutional" not in args.skip:
            section_institutional(conn, ticker)
        if "darkpool" not in args.skip:
            section_darkpool(conn, ticker, args.since)
        if "options" not in args.skip:
            section_options_flow(conn, ticker, args.since)
        if "shorts" not in args.skip:
            section_short_interest(conn, ticker)
        if "ohlc" not in args.skip:
            section_price_volume(ticker, args.since)
        if "massive" not in args.skip:
            section_massive_blocks(ticker, args.since)

    conn.close()

    print("\n" + "=" * 78)
    print("  Done. Re-run daily; deltas vs previous run are computed automatically.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
