#!/usr/bin/env python3
"""
Watchlist monitor — daily position-health & pre-position research script.

Designed for held / candidate positions held over weeks-to-months
(PLTR, MP, CRWD, OKLO, SNOW are first set, but works for any ticker).

Differs from monitor_earnings.py in emphasis:
- More weight on long-term institutional flow, cohort relative strength,
  drawdown from 52w high, multi-window relative performance.
- Less weight on implied earnings move, near-term option positioning.
- Per-ticker config tunes which sections are most actionable.

Usage:
    python scripts/monitor_watchlist.py PLTR
    python scripts/monitor_watchlist.py PLTR MP CRWD       # multiple
    python scripts/monitor_watchlist.py PLTR --quick       # daily-morning view
    python scripts/monitor_watchlist.py PLTR --full        # deeper research view
    python scripts/monitor_watchlist.py --all              # all configured tickers

DB:    watchlist_monitor.db (separate from earnings_monitor.db)
Keys:  reuses UW_API_KEY and MASSIVE_API_KEY env vars
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

warnings.filterwarnings("ignore")

# Repo root: this file lives at scripts/, project at parent
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if HERE.name == "scripts" else HERE
DB_PATH = PROJECT_ROOT / "watchlist_monitor.db"

# ─── Required keys (reused from monitor_earnings.py) ────────────────────────
UW_KEY = os.environ.get("UW_API_KEY", "").strip()
MASSIVE_KEY = os.environ.get("MASSIVE_API_KEY", "").strip()

if not UW_KEY:
    print("ERROR: UW_API_KEY not set in environment.")
    print("       export UW_API_KEY='...' first (same key used by monitor_earnings.py)")
    sys.exit(1)

UW_BASE = "https://api.unusualwhales.com/api"
MASSIVE_BASE = "https://api.massive.com/v3"

# ─── Per-ticker config ──────────────────────────────────────────────────────
# Each entry overrides defaults for tuning per ticker.
#   sector_etf:     ETF used for relative-strength comparison
#   cohort:         peer tickers for cohort divergence
#   shares_out:     hard-coded shares outstanding (avoids API failures)
#   notes:          one-line context that prints at the top
#   tier:           position tier from your AI playbook (🟢/🟡/🔴)
#   thesis_kill:    free-text reminder of the bucket-level kill switch
TICKER_CONFIG: Dict[str, Dict[str, Any]] = {
    "PLTR": {
        "sector_etf": "IGV",
        "cohort": ["CRWD", "NOW", "SNOW", "DDOG"],
        "shares_out": 2_300_000_000,
        "tier": "🟡 Tactical",
        "notes": "Enterprise SaaS / AI capture. High multiple, sentiment-driven.",
        "thesis_kill": "2 consecutive Q of software NRR <110% on average OR IGV breaks prior cycle low.",
    },
    "MP": {
        "sector_etf": "XLB",
        "cohort": ["LIN", "APD", "ALB", "FCX"],
        "shares_out": 180_000_000,
        "tier": "🔴 Lotto",
        "notes": "Rare earths / supply chain story. Government contract dependent.",
        "thesis_kill": "Loss of DoD contract OR China rare-earth export floor announcement.",
    },
    "CRWD": {
        "sector_etf": "IGV",
        "cohort": ["PANW", "ZS", "S", "FTNT"],
        "shares_out": 247_000_000,
        "tier": "🟢 Core",
        "notes": "Cybersecurity, AI-driven SOC moat. Watch competitor multiples.",
        "thesis_kill": "Major AI-driven cyber incident → sector-wide multiple reset.",
    },
    "OKLO": {
        "sector_etf": "XLU",
        "cohort": ["SMR", "CCJ", "LEU", "VST"],
        "shares_out": 145_000_000,
        "tier": "🔴 Lotto",
        "notes": "SMR nuclear, pre-revenue. Binary regulatory + customer risk.",
        "thesis_kill": "Major NRC setback OR hyperscaler nuclear deal cancellation.",
    },
    "SNOW": {
        "sector_etf": "IGV",
        "cohort": ["DDOG", "MDB", "NET", "TEAM"],
        "shares_out": 333_000_000,
        "tier": "🟡 Tactical",
        "notes": "Data SaaS. AI capture vs. seat-count pressure narrative.",
        "thesis_kill": "Same as PLTR — software NRR + IGV breakdown.",
    },
}

DEFAULT_CONFIG = {
    "sector_etf": "SPY",
    "cohort": [],
    "shares_out": None,
    "tier": "—",
    "notes": "—",
    "thesis_kill": "—",
}

# ─── Output severity ────────────────────────────────────────────────────────
@dataclass
class Flag:
    sev: str  # HIGH, MED, INFO
    ticker: str
    text: str

FLAGS: List[Flag] = []

def flag(sev: str, ticker: str, text: str) -> None:
    FLAGS.append(Flag(sev, ticker, text))

# ─── HTTP layer ─────────────────────────────────────────────────────────────
import requests

_session = requests.Session()
_session.headers.update({"Accept": "application/json", "User-Agent": "ml_quant_fund_watchlist/1.0"})

def uw_get(path: str, params: Optional[Dict[str, Any]] = None,
           timeout: int = 15, silent_404: bool = False) -> Optional[Any]:
    url = f"{UW_BASE}{path}"
    headers = {"Authorization": f"Bearer {UW_KEY}"}
    try:
        r = _session.get(url, params=params or {}, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404 and silent_404:
            return None
        msg = str(e)[:120]
        print(f"  [warn] UW {path} failed: {msg}")
        return None
    except Exception as e:
        print(f"  [warn] UW {path} failed: {str(e)[:120]}")
        return None

def massive_get(path: str, params: Optional[Dict[str, Any]] = None,
                timeout: int = 15) -> Optional[Any]:
    if not MASSIVE_KEY:
        return None
    url = f"{MASSIVE_BASE}{path}"
    p = dict(params or {})
    p["apikey"] = MASSIVE_KEY
    try:
        r = _session.get(url, params=p, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] Massive {path} failed: {str(e)[:120]}")
        return None

# ─── DB layer ───────────────────────────────────────────────────────────────
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db() -> None:
    with db() as conn:
        c = conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS run_log (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            tickers TEXT NOT NULL,
            mode TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dark_pool_prints (
            ticker TEXT NOT NULL,
            executed_at TEXT NOT NULL,
            size INTEGER,
            price REAL,
            value REAL,
            tape TEXT,
            PRIMARY KEY (ticker, executed_at, size, price)
        );
        CREATE TABLE IF NOT EXISTS price_snapshots (
            ticker TEXT NOT NULL,
            asof_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            PRIMARY KEY (ticker, asof_date)
        );
        CREATE TABLE IF NOT EXISTS inst_holdings (
            ticker TEXT NOT NULL,
            asof_run_id INTEGER NOT NULL,
            institution TEXT NOT NULL,
            shares INTEGER,
            value REAL,
            PRIMARY KEY (ticker, asof_run_id, institution)
        );
        CREATE INDEX IF NOT EXISTS ix_dp_ticker_dt ON dark_pool_prints(ticker, executed_at);
        CREATE INDEX IF NOT EXISTS ix_inst_ticker ON inst_holdings(ticker, institution);
        """)

def new_run(tickers: List[str], mode: str) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO run_log(run_ts, tickers, mode) VALUES(?,?,?)",
            (datetime.now(timezone.utc).isoformat(), ",".join(tickers), mode),
        )
        return cur.lastrowid

# ─── Formatting helpers ─────────────────────────────────────────────────────
def hr(char: str = "─", width: int = 78) -> str:
    return char * width

def section_header(title: str) -> None:
    print(f"\n=== {title} ===")

def banner(title: str, width: int = 78) -> None:
    print()
    print("#" * width)
    print(f"#  {title}")
    print("#" * width)

def fmt_money(v: Optional[float], wide: bool = True) -> str:
    if v is None or v == 0:
        return "—" if not wide else "—"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e12:
        return f"{sign}${av/1e12:.2f}T"
    if av >= 1e9:
        return f"{sign}${av/1e9:.2f}B"
    if av >= 1e6:
        return f"{sign}${av/1e6:.1f}M"
    if av >= 1e3:
        return f"{sign}${av/1e3:.1f}K"
    return f"{sign}${av:.2f}"

def fmt_pct(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "  —"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "—"
    return f"{v:,}"

# ─── Section: Position context ──────────────────────────────────────────────
def section_position_context(ticker: str, cfg: Dict[str, Any]) -> None:
    section_header(f"POSITION CONTEXT — {ticker}")
    print(f"  Tier:           {cfg.get('tier', '—')}")
    print(f"  Notes:          {cfg.get('notes', '—')}")
    print(f"  Sector ETF:     {cfg.get('sector_etf', '—')}")
    cohort = cfg.get("cohort") or []
    if cohort:
        print(f"  Cohort:         {', '.join(cohort)}")
    if cfg.get("shares_out"):
        print(f"  Shares out:     ~{cfg['shares_out']:,}")
    print(f"  Thesis kill:    {cfg.get('thesis_kill', '—')}")

# ─── Section: Price / volume / drawdown ─────────────────────────────────────
def fetch_uw_ohlc(ticker: str, days: int = 365) -> List[Dict[str, Any]]:
    """Fetch daily OHLCV for a ticker via UW."""
    end = datetime.now().date()
    start = end - timedelta(days=days)
    data = uw_get(
        f"/stock/{ticker}/ohlc/1d",
        params={"date_from": start.isoformat(), "date_to": end.isoformat(), "limit": 500},
    )
    if not data or "data" not in data:
        return []
    rows = data["data"]
    # Normalize → sort ascending by date
    out = []
    for r in rows:
        try:
            out.append({
                "date": r.get("market_time", "")[:10] or r.get("date", "")[:10],
                "open": float(r["open"]) if r.get("open") is not None else None,
                "high": float(r["high"]) if r.get("high") is not None else None,
                "low":  float(r["low"]) if r.get("low") is not None else None,
                "close": float(r["close"]) if r.get("close") is not None else None,
                "volume": int(r["volume"]) if r.get("volume") is not None else 0,
            })
        except (KeyError, TypeError, ValueError):
            continue
    out = [r for r in out if r["date"]]
    out.sort(key=lambda x: x["date"])
    return out

def fetch_uw_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """Live intraday quote (best-effort)."""
    for path in (f"/stock/{ticker}/realtime", f"/stock/{ticker}/quote"):
        d = uw_get(path, silent_404=True)
        if d and isinstance(d, dict):
            inner = d.get("data") if "data" in d else d
            if isinstance(inner, dict) and inner:
                return inner
    return None

def section_price_volume(ticker: str, cfg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    section_header(f"PRICE / VOLUME / DRAWDOWN — {ticker}")
    bars = fetch_uw_ohlc(ticker, days=365)
    if not bars:
        print("  No OHLC data returned.")
        return None

    closes = [b["close"] for b in bars if b["close"] is not None]
    last = bars[-1]
    last_close = last["close"]
    last_vol = last["volume"]

    quote = fetch_uw_quote(ticker)
    live_px = None
    if quote:
        for k in ("last", "price", "current_price", "regular_market_price"):
            v = quote.get(k)
            if v is not None:
                try:
                    live_px = float(v)
                    break
                except (TypeError, ValueError):
                    pass

    if live_px and last_close:
        chg = (live_px - last_close) / last_close * 100
        print(f"  Live (intraday):  ${live_px:.2f}   ({fmt_pct(chg)} vs last close ${last_close:.2f})")
    print(f"  Last close:       ${last_close:.2f}    Volume: {fmt_int(last_vol)}")

    # 20d avg volume
    last_20 = [b["volume"] for b in bars[-20:] if b["volume"]]
    if last_20:
        avg20 = sum(last_20) / len(last_20)
        ratio = last_vol / avg20 if avg20 else 0
        print(f"  20d avg volume:   {fmt_int(int(avg20))}   Today vs avg: {ratio:.2f}x")
        if ratio >= 2.0:
            flag("MED", ticker, f"Volume {ratio:.1f}x 20d avg")

    # 52w high / low + drawdown
    if closes:
        hi52 = max(closes)
        lo52 = min(closes)
        dd = (last_close - hi52) / hi52 * 100 if hi52 else 0
        runup = (last_close - lo52) / lo52 * 100 if lo52 else 0
        print(f"  52w high:         ${hi52:.2f}    Drawdown from high: {fmt_pct(dd)}")
        print(f"  52w low:          ${lo52:.2f}    Run-up from low:    {fmt_pct(runup)}")
        # Per playbook section 8.2: pullback thresholds matter for entry
        if dd <= -25:
            flag("HIGH", ticker, f"Drawdown {dd:.1f}% from 52w high — meaningful pullback zone")
        elif dd <= -15:
            flag("MED", ticker, f"Drawdown {dd:.1f}% from 52w high")

    # 5 highest-volume days (context)
    top_vol = sorted(bars, key=lambda b: b["volume"], reverse=True)[:5]
    print(f"  Top 5 volume days (last 12mo):")
    for b in top_vol:
        chg = ((b["close"] - b["open"]) / b["open"] * 100) if b["open"] else 0
        print(f"    {b['date']}  vol={fmt_int(b['volume']):>14}  "
              f"open=${b['open']:.2f} close=${b['close']:.2f}  ({fmt_pct(chg)})")

    return bars

# ─── Section: Multi-window relative strength ────────────────────────────────
def _ret(bars: List[Dict[str, Any]], days: int) -> Optional[float]:
    if len(bars) < days + 1:
        return None
    end = bars[-1]["close"]
    start = bars[-days - 1]["close"]
    if not start or not end:
        return None
    return (end - start) / start * 100

def section_relative_strength(ticker: str, cfg: Dict[str, Any],
                              bars: List[Dict[str, Any]]) -> None:
    section_header(f"MULTI-WINDOW RELATIVE STRENGTH — {ticker}")
    if not bars:
        print("  No bars; skipping.")
        return

    sector = cfg.get("sector_etf", "SPY")
    sector_bars = fetch_uw_ohlc(sector, days=180)
    spy_bars = fetch_uw_ohlc("SPY", days=180) if sector != "SPY" else sector_bars

    windows = [("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60), ("120d", 120)]

    print(f"                {'1d':>10} {'5d':>10} {'20d':>10} {'60d':>10} {'120d':>10}")
    own = [_ret(bars, d) for _, d in windows]
    sec = [_ret(sector_bars, d) for _, d in windows] if sector_bars else [None] * 5
    spy = [_ret(spy_bars, d) for _, d in windows] if spy_bars else [None] * 5

    print(f"  {ticker:<10}    " + " ".join(f"{fmt_pct(v):>10}" for v in own))
    print(f"  {sector:<10}    " + " ".join(f"{fmt_pct(v):>10}" for v in sec))
    if sector != "SPY":
        print(f"  SPY           " + " ".join(f"{fmt_pct(v):>10}" for v in spy))
    print(f"  {hr('-', 64)}")
    diff_sec = [(o - s) if (o is not None and s is not None) else None for o, s in zip(own, sec)]
    diff_spy = [(o - s) if (o is not None and s is not None) else None for o, s in zip(own, spy)]
    print(f"  vs {sector:<7} " + " ".join(f"{fmt_pct(v):>10}" for v in diff_sec))
    if sector != "SPY":
        print(f"  vs SPY        " + " ".join(f"{fmt_pct(v):>10}" for v in diff_spy))

    # Flag persistent multi-window underperformance vs sector
    bad_wins = sum(1 for v in diff_sec[1:4] if v is not None and v < -3.0)  # 5d/20d/60d
    if bad_wins >= 2:
        flag("MED", ticker, f"Persistent underperformance vs {sector} across multiple windows")

# ─── Section: Sector cohort ─────────────────────────────────────────────────
def section_cohort(ticker: str, cfg: Dict[str, Any], bars: List[Dict[str, Any]]) -> None:
    cohort = cfg.get("cohort") or []
    if not cohort:
        return
    section_header(f"SECTOR COHORT — {ticker}")
    print(f"  Ticker        {'1d':>10} {'5d':>10} {'20d':>10}")
    own_1, own_5, own_20 = _ret(bars, 1), _ret(bars, 5), _ret(bars, 20)
    print(f"  {ticker:<10}    {fmt_pct(own_1):>10} {fmt_pct(own_5):>10} {fmt_pct(own_20):>10}")

    cohort_20 = []
    for peer in cohort:
        pb = fetch_uw_ohlc(peer, days=60)
        if not pb:
            print(f"  {peer:<10}    " + "  ".join("—" for _ in range(3)))
            continue
        r1, r5, r20 = _ret(pb, 1), _ret(pb, 5), _ret(pb, 20)
        print(f"  {peer:<10}    {fmt_pct(r1):>10} {fmt_pct(r5):>10} {fmt_pct(r20):>10}")
        if r20 is not None:
            cohort_20.append(r20)

    if cohort_20 and own_20 is not None:
        avg = sum(cohort_20) / len(cohort_20)
        div = own_20 - avg
        verdict = "OUTPERFORMING" if div > 0 else "UNDERPERFORMING"
        print(f"  20d divergence vs cohort avg: {fmt_pct(div)} ({verdict})")
        if div <= -10:
            flag("MED", ticker, f"20d cohort divergence {div:.1f}pp — underperforming")
        elif div >= 10:
            flag("MED", ticker, f"20d cohort divergence {div:.1f}pp — outperforming")

# ─── Section: Institutional ownership ───────────────────────────────────────
def section_institutional(ticker: str, cfg: Dict[str, Any], run_id: int) -> None:
    section_header(f"INSTITUTIONAL OWNERSHIP — {ticker}")
    data = uw_get(f"/institution/{ticker}/ownership", params={"limit": 200})
    holders = data.get("data") if data and "data" in data else data
    if not holders or not isinstance(holders, list):
        print("  No institutional holders returned.")
        return

    sh_out = cfg.get("shares_out")
    threshold_pct = 0.5
    threshold_shares = int(sh_out * threshold_pct / 100) if sh_out else None

    print(f"  Total holders reported: {len(holders)}")
    if sh_out:
        print(f"  Shares outstanding: ~{sh_out:,}  (NEW threshold: {threshold_pct}% = "
              f"~{threshold_shares:,} shares)")

    # Detect deltas vs prior run for same ticker
    prior_map: Dict[str, int] = {}
    with db() as conn:
        rows = conn.execute(
            """SELECT institution, shares
               FROM inst_holdings
               WHERE ticker = ?
                 AND asof_run_id = (
                   SELECT MAX(asof_run_id) FROM inst_holdings
                   WHERE ticker = ? AND asof_run_id < ?
                 )""",
            (ticker, ticker, run_id),
        ).fetchall()
        prior_map = {r["institution"]: r["shares"] or 0 for r in rows}

    fmt_rows = []
    inserts = []
    for h in holders[:25]:
        name = (h.get("name") or h.get("institution_name") or "?").upper()[:42]
        shares = int(h.get("shares") or h.get("total_shares") or 0)
        value = float(h.get("value") or h.get("market_value") or 0)
        prior = prior_map.get(name, 0)
        delta = shares - prior
        is_new = (prior == 0) and (threshold_shares is not None) and (shares >= threshold_shares)
        pct = (shares / sh_out * 100) if sh_out else None
        fmt_rows.append((name, shares, delta, value, pct, is_new))
        inserts.append((ticker, run_id, name, shares, value))

    # Persist
    with db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO inst_holdings VALUES(?,?,?,?,?)",
            inserts,
        )

    print(f"  {'Institution':<44} {'Shares':>14} {'Δ vs prev':>12} {'Value':>16}  % s/o")
    for name, shares, delta, value, pct, is_new in fmt_rows[:15]:
        tag = "  NEW" if is_new else ""
        pct_str = f"{pct:.2f}%" if pct is not None else "  —"
        print(f"  {name:<44} {fmt_int(shares):>14} {fmt_int(delta):>12} "
              f"{fmt_money(value):>16}  {pct_str}{tag}")
        if is_new:
            flag("MED", ticker, f"New institutional holder: {name} ({pct_str} s/o, {fmt_money(value)})")

# ─── Section: Dark pool prints ──────────────────────────────────────────────
def fetch_uw_darkpool_day(ticker: str, day: str) -> List[Dict[str, Any]]:
    data = uw_get(
        f"/darkpool/{ticker}",
        params={"date": day, "limit": 500},
        silent_404=True,
    )
    if not data:
        return []
    items = data.get("data") if isinstance(data, dict) and "data" in data else data
    return items if isinstance(items, list) else []

def section_dark_pool(ticker: str, cfg: Dict[str, Any], bars: List[Dict[str, Any]],
                      lookback_days: int = 20) -> None:
    section_header(f"DARK POOL PRINTS — {ticker} (last {lookback_days} trading days)")

    # Fetch from local DB; only call UW for missing days
    end = datetime.now().date()
    start = end - timedelta(days=lookback_days * 2)  # buffer for weekends

    # Pull trading days from bars (more reliable than calendar arithmetic)
    if bars:
        recent_dates = [b["date"] for b in bars[-lookback_days:]]
    else:
        recent_dates = []

    new_prints = 0
    for day in recent_dates:
        # Have we already fetched this day?
        with db() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) c FROM dark_pool_prints WHERE ticker=? AND substr(executed_at,1,10)=?",
                (ticker, day),
            ).fetchone()["c"]
        if existing > 0:
            continue
        items = fetch_uw_darkpool_day(ticker, day)
        rows = []
        for it in items:
            ts = it.get("executed_at") or it.get("date") or ""
            sz = int(it.get("size") or it.get("shares") or 0)
            px = float(it.get("price") or 0)
            val = float(it.get("value") or sz * px)
            tape = (it.get("tape") or "L")
            if not ts or sz == 0 or px == 0:
                continue
            rows.append((ticker, ts, sz, px, val, tape))
        if rows:
            with db() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO dark_pool_prints VALUES(?,?,?,?,?,?)",
                    rows,
                )
            new_prints += len(rows)

    # Daily summary
    with db() as conn:
        rows = conn.execute(
            """SELECT substr(executed_at,1,10) AS d,
                      COUNT(*) AS prints,
                      SUM(size) AS shares,
                      SUM(value) AS value,
                      MAX(value) AS max_print
               FROM dark_pool_prints
               WHERE ticker = ?
                 AND substr(executed_at,1,10) >= ?
               GROUP BY substr(executed_at,1,10)
               ORDER BY substr(executed_at,1,10) DESC
               LIMIT ?""",
            (ticker, (end - timedelta(days=lookback_days * 2)).isoformat(), lookback_days),
        ).fetchall()

    if not rows:
        print("  No dark pool data.")
        return

    print(f"  {'Date':<12} {'Prints':>8} {'Shares':>14} {'Value $':>16} {'Max print $':>14}")
    for r in rows:
        marker = "  BLOCK" if (r["max_print"] or 0) >= 1_000_000 else ""
        heavy = "  HEAVY" if (r["max_print"] or 0) >= 50_000_000 else ""
        print(f"  {r['d']:<12} {fmt_int(r['prints']):>8} {fmt_int(r['shares']):>14} "
              f"{fmt_money(r['value']):>16} {fmt_money(r['max_print']):>14}{heavy}{marker}")

    # 7-day signed-flow heuristic
    section_signed_flow(ticker, bars, rows[:7])

def section_signed_flow(ticker: str, bars: List[Dict[str, Any]],
                        recent_rows: List[sqlite3.Row]) -> None:
    """Compute VWAP-based signed flow for the recent days."""
    if not bars or not recent_rows:
        return
    bar_by_date = {b["date"]: b for b in bars}

    print()
    print("  Signed flow estimate (VWAP heuristic, NOT Lee-Ready):")
    print(f"  {'Date':<12} {'Buy $':>16} {'Sell $':>16} {'Net':>16} {'Skew':>8}")

    total_buy = 0.0
    total_sell = 0.0
    for r in recent_rows:
        d = r["d"]
        bar = bar_by_date.get(d)
        if not bar:
            continue
        # VWAP approx (typical price)
        if bar["high"] and bar["low"] and bar["close"]:
            vwap = (bar["high"] + bar["low"] + bar["close"]) / 3
        else:
            continue

        with db() as conn:
            prints = conn.execute(
                """SELECT price, value FROM dark_pool_prints
                   WHERE ticker=? AND substr(executed_at,1,10)=?""",
                (ticker, d),
            ).fetchall()

        buy = sum(p["value"] for p in prints if p["price"] >= vwap)
        sell = sum(p["value"] for p in prints if p["price"] < vwap)
        net = buy - sell
        skew = (net / (buy + sell) * 100) if (buy + sell) else 0
        total_buy += buy
        total_sell += sell
        print(f"  {d:<12} {fmt_money(buy):>16} {fmt_money(sell):>16} "
              f"{fmt_money(net):>16} {fmt_pct(skew, 1):>8}")

    total = total_buy + total_sell
    agg_skew = ((total_buy - total_sell) / total * 100) if total else 0
    print(f"  7-day aggregate skew: {fmt_pct(agg_skew, 1)}  "
          f"(buy {fmt_money(total_buy)} vs sell {fmt_money(total_sell)})")
    if abs(agg_skew) >= 15 and total >= 10_000_000:
        direction = "buy" if agg_skew > 0 else "sell"
        flag("MED", ticker, f"7d signed dark pool {direction} skew {agg_skew:.1f}% on {fmt_money(total)} flow")
    print("  ⚠️  VWAP heuristic — coarse approximation. Above-VWAP prints may still be sells.")

# ─── Section: Options flow ──────────────────────────────────────────────────
def section_options_flow(ticker: str) -> None:
    section_header(f"OPTIONS FLOW — {ticker} (recent)")
    data = uw_get(f"/stock/{ticker}/flow-alerts", params={"limit": 100}, silent_404=True)
    items = data.get("data") if isinstance(data, dict) and "data" in data else data
    if not isinstance(items, list) or not items:
        print("  No options flow data.")
        return

    call_prem = 0.0
    put_prem = 0.0
    sorted_items = []
    for it in items:
        try:
            side = (it.get("type") or it.get("option_type") or "").upper()
            prem = float(it.get("total_premium") or it.get("premium") or 0)
            if side.startswith("C"):
                call_prem += prem
            elif side.startswith("P"):
                put_prem += prem
            sorted_items.append((prem, side, it))
        except (TypeError, ValueError):
            continue

    print(f"  Flow alerts in window: {len(items)}")
    print(f"  Call premium: {fmt_money(call_prem)}")
    print(f"  Put premium:  {fmt_money(put_prem)}")
    if call_prem > 0:
        ratio = put_prem / call_prem
        print(f"  Put/Call premium ratio: {ratio:.2f}")
        if ratio >= 1.5:
            flag("MED", ticker, f"Options P/C premium ratio {ratio:.2f} — bearish positioning")
        elif ratio <= 0.25:
            flag("MED", ticker, f"Options P/C premium ratio {ratio:.2f} — heavy call skew")

    sorted_items.sort(key=lambda x: x[0], reverse=True)
    print("  Top 5 alerts by premium:")
    for prem, side, it in sorted_items[:5]:
        ts = (it.get("created_at") or it.get("alert_time") or "")[:19]
        strike = it.get("strike") or it.get("strike_price") or "?"
        expiry = (it.get("expiry") or it.get("expiration_date") or "?")[:10]
        vol = it.get("total_volume") or it.get("volume") or "?"
        oi = it.get("total_oi") or it.get("open_interest") or "?"
        print(f"    {ts}  {side:<4}  ${strike} {expiry}  prem={fmt_money(prem)}  vol={vol}  oi={oi}")

# ─── Section: Short interest ────────────────────────────────────────────────
def section_short_interest(ticker: str) -> None:
    section_header(f"SHORT INTEREST — {ticker}")
    data = uw_get(f"/shorts/{ticker}/interest-float/v2", silent_404=True)
    items = data.get("data") if isinstance(data, dict) and "data" in data else data
    if not isinstance(items, list) or not items:
        print("  No short interest data.")
        return
    print(f"  {'Date':<14} {'Short Int':>14} {'% Float':>10} {'DTC':>8}")
    for r in items[:8]:
        date = (r.get("settlement_date") or r.get("date") or "?")[:10]
        si = int(r.get("short_interest") or 0)
        pct = float(r.get("short_interest_percent_of_float") or 0)
        dtc = float(r.get("days_to_cover") or 0)
        print(f"  {date:<14} {fmt_int(si):>14} {pct:>9.2f}% {dtc:>8.1f}")
    # Detect change
    if len(items) >= 2:
        latest = int(items[0].get("short_interest") or 0)
        prior = int(items[1].get("short_interest") or 0)
        if prior > 0:
            ch = (latest - prior) / prior * 100
            print(f"  Δ short shares (latest vs prior): {fmt_int(latest - prior)} ({fmt_pct(ch)})")
            if ch >= 15:
                flag("MED", ticker, f"Short interest +{ch:.1f}% — bears building")
            elif ch <= -15:
                flag("MED", ticker, f"Short cover {ch:.1f}% — shorts unwinding")

# ─── Section: Recent news (UW + macro context) ──────────────────────────────
def section_news(ticker: str, cfg: Dict[str, Any]) -> None:
    section_header(f"RECENT NEWS — {ticker}")
    # Try ticker-specific UW news
    for path in (f"/stock/{ticker}/news-headlines",
                 f"/stock/{ticker}/news",
                 f"/news/{ticker}"):
        data = uw_get(path, params={"limit": 15}, silent_404=True)
        if data:
            items = data.get("data") if isinstance(data, dict) and "data" in data else data
            if isinstance(items, list) and items:
                print(f"  Source: UnusualWhales (last 14 days)")
                for it in items[:10]:
                    date = (it.get("created_at") or it.get("published_at") or "")[:10]
                    src = (it.get("source") or "")[:20]
                    title = (it.get("headline") or it.get("title") or "")[:90]
                    print(f"  {date}   [{src:<20}] {title}")
                return
    print("  No UW news returned.")

# ─── Section: Today's flags summary ─────────────────────────────────────────
def section_flags_summary() -> None:
    if not FLAGS:
        return
    print()
    print(hr("="))
    print("  TODAY'S FLAGS")
    print(hr("="))
    by_sev = defaultdict(list)
    for f in FLAGS:
        by_sev[f.sev].append(f)
    for sev in ("HIGH", "MED", "INFO"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        print(f"  [{sev}]  {len(items)} flag(s)")
        for f in sorted(items, key=lambda x: x.ticker):
            print(f"    • {f.ticker}  {f.text}")

# ─── Per-ticker pipeline ────────────────────────────────────────────────────
def run_ticker(ticker: str, mode: str, run_id: int) -> None:
    cfg = TICKER_CONFIG.get(ticker, DEFAULT_CONFIG.copy())
    banner(ticker)
    section_position_context(ticker, cfg)
    bars = section_price_volume(ticker, cfg)
    section_relative_strength(ticker, cfg, bars or [])
    if mode in ("full", "both"):
        section_cohort(ticker, cfg, bars or [])
    section_institutional(ticker, cfg, run_id)
    section_dark_pool(ticker, cfg, bars or [], lookback_days=15)
    section_options_flow(ticker)
    if mode in ("full", "both"):
        section_short_interest(ticker)
        section_news(ticker, cfg)

# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Watchlist monitor for held / candidate positions.")
    ap.add_argument("tickers", nargs="*", help="Ticker symbols to scan")
    ap.add_argument("--all", action="store_true", help="Scan all configured tickers")
    ap.add_argument("--quick", action="store_true", help="Quick daily view (skip cohort/news/short)")
    ap.add_argument("--full", action="store_true", help="Full pre-position research view (default)")
    args = ap.parse_args()

    init_db()

    # Resolve ticker list
    if args.all:
        tickers = sorted(TICKER_CONFIG.keys())
    elif args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
    else:
        ap.print_help()
        return 1

    mode = "quick" if args.quick else "full"

    print(hr("="))
    print(f"  Watchlist monitor")
    print(f"  Run UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  Tickers: {', '.join(tickers)}    Mode: {mode}")
    print(f"  DB:      {DB_PATH}")
    print(hr("="))

    run_id = new_run(tickers, mode)

    for t in tickers:
        try:
            run_ticker(t, mode, run_id)
        except Exception as e:
            print(f"\n[ERROR] {t} pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    section_flags_summary()
    print()
    print(hr("="))
    print("  Done.  Re-run during US market hours for freshest data.")
    print(hr("="))
    return 0

if __name__ == "__main__":
    sys.exit(main())
