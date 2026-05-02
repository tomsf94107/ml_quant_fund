# features/institutional_ingest.py
# ─────────────────────────────────────────────────────────────────────────────
# Institutional / dark-pool flow ingestion via Unusual Whales /api/darkpool.
#
# UW's darkpool endpoint returns curated dark-pool prints with NBBO already
# attached, which means:
#   - No separate quotes API call (no 403, no extra cost)
#   - Full Lee-Ready side inference using the NBBO carried with each trade
#   - Pre-computed `premium` for notional (no math needed)
#   - `tracking_id` per trade = perfect dedup key
#
# Schema is fresh — drop institutional_trades.db before running this for
# the first time (it had nothing useful from the failed Massive run anyway).
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

# ─── Auth ────────────────────────────────────────────────────────────────────
# Try config.keys first (matches the pattern used by massive_client.py), then env.
try:
    from config.keys import UW_API_KEY  # type: ignore
except ImportError:
    UW_API_KEY = os.environ.get("UW_API_KEY", "")

UW_BASE = "https://api.unusualwhales.com"
UW_DARKPOOL_PATH = "/api/darkpool/{ticker}"

# ─── Paths & constants ───────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "institutional_trades.db"
NOTIONAL_FLOOR = 250_000.0
BLOCK_SHARE_THRESHOLD = 10_000

UW_PAGE_LIMIT = 500           # UW typical max per page
UW_MAX_PAGES_PER_DAY = 100    # safety cap; 50k prints/day per ticker

# Sale condition codes (UW uses snake_case strings)
# NOTE: sweeps and crosses don't appear on FINRA TRF (dark pool) prints — those
# are lit-tape conditions reserved for /api/lit-blocks (future work). The codes
# we DO see on UW darkpool are algorithmic-execution markers:
ALGO_CONDITIONS = {
    "average_price_trade",      # VWAP/TWAP fill represented as one print
    "prior_reference_price",    # executed at a previously-quoted price
    "contingent_trade",         # multi-leg trade
}
SWEEP_CONDITIONS: set[str] = set()  # populated when lit-tape ingest is added
CROSS_CONDITIONS: set[str] = set()  # populated when lit-tape ingest is added

# Closing auction window: 4:00 PM ET ± 2 minutes = 19:58–20:02 UTC during DST,
# 20:58–21:02 UTC otherwise. Use a slightly broad UTC window to capture both.
def _is_closing_auction(ts_iso: str) -> bool:
    """True if trade_ts falls within the US equity closing auction window."""
    if not ts_iso or len(ts_iso) < 19:
        return False
    hh = ts_iso[11:13]
    mm = ts_iso[14:16]
    # 19:58–20:02 UTC (EDT close) OR 20:58–21:02 UTC (EST close)
    if hh == "19" and mm >= "58":
        return True
    if hh == "20" and mm <= "02":
        return True
    if hh == "20" and mm >= "58":
        return True
    if hh == "21" and mm <= "02":
        return True
    return False

# ─── Schema ──────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS institutional_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_id     INTEGER UNIQUE,                 -- UW per-trade unique ID
    ticker          TEXT    NOT NULL,
    trade_ts        TEXT    NOT NULL,               -- ISO 8601 UTC
    trade_date      TEXT    NOT NULL,
    sip_ts_ns       INTEGER,
    side            TEXT    NOT NULL DEFAULT 'UNKNOWN',
    shares          REAL    NOT NULL,
    price           REAL    NOT NULL,
    notional_usd    REAL    NOT NULL,
    nbbo_bid        REAL,
    nbbo_ask        REAL,
    exchange_code   TEXT,                            -- UW market_center letter
    exchange_name   TEXT,
    is_dark_pool    INTEGER NOT NULL DEFAULT 1,
    is_block        INTEGER NOT NULL DEFAULT 0,
    is_sweep        INTEGER NOT NULL DEFAULT 0,
    is_cross        INTEGER NOT NULL DEFAULT 0,
    is_algo         INTEGER NOT NULL DEFAULT 0,
    is_closing_auction INTEGER NOT NULL DEFAULT 0,
    is_canceled     INTEGER NOT NULL DEFAULT 0,
    sale_cond_codes TEXT,
    provider        TEXT    NOT NULL DEFAULT 'uw',
    fetched_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inst_ticker_ts   ON institutional_trades(ticker, trade_ts DESC);
CREATE INDEX IF NOT EXISTS idx_inst_date_ticker ON institutional_trades(trade_date, ticker);
CREATE INDEX IF NOT EXISTS idx_inst_notional    ON institutional_trades(notional_usd DESC);
CREATE INDEX IF NOT EXISTS idx_inst_block       ON institutional_trades(is_block, trade_date DESC) WHERE is_block = 1;

CREATE TABLE IF NOT EXISTS institutional_scraper_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    last_poll_at      TEXT,
    last_provider     TEXT,
    last_row_count    INTEGER,
    last_ticker_count INTEGER,
    last_error        TEXT,
    updated_at        TEXT
);
INSERT OR IGNORE INTO institutional_scraper_state (id) VALUES (1);

CREATE TABLE IF NOT EXISTS ingest_cursor (
    ticker           TEXT PRIMARY KEY,
    last_trade_ts    TEXT NOT NULL,
    last_tracking_id INTEGER,
    rows_total       INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL
);
"""

# ─── DB helpers ──────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.executescript(SCHEMA_SQL)
    return conn


# ─── HTTP wrapper ────────────────────────────────────────────────────────────

def _check_uw_key() -> None:
    if not UW_API_KEY:
        raise RuntimeError(
            "UW_API_KEY not found. Add it to config/keys.py or set the "
            "UW_API_KEY environment variable."
        )


def _uw_get(path: str, params: Optional[dict] = None,
            timeout: int = 30, max_retries: int = 3) -> dict:
    _check_uw_key()
    headers = {"Authorization": f"Bearer {UW_API_KEY}",
               "Accept": "application/json"}
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{UW_BASE}{path}", headers=headers,
                             params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:  # rate limited
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (401, 403):
                raise RuntimeError(f"UW auth error {r.status_code}: {r.text[:300]}")
            # 4xx other than auth/rate — bad params, don't retry
            raise RuntimeError(f"UW {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"UW network error after {max_retries} retries: {last_exc}")


# ─── Parsing helpers ─────────────────────────────────────────────────────────

def _f(s) -> Optional[float]:
    """Parse string-or-None to float-or-None."""
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _iso_to_ns(iso_str: str) -> int:
    """Parse '2026-05-01T15:15:04Z' to nanosecond UTC timestamp."""
    if not iso_str:
        return 0
    if iso_str.endswith("Z"):
        iso_str = iso_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


# ─── UW darkpool fetch (with smart filter detection) ─────────────────────────

# Module-level memo so we don't re-probe server-side filter for every ticker.
_SERVER_FILTER_KNOWN: Optional[str] = None  # None = unprobed, "min_premium" or "none"


def fetch_darkpool_day(ticker: str, day: date,
                       min_premium: float = 0.0,
                       max_pages: int = UW_MAX_PAGES_PER_DAY,
                       verbose: bool = False) -> list[dict]:
    """
    Fetch all dark-pool prints for ticker on `day`, filtered to ≥ min_premium.

    Tries server-side `min_premium` filter first; falls back to client-side.
    """
    global _SERVER_FILTER_KNOWN
    date_str = day.isoformat()
    base_params = {"limit": UW_PAGE_LIMIT, "date": date_str}

    use_server_filter = False
    if min_premium > 0:
        if _SERVER_FILTER_KNOWN is None:
            probe_params = dict(base_params, limit=20, min_premium=int(min_premium))
            try:
                probe = _uw_get(UW_DARKPOOL_PATH.format(ticker=ticker.upper()), probe_params)
                items = probe.get("data", []) if isinstance(probe, dict) else []
                if items:
                    min_seen = min((_f(it.get("premium")) or 0) for it in items)
                    if min_seen >= min_premium * 0.99:
                        _SERVER_FILTER_KNOWN = "min_premium"
                        use_server_filter = True
                        if verbose:
                            print(f"    server-side min_premium filter active "
                                  f"(probe min ${min_seen:,.0f})")
                    else:
                        _SERVER_FILTER_KNOWN = "none"
                        if verbose:
                            print(f"    server filter not respected "
                                  f"(returned ${min_seen:,.0f}); using client-side")
                else:
                    _SERVER_FILTER_KNOWN = "none"
            except RuntimeError:
                _SERVER_FILTER_KNOWN = "none"
        elif _SERVER_FILTER_KNOWN == "min_premium":
            use_server_filter = True

    params = dict(base_params)
    if use_server_filter:
        params["min_premium"] = int(min_premium)

    all_items: list[dict] = []
    older_than: Optional[str] = None

    for page_idx in range(max_pages):
        p = dict(params)
        if older_than:
            p["older_than"] = older_than
        try:
            data = _uw_get(UW_DARKPOOL_PATH.format(ticker=ticker.upper()), p)
        except RuntimeError as e:
            if verbose:
                print(f"    {ticker} {date_str} page {page_idx}: {e}")
            break

        items = data.get("data", []) if isinstance(data, dict) else []
        if not items:
            break

        # UW's `date` param appears to be ignored / soft — pagination via
        # `older_than` walks backward across day boundaries. Clamp client-side
        # to only items whose executed_at falls on the requested day.
        items_on_day = [
            it for it in items
            if (it.get("executed_at") or "")[:10] == date_str
        ]
        all_items.extend(items_on_day)

        # If we've started seeing items from an earlier day, we've crossed the
        # boundary — stop paginating, we have everything for this day.
        if len(items_on_day) < len(items):
            break
        if len(items) < UW_PAGE_LIMIT:
            break
        older_than = items[-1].get("executed_at")
        if not older_than:
            break
        time.sleep(0.05)

    if not use_server_filter and min_premium > 0:
        before = len(all_items)
        all_items = [it for it in all_items
                     if (_f(it.get("premium")) or 0) >= min_premium]
        if verbose:
            print(f"    client-side filter: {before} → {len(all_items)} ≥ ${min_premium:,.0f}")

    return all_items


# ─── Side inference (Lee-Ready) ──────────────────────────────────────────────

def lee_ready_classify(price: float,
                       bid: Optional[float], ask: Optional[float],
                       prev_price: Optional[float]) -> str:
    """BUY / SELL / UNKNOWN. Quote rule first, tick rule fallback."""
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if price > mid:
            return "BUY"
        if price < mid:
            return "SELL"
    if prev_price is not None:
        if price > prev_price:
            return "BUY"
        if price < prev_price:
            return "SELL"
    return "UNKNOWN"


# ─── Row transformation ──────────────────────────────────────────────────────

def transform_row(item: dict, side: str) -> dict:
    ts_iso = item.get("executed_at") or ""
    sip_ns = _iso_to_ns(ts_iso)
    trade_dt = (datetime.fromtimestamp(sip_ns / 1e9, tz=timezone.utc)
                if sip_ns else datetime.now(timezone.utc))

    shares = float(item.get("size") or 0)
    price = _f(item.get("price")) or 0.0
    premium = _f(item.get("premium")) or (shares * price)

    sale_cond = item.get("sale_cond_codes")
    sale_codes: list[str] = []
    if sale_cond:
        if isinstance(sale_cond, str):
            sale_codes = [sale_cond]
        elif isinstance(sale_cond, list):
            sale_codes = [str(c) for c in sale_cond]

    is_sweep = 1 if any(c in SWEEP_CONDITIONS for c in sale_codes) else 0
    is_cross = 1 if any(c in CROSS_CONDITIONS for c in sale_codes) else 0
    is_algo = 1 if any(c in ALGO_CONDITIONS for c in sale_codes) else 0
    is_block = 1 if shares >= BLOCK_SHARE_THRESHOLD else 0
    is_canceled = 1 if item.get("canceled") else 0
    is_closing_auction = 1 if _is_closing_auction(ts_iso) else 0

    market_center = item.get("market_center")

    return {
        "tracking_id": item.get("tracking_id"),
        "ticker": (item.get("ticker") or "").upper(),
        "trade_ts": trade_dt.isoformat(),
        "trade_date": trade_dt.date().isoformat(),
        "sip_ts_ns": sip_ns,
        "side": side,
        "shares": shares,
        "price": price,
        "notional_usd": premium,
        "nbbo_bid": _f(item.get("nbbo_bid")),
        "nbbo_ask": _f(item.get("nbbo_ask")),
        "exchange_code": market_center,
        "exchange_name": f"DARK_{market_center}" if market_center else "DARK",
        "is_dark_pool": 1,
        "is_block": is_block,
        "is_sweep": is_sweep,
        "is_cross": is_cross,
        "is_algo": is_algo,
        "is_closing_auction": is_closing_auction,
        "is_canceled": is_canceled,
        "sale_cond_codes": json.dumps(sale_codes) if sale_codes else None,
        "provider": "uw",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


_INSERT_COLS = [
    "tracking_id", "ticker", "trade_ts", "trade_date", "sip_ts_ns",
    "side", "shares", "price", "notional_usd", "nbbo_bid", "nbbo_ask",
    "exchange_code", "exchange_name", "is_dark_pool", "is_block",
    "is_sweep", "is_cross", "is_algo", "is_closing_auction",
    "is_canceled", "sale_cond_codes",
    "provider", "fetched_at",
]
_INSERT_SQL = (
    f"INSERT OR IGNORE INTO institutional_trades ({','.join(_INSERT_COLS)}) "
    f"VALUES ({','.join('?' * len(_INSERT_COLS))})"
)


def upsert_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = [tuple(r.get(c) for c in _INSERT_COLS) for r in rows]
    cur = conn.executemany(_INSERT_SQL, payload)
    conn.commit()
    return max(cur.rowcount or 0, 0)


# ─── Cursor + state ──────────────────────────────────────────────────────────

def get_cursor(conn: sqlite3.Connection, ticker: str) -> Optional[datetime]:
    row = conn.execute(
        "SELECT last_trade_ts FROM ingest_cursor WHERE ticker = ?", (ticker,)
    ).fetchone()
    if row and row[0]:
        try:
            dt = datetime.fromisoformat(row[0])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def update_cursor(conn: sqlite3.Connection, ticker: str,
                  last_ts: datetime, last_tracking_id: Optional[int],
                  n_new_rows: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_cursor (ticker, last_trade_ts, last_tracking_id, rows_total, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
          last_trade_ts    = excluded.last_trade_ts,
          last_tracking_id = excluded.last_tracking_id,
          rows_total       = ingest_cursor.rows_total + excluded.rows_total,
          updated_at       = excluded.updated_at
        """,
        (
            ticker, last_ts.isoformat(), last_tracking_id, n_new_rows,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def update_scraper_state(conn: sqlite3.Connection, total_rows: int,
                         total_tickers: int, error: Optional[str] = None) -> None:
    conn.execute(
        """
        UPDATE institutional_scraper_state
           SET last_poll_at      = ?,
               last_provider     = 'uw',
               last_row_count    = ?,
               last_ticker_count = ?,
               last_error        = ?,
               updated_at        = ?
         WHERE id = 1
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            total_rows, total_tickers, error,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ─── Per-ticker ingest ───────────────────────────────────────────────────────

def ingest_ticker(conn: sqlite3.Connection, ticker: str,
                  start_dt: datetime, end_dt: datetime,
                  notional_floor: float = NOTIONAL_FLOOR,
                  inner_progress_cb: Optional[Callable[[int, int], None]] = None,
                  update_cursor_after: bool = True,
                  ) -> int:
    ticker = ticker.upper().strip()
    if start_dt >= end_dt:
        return 0

    cur_day = start_dt.date()
    end_day = end_dt.date()
    n_days = max((end_day - cur_day).days + 1, 1)
    total_inserted = 0
    last_ts: Optional[datetime] = None
    last_tracking_id: Optional[int] = None

    day_idx = 0
    while cur_day <= end_day:
        try:
            items = fetch_darkpool_day(ticker, cur_day, min_premium=notional_floor)
        except Exception as e:
            print(f"    ⚠ {ticker} {cur_day}: {e}")
            cur_day += timedelta(days=1)
            day_idx += 1
            if inner_progress_cb:
                try: inner_progress_cb(day_idx, n_days)
                except Exception: pass
            continue

        if items:
            items.sort(key=lambda x: x.get("executed_at", ""))
            rows = []
            prev_price: Optional[float] = None
            for it in items:
                price = _f(it.get("price")) or 0.0
                bid = _f(it.get("nbbo_bid"))
                ask = _f(it.get("nbbo_ask"))
                side = lee_ready_classify(price, bid, ask, prev_price)
                rows.append(transform_row(it, side))
                prev_price = price

            n = upsert_rows(conn, rows)
            total_inserted += n

            if rows:
                last_row = rows[-1]
                try:
                    last_ts = datetime.fromisoformat(last_row["trade_ts"])
                except ValueError:
                    last_ts = end_dt
                last_tracking_id = last_row.get("tracking_id")

        day_idx += 1
        if inner_progress_cb:
            try: inner_progress_cb(day_idx, n_days)
            except Exception: pass

        cur_day += timedelta(days=1)

    final_ts = last_ts if last_ts else end_dt
    if update_cursor_after:
        update_cursor(
            conn, ticker,
            max(final_ts, end_dt - timedelta(seconds=1)),
            last_tracking_id, total_inserted,
        )
    return total_inserted


# ─── Top-level entrypoint ────────────────────────────────────────────────────

def run_institutional_ingest(
    tickers: list[str],
    days_back: int = 30,
    db_path: Path = DB_PATH,
    notional_floor: float = NOTIONAL_FLOOR,
    use_cursor: bool = True,
    progress_cb: Optional[Callable[[str, int, int, float], None]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, int]:
    """
    Main entrypoint, called from UI.
    Returns: dict mapping ticker → rows inserted.
    """
    conn = init_db(db_path)

    historical_mode = (start_date is not None or end_date is not None)
    if historical_mode:
        use_cursor = False
        explicit_start = (datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
                          if start_date else datetime.now(timezone.utc) - timedelta(days=days_back))
        explicit_end = (datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
                        if end_date else datetime.now(timezone.utc))
        end_dt = explicit_end
        default_start = explicit_start
    else:
        end_dt = datetime.now(timezone.utc)
        default_start = end_dt - timedelta(days=days_back)

    results: dict[str, int] = {}
    n_tickers = len(tickers)
    last_error: Optional[str] = None

    try:
        for idx, ticker in enumerate(tickers):
            ticker = ticker.upper().strip()

            if use_cursor:
                cursor_ts = get_cursor(conn, ticker)
                if cursor_ts is not None and cursor_ts.tzinfo is None:
                    cursor_ts = cursor_ts.replace(tzinfo=timezone.utc)
                start_dt = cursor_ts if cursor_ts else default_start
            else:
                start_dt = default_start

            if start_dt >= end_dt:
                results[ticker] = 0
                if progress_cb:
                    progress_cb(ticker, idx, n_tickers, 1.0)
                continue

            try:
                def _icb(done: int, total: int) -> None:
                    if progress_cb and total > 0:
                        progress_cb(ticker, idx, n_tickers, done / total)

                n = ingest_ticker(conn, ticker, start_dt, end_dt,
                                  notional_floor=notional_floor,
                                  inner_progress_cb=_icb,
                                  update_cursor_after=not historical_mode)
                results[ticker] = n
            except Exception as e:
                last_error = f"{ticker}: {e}"
                print(f"  ⚠ {ticker} failed: {e}")
                results[ticker] = 0

            if progress_cb:
                progress_cb(ticker, idx, n_tickers, 1.0)
            time.sleep(0.05)

    finally:
        total_rows = sum(results.values())
        active = sum(1 for v in results.values() if v > 0)
        update_scraper_state(conn, total_rows, active, last_error)
        conn.close()

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UW darkpool flow ingest")
    parser.add_argument("--tickers-file", default="tickers.txt")
    parser.add_argument("--days-back", type=int, default=30,
                        help="lookback window if --start-date not given (default 30)")
    parser.add_argument("--start-date",
                        help="explicit window start, YYYY-MM-DD (overrides --days-back, bypasses cursor)")
    parser.add_argument("--end-date",
                        help="explicit window end, YYYY-MM-DD (default = today)")
    parser.add_argument("--notional-floor", type=float, default=NOTIONAL_FLOOR)
    parser.add_argument("--no-cursor", action="store_true")
    parser.add_argument("--ticker", help="single ticker (overrides tickers-file)")
    args = parser.parse_args()

    sd = date.fromisoformat(args.start_date) if args.start_date else None
    ed = date.fromisoformat(args.end_date) if args.end_date else None

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [
            line.strip().upper()
            for line in Path(args.tickers_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if sd or ed:
        window_str = f"{sd or '...'} → {ed or 'today'}"
    else:
        window_str = f"last {args.days_back}d"
    print(f"UW darkpool ingest: {len(tickers)} tickers, {window_str}, "
          f"floor ${args.notional_floor:,.0f}")

    def cb(t: str, i: int, n: int, frac: float) -> None:
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{i+1}/{n}] {t:8s} [{bar}] {frac*100:.0f}%   ",
              end="", flush=True)

    res = run_institutional_ingest(
        tickers, days_back=args.days_back,
        notional_floor=args.notional_floor,
        use_cursor=not args.no_cursor, progress_cb=cb,
        start_date=sd, end_date=ed,
    )
    print()
    total = sum(res.values())
    active = sum(1 for v in res.values() if v > 0)
    print(f"Done: {total:,} new prints across {active} tickers.")
