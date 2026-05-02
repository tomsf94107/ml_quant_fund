# features/massive_client.py
# ─────────────────────────────────────────────────────────────────────────────
# Massive (formerly Polygon.io) Stocks API client.
# Drop-in-ish replacement for yfinance for OHLCV data.
#
# Uses the Stocks Developer plan ($79/mo). Single API key (MASSIVE_API_KEY)
# already configured in env from prior Options plan subscription.
#
# API surface mimics yfinance where reasonable so call sites change minimally:
#   yf.download(ticker, start=, end=, auto_adjust=True)
#       -> mc.download(ticker, start=, end=, auto_adjust=True)
#   yf.download([t1, t2, ...], start=, end=)
#       -> mc.download([t1, t2, ...], start=, end=)
#   yf.Ticker(ticker).info
#       -> mc.get_ticker_info(ticker)
#
# Returns pandas DataFrames with the same column shape as yfinance:
#   single ticker:  ['Open', 'High', 'Low', 'Close', 'Volume']
#   multi ticker:   MultiIndex columns same as yf.download(list)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Union, List

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"
from config.keys import MASSIVE_API_KEY as API_KEY

# Index symbols not available on Massive Stocks Developer plan.
# These auto-route to yfinance.
INDEX_SYMBOLS = {
    "^VIX", "^VIX3M", "^VIX9D",        # CBOE volatility indices
    "^TNX", "^TYX", "^FVX", "^IRX",    # Treasury yield indices
    "^GSPC", "^DJI", "^IXIC", "^RUT",  # Major US equity indices
    "^FTSE", "^N225", "^HSI", "^GDAXI", # International indices
    "ES=F", "NQ=F", "YM=F", "RTY=F",   # Equity index futures
    "CL=F", "GC=F", "SI=F", "HG=F",    # Commodity futures
    "DX-Y.NYB",                         # US Dollar Index (Yahoo-only format)
}


def _is_index(symbol):
    """Return True if symbol should route to yfinance instead of Massive."""
    if symbol in INDEX_SYMBOLS:
        return True
    if symbol.startswith("^"):
        return True
    if symbol.endswith("=F"):
        return True
    if symbol.endswith(".NYB"):  # Yahoo Board (currencies, commodities)
        return True
    return False


def _check_key():
    if not API_KEY:
        raise RuntimeError(
            "MASSIVE_API_KEY (or POLYGON_API_KEY) not set in environment. "
            "Subscribe to Massive Stocks Developer plan and set the key."
        )


def _normalize_date(d) -> str:
    """Accept str/date/datetime, return 'YYYY-MM-DD'."""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y-%m-%d")
    raise ValueError(f"Cannot normalize date: {type(d)} {d}")


def _request_with_retry(url, params, timeout=15, max_retries=6):
    """GET with retry on 429/5xx and exponential backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or 500 <= r.status_code < 600:
                wait = 2 ** attempt
                log.warning(f"massive.retry status={r.status_code} attempt={attempt} wait={wait}s")
                time.sleep(wait)
                continue
            log.error(f"massive.error status={r.status_code} url={url[:80]} body={r.text[:200]}")
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            wait = 2 ** attempt
            log.warning(f"massive.network attempt={attempt} wait={wait}s err={e}")
            time.sleep(wait)
    raise RuntimeError(f"Massive API failed after {max_retries} retries: {last_err}")


def download(
    tickers,
    start=None,
    end=None,
    period=None,
    interval="1d",
    auto_adjust=True,
    progress=False,
    **kwargs,
):
    """
    Download OHLCV data. Drop-in replacement for yfinance.download().

    HYBRID ROUTING:
    - Index symbols (^VIX, ^GSPC, futures like ES=F) auto-route to yfinance
    - Stocks/ETFs route to Massive
    - Mixed list: each ticker fetched from correct source, merged into
      yfinance-shape MultiIndex DataFrame

    Returns a DataFrame with:
      - single ticker: columns ['Open', 'High', 'Low', 'Close', 'Volume']
      - multi ticker: MultiIndex columns [(field, ticker), ...]
    """
    if period and not start:
        end = end or date.today()
        if isinstance(end, str):
            end = pd.Timestamp(end).date()
        start = _resolve_period(end, period)

    if not start or not end:
        raise ValueError("download requires either (start, end) or period")

    start_str = _normalize_date(start)
    end_str = _normalize_date(end)

    if isinstance(tickers, str):
        # Single ticker — route based on type
        if _is_index(tickers):
            return _download_yfinance(tickers, start_str, end_str, interval, auto_adjust)
        else:
            _check_key()
            mult, span = _interval_to_massive(interval)
            return _download_single(tickers, start_str, end_str, mult, span, auto_adjust)
    else:
        # List of tickers — split, fetch each from correct source, merge
        indices = [t for t in tickers if _is_index(t)]
        stocks = [t for t in tickers if not _is_index(t)]

        frames = {}

        # Fetch indices from yfinance
        for t in indices:
            try:
                df = _download_yfinance(t, start_str, end_str, interval, auto_adjust)
                if not df.empty:
                    frames[t] = df
            except Exception as e:
                log.warning(f"yfinance failed for index {t}: {e}")

        # Fetch stocks from Massive
        if stocks:
            _check_key()
            mult, span = _interval_to_massive(interval)
            for t in stocks:
                try:
                    df = _download_single(t, start_str, end_str, mult, span, auto_adjust)
                    if not df.empty:
                        frames[t] = df
                except Exception as e:
                    log.warning(f"massive failed for {t}: {e}")

        if not frames:
            return pd.DataFrame()

        # Merge into yfinance-shape MultiIndex
        out = pd.concat(frames, axis=1)
        out.columns = out.columns.swaplevel(0, 1)
        out = out.sort_index(axis=1)
        return out


def _download_yfinance(ticker, start_str, end_str, interval, auto_adjust):
    """Fallback fetch via yfinance for index symbols not in Massive."""
    import yfinance as yf
    df = yf.download(ticker, start=start_str, end=end_str,
                     interval=interval, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    # Flatten MultiIndex if yfinance returned one
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    if interval == "1d":
        df.index = df.index.normalize()
    return df


def _resolve_period(end_date, period):
    """Convert yfinance period string to start date."""
    period = period.lower()
    today = end_date if isinstance(end_date, date) else pd.Timestamp(end_date).date()
    if period.endswith("d"):
        days = int(period[:-1])
        return today - timedelta(days=days)
    if period.endswith("mo"):
        months = int(period[:-2])
        return today - timedelta(days=months * 31)
    if period.endswith("y"):
        years = int(period[:-1])
        return today - timedelta(days=years * 366)
    raise ValueError(f"Unsupported period: {period}")


def _interval_to_massive(interval):
    """Convert yfinance interval to Massive (multiplier, timespan)."""
    iv = interval.lower()
    mapping = {
        "1m":  (1,  "minute"),
        "2m":  (2,  "minute"),
        "5m":  (5,  "minute"),
        "15m": (15, "minute"),
        "30m": (30, "minute"),
        "60m": (1,  "hour"),
        "1h":  (1,  "hour"),
        "1d":  (1,  "day"),
        "1wk": (1,  "week"),
        "1mo": (1,  "month"),
    }
    if iv not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[iv]


def _download_single(ticker, start_str, end_str, mult, span, auto_adjust):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start_str}/{end_str}"
    params = {
        "adjusted": "true" if auto_adjust else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }
    try:
        data = _request_with_retry(url, params)
    except Exception as e:
        # Massive failed all retries — try yfinance fallback for this ticker
        log.warning(f"massive failed for {ticker} after retries, falling back to yfinance: {e}")
        return _download_yfinance(ticker, start_str, end_str,
                                  f"{mult}{span[0]}" if span != "day" else "1d",
                                  auto_adjust)

    if data.get("resultsCount", 0) == 0 or not data.get("results"):
        log.warning(f"massive: no data for {ticker} {start_str} to {end_str}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    rows = []
    for bar in data["results"]:
        rows.append({
            "Date":   pd.Timestamp(bar["t"], unit="ms", tz="UTC").tz_localize(None),
            "Open":   bar["o"],
            "High":   bar["h"],
            "Low":    bar["l"],
            "Close":  bar["c"],
            "Volume": bar["v"],
        })

    df = pd.DataFrame(rows).set_index("Date")
    if span == "day":
        df.index = df.index.normalize()
    df.index.name = "Date"
    return df


def _download_batch(tickers, start_str, end_str, mult, span, auto_adjust):
    """
    Download multiple tickers. Returns yfinance-style MultiIndex DataFrame.
    Columns: [(field, ticker), ...] where field is Open/High/Low/Close/Volume.
    """
    frames = {}
    for t in tickers:
        try:
            df = _download_single(t, start_str, end_str, mult, span, auto_adjust)
            if not df.empty:
                frames[t] = df
        except Exception as e:
            log.warning(f"massive batch: failed to fetch {t}: {e}")
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out.columns = out.columns.swaplevel(0, 1)
    out = out.sort_index(axis=1)
    return out


def get_ticker_info(ticker):
    """
    Replacement for yf.Ticker(ticker).info.
    """
    _check_key()
    url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
    params = {"apiKey": API_KEY}
    try:
        data = _request_with_retry(url, params)
        r = data.get("results", {}) or {}
        return {
            "symbol":              r.get("ticker", ticker),
            "longName":            r.get("name"),
            "shortName":           r.get("name"),
            "marketCap":           r.get("market_cap"),
            "sharesOutstanding":   r.get("share_class_shares_outstanding") or r.get("weighted_shares_outstanding"),
            "currency":            r.get("currency_name"),
            "exchange":            r.get("primary_exchange"),
            "sector":              None,
            "industry":            None,
            "website":             r.get("homepage_url"),
            "description":         r.get("description"),
        }
    except Exception as e:
        log.warning(f"massive get_ticker_info failed for {ticker}: {e}")
        return {}


def get_short_interest(ticker):
    """
    Short interest data. NOT in Massive Stocks Developer plan.
    Falls back to yfinance for this specific data point.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "shortPercentOfFloat":   info.get("shortPercentOfFloat"),
            "shortRatio":            info.get("shortRatio"),
            "sharesShort":           info.get("sharesShort"),
            "sharesShortPriorMonth": info.get("sharesShortPriorMonth"),
        }
    except Exception as e:
        log.warning(f"yfinance short interest failed for {ticker}: {e}")
        return {}


def get_prev_close(ticker):
    """Convenience: previous close (intraday widget needs this)."""
    _check_key()
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev"
    params = {"adjusted": "true", "apiKey": API_KEY}
    try:
        data = _request_with_retry(url, params)
        results = data.get("results", [])
        if results:
            return results[0].get("c")
    except Exception as e:
        log.warning(f"prev_close failed for {ticker}: {e}")
    return None
# ─────────────────────────────────────────────────────────────────────────────
# APPEND THIS BLOCK to the END of features/massive_client.py
# Adds /v3/trades and /v3/quotes methods used by features/institutional_ingest.py
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime as _dt, timezone as _tz

TRADES_PAGE_LIMIT = 50000  # Massive max per request
QUOTES_PAGE_LIMIT = 50000


def _to_ns(dt: _dt) -> int:
    """Convert UTC datetime to nanosecond timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc)
    return int(dt.timestamp() * 1e9)


def list_trades(ticker: str, start: _dt, end: _dt) -> List[dict]:
    """
    Fetch all trades for ticker in [start, end) UTC window via Massive /v3/trades.

    Pagination handled via next_url. Returns list of trade dicts with at minimum:
        sip_timestamp (int ns), price, size, exchange, conditions
    """
    _check_key()

    start_ns = _to_ns(start)
    end_ns = _to_ns(end)

    url = f"{BASE_URL}/v3/trades/{ticker.upper()}"
    params = {
        "timestamp.gte": start_ns,
        "timestamp.lt":  end_ns,
        "limit":         TRADES_PAGE_LIMIT,
        "order":         "asc",
        "sort":          "timestamp",
        "apiKey":        API_KEY,
    }

    all_trades: List[dict] = []
    page_count = 0
    max_pages = 1000  # safety cap; ~50M trades

    while page_count < max_pages:
        data = _request_with_retry(url, params, timeout=30)
        if not data:
            break
        results = data.get("results", []) or []
        all_trades.extend(results)
        page_count += 1

        next_url = data.get("next_url")
        if not next_url or not results:
            break
        # next_url already encodes filter params; we just append the API key
        url = next_url
        params = {"apiKey": API_KEY}

    return all_trades


def get_quote_at(ticker: str, sip_ts_ns: int,
                 lookback_ns: int = 60_000_000_000) -> Optional[dict]:
    """
    Fetch the most recent NBBO quote at-or-before sip_ts_ns (nanoseconds).

    Looks back up to `lookback_ns` (default 60s) for an active quote.
    Returns dict with bid_price / ask_price (and other fields) or None.
    """
    _check_key()

    url = f"{BASE_URL}/v3/quotes/{ticker.upper()}"
    params = {
        "timestamp.lte": sip_ts_ns,
        "timestamp.gte": sip_ts_ns - lookback_ns,
        "limit":         1,
        "order":         "desc",
        "sort":          "timestamp",
        "apiKey":        API_KEY,
    }

    data = _request_with_retry(url, params, timeout=10)
    if not data:
        return None
    results = data.get("results", []) or []
    return results[0] if results else None
