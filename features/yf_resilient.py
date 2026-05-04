"""
features/yf_resilient.py
─────────────────────────────────────────────────────────────────────────────
Resilient wrapper for yfinance calls.

Why this exists:
  - yfinance + DNS failures can SEGFAULT the Python process (no traceback)
  - This killed Pipeline C on Apr 28 + 29 with no error message
  - Native (C-level) crashes bypass Python's exception handling

Strategy:
  1. PRE-CHECK DNS to query1.finance.yahoo.com before any yfinance call
  2. If DNS fails → return None immediately (don't even invoke yfinance)
  3. If DNS succeeds → call yfinance with retry logic
  4. Cache DNS result for 60 seconds (avoid hammering DNS resolver)

Usage:
  from features.yf_resilient import safe_ticker_info, safe_ticker_options, \
                                     safe_ticker_history, safe_yf_download, \
                                     yahoo_dns_check

  info = safe_ticker_info("AAPL")            # dict or None
  expiries = safe_ticker_options("AAPL")      # list or []
  hist = safe_ticker_history("AAPL", period="1d")  # DataFrame or None
  df = safe_yf_download(["^VIX"], start="2026-01-01", end="2026-04-29")
"""
from __future__ import annotations

import logging
import socket
import time
from functools import lru_cache
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# DNS check cache (avoid hammering resolver — cache for 60 sec)
_dns_cache: dict[str, tuple[bool, float]] = {}
_DNS_CACHE_TTL = 60.0  # seconds

# Max retries for yfinance calls when DNS is healthy
_MAX_YF_RETRIES = 3
_RETRY_BACKOFF = 1.0  # 1s, 2s, 4s


def yahoo_dns_check(host: str = "query1.finance.yahoo.com",
                    timeout: float = 2.0) -> bool:
    """
    Check if a Yahoo Finance host is DNS-resolvable.
    Cached for 60 seconds to avoid resolver pressure.
    Returns True if resolvable, False otherwise.
    """
    now = time.time()
    cached = _dns_cache.get(host)
    if cached:
        ok, ts = cached
        if now - ts < _DNS_CACHE_TTL:
            return ok

    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        _dns_cache[host] = (True, now)
        return True
    except (socket.gaierror, socket.timeout, OSError) as e:
        log.warning(f"yfinance DNS pre-check failed for {host}: {e}")
        _dns_cache[host] = (False, now)
        return False
    finally:
        socket.setdefaulttimeout(None)


def _retry_yf_call(func, *args, label: str = "yf_call", **kwargs):
    """
    Generic retry wrapper for yfinance calls.
    Catches ALL exceptions and returns None if all retries fail.
    Pre-checks DNS to avoid C-level segfaults.
    """
    if not yahoo_dns_check():
        log.warning(f"{label}: skipped (Yahoo DNS unreachable)")
        return None

    last_exc = None
    for attempt in range(_MAX_YF_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            wait = _RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"{label}: attempt {attempt+1}/{_MAX_YF_RETRIES} failed: {e}")
            if attempt < _MAX_YF_RETRIES - 1:
                time.sleep(wait)

    log.warning(f"{label}: all {_MAX_YF_RETRIES} retries failed: {last_exc}")
    return None


def safe_ticker_info(ticker: str) -> Optional[dict]:
    """
    Replacement for yf.Ticker(ticker).info
    Returns None if DNS fails or all retries fail.
    """
    def _call():
        import yfinance as yf
        return yf.Ticker(ticker).info

    return _retry_yf_call(_call, label=f"yf.Ticker({ticker}).info")


def safe_ticker_options(ticker: str) -> list:
    """
    Replacement for yf.Ticker(ticker).options
    Returns [] if DNS fails or all retries fail.
    """
    def _call():
        import yfinance as yf
        return list(yf.Ticker(ticker).options or [])

    result = _retry_yf_call(_call, label=f"yf.Ticker({ticker}).options")
    return result if result is not None else []


def safe_ticker_history(ticker: str, period: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
    """
    Replacement for yf.Ticker(ticker).history(period=...)
    Returns None if DNS fails or all retries fail.
    """
    def _call():
        import yfinance as yf
        return yf.Ticker(ticker).history(period=period, **kwargs)

    return _retry_yf_call(_call, label=f"yf.Ticker({ticker}).history(period={period})")


def safe_ticker_option_chain(ticker: str, expiry: str):
    """
    Replacement for yf.Ticker(ticker).option_chain(expiry)
    Returns None if DNS fails or all retries fail.
    """
    def _call():
        import yfinance as yf
        return yf.Ticker(ticker).option_chain(expiry)

    return _retry_yf_call(_call, label=f"yf.Ticker({ticker}).option_chain({expiry})")


def safe_yf_download(tickers, start=None, end=None, **kwargs) -> Optional[pd.DataFrame]:
    """
    Replacement for yf.download(tickers, ...)
    Returns None if DNS fails or all retries fail.
    """
    def _call():
        import yfinance as yf
        return yf.download(tickers, start=start, end=end, **kwargs)

    return _retry_yf_call(_call, label=f"yf.download({tickers})")
