"""
features/uw_client.py -- Centralized Unusual Whales (UW) API client.

ALL UW API access must go through this module. Enforces:
  - Market-hours gate (9:30 AM - 4:00 PM ET, Mon-Fri)
  - Daily rate-limit awareness (20k/day, with safety buffer)
  - Retry with exponential backoff on 429 / 5xx
  - Structured logging (all events prefixed `uw.`)
  - Optional DB fallback when market is closed or call fails

Usage:
    from features.uw_client import uw_get

    # Inside market hours: hits UW. Outside: returns None (or db_fallback).
    data = uw_get("/api/darkpool/AAPL", params={"date": "2026-04-24"})

    # With DB fallback for retrain / off-hours jobs:
    data = uw_get(
        "/api/darkpool/AAPL",
        db_fallback=lambda: load_dark_pool_from_sqlite("AAPL"),
    )

    # Ingest scripts that MUST run off-hours (daily_uw_snapshot.py, etl_earnings.py):
    data = uw_get("/api/earnings/AAPL", allow_outside_market=True)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, time as dtime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

# Host only -- endpoint paths include /api/ prefix.
UW_BASE_URL = os.getenv("UW_BASE_URL", "https://api.unusualwhales.com").rstrip("/")
UW_API_KEY = os.getenv("UW_API_KEY")
ET = ZoneInfo("America/New_York")

MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

DAILY_LIMIT = 20_000
SAFETY_BUFFER = 500  # stop calling when within this many of the daily limit


def is_market_open(now: Optional[datetime] = None) -> bool:
    """True if US equity regular session is open. Does NOT account for holidays."""
    if now is None:
        now = datetime.now(ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    if now.weekday() >= 5:  # Sat, Sun
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


class UWDailyLimitError(RuntimeError):
    """Raised internally when the daily UW call budget is exhausted."""


class UWClient:
    def __init__(self) -> None:
        self._calls_today: int = 0
        self._count_date = None
        self._session = requests.Session()
        if UW_API_KEY:
            self._session.headers.update({
                "Authorization": f"Bearer {UW_API_KEY}",
                "UW-CLIENT-API-ID": "100001",
                "Accept": "application/json",
            })

    @property
    def calls_today(self) -> int:
        self._roll_date()
        return self._calls_today

    def _roll_date(self) -> None:
        today = datetime.now(ET).date()
        if self._count_date != today:
            self._count_date = today
            self._calls_today = 0

    def _reserve_call(self) -> None:
        self._roll_date()
        if self._calls_today >= DAILY_LIMIT - SAFETY_BUFFER:
            raise UWDailyLimitError(
                f"UW daily budget near limit: {self._calls_today}/{DAILY_LIMIT}"
            )
        self._calls_today += 1

    def get(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        *,
        db_fallback: Optional[Callable[[], Any]] = None,
        max_retries: int = 3,
        timeout: float = 15.0,
        allow_outside_market: bool = False,
    ) -> Any:
        # 1) Market-hours gate
        if not allow_outside_market and not is_market_open():
            logger.info(
                "uw.skip reason=market_closed endpoint=%s fallback=%s",
                endpoint, db_fallback is not None,
            )
            return db_fallback() if db_fallback else None

        # 2) Daily budget gate
        try:
            self._reserve_call()
        except UWDailyLimitError as e:
            logger.warning("uw.skip reason=daily_limit endpoint=%s err=%s", endpoint, e)
            return db_fallback() if db_fallback else None

        # 3) Retry loop (429 / 5xx / network errors)
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{UW_BASE_URL}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                r = self._session.get(url, params=params, timeout=timeout)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    wait = 2 ** attempt
                    logger.warning(
                        "uw.retry status=%d endpoint=%s attempt=%d wait=%ds",
                        r.status_code, endpoint, attempt, wait,
                    )
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                logger.info(
                    "uw.ok endpoint=%s calls_today=%d",
                    endpoint, self._calls_today,
                )
                return r.json()
            except requests.RequestException as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    "uw.err endpoint=%s attempt=%d wait=%ds err=%s",
                    endpoint, attempt, wait, e,
                )
                time.sleep(wait)

        logger.error("uw.fail endpoint=%s err=%s", endpoint, last_err)
        return db_fallback() if db_fallback else None


# Module-level singleton
_client: Optional[UWClient] = None


def get_client() -> UWClient:
    global _client
    if _client is None:
        _client = UWClient()
    return _client


def uw_get(
    endpoint: str,
    params: Optional[dict] = None,
    *,
    db_fallback: Optional[Callable[[], Any]] = None,
    max_retries: int = 3,
    timeout: float = 15.0,
    allow_outside_market: bool = False,
) -> Any:
    """Single entry point for all UW GET calls.

    Returns None (or db_fallback() result) when:
      - market is closed (and allow_outside_market is False),
      - daily budget is exhausted,
      - all retries fail.
    """
    return get_client().get(
        endpoint,
        params,
        db_fallback=db_fallback,
        max_retries=max_retries,
        timeout=timeout,
        allow_outside_market=allow_outside_market,
    )
