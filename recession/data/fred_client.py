"""
Low-level FRED / ALFRED API client.

Responsibilities:
- Authenticated HTTP calls to api.stlouisfed.org (v1)
- Local file cache (JSON) to respect 120 req/min rate limit
- Token-bucket rate limiter
- Exponential-backoff retry on transient failures
- Vintage-aware queries: returns full revision history for revisable series

Does NOT:
- Know anything about our schema
- Transform values (frequency conversion, detrending, etc. live in series_specs.py)
- Write to the DB (that's ingest.py)

API reference:
- Observations endpoint: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- Vintage handling: realtime_start / realtime_end parameters on observations endpoint.
  With realtime_start=earliest and realtime_end=latest, output_type=2 returns
  one row per (observation_date, vintage_period) where the value differed
  across vintages — i.e. the full revision history.

Usage:
    client = FredClient.from_env()
    obs = client.observations("T10Y3M", start="2020-01-01")
    obs_with_vintages = client.observations_all_vintages("USSLIND", start="2020-01-01")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# API-key log masking
# -----------------------------------------------------------------------------
# The `requests` library uses urllib3's connection pool, which logs the full
# URL (including query string) at DEBUG level. That leaks api_key into logs
# whenever DEBUG logging is enabled. This filter scrubs api_key=<value> from
# all log messages that flow through any handler.
#
# Implementation note: filters attached to a Logger only run for records
# originating at that logger; they do NOT run for records propagating up from
# child loggers. To catch records from urllib3 etc., the filter must be
# attached to the HANDLERS that actually emit output, not to loggers.

class _ApiKeyMaskFilter(logging.Filter):
    """Logging filter that redacts api_key= query parameters from log messages."""
    _PATTERN = re.compile(r"(api_key=)([0-9a-fA-F]{8,})", re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._PATTERN.sub(r"\1<REDACTED>", record.msg)
        if record.args:
            args_iter = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(
                self._PATTERN.sub(r"\1<REDACTED>", a) if isinstance(a, str) else a
                for a in args_iter
            )
        return True


_API_KEY_MASK_FILTER = _ApiKeyMaskFilter()


def _install_api_key_mask() -> None:
    """Install the mask filter on every handler currently attached to the root
    logger. Idempotent — re-attaching the same filter is a no-op.
    Should be called after logging.basicConfig() or any handler setup.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        # Don't attach if our exact instance is already there
        if _API_KEY_MASK_FILTER not in handler.filters:
            handler.addFilter(_API_KEY_MASK_FILTER)


# Attach immediately for any handlers already configured at import time.
_install_api_key_mask()


# Also install a hook so future basicConfig/handler additions get masked.
# We monkey-patch addHandler on the root logger to auto-install the filter.
_original_addHandler = logging.Logger.addHandler

def _patched_addHandler(self, hdlr):
    _original_addHandler(self, hdlr)
    if self is logging.getLogger() or self.name == "root":
        if _API_KEY_MASK_FILTER not in hdlr.filters:
            hdlr.addFilter(_API_KEY_MASK_FILTER)

if logging.Logger.addHandler is not _patched_addHandler:
    logging.Logger.addHandler = _patched_addHandler

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# FRED rate limit is 120 req/min per key. We stay well under to avoid 429s.
DEFAULT_REQUESTS_PER_MINUTE = 100

# Default cache directory: <repo_root>/recession/cache/fred
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "fred"


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class FredApiError(Exception):
    """Raised when FRED API returns an error response."""


class FredSeriesNotFoundError(FredApiError):
    """Raised when a series ID is not found in FRED."""


# -----------------------------------------------------------------------------
# Rate limiter (simple token bucket)
# -----------------------------------------------------------------------------

@dataclass
class TokenBucket:
    """Token-bucket rate limiter. Sleeps when out of tokens."""
    rate_per_minute: int
    capacity: int = field(init=False)
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.capacity = self.rate_per_minute
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def acquire(self, tokens: int = 1) -> None:
        """Block until `tokens` are available, then consume them."""
        while True:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            # Compute sleep time until next token is available
            tokens_needed = tokens - self.tokens
            sleep_seconds = (tokens_needed / self.rate_per_minute) * 60.0
            sleep_seconds = max(sleep_seconds, 0.1)
            logger.debug("Rate limit: sleeping %.2fs for %d tokens", sleep_seconds, tokens_needed)
            time.sleep(sleep_seconds)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill = (elapsed / 60.0) * self.rate_per_minute
        self.tokens = min(self.capacity, self.tokens + refill)
        self.last_refill = now


# -----------------------------------------------------------------------------
# File-based cache
# -----------------------------------------------------------------------------

class JsonFileCache:
    """
    Simple JSON file cache keyed on a hash of the request URL.

    For FRED data, the cache is effectively permanent for past observations —
    historical vintages don't change retroactively. To force a refresh we
    delete the cache file (or pass force_refresh=True).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str, params: dict) -> str:
        """Hash the URL + sorted params (excluding api_key) into a stable key."""
        safe_params = {k: v for k, v in params.items() if k != "api_key"}
        canonical = url + "?" + urlencode(sorted(safe_params.items()))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _path(self, url: str, params: dict) -> Path:
        # Use the FRED endpoint name as a folder for human-browsability
        endpoint = url.replace(FRED_BASE_URL + "/", "").replace("/", "_")
        return self.cache_dir / endpoint / f"{self._key(url, params)}.json"

    def get(self, url: str, params: dict) -> Optional[dict]:
        path = self._path(url, params)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt cache file %s; deleting", path)
            path.unlink(missing_ok=True)
            return None

    def put(self, url: str, params: dict, response: dict) -> None:
        path = self._path(url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response, indent=2))

    def invalidate(self, url: str, params: dict) -> None:
        path = self._path(url, params)
        path.unlink(missing_ok=True)

    def stats(self) -> dict:
        if not self.cache_dir.exists():
            return {"n_files": 0, "total_bytes": 0}
        files = list(self.cache_dir.rglob("*.json"))
        return {
            "n_files": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
            "cache_dir": str(self.cache_dir),
        }


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------

class FredClient:
    """
    FRED + ALFRED API client.

    Construct via from_env() to load FRED_API_KEY from environment / .env file,
    or pass api_key directly for testing.
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[Path] = None,
        rate_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> None:
        if not api_key or len(api_key) != 32:
            raise ValueError(f"FRED_API_KEY must be a 32-char hex string; got len={len(api_key) if api_key else 0}")
        self.api_key       = api_key
        self.cache         = JsonFileCache(cache_dir or DEFAULT_CACHE_DIR)
        self.bucket        = TokenBucket(rate_per_minute)
        self.max_retries   = max_retries
        self.timeout       = timeout_seconds
        self._session      = requests.Session()
        # FRED returns gzip; requests handles it automatically with this header
        self._session.headers.update({
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "ml_quant_fund-recession/0.1.0",
        })

    @classmethod
    def from_env(cls, **kwargs: Any) -> "FredClient":
        """Construct client with FRED_API_KEY from env (loads .env if present)."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.debug("python-dotenv not installed; relying on environment only")

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FRED_API_KEY not found. Set it in .env or export it. "
                "Get a key at https://fredaccount.stlouisfed.org/apikeys"
            )
        return cls(api_key=api_key, **kwargs)

    # -------------------------------------------------------------------------
    # Internal: HTTP with cache, rate-limit, retry
    # -------------------------------------------------------------------------
    def _request(
        self,
        endpoint: str,
        params: dict,
        force_refresh: bool = False,
    ) -> dict:
        """Make a GET request with caching, rate-limiting, and retries."""
        # Enforce required defaults
        params = {**params, "api_key": self.api_key, "file_type": "json"}
        url = f"{FRED_BASE_URL}/{endpoint}"

        # Cache lookup
        if not force_refresh:
            cached = self.cache.get(url, params)
            if cached is not None:
                logger.debug("Cache hit: %s %s", endpoint, params.get("series_id"))
                return cached

        # Rate-limited request with retry
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self.bucket.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    self.cache.put(url, params, data)
                    return data
                if resp.status_code == 400:
                    # FRED returns 400 for unknown series IDs
                    err = resp.json().get("error_message", "(no error_message)")
                    if "does not exist" in err.lower() or "not found" in err.lower():
                        raise FredSeriesNotFoundError(
                            f"Series '{params.get('series_id')}' not found: {err}"
                        )
                    raise FredApiError(f"400 Bad Request: {err}")
                if resp.status_code == 429:
                    # Rate limited; back off exponentially
                    wait = 2 ** attempt
                    logger.warning("FRED 429, sleeping %ds (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("FRED %d, retrying in %ds (attempt %d)",
                                   resp.status_code, wait, attempt + 1)
                    time.sleep(wait)
                    continue
                raise FredApiError(f"FRED returned {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("Network error: %s, retrying in %ds", e, wait)
                time.sleep(wait)

        raise FredApiError(
            f"FRED request failed after {self.max_retries} retries: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Public: single-vintage observations (latest revised data)
    # -------------------------------------------------------------------------
    def observations(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        frequency: Optional[str] = None,
        aggregation_method: str = "avg",
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Latest-vintage observations. One row per observation_date.

        For non-revisable series (yields, spreads, prices), this is the only
        information needed — vintage_date is set to observation_date + publication_lag
        downstream by the ingester.

        Args:
            series_id: FRED series ID (e.g. 'T10Y3M').
            start, end: ISO date strings 'YYYY-MM-DD'.
            frequency: 'd', 'w', 'm', 'q', 'a'. None = native frequency.
            aggregation_method: 'avg', 'sum', 'eop' (end of period). Used when
                frequency conversion is applied.
            force_refresh: bypass cache.

        Returns:
            List of {date, value} dicts. Missing values come back as '.' from
            FRED; we convert to None.
        """
        params: dict[str, str] = {"series_id": series_id}
        if start:               params["observation_start"]   = start
        if end:                 params["observation_end"]     = end
        if frequency:           params["frequency"]           = frequency
        if frequency:           params["aggregation_method"]  = aggregation_method

        data = self._request("series/observations", params, force_refresh)
        out = []
        for obs in data.get("observations", []):
            value_str = obs.get("value", ".")
            value = None if value_str in (".", "", None) else float(value_str)
            out.append({"date": obs["date"], "value": value})
        return out

    # -------------------------------------------------------------------------
    # Public: full vintage history (ALFRED-style)
    # -------------------------------------------------------------------------
    def observations_all_vintages(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Full vintage history. Multiple rows per observation_date if the series
        has been revised.

        Uses output_type=2 (observations by vintage date). For each
        observation_date we get the value as known at each vintage period.

        Returns:
            List of {date, vintage_date, value} dicts, where vintage_date
            is the realtime_start of the period during which `value` was
            the published estimate of the observation for `date`.
        """
        params: dict[str, str] = {
            "series_id":      series_id,
            "realtime_start": "1776-07-04",   # FRED accepts this as "earliest"
            "realtime_end":   "9999-12-31",   # FRED accepts this as "latest"
            "output_type":    "1",            # one row per (date, realtime_period)
        }
        if start: params["observation_start"] = start
        if end:   params["observation_end"]   = end

        data = self._request("series/observations", params, force_refresh)
        out = []
        for obs in data.get("observations", []):
            value_str = obs.get("value", ".")
            value = None if value_str in (".", "", None) else float(value_str)
            out.append({
                "date":          obs["date"],
                "vintage_date":  obs["realtime_start"],   # when this value was first known
                "value":         value,
            })
        return out

    # -------------------------------------------------------------------------
    # Public: series metadata
    # -------------------------------------------------------------------------
    def series_info(self, series_id: str, force_refresh: bool = False) -> dict:
        """
        Fetch series metadata (title, units, frequency, last_updated, etc).

        Useful for validating that a series ID exists before doing a full pull.
        """
        data = self._request(
            "series", {"series_id": series_id}, force_refresh,
        )
        seriess = data.get("seriess", [])
        if not seriess:
            raise FredSeriesNotFoundError(f"Series '{series_id}' returned no metadata")
        return seriess[0]

    def validate_series_ids(self, series_ids: list[str]) -> dict[str, bool]:
        """Bulk validation. Returns {series_id: exists?}."""
        out = {}
        for sid in series_ids:
            try:
                self.series_info(sid)
                out[sid] = True
            except FredSeriesNotFoundError:
                out[sid] = False
            except FredApiError as e:
                logger.warning("Could not validate %s: %s", sid, e)
                out[sid] = False
        return out

    # -------------------------------------------------------------------------
    # Public: cache management
    # -------------------------------------------------------------------------
    def cache_stats(self) -> dict:
        return self.cache.stats()
