"""Point-in-time honesty helpers.

Single source of truth for the `knowable_at` convention used across ETLs
to populate `created_at` columns. Matches the SEC Form 4 filing window
(2 business days) and is consistent with the insider_flows gamma backfill
(Memory #13).

The contract:
    knowable_at(event_date) -> ISO timestamp string
        Returns event_date + 2 business days as an ISO 8601 string.
        Used by data ingestion writers to populate created_at, enabling
        PIT-honest walk-forward training.

Why 2 BD: SEC requires Form 4 to be filed within 2 BD of the trade.
Earnings reports become publicly knowable similarly (after-close release
on report_date, processed and ingested over the next 1-2 BD).

Use with the matching SQL filter pattern:
    WHERE (created_at IS NULL OR DATE(created_at) <= ?)
The DATE() cast strips the timestamp portion to avoid string-comparison
bugs where 'YYYY-MM-DDTHH:MM:SS' > 'YYYY-MM-DD'.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd


DateLike = Union[str, date, datetime, pd.Timestamp]


def knowable_at(event_date: DateLike) -> str:
    """Return ISO timestamp for when a corporate fact became publicly knowable.

    Convention: event_date + 2 business days (skips Sat/Sun, ignores
    market holidays).

    Args:
        event_date: The underlying event date (earnings report, Form 4
            trade date, etc.). Accepts str, date, datetime, or pd.Timestamp.

    Returns:
        ISO 8601 timestamp string (e.g., '2026-02-27T00:00:00').

    Example:
        >>> knowable_at("2026-02-25")  # Wednesday
        '2026-02-27T00:00:00'
        >>> knowable_at("2026-01-30")  # Friday (skips weekend)
        '2026-02-03T00:00:00'
    """
    d = pd.to_datetime(event_date) + pd.tseries.offsets.BusinessDay(2)
    return d.isoformat()
