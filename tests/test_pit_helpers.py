"""
Tests for utils/pit_helpers.knowable_at.

The knowable_at helper returns an ISO timestamp marking when a corporate
fact (earnings report, insider Form 4) became publicly knowable. The
convention is event_date + 2 business days at market close (16:00 ET).

This matches SEC Form 4 filing rules (2 BD reporting window) and is used
across data ingestion to populate `created_at` columns for PIT honesty.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pit_helpers import knowable_at


# ─── Basic behavior ──────────────────────────────────────────────────────────

def test_returns_iso_string():
    """Returns ISO 8601 format string."""
    result = knowable_at("2026-02-25")
    assert isinstance(result, str)
    # Parseable as datetime
    pd.to_datetime(result)


def test_skip_weekend_friday_event():
    """Friday + 2 BD = Tuesday (skip Sat, Sun)."""
    result = knowable_at("2026-01-30")  # Friday
    parsed = pd.to_datetime(result).date()
    assert parsed == date(2026, 2, 3)  # Tuesday


def test_midweek_no_weekend_skip():
    """Wednesday + 2 BD = Friday (no weekend involved)."""
    result = knowable_at("2026-02-25")  # Wednesday
    parsed = pd.to_datetime(result).date()
    assert parsed == date(2026, 2, 27)  # Friday


def test_monday_event():
    """Monday + 2 BD = Wednesday."""
    result = knowable_at("2026-03-02")  # Monday
    parsed = pd.to_datetime(result).date()
    assert parsed == date(2026, 3, 4)  # Wednesday


def test_thursday_event():
    """Thursday + 2 BD = Monday next week."""
    result = knowable_at("2026-05-14")  # Thursday
    parsed = pd.to_datetime(result).date()
    assert parsed == date(2026, 5, 18)  # Monday


# ─── Input type flexibility ──────────────────────────────────────────────────

def test_accepts_string():
    result = knowable_at("2026-02-25")
    assert "2026-02-27" in result


def test_accepts_date():
    result = knowable_at(date(2026, 2, 25))
    assert "2026-02-27" in result


def test_accepts_datetime():
    result = knowable_at(datetime(2026, 2, 25, 15, 30))
    assert "2026-02-27" in result


def test_accepts_timestamp():
    result = knowable_at(pd.Timestamp("2026-02-25"))
    assert "2026-02-27" in result


# ─── PIT filter integration (the use case) ───────────────────────────────────

def test_pit_filter_compatibility_with_date_only_as_of():
    """The knowable_at value must work with date-only SQL filter via DATE() cast.
    
    Real-world scenario: a row created with knowable_at='2026-02-27T00:00:00'
    must be INCLUDED when filtered by DATE(created_at) <= '2026-02-27'.
    """
    created_at = knowable_at("2026-02-25")
    # Simulate the SQL filter: DATE(created_at) <= '2026-02-27'
    date_portion = pd.to_datetime(created_at).strftime("%Y-%m-%d")
    assert date_portion <= "2026-02-27"
    assert date_portion >= "2026-02-26"  # Two BD later than Wed Feb 25


def test_pit_filter_excludes_future_events():
    """A row from a future event must be EXCLUDED when filtered by an earlier as_of."""
    created_at = knowable_at("2026-05-20")
    date_portion = pd.to_datetime(created_at).strftime("%Y-%m-%d")
    # If as_of is '2026-05-13' (one week before event), this row should be excluded
    assert date_portion > "2026-05-13"


# ─── Edge cases ──────────────────────────────────────────────────────────────

def test_year_boundary():
    """Last business day of year + 2 BD into next year."""
    result = knowable_at("2025-12-30")  # Tuesday
    parsed = pd.to_datetime(result).date()
    assert parsed == date(2026, 1, 1)  # Thursday — note: holidays not considered


def test_consistency_across_calls():
    """Same input → same output (idempotent)."""
    a = knowable_at("2026-02-25")
    b = knowable_at("2026-02-25")
    assert a == b
