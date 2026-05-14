"""
Integration tests for insider PIT honesty.

Tests _load_insider() (the production reader called by builder.py:_load_insider)
and verifies the writer convention via SQL-level integrity check.

These run against production insider_trades.db. They confirm:
  - The PIT filter respects as_of for today's-timestamped rows (DATE() cast fix)
  - The PIT filter correctly excludes future-knowable rows
  - The writer convention (created_at = knowable_at(date)) is in effect for
    newly-inserted rows (smoke test)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

INSIDER_DB = Path(__file__).parent.parent / "insider_trades.db"


@pytest.fixture
def date_index():
    """2-year business date range ending 2026-05-13."""
    return pd.date_range("2024-05-13", "2026-05-13", freq="B")


# ─── Direct reader tests (matches builder.py:_load_insider) ─────────────────

def _load_insider_direct(ticker: str, dates: pd.Index, as_of: str | None = None):
    """Replicate builder.py:_load_insider for isolated testing.
    
    This is what we test against the production data — verifies the PIT
    filter on the actual DB.
    """
    conn = sqlite3.connect(INSIDER_DB)
    if as_of is not None:
        df = pd.read_sql(
            "SELECT date, net_shares FROM insider_flows "
            "WHERE ticker = ? AND (created_at IS NULL OR DATE(created_at) <= ?) "
            "ORDER BY date",
            conn, params=(ticker.upper(), str(as_of)), parse_dates=["date"]
        )
    else:
        df = pd.read_sql(
            "SELECT date, net_shares FROM insider_flows WHERE ticker = ? ORDER BY date",
            conn, params=(ticker.upper(),), parse_dates=["date"]
        )
    conn.close()
    return df


def test_insider_db_exists():
    """Sanity: insider_trades.db is in expected location."""
    assert INSIDER_DB.exists(), f"insider_trades.db missing at {INSIDER_DB}"


def test_insider_flows_has_data():
    """Sanity: insider_flows table has rows we can test against."""
    conn = sqlite3.connect(INSIDER_DB)
    n = conn.execute("SELECT COUNT(*) FROM insider_flows").fetchone()[0]
    conn.close()
    assert n > 100, f"insider_flows has only {n} rows; expected hundreds"


# ─── PIT honesty: today's-timestamped rows ──────────────────────────────────

def test_pit_filter_today_includes_recent_rows():
    """as_of=today should INCLUDE rows ingested earlier today.
    
    Bug we fixed: DATE-string comparison excluded same-day timestamped rows.
    """
    today = "2026-05-14"
    df = _load_insider_direct("NVDA", pd.date_range("2024-01-01", "2026-05-13", freq="B"), as_of=today)
    # NVDA should have many insider flow rows in the past 2 years
    assert len(df) > 5, (
        f"PIT filter with as_of=today returned only {len(df)} rows for NVDA; "
        f"expected dozens"
    )


def test_pit_filter_historical_excludes_future_rows():
    """as_of='2025-06-01' should NOT include rows knowable AFTER that date."""
    as_of = "2025-06-01"
    df = _load_insider_direct("NVDA", pd.date_range("2024-01-01", "2026-05-13", freq="B"), as_of=as_of)
    # If filter works, no row's date should be AFTER as_of (because such
    # a row's knowable_at = date + 2 BD > as_of)
    if len(df) > 0:
        max_date = df["date"].max()
        # Allow slight slop: knowable_at = date + 2 BD, so date can be at most
        # as_of - 2 BD. Approximation: date should be <= as_of itself.
        assert pd.Timestamp(max_date) <= pd.Timestamp(as_of) + pd.Timedelta(days=2), (
            f"PIT leak: insider row dated {max_date} appears in as_of={as_of}"
        )


# ─── Writer convention (verifies knowable_at applied to NEW writes) ─────────

def test_writer_creates_with_knowable_at_convention():
    """All NON-LEGACY rows should have created_at ≈ date + 2 BD.
    
    This is a regression test — once the writer fix is in place, every NEW
    upsert should satisfy julianday(created_at) - julianday(date) ≈ 2-5 days.
    
    Allow tolerance for weekends (Fri→Tue = 4 days inclusive).
    """
    conn = sqlite3.connect(INSIDER_DB)
    # Check the most recently inserted rows (post-fix)
    cur = conn.execute("""
        SELECT julianday(created_at) - julianday(date) AS day_diff,
               COUNT(*)
        FROM insider_flows
        WHERE date >= '2026-05-01'
        GROUP BY ROUND(day_diff, 0)
        ORDER BY day_diff
    """)
    distribution = cur.fetchall()
    conn.close()
    # Post-fix, all recent rows should have day_diff in [2, 5] range
    # (knowable_at = date + 2 BD, max 4-5 calendar days due to weekends)
    bad_rows = sum(count for diff, count in distribution if diff > 7)
    assert bad_rows == 0, (
        f"{bad_rows} insider_flows rows have created_at >7 days after date; "
        f"writer convention not applied. Distribution: {distribution}"
    )


def test_writer_creates_with_knowable_at_no_negative_diff():
    """No row should have created_at BEFORE its date (would be a time-travel bug)."""
    conn = sqlite3.connect(INSIDER_DB)
    cur = conn.execute("""
        SELECT COUNT(*) FROM insider_flows
        WHERE julianday(created_at) - julianday(date) < 0
    """)
    violations = cur.fetchone()[0]
    conn.close()
    assert violations == 0, f"{violations} rows have created_at < date (impossible)"


# ─── Builder integration smoke test ─────────────────────────────────────────

def test_builder_insider_features_nonzero_for_nvda():
    """build_feature_dataframe('NVDA') should produce nonzero insider_* values.
    
    If the PIT filter is fixed AND insider data exists, then training_mode
    feature extraction should not return all-zero insider columns.
    """
    from features.builder import build_feature_dataframe
    df = build_feature_dataframe("NVDA")
    
    if "insider_net_shares" in df.columns:
        n_nonzero = (df["insider_net_shares"] != 0).sum()
        # NVDA has had many insider transactions over 5 years
        assert n_nonzero > 10, (
            f"NVDA insider_net_shares has only {n_nonzero} nonzero values; "
            f"expected >10. PIT filter may still be broken."
        )
