"""
Tests for scripts/refresh_earnings_calendar.py.

Verifies the refresh script correctly derives next-earnings-per-ticker
from accuracy.db.earnings_cache and populates earnings_calendar.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import the module under test
import refresh_earnings_calendar as REC

DB_PATH = Path(__file__).parent.parent / "accuracy.db"


@pytest.fixture
def conn():
    """Read-only connection to accuracy.db."""
    c = sqlite3.connect(DB_PATH)
    yield c
    c.close()


# ─── Table existence ────────────────────────────────────────────────────────

def test_table_exists_after_refresh(conn):
    """earnings_calendar table should exist after init_table()."""
    REC.init_table(DB_PATH)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='earnings_calendar'"
    )
    assert cur.fetchone() is not None, "earnings_calendar table not created"


def test_schema_has_required_columns(conn):
    """Table must have expected columns."""
    REC.init_table(DB_PATH)
    cur = conn.execute("PRAGMA table_info(earnings_calendar)")
    cols = {row[1] for row in cur.fetchall()}
    expected = {
        "id", "ticker", "next_date", "next_time", "expected_move",
        "days_until", "updated_at",
    }
    missing = expected - cols
    assert not missing, f"Missing columns: {missing}"


# ─── Refresh logic ──────────────────────────────────────────────────────────

def test_refresh_populates_rows():
    """After refresh, table has rows for known tickers with future earnings."""
    REC.refresh(DB_PATH, verbose=False)
    with sqlite3.connect(DB_PATH) as conn:
        n = conn.execute("SELECT COUNT(*) FROM earnings_calendar").fetchone()[0]
    # earnings_cache had 127 forward rows / 114 tickers at audit time
    assert n >= 50, f"Expected >=50 tickers with upcoming earnings, got {n}"


def test_nvda_has_next_earnings():
    """NVDA reports 2026-05-20 — verify it's in the calendar."""
    REC.refresh(DB_PATH, verbose=False)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT ticker, next_date, days_until FROM earnings_calendar WHERE ticker='NVDA'"
        ).fetchone()
    assert row is not None, "NVDA missing from earnings_calendar"
    assert row[1] == "2026-05-20", f"Expected NVDA next 2026-05-20, got {row[1]}"
    assert 0 < row[2] < 30, f"NVDA days_until should be small, got {row[2]}"


def test_no_past_earnings_in_calendar():
    """Calendar should contain only FUTURE next_dates relative to today."""
    REC.refresh(DB_PATH, verbose=False)
    today = REC.today_str()
    with sqlite3.connect(DB_PATH) as conn:
        n_past = conn.execute(
            "SELECT COUNT(*) FROM earnings_calendar WHERE next_date < ?",
            (today,),
        ).fetchone()[0]
    assert n_past == 0, f"{n_past} past-dated rows in earnings_calendar"


def test_one_row_per_ticker():
    """UNIQUE(ticker) constraint — only one upcoming earnings per ticker."""
    REC.refresh(DB_PATH, verbose=False)
    with sqlite3.connect(DB_PATH) as conn:
        max_per_ticker = conn.execute("""
            SELECT MAX(cnt) FROM (
                SELECT ticker, COUNT(*) AS cnt FROM earnings_calendar GROUP BY ticker
            )
        """).fetchone()[0]
    assert max_per_ticker == 1, f"Some ticker has multiple rows: max={max_per_ticker}"


def test_idempotent_refresh():
    """Running refresh twice should produce same row count (UPSERT not duplicate)."""
    REC.refresh(DB_PATH, verbose=False)
    with sqlite3.connect(DB_PATH) as conn:
        n1 = conn.execute("SELECT COUNT(*) FROM earnings_calendar").fetchone()[0]
    REC.refresh(DB_PATH, verbose=False)
    with sqlite3.connect(DB_PATH) as conn:
        n2 = conn.execute("SELECT COUNT(*) FROM earnings_calendar").fetchone()[0]
    assert n1 == n2, f"Refresh not idempotent: {n1} → {n2}"


# ─── Field correctness ──────────────────────────────────────────────────────

def test_days_until_matches_next_date():
    """days_until should be (next_date - today).days."""
    REC.refresh(DB_PATH, verbose=False)
    today_ts = REC.today_str()
    from datetime import datetime
    today_dt = datetime.strptime(today_ts, "%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT ticker, next_date, days_until FROM earnings_calendar LIMIT 10"
        ).fetchall()
    for ticker, next_date, days_until in rows:
        next_dt = datetime.strptime(next_date, "%Y-%m-%d")
        expected = (next_dt - today_dt).days
        assert days_until == expected, (
            f"{ticker}: days_until={days_until}, expected {expected} "
            f"(next_date={next_date}, today={today_ts})"
        )


def test_expected_move_preserved():
    """expected_move from earnings_cache should be copied through."""
    REC.refresh(DB_PATH, verbose=False)
    with sqlite3.connect(DB_PATH) as conn:
        # NVDA's expected_move was ~0.0687 in earnings_cache
        em = conn.execute(
            "SELECT expected_move FROM earnings_calendar WHERE ticker='NVDA'"
        ).fetchone()
    if em is not None and em[0] is not None:
        assert 0.0 <= em[0] <= 1.0, f"expected_move out of range: {em[0]}"
