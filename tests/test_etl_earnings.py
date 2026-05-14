"""
Unit/integration tests for data/etl_earnings.py:load_earnings_features.

These run against production accuracy.db. They verify the function
returns real earnings surprise values for tickers in earnings_cache,
defaults for ETFs (no earnings), and the right schema in all cases.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.etl_earnings import load_earnings_features


@pytest.fixture
def date_index():
    """2-year business date range ending today's known data."""
    return pd.date_range("2024-05-13", "2026-05-13", freq="B")


# ─── Schema and shape ────────────────────────────────────────────────────────

def test_returns_correct_schema(date_index):
    df = load_earnings_features("NVDA", date_index)
    expected_cols = {
        "eps_surprise", "rev_surprise", "days_to_earnings",
        "post_earnings_1d", "post_earnings_3d", "post_earnings_5d",
    }
    assert set(df.columns) == expected_cols


def test_index_length_matches_input(date_index):
    df = load_earnings_features("NVDA", date_index)
    assert len(df) == len(date_index)


# ─── eps_surprise (the bug we're fixing) ─────────────────────────────────────

def test_nvda_has_nonzero_eps_surprise(date_index):
    """NVDA reports quarterly — must have nonzero eps_surprise post-fix."""
    df = load_earnings_features("NVDA", date_index)
    nonzero = (df["eps_surprise"] != 0).sum()
    assert nonzero > 100, (
        f"NVDA should have many days with nonzero eps_surprise; got {nonzero}"
    )


def test_eps_surprise_clipped_range(date_index):
    df = load_earnings_features("NVDA", date_index)
    assert df["eps_surprise"].min() >= -3.0
    assert df["eps_surprise"].max() <= 3.0


def test_eps_surprise_computation_sign():
    """Spot check: CEG 2026-05-11 actual=2.74 est=2.56 → positive surprise."""
    # Use a date range covering 2026-05-11 + a few days after
    date_index = pd.date_range("2026-05-11", "2026-05-15", freq="B")
    df = load_earnings_features("CEG", date_index)
    # On or after 2026-05-11, eps_surprise should be POSITIVE (beat)
    post_earnings = df.loc[df.index >= "2026-05-11", "eps_surprise"]
    assert (post_earnings > 0).any(), "CEG had a beat — surprise should be positive"


# ─── days_to_earnings ────────────────────────────────────────────────────────

def test_nvda_days_to_earnings_is_finite():
    """NVDA reports 2026-05-20 — days_to_earnings should be small near today."""
    date_index = pd.date_range("2026-05-01", "2026-05-13", freq="B")
    df = load_earnings_features("NVDA", date_index)
    # NVDA reports 2026-05-20, so May 13 should show ~5-7 days
    last_value = df["days_to_earnings"].iloc[-1]
    assert 0 <= last_value < 30, f"Expected NVDA ~7 days to earnings, got {last_value}"


def test_etf_has_default_days_to_earnings(date_index):
    """ETFs don't have earnings — days_to_earnings should be sentinel 999."""
    df = load_earnings_features("SPY", date_index)
    assert (df["days_to_earnings"] == 999.0).all()


# ─── Defaults / ETFs / Unknown tickers ───────────────────────────────────────

def test_etf_returns_zero_surprises(date_index):
    df = load_earnings_features("SPY", date_index)
    assert (df["eps_surprise"] == 0.0).all()
    assert (df["rev_surprise"] == 0.0).all()


def test_unknown_ticker_returns_defaults(date_index):
    df = load_earnings_features("ZZZZNOTREAL", date_index)
    assert (df["eps_surprise"] == 0.0).all()
    assert (df["days_to_earnings"] == 999.0).all()


def test_rev_surprise_is_zero_v1(date_index):
    """v1 — rev_surprise placeholder (no revenue data in earnings_cache yet).
    
    Will become nonzero in D3 (UW revenue fetch). For now, document
    that we're returning zeros intentionally.
    """
    df = load_earnings_features("NVDA", date_index)
    assert (df["rev_surprise"] == 0.0).all()


# ─── Post-earnings flags ─────────────────────────────────────────────────────

def test_post_earnings_flags_binary(date_index):
    df = load_earnings_features("NVDA", date_index)
    for col in ["post_earnings_1d", "post_earnings_3d", "post_earnings_5d"]:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0.0, 1.0}), f"{col} has non-binary values"


def test_post_earnings_1d_sometimes_fires(date_index):
    """Over 2 years there must be ≥4 quarters of post-earnings 1d windows."""
    df = load_earnings_features("NVDA", date_index)
    n_post_days = df["post_earnings_1d"].sum()
    # Each earnings has 2 days in 1d-window (day-of and day-after), times ~8 quarters
    assert n_post_days >= 4, f"Expected post-earnings flag days, got {n_post_days}"


# ─── PIT honesty ─────────────────────────────────────────────────────────────

def test_pit_as_of_includes_today_rows(date_index):
    """as_of=today must INCLUDE rows ingested earlier today.
    
    Bug we fixed: WHERE created_at <= '2026-05-14' (date-only) excluded
    rows with created_at = '2026-05-14T00:07:31' (ISO timestamp) due to
    string comparison. DATE() cast fixes this.
    """
    # Use today's date as as_of — should still get historical NVDA earnings
    today = "2026-05-14"
    df = load_earnings_features("NVDA", date_index, as_of=today)
    nonzero = (df["eps_surprise"] != 0).sum()
    assert nonzero > 100, (
        f"PIT filter with as_of=today excluded too many rows; "
        f"got {nonzero} nonzero (expected >100 for NVDA over 2 years)"
    )


def test_pit_as_of_historical_returns_historical_rows():
    """PIT honest filter: as_of='2025-08-01' should see NVDA earnings 
    knowable BY that date (Apr 30 2025) but NOT later ones (Jul 31 2025 
    knowable Aug 4, Oct 31 knowable Nov 4, etc).
    
    NVDA reports: Apr 30 (knowable May 4), Jul 31 (knowable Aug 4),
                  Oct 31 (knowable Nov 4), Jan 31 (knowable Feb 4)
    
    With as_of='2025-08-01': only the Apr 30 report is knowable.
    """
    import pandas as pd
    date_index = pd.date_range("2024-05-13", "2026-05-13", freq="B")
    df_aug = load_earnings_features("NVDA", date_index, as_of="2025-08-01")
    
    # On 2025-09-01 (after as_of), no FUTURE-vintage earnings should appear
    # eps_surprise on that date should reflect only Apr 30 surprise (the
    # only one knowable by Aug 1)
    val_sep = df_aug.loc[df_aug.index == "2025-09-02", "eps_surprise"]
    # Should be NVDA's Apr 30 2025 surprise value (~3-8% range)
    if len(val_sep) > 0:
        v = val_sep.iloc[0]
        assert abs(v) < 0.3, f"PIT leak: as_of=Aug 1 shows {v} on Sep 2 (only Apr 30 should be visible)"


def test_pit_as_of_filter_smoke(date_index):
    """Smoke: function runs cleanly with as_of, returns right schema."""
    df = load_earnings_features("NVDA", date_index, as_of="2024-06-01")
    assert "eps_surprise" in df.columns
    assert len(df) == len(date_index)
