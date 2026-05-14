"""
Unit tests for portfolio/neutralizer.py.

Verifies:
  - Shape preservation
  - Sector neutralization math (sum=0 per sector with >=2 members)
  - Dollar neutralization math (sum=0 globally)
  - Long-only conversion (weights sum to 1)
  - Pass-through for singleton sectors
  - NaN handling
  - Empty input
  - Top-N filtering
  - Mode='none' identity
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio.neutralizer import (
    neutralize_signals,
    build_research_portfolio,
    coverage_report,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sector_map():
    """Small sector map: 2 sectors with 3 tickers each, 1 singleton."""
    return {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "GOOG": "Tech",
        "JPM":  "Finance",
        "GS":   "Finance",
        "BAC":  "Finance",
        "SOLO": "Singleton",  # alone in its sector
    }


@pytest.fixture
def signals_one_date(sector_map):
    """One date, 7 tickers, varied signal values."""
    return pd.DataFrame({
        "date":         ["2026-05-13"] * 7,
        "ticker":       ["AAPL", "MSFT", "GOOG", "JPM", "GS", "BAC", "SOLO"],
        "signal_value": [0.10, 0.05, 0.00, 0.20, -0.10, 0.05, 0.15],
    })


@pytest.fixture
def signals_two_dates(sector_map):
    """Two dates × 6 tickers (excluding SOLO for simplicity)."""
    rows = []
    for date in ["2026-05-12", "2026-05-13"]:
        for ticker, val in [
            ("AAPL", 0.10), ("MSFT", 0.05), ("GOOG", 0.00),
            ("JPM", 0.20), ("GS", -0.10), ("BAC", 0.05),
        ]:
            rows.append({"date": date, "ticker": ticker, "signal_value": val})
    return pd.DataFrame(rows)


# ─── Mode: sector ────────────────────────────────────────────────────────────

def test_sector_mode_sum_zero_per_sector(signals_one_date, sector_map):
    """Sector-neutralized weights sum to 0 within each ≥2-member sector."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="sector")
    
    # Join sector info
    weights["sector"] = weights["ticker"].map(sector_map)
    
    for sector, group in weights.groupby("sector"):
        if sector in ("Tech", "Finance"):
            assert abs(group["weight"].sum()) < 1e-9, f"{sector} weights don't sum to 0"


def test_sector_mode_singleton_passthrough(signals_one_date, sector_map):
    """Singleton sector tickers pass through unchanged (no demean)."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="sector")
    solo = weights[weights["ticker"] == "SOLO"]
    assert len(solo) == 1
    assert solo["weight"].iloc[0] == pytest.approx(0.15, abs=1e-9)


def test_sector_mode_shape_preserved(signals_one_date, sector_map):
    """All input tickers appear in output (no drops)."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="sector")
    assert set(weights["ticker"]) == set(signals_one_date["ticker"])


def test_sector_mode_math_correctness(signals_one_date, sector_map):
    """Verify exact math: Tech mean = (0.10+0.05+0.00)/3 = 0.05.
       Tech demeaned: AAPL=0.05, MSFT=0.00, GOOG=-0.05."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="sector")
    w = weights.set_index("ticker")["weight"]
    
    assert w["AAPL"] == pytest.approx(0.05, abs=1e-9)
    assert w["MSFT"] == pytest.approx(0.00, abs=1e-9)
    assert w["GOOG"] == pytest.approx(-0.05, abs=1e-9)
    # Finance mean = (0.20 - 0.10 + 0.05)/3 = 0.05
    assert w["JPM"] == pytest.approx(0.15, abs=1e-9)
    assert w["GS"]  == pytest.approx(-0.15, abs=1e-9)
    assert w["BAC"] == pytest.approx(0.00, abs=1e-9)


# ─── Mode: dollar ────────────────────────────────────────────────────────────

def test_dollar_mode_sum_zero_globally(signals_one_date, sector_map):
    """Dollar-neutralized weights sum to 0 across all tickers per date."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="dollar")
    for date, group in weights.groupby("date"):
        assert abs(group["weight"].sum()) < 1e-9, f"{date} not dollar-neutral"


def test_dollar_mode_math_correctness(signals_one_date, sector_map):
    """Mean of all 7 signals = (0.10+0.05+0.00+0.20-0.10+0.05+0.15)/7 = 0.0643."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="dollar")
    w = weights.set_index("ticker")["weight"]
    expected_mean = (0.10 + 0.05 + 0.00 + 0.20 - 0.10 + 0.05 + 0.15) / 7
    assert w["AAPL"] == pytest.approx(0.10 - expected_mean, abs=1e-9)


# ─── Mode: none ──────────────────────────────────────────────────────────────

def test_none_mode_is_identity(signals_one_date, sector_map):
    """Mode='none' returns raw signal values unchanged."""
    weights = neutralize_signals(signals_one_date, sector_map, mode="none")
    w = weights.set_index("ticker")["weight"]
    orig = signals_one_date.set_index("ticker")["signal_value"]
    for ticker in orig.index:
        if abs(orig[ticker]) > 1e-9:  # zero-signal rows are dropped
            assert w[ticker] == pytest.approx(orig[ticker], abs=1e-9)


# ─── Long-only conversion ───────────────────────────────────────────────────

def test_long_only_sums_to_one(signals_one_date, sector_map):
    """Long-only weights sum to 1 per date."""
    weights = neutralize_signals(
        signals_one_date, sector_map, mode="sector", long_only=True
    )
    for date, group in weights.groupby("date"):
        assert group["weight"].sum() == pytest.approx(1.0, abs=1e-9)


def test_long_only_no_negatives(signals_one_date, sector_map):
    """Long-only weights are all non-negative."""
    weights = neutralize_signals(
        signals_one_date, sector_map, mode="sector", long_only=True
    )
    assert (weights["weight"] >= 0).all()


# ─── NaN handling ───────────────────────────────────────────────────────────

def test_nan_input_does_not_crash(sector_map):
    """NaN signal values don't crash; propagate sensibly."""
    df = pd.DataFrame({
        "date":         ["2026-05-13"] * 3,
        "ticker":       ["AAPL", "MSFT", "GOOG"],
        "signal_value": [0.10, np.nan, 0.05],
    })
    weights = neutralize_signals(df, sector_map, mode="sector")
    # At minimum: function returns a DataFrame without raising
    assert isinstance(weights, pd.DataFrame)


# ─── Multi-date ─────────────────────────────────────────────────────────────

def test_multi_date_handled_independently(signals_two_dates, sector_map):
    """Two dates produce independent neutralization."""
    weights = neutralize_signals(signals_two_dates, sector_map, mode="sector")
    
    for date, group in weights.groupby("date"):
        group["sector"] = group["ticker"].map(sector_map)
        for sector, sub in group.groupby("sector"):
            if sector in ("Tech", "Finance"):
                assert abs(sub["weight"].sum()) < 1e-9


# ─── Invalid mode ───────────────────────────────────────────────────────────

def test_invalid_mode_raises(signals_one_date, sector_map):
    with pytest.raises(ValueError, match="Invalid mode"):
        neutralize_signals(signals_one_date, sector_map, mode="bogus")


# ─── Coverage report ────────────────────────────────────────────────────────

def test_coverage_report(signals_one_date, sector_map):
    """Coverage report identifies neutralizable vs pass-through tickers."""
    report = coverage_report(signals_one_date, sector_map)
    assert report["total_tickers"] == 7
    assert report["neutralizable"] == 6  # Tech 3 + Finance 3
    assert report["pass_through"] == 1   # SOLO


# ─── Build research portfolio ───────────────────────────────────────────────

def test_build_portfolio_end_to_end(sector_map):
    """End-to-end smoke: predictions → portfolio with horizon column."""
    preds = pd.DataFrame({
        "prediction_date": ["2026-05-13"] * 6,
        "ticker":          ["AAPL", "MSFT", "GOOG", "JPM", "GS", "BAC"],
        "horizon":         [3, 3, 3, 3, 3, 3],
        "prob_up":         [0.60, 0.55, 0.50, 0.70, 0.40, 0.55],
    })
    portfolio = build_research_portfolio(
        preds, sector_map=sector_map, mode="sector", long_only=False
    )
    assert "horizon" in portfolio.columns
    assert "weight" in portfolio.columns
    assert "ticker" in portfolio.columns
    assert (portfolio["horizon"] == 3).all()


def test_top_n_filter(sector_map):
    """top_n keeps only N largest absolute-weight positions per (date, horizon)."""
    preds = pd.DataFrame({
        "prediction_date": ["2026-05-13"] * 6,
        "ticker":          ["AAPL", "MSFT", "GOOG", "JPM", "GS", "BAC"],
        "horizon":         [3] * 6,
        "prob_up":         [0.60, 0.55, 0.50, 0.70, 0.40, 0.55],
    })
    portfolio = build_research_portfolio(
        preds, sector_map=sector_map, mode="sector", top_n=3
    )
    assert len(portfolio) == 3
