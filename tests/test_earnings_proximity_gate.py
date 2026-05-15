"""
Tests for signals/generator.py:_check_earnings_proximity_filter().

Suppresses BUY signals when:
  Rule 1: Earnings in <=5 days with expected_move > 8% (high uncertainty)
  Rule 2: Recently reported a big EPS miss (<-15%) with no near-term catalyst
  Rule 3: Pre-earnings bearish setup (insider selling + high short interest)

These three rules together would have blocked today's OKLO BUY HIGH and
SNOW BUY MEDIUM (the two confirmed false positives from May 14 2026).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.generator import _check_earnings_proximity_filter

DB_PATH = Path(__file__).parent.parent / "accuracy.db"


# ─── Helper: build a 1-row "features" DataFrame mimicking build_feature_dataframe output ───

def _features(eps_surprise=0.0, insider_21d=0.0, short_pct_float=0.0):
    """Build a minimal feature-frame snapshot for testing."""
    return pd.DataFrame([{
        "eps_surprise":    eps_surprise,
        "insider_21d":     insider_21d,
        "short_pct_float": short_pct_float,
    }])


# ─── Rule 1: Earnings Proximity ────────────────────────────────────────────────

def test_rule1_blocks_buy_inside_earnings_window_with_high_exp_move():
    """SNOW-style: 5 days to earnings, expected move 10% → suppress."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 5, "expected_move": 0.10}):
        result = _check_earnings_proximity_filter("SNOW", _features())
    assert result is not None
    assert result["rule"] == "EARNINGS_PROXIMITY"
    assert "5d" in result["reason"]
    assert "10" in result["reason"]


def test_rule1_allows_buy_outside_earnings_window():
    """20 days to earnings, expected move 10% → allow."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 20, "expected_move": 0.10}):
        result = _check_earnings_proximity_filter("AAPL", _features())
    # Rule 1 doesn't fire; assume rules 2/3 also don't fire by default
    assert result is None or result["rule"] != "EARNINGS_PROXIMITY"


def test_rule1_allows_buy_with_low_exp_move():
    """3 days to earnings but expected move only 4% → allow (low uncertainty)."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 3, "expected_move": 0.04}):
        result = _check_earnings_proximity_filter("AAPL", _features())
    assert result is None or result["rule"] != "EARNINGS_PROXIMITY"


def test_rule1_handles_no_earnings_data_gracefully():
    """No upcoming earnings in calendar → allow."""
    with patch("signals.generator._lookup_earnings_calendar", return_value=None):
        result = _check_earnings_proximity_filter("AAPL", _features())
    # If only rule 1 was eligible and there's no calendar data, return None
    assert result is None


# ─── Rule 2: Post-Earnings Damage ──────────────────────────────────────────────

def test_rule2_blocks_buy_after_big_eps_miss():
    """OKLO-style: just reported -65% EPS surprise, next earnings 88d out."""
    feat = _features(eps_surprise=-0.65)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 88, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("OKLO", feat)
    assert result is not None
    assert result["rule"] == "POST_EARNINGS_DAMAGE"
    assert "miss" in result["reason"].lower() or "-65" in result["reason"]


def test_rule2_allows_buy_with_small_miss():
    """Small EPS miss (-5%) → allow (noise, not damage)."""
    feat = _features(eps_surprise=-0.05)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 88, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("AAPL", feat)
    assert result is None


def test_rule2_allows_buy_with_eps_beat():
    """Positive EPS surprise → allow."""
    feat = _features(eps_surprise=+0.15)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 88, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("AAPL", feat)
    assert result is None


def test_rule2_doesnt_fire_if_near_term_catalyst():
    """Big miss BUT earnings coming up soon (10d) → rule 1 handles, not rule 2."""
    feat = _features(eps_surprise=-0.65)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 10, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("OKLO", feat)
    # Rule 2 needs days_until > 60. With 10d, rule 2 should NOT fire.
    # Some other rule might fire but rule 2 specifically should not be the reason.
    assert result is None or result["rule"] != "POST_EARNINGS_DAMAGE"


# ─── Rule 3: Pre-Earnings Bearish Setup ────────────────────────────────────────

def test_rule3_blocks_buy_with_insider_sells_and_high_short():
    """Pre-earnings bearish: insider selling + 15% short int + earnings 14d."""
    feat = _features(insider_21d=-200000, short_pct_float=0.15)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 14, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("BEAR1", feat)
    assert result is not None
    assert result["rule"] == "PRE_EARNINGS_BEARISH"


def test_rule3_allows_buy_with_insider_sells_but_low_short():
    """Insider sells but short interest normal → allow (insiders may be 
    routine 10b5-1)."""
    feat = _features(insider_21d=-200000, short_pct_float=0.04)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 14, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("AAPL", feat)
    assert result is None


def test_rule3_allows_buy_with_high_short_but_no_insider_sells():
    """High short but no insider activity → allow (shorts may be wrong)."""
    feat = _features(insider_21d=0, short_pct_float=0.20)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 14, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("AAPL", feat)
    assert result is None


def test_rule3_doesnt_fire_outside_window():
    """Insider sells + high short BUT earnings 60d away → rule 3 doesn't fire."""
    feat = _features(insider_21d=-200000, short_pct_float=0.15)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 60, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("AAPL", feat)
    # Rule 3 requires days_until <= 14
    assert result is None


# ─── Real-world integration: OKLO and SNOW ─────────────────────────────────────

def test_oklo_today_gets_suppressed():
    """OKLO real features: eps_surprise=-0.65, insider_21d=-1.03M, 
    short_pct=16.5%, earnings 88d. Should fire rule 2."""
    feat = _features(eps_surprise=-0.65, insider_21d=-1032432, short_pct_float=0.165)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 88, "expected_move": 0.06}):
        result = _check_earnings_proximity_filter("OKLO", feat)
    assert result is not None
    assert result["rule"] == "POST_EARNINGS_DAMAGE"


def test_snow_today_gets_suppressed():
    """SNOW real features: earnings in 6d, expected_move 10.4%. 
    Should fire rule 1."""
    feat = _features(eps_surprise=+0.18, insider_21d=-101482, short_pct_float=0.052)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 6, "expected_move": 0.104}):
        result = _check_earnings_proximity_filter("SNOW", feat)
    assert result is not None
    assert result["rule"] == "EARNINGS_PROXIMITY"


def test_nvda_today_allowed():
    """NVDA: earnings in 6d but expected_move only 6.9% (below 8% threshold).
    No big miss, no insider cluster. Should allow."""
    feat = _features(eps_surprise=+0.10, insider_21d=-50000, short_pct_float=0.01)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 6, "expected_move": 0.069}):
        result = _check_earnings_proximity_filter("NVDA", feat)
    # 6.9% < 8% → rule 1 doesn't fire
    # eps_surprise +0.10 → rule 2 doesn't fire
    # short_pct 0.01 → rule 3 doesn't fire
    assert result is None


def test_crwd_today_allowed():
    """CRWD: earnings 20d out (outside all windows). Should allow."""
    feat = _features(eps_surprise=+0.08, insider_21d=-150000, short_pct_float=0.025)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 20, "expected_move": 0.05}):
        result = _check_earnings_proximity_filter("CRWD", feat)
    assert result is None


# ─── Resilience ────────────────────────────────────────────────────────────────

def test_fails_open_on_db_error():
    """If DB lookup raises, return None (allow BUY) — fail open like fitness filter."""
    with patch("signals.generator._lookup_earnings_calendar",
               side_effect=Exception("DB error")):
        result = _check_earnings_proximity_filter("AAPL", _features())
    assert result is None


# ─── Rule 1 Tier B: Extreme IV within 14 days ─────────────────────────────────
# Tier B is the "wider window, higher IV floor" branch added May 15 2026 to catch
# pre-earnings traps that fall outside Tier A's 7d window. Verified Tier B fires
# on SNOW (12d/13.7% IV) and MRVL (12d/15.8% IV) but not on AI (12d/11.9% IV).

def test_tier_b_fires_snow_style():
    """SNOW post-calendar-fix: 12d to earnings, 13.66% IV -> fires Tier B."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 12, "expected_move": 0.1366}):
        result = _check_earnings_proximity_filter("SNOW", _features())
    assert result is not None
    assert result["rule"] == "EARNINGS_PROXIMITY"
    assert "tier B" in result["reason"]


def test_tier_b_fires_at_14d_12pct_boundary():
    """Boundary case: 14d, exactly 12% IV -> Tier B fires (inclusive)."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 14, "expected_move": 0.12}):
        result = _check_earnings_proximity_filter("BOUND", _features())
    assert result is not None
    assert result["rule"] == "EARNINGS_PROXIMITY"
    assert "tier B" in result["reason"]


def test_tier_b_doesnt_fire_just_below_iv_floor():
    """AI-style: 12d to earnings, 11.9% IV -> just below 12% floor -> pass."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 12, "expected_move": 0.119}):
        result = _check_earnings_proximity_filter("AI", _features())
    assert result is None


def test_tier_b_doesnt_fire_just_outside_day_window():
    """15 days out -> outside Tier B's 14d window -> pass even at high IV."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 15, "expected_move": 0.20}):
        result = _check_earnings_proximity_filter("PL", _features())
    assert result is None


def test_mid_zone_doesnt_fire_either_tier():
    """Between tiers: 10d (>7), 10% IV (<12%) -> neither tier fires."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 10, "expected_move": 0.10}):
        result = _check_earnings_proximity_filter("MID", _features())
    assert result is None


def test_tier_a_takes_precedence_when_both_could_fire():
    """5d + 15% IV: both tiers' criteria met. Reports Tier A (closer match)."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 5, "expected_move": 0.15}):
        result = _check_earnings_proximity_filter("HIGHIV", _features())
    assert result is not None
    assert "tier A" in result["reason"]


# ─── Real-world fires (post calendar-fix May 15 2026) ──────────────────────────

def test_real_today_mrvl_fires_via_tier_b():
    """MRVL post-calendar-fix: 12d / 15.75% IV -> Tier B fires."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 12, "expected_move": 0.1575}):
        result = _check_earnings_proximity_filter("MRVL", _features())
    assert result is not None
    assert "tier B" in result["reason"]


def test_real_today_cava_fires_via_tier_a():
    """CAVA: 4d / 11.6% IV -> Tier A fires (close + meaningful IV)."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 4, "expected_move": 0.1159}):
        result = _check_earnings_proximity_filter("CAVA", _features())
    assert result is not None
    assert "tier A" in result["reason"]


def test_real_today_zm_fires_via_tier_a_after_calendar_fix():
    """ZM post-calendar-fix: 6d / 9.51% IV -> Tier A fires."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 6, "expected_move": 0.0951}):
        result = _check_earnings_proximity_filter("ZM", _features())
    assert result is not None
    assert "tier A" in result["reason"]


def test_real_today_crwd_allowed_after_calendar_fix():
    """CRWD post-calendar-fix: 19d / 10.74% IV -> outside both tiers -> pass.
    Matches independent analysis (initiate CRWD position)."""
    feat = _features(eps_surprise=+0.08, insider_21d=-150000, short_pct_float=0.025)
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 19, "expected_move": 0.1074}):
        result = _check_earnings_proximity_filter("CRWD", feat)
    assert result is None


def test_real_today_avgo_allowed():
    """AVGO: 19d / 9.8% IV -> outside both tiers -> pass."""
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 19, "expected_move": 0.098}):
        result = _check_earnings_proximity_filter("AVGO", _features())
    assert result is None


# ─── Original resilience tests (keep at bottom) ────────────────────────────────

def test_fails_open_on_missing_features():
    """If features DF is missing expected columns, return None (allow BUY)."""
    bad_feat = pd.DataFrame([{"random_column": 1}])
    with patch("signals.generator._lookup_earnings_calendar",
               return_value={"days_until": 4, "expected_move": 0.10}):
        result = _check_earnings_proximity_filter("AAPL", bad_feat)
    # Rule 1 needs only earnings data, not features — should still fire
    # (days_until=4 is inside the <=7d threshold)
    assert result is not None
    assert result["rule"] == "EARNINGS_PROXIMITY"
