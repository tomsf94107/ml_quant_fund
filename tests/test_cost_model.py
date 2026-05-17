"""
Sprint W1 — transaction cost model tests for analysis/fitness_scorer.py.

Validates that compute_group_fitness now deducts a per-bar transaction
cost, that net <= gross, that zero-turnover => net == gross, and that the
ML_QUANT_COST_BPS env override works.

Run:  python -m pytest tests/test_cost_model.py -v
"""
import importlib
import math
import os
import pandas as pd
import pytest

import analysis.fitness_scorer as fs


def _group(prob_ups, returns, horizon=1):
    """Build a (ticker, horizon) group DataFrame like load_predictions yields.
    n must be >= MIN_OBS or compute_group_fitness returns None."""
    n = len(prob_ups)
    return pd.DataFrame({
        "ticker": ["TEST"] * n,
        "horizon": [horizon] * n,
        "prediction_date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "prob_up": prob_ups,
        "actual_return": returns,
    })


# ── constants ─────────────────────────────────────────────────────────────

def test_cost_constant_exists():
    assert hasattr(fs, "COST_PER_UNIT_TURNOVER_BPS")
    assert hasattr(fs, "_COST_RATE")
    # default 10 bps -> 0.001
    assert fs._COST_RATE == pytest.approx(0.001)


def test_fitnessrow_has_gross_field():
    """FitnessRow must carry gross_annualized_return."""
    fields = fs.FitnessRow.__dataclass_fields__
    assert "gross_annualized_return" in fields


# ── core cost behaviour ───────────────────────────────────────────────────

def test_net_below_gross_when_trading():
    """With real position changes, net annualized return < gross."""
    # alternating strong long / flat -> lots of turnover
    n = 40
    probs = [0.9 if i % 2 == 0 else 0.5 for i in range(n)]
    rets = [0.01] * n
    row = fs.compute_group_fitness(_group(probs, rets), mode="long_only")
    assert row is not None
    assert row.gross_annualized_return > row.annualized_return, \
        "cost should pull net below gross"


def test_zero_turnover_means_net_equals_gross_minus_entry():
    """A constant position has no ongoing turnover — net should equal gross
    except for the single initial-entry cost charged on bar 0."""
    n = 40
    probs = [0.9] * n           # constant strong-long => position never changes
    rets = [0.01] * n
    row = fs.compute_group_fitness(_group(probs, rets), mode="long_only")
    assert row is not None
    # only bar 0 is charged: total cost = _COST_RATE * 1.0 (entry), spread
    # across n bars then annualized.
    bars_per_year = fs.TRADING_DAYS_PER_YEAR / fs.HORIZON_DAYS[1]
    expected_drag = (fs._COST_RATE / n) * bars_per_year
    actual_drag = row.gross_annualized_return - row.annualized_return
    assert actual_drag == pytest.approx(expected_drag, rel=1e-6)


def test_higher_cost_reduces_net_more():
    """Raising the cost rate must widen the gross-net gap."""
    n = 40
    probs = [0.9 if i % 2 == 0 else 0.5 for i in range(n)]
    rets = [0.01] * n
    g = _group(probs, rets)

    row_default = fs.compute_group_fitness(g, mode="long_only")
    drag_default = row_default.gross_annualized_return - row_default.annualized_return

    # bump cost to 50 bps via env + reload
    os.environ["ML_QUANT_COST_BPS"] = "50.0"
    importlib.reload(fs)
    row_high = fs.compute_group_fitness(g, mode="long_only")
    drag_high = row_high.gross_annualized_return - row_high.annualized_return

    # restore
    os.environ.pop("ML_QUANT_COST_BPS", None)
    importlib.reload(fs)

    assert drag_high > drag_default
    # 50 bps is 5x of 10 bps -> drag ~5x
    assert drag_high == pytest.approx(5 * drag_default, rel=1e-6)


def test_env_override_changes_rate():
    """ML_QUANT_COST_BPS must drive _COST_RATE after reload."""
    os.environ["ML_QUANT_COST_BPS"] = "25.0"
    importlib.reload(fs)
    try:
        assert fs._COST_RATE == pytest.approx(0.0025)
    finally:
        os.environ.pop("ML_QUANT_COST_BPS", None)
        importlib.reload(fs)


def test_gross_unaffected_by_cost_rate():
    """Gross annualized return is pre-cost — changing the rate must NOT
    move it (only net should move)."""
    n = 40
    probs = [0.9 if i % 2 == 0 else 0.5 for i in range(n)]
    rets = [0.01] * n
    g = _group(probs, rets)

    gross_default = fs.compute_group_fitness(g, "long_only").gross_annualized_return

    os.environ["ML_QUANT_COST_BPS"] = "50.0"
    importlib.reload(fs)
    gross_high = fs.compute_group_fitness(g, "long_only").gross_annualized_return
    os.environ.pop("ML_QUANT_COST_BPS", None)
    importlib.reload(fs)

    assert gross_default == pytest.approx(gross_high), \
        "gross must not depend on the cost rate"


def test_min_obs_still_enforced():
    """Groups below MIN_OBS still return None (cost model must not change
    the sample-size gate)."""
    short = _group([0.9] * 5, [0.01] * 5)
    assert fs.compute_group_fitness(short, "long_only") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
