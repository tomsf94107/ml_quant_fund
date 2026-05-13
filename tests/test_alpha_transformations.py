"""
Unit tests for features/alpha_transformations.py.

Verifies:
  - Shape preservation (output panel has same shape as input)
  - NaN handling (NaN inputs don't crash; appropriate NaN propagation)
  - Mathematical correctness on small known examples
  - No future leakage in ts_* operators (causal/rolling only)
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.alpha_transformations import (
    cs_rank, cs_zscore, cs_mean, cs_std, cs_demean,
    ts_mean, ts_std, ts_rank, ts_delta, ts_corr,
    ts_max, ts_min, ts_argmax, ts_decay_linear, ts_sma,
    signed_power, scale,
    group_neutralize, indneutralize,
    ALPHA_OPS,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def panel():
    """30 days × 5 tickers panel, mix of values + some NaNs."""
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(30, 5),
        index=dates, columns=tickers,
    )
    # Inject a couple NaNs
    df.iloc[0, 0] = np.nan
    df.iloc[5, 2] = np.nan
    return df


@pytest.fixture
def tiny_panel():
    """Small, deterministic panel for math checks."""
    return pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0],
         [4.0, 3.0, 2.0, 1.0],
         [2.0, 2.0, 2.0, 2.0]],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
        columns=["A", "B", "C", "D"],
    )


# ─── Cross-sectional tests ───────────────────────────────────────────────────

class TestCrossSectional:
    def test_cs_rank_shape(self, panel):
        out = cs_rank(panel)
        assert out.shape == panel.shape

    def test_cs_rank_pct(self, tiny_panel):
        # Row 1: [1, 2, 3, 4] → percentile ranks [0.25, 0.5, 0.75, 1.0]
        out = cs_rank(tiny_panel, pct=True)
        np.testing.assert_allclose(out.iloc[0].values, [0.25, 0.5, 0.75, 1.0])

    def test_cs_rank_ties(self, tiny_panel):
        # Row 3: all 2.0 → all rank 0.625 (average rank when tied)
        out = cs_rank(tiny_panel, pct=True)
        assert (out.iloc[2] == out.iloc[2, 0]).all()  # all equal

    def test_cs_zscore_zero_mean(self, panel):
        out = cs_zscore(panel)
        # Each row should have mean ~0 (within float tolerance)
        row_means = out.mean(axis=1).dropna()
        np.testing.assert_allclose(row_means.values, 0, atol=1e-10)

    def test_cs_demean(self, tiny_panel):
        # Row 1: [1,2,3,4], mean=2.5 → demeaned [-1.5,-0.5,0.5,1.5]
        out = cs_demean(tiny_panel)
        np.testing.assert_allclose(
            out.iloc[0].values, [-1.5, -0.5, 0.5, 1.5]
        )

    def test_cs_mean_returns_series(self, panel):
        out = cs_mean(panel)
        assert isinstance(out, pd.Series)
        assert len(out) == len(panel)

    def test_cs_std_returns_series(self, panel):
        out = cs_std(panel)
        assert isinstance(out, pd.Series)


# ─── Time-series tests ──────────────────────────────────────────────────────

class TestTimeSeries:
    def test_ts_mean_shape(self, panel):
        out = ts_mean(panel, window=5)
        assert out.shape == panel.shape

    def test_ts_mean_no_lookahead(self, panel):
        # Value at index t should depend only on rows [t-window+1, t].
        # Modify a future value and check past output unchanged.
        out_before = ts_mean(panel, window=5).iloc[10].copy()
        panel.iloc[20, 0] = 9999.0  # mutate future
        out_after = ts_mean(panel, window=5).iloc[10]
        np.testing.assert_allclose(out_before.values, out_after.values)

    def test_ts_delta(self, tiny_panel):
        # Row 1 - Row 0: [4-1, 3-2, 2-3, 1-4] = [3, 1, -1, -3]
        out = ts_delta(tiny_panel, window=1)
        np.testing.assert_allclose(out.iloc[1].values, [3, 1, -1, -3])

    def test_ts_max(self, tiny_panel):
        # Row 1 max of [Row0, Row1] for each col
        # A: max(1, 4) = 4, B: max(2, 3) = 3, C: max(3, 2) = 3, D: max(4, 1) = 4
        out = ts_max(tiny_panel, window=2)
        np.testing.assert_allclose(out.iloc[1].values, [4, 3, 3, 4])

    def test_ts_min(self, tiny_panel):
        out = ts_min(tiny_panel, window=2)
        np.testing.assert_allclose(out.iloc[1].values, [1, 2, 2, 1])

    def test_ts_rank_returns_pct(self, panel):
        out = ts_rank(panel, window=10)
        # ts_rank returns values in [0, 1] (percentile)
        non_nan = out.dropna()
        assert (non_nan.values >= 0).all()
        assert (non_nan.values <= 1).all()

    def test_ts_corr_self_is_one(self, panel):
        # corr(x, x) = 1 wherever defined
        out = ts_corr(panel, panel, window=10)
        non_nan = out.dropna()
        # Some near-zero variance windows may produce NaN — filter
        finite = non_nan.replace([np.inf, -np.inf], np.nan).dropna()
        np.testing.assert_allclose(finite.values, 1.0, atol=1e-6)

    def test_ts_argmax_today(self, tiny_panel):
        # In tiny_panel, ts_argmax(window=3) at row 2: 
        # A col history [1, 4, 2] → max is at index 1, which is 1 day ago → 1
        out = ts_argmax(tiny_panel, window=3)
        # Row 2 A: history [1.0, 4.0, 2.0], max at position 1, so days-ago=1
        assert out.iloc[2]["A"] == 1.0

    def test_ts_decay_linear_emphasizes_recent(self, tiny_panel):
        # A col history [1, 4, 2] with window=3
        # weights = [1,2,3]/6, decay = (1*1 + 4*2 + 2*3)/6 = (1+8+6)/6 = 2.5
        out = ts_decay_linear(tiny_panel, window=3)
        assert abs(out.iloc[2]["A"] - 2.5) < 1e-9

    def test_ts_sma_alias(self, panel):
        # ts_sma should be identical to ts_mean
        a = ts_sma(panel, window=5)
        b = ts_mean(panel, window=5)
        pd.testing.assert_frame_equal(a, b)


# ─── Pointwise tests ────────────────────────────────────────────────────────

class TestPointwise:
    def test_signed_power_preserves_sign(self, panel):
        out = signed_power(panel, p=0.5)
        # Signs match input
        sign_match = (np.sign(out.fillna(0)) == np.sign(panel.fillna(0)))
        assert sign_match.values.all()

    def test_signed_power_zero(self):
        df = pd.DataFrame({"A": [0.0, 1.0, -1.0, 4.0, -4.0]})
        out = signed_power(df, p=0.5)
        # |4|^0.5 = 2, signed → 2; |-4|^0.5 = 2, signed → -2
        np.testing.assert_allclose(
            out["A"].values, [0.0, 1.0, -1.0, 2.0, -2.0]
        )

    def test_scale_unit_sum(self, panel):
        out = scale(panel, target_sum=1.0)
        abs_sums = out.abs().sum(axis=1).dropna()
        np.testing.assert_allclose(abs_sums.values, 1.0, atol=1e-10)


# ─── Group / neutralization tests ───────────────────────────────────────────

class TestGroupNeutralize:
    def test_group_neutralize_subtracts_group_mean(self):
        df = pd.DataFrame({
            "AAPL": [10.0, 20.0],
            "MSFT": [12.0, 22.0],
            "XOM":  [50.0, 60.0],
            "CVX":  [52.0, 62.0],
        })
        groups = {"AAPL": "tech", "MSFT": "tech", "XOM": "energy", "CVX": "energy"}
        out = group_neutralize(df, groups)
        # tech mean row 0 = 11, energy mean row 0 = 51
        # AAPL[0] = 10 - 11 = -1, MSFT[0] = 12 - 11 = 1
        # XOM[0] = 50 - 51 = -1, CVX[0] = 52 - 51 = 1
        assert out["AAPL"][0] == -1
        assert out["MSFT"][0] == 1
        assert out["XOM"][0]  == -1
        assert out["CVX"][0]  == 1

    def test_indneutralize_alias(self):
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        groups = {"A": "g1", "B": "g1"}
        a = group_neutralize(df, groups)
        b = indneutralize(df, groups)
        pd.testing.assert_frame_equal(a, b)

    def test_group_neutralize_passthrough(self):
        # Ticker not in groups dict → passes through unchanged
        df = pd.DataFrame({"A": [1.0], "ORPHAN": [99.0]})
        groups = {"A": "g1"}
        out = group_neutralize(df, groups)
        assert out["ORPHAN"][0] == 99.0


# ─── Registry test ──────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_registry_ops_callable(self, panel):
        """Every op in ALPHA_OPS should be callable with its default window."""
        for op_name, (fn, default_windows) in ALPHA_OPS.items():
            if default_windows is None:
                # cs_* and pointwise ops
                try:
                    out = fn(panel) if op_name != "signed_power" else fn(panel, p=0.5)
                except Exception as e:
                    pytest.fail(f"{op_name} failed without window: {e}")
            else:
                # ts_* ops — try with first default window
                window = default_windows[0]
                try:
                    out = fn(panel, window=window)
                except Exception as e:
                    pytest.fail(f"{op_name}(window={window}) failed: {e}")
