# tests/test_insider_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for scripts/insider_classifier.py.
# Run with:  python -m unittest tests.test_insider_classifier -v
# Or:        python tests/test_insider_classifier.py
#
# Pure-logic tests, no DB, no network. Should run in <1 second.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import unittest

# Allow running both as a module and as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from insider_classifier import (
    classify, classify_cluster,
    GREEN_STRONG, GREEN_WEAK, RED_STRONG, RED_WEAK, SIGNAL_RANK,
    GREEN_STRONG_CSUITE_BUY_USD, GREEN_WEAK_CSUITE_BUY_USD,
    GREEN_WEAK_ANY_BUY_USD,
    RED_STRONG_CSUITE_SELL_USD, RED_WEAK_CSUITE_SELL_USD,
)


def _filing(**overrides) -> dict:
    """Helper: build a baseline filing and override fields per test."""
    base = {
        "ticker":            "AAPL",
        "insider_name":      "Cook, Timothy D.",
        "insider_title":     "Chief Executive Officer",
        "role_weight":       3.0,
        "transaction_code":  "P",
        "shares":            1000.0,
        "price_per_share":   200.0,
        "notional_usd":      200_000.0,
        "acquired_disposed": "A",
        "is_csuite":         1,
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
#  GREEN signals — buys
# ══════════════════════════════════════════════════════════════════════════════

class TestGreenSignals(unittest.TestCase):

    def test_csuite_big_buy_is_green_strong(self):
        f = _filing(notional_usd=500_000.0)  # well above STRONG threshold
        result = classify(f)
        self.assertIsNotNone(result)
        signal, rationale = result
        self.assertEqual(signal, GREEN_STRONG)
        self.assertIn("AAPL", rationale)
        self.assertIn("$500,000", rationale)

    def test_csuite_at_strong_threshold_is_green_strong(self):
        f = _filing(notional_usd=GREEN_STRONG_CSUITE_BUY_USD)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_STRONG)

    def test_csuite_just_below_strong_threshold_is_green_weak(self):
        f = _filing(notional_usd=GREEN_STRONG_CSUITE_BUY_USD - 1)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_WEAK)

    def test_csuite_at_weak_threshold_is_green_weak(self):
        f = _filing(notional_usd=GREEN_WEAK_CSUITE_BUY_USD)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_WEAK)

    def test_csuite_just_below_weak_threshold_is_none(self):
        f = _filing(notional_usd=GREEN_WEAK_CSUITE_BUY_USD - 1)
        result = classify(f)
        self.assertIsNone(result)

    def test_huge_buy_by_director_is_green_weak(self):
        # Director (role_weight 1.0, is_csuite 0) buying $2M = still meaningful
        f = _filing(
            insider_title="Director",
            role_weight=1.0,
            is_csuite=0,
            notional_usd=2_000_000.0,
        )
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_WEAK)

    def test_small_buy_by_director_is_none(self):
        f = _filing(
            insider_title="Director",
            role_weight=1.0,
            is_csuite=0,
            notional_usd=100_000.0,
        )
        result = classify(f)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
#  RED signals — sells
# ══════════════════════════════════════════════════════════════════════════════

class TestRedSignals(unittest.TestCase):

    def test_csuite_big_sell_is_red_strong(self):
        f = _filing(
            transaction_code="S",
            acquired_disposed="D",
            notional_usd=2_000_000.0,
        )
        signal, rationale = classify(f)
        self.assertEqual(signal, RED_STRONG)
        self.assertIn("SELL", rationale)

    def test_csuite_medium_sell_is_red_weak(self):
        f = _filing(
            transaction_code="S",
            acquired_disposed="D",
            notional_usd=500_000.0,
        )
        signal, _ = classify(f)
        self.assertEqual(signal, RED_WEAK)

    def test_csuite_small_sell_is_none(self):
        # $100k sell doesn't clear even RED_WEAK threshold
        f = _filing(
            transaction_code="S",
            acquired_disposed="D",
            notional_usd=100_000.0,
        )
        result = classify(f)
        self.assertIsNone(result)

    def test_director_sell_never_alerts(self):
        # Non-csuite sells are noise (RSU vests). Even a $5M director sell is None.
        f = _filing(
            insider_title="Director",
            role_weight=1.0,
            is_csuite=0,
            transaction_code="S",
            acquired_disposed="D",
            notional_usd=5_000_000.0,
        )
        result = classify(f)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Transaction code filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestTransactionCodeFilter(unittest.TestCase):
    """Only P (purchase) and S (sale) trigger alerts. Skip A/G/F/M/etc."""

    def test_grant_code_A_is_none(self):
        f = _filing(transaction_code="A", acquired_disposed="A",
                    notional_usd=10_000_000.0)
        self.assertIsNone(classify(f))

    def test_gift_code_G_is_none(self):
        f = _filing(transaction_code="G", acquired_disposed="A")
        self.assertIsNone(classify(f))

    def test_exercise_code_M_is_none(self):
        f = _filing(transaction_code="M", acquired_disposed="A")
        self.assertIsNone(classify(f))

    def test_tax_withholding_F_is_none(self):
        f = _filing(transaction_code="F", acquired_disposed="D")
        self.assertIsNone(classify(f))

    def test_lowercase_code_handled(self):
        f = _filing(transaction_code="p", acquired_disposed="a",
                    notional_usd=500_000.0)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_STRONG)


# ══════════════════════════════════════════════════════════════════════════════
#  Notional fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionalFallback(unittest.TestCase):
    """When notional_usd is missing, fall back to shares * price_per_share,
    then shares * current_price."""

    def test_uses_filing_price_when_notional_missing(self):
        f = _filing(notional_usd=None, shares=1000, price_per_share=300.0)
        # 1000 * 300 = 300k → GREEN_STRONG (>= 250k)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_STRONG)

    def test_uses_current_price_when_filing_price_also_missing(self):
        f = _filing(notional_usd=None, shares=1000, price_per_share=None)
        signal, _ = classify(f, current_price=300.0)
        self.assertEqual(signal, GREEN_STRONG)

    def test_returns_none_when_no_price_available(self):
        f = _filing(notional_usd=None, shares=1000, price_per_share=None)
        result = classify(f, current_price=None)
        self.assertIsNone(result)

    def test_returns_none_when_shares_zero(self):
        f = _filing(shares=0, notional_usd=0)
        self.assertIsNone(classify(f))


# ══════════════════════════════════════════════════════════════════════════════
#  C-suite detection
# ══════════════════════════════════════════════════════════════════════════════

class TestCsuiteDetection(unittest.TestCase):
    """is_csuite=1 flag OR role_weight >= 2.5 should both qualify as C-suite."""

    def test_role_weight_3_qualifies_even_if_flag_zero(self):
        f = _filing(is_csuite=0, role_weight=3.0, notional_usd=300_000.0)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_STRONG)

    def test_role_weight_2_5_qualifies(self):
        f = _filing(is_csuite=0, role_weight=2.5, notional_usd=300_000.0)
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_STRONG)

    def test_role_weight_2_does_not_qualify_as_csuite(self):
        # role_weight 2.0 = CTO/CIO. Important officer but below csuite cutoff.
        # A 300k buy needs csuite for GREEN_STRONG. Should fall through to checks.
        f = _filing(is_csuite=0, role_weight=2.0, notional_usd=300_000.0)
        # 300k is below GREEN_WEAK_ANY_BUY_USD (1M) — should be None
        result = classify(f)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Cluster detection
# ══════════════════════════════════════════════════════════════════════════════

class TestClusterSignal(unittest.TestCase):

    def test_three_csuite_buyers_triggers_green_strong(self):
        filings = [
            _filing(insider_name="A", notional_usd=100_000),
            _filing(insider_name="B", notional_usd=100_000),
            _filing(insider_name="C", notional_usd=100_000),
        ]
        result = classify_cluster(filings)
        self.assertIsNotNone(result)
        signal, rationale = result
        self.assertEqual(signal, GREEN_STRONG)
        self.assertIn("Cluster", rationale)
        self.assertIn("3", rationale)

    def test_two_csuite_buyers_does_not_trigger(self):
        filings = [
            _filing(insider_name="A", notional_usd=100_000),
            _filing(insider_name="B", notional_usd=100_000),
        ]
        self.assertIsNone(classify_cluster(filings))

    def test_three_buys_by_one_insider_does_not_trigger(self):
        # Cluster requires DISTINCT insiders, not multiple trades by one person
        filings = [
            _filing(insider_name="A", notional_usd=100_000),
            _filing(insider_name="A", notional_usd=100_000),
            _filing(insider_name="A", notional_usd=100_000),
        ]
        self.assertIsNone(classify_cluster(filings))

    def test_cluster_ignores_non_open_market(self):
        # 3 grants (code A) by 3 csuite people — not a real cluster
        filings = [
            _filing(insider_name="A", transaction_code="A"),
            _filing(insider_name="B", transaction_code="A"),
            _filing(insider_name="C", transaction_code="A"),
        ]
        self.assertIsNone(classify_cluster(filings))

    def test_cluster_ignores_directors(self):
        # 3 directors buying isn't as strong; cluster signal requires csuite
        filings = [
            _filing(insider_name="A", role_weight=1.0, is_csuite=0,
                    notional_usd=100_000),
            _filing(insider_name="B", role_weight=1.0, is_csuite=0,
                    notional_usd=100_000),
            _filing(insider_name="C", role_weight=1.0, is_csuite=0,
                    notional_usd=100_000),
        ]
        self.assertIsNone(classify_cluster(filings))

    def test_empty_list_returns_none(self):
        self.assertIsNone(classify_cluster([]))


# ══════════════════════════════════════════════════════════════════════════════
#  Signal ranking
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalRank(unittest.TestCase):
    def test_strong_outranks_weak(self):
        self.assertGreater(SIGNAL_RANK[GREEN_STRONG], SIGNAL_RANK[GREEN_WEAK])
        self.assertGreater(SIGNAL_RANK[RED_STRONG], SIGNAL_RANK[RED_WEAK])

    def test_green_strong_outranks_red_strong(self):
        # Buys are rarer & more meaningful → if both fire, GREEN wins.
        self.assertGreater(SIGNAL_RANK[GREEN_STRONG], SIGNAL_RANK[RED_STRONG])


# ══════════════════════════════════════════════════════════════════════════════
#  Realistic scenarios from your DB
# ══════════════════════════════════════════════════════════════════════════════

class TestRealisticScenarios(unittest.TestCase):
    """Cases inspired by the real data in your insider_trades.db."""

    def test_apple_cook_typical_rsu_sell(self):
        # Cook does Rule 10b5-1 sells regularly. Big notional → RED_STRONG
        # (caller will decide whether to suppress based on 10b5-1 metadata
        #  in a future enhancement)
        f = _filing(
            ticker="AAPL",
            insider_name="Cook, Timothy D.",
            insider_title="Chief Executive Officer",
            role_weight=3.0,
            transaction_code="S",
            acquired_disposed="D",
            shares=50_000,
            price_per_share=200.0,
            notional_usd=10_000_000.0,
            is_csuite=1,
        )
        signal, _ = classify(f)
        self.assertEqual(signal, RED_STRONG)

    def test_meta_director_small_buy_filtered(self):
        # Director buying $20k. Below all thresholds. None.
        f = _filing(
            ticker="META",
            insider_title="Director",
            role_weight=1.0,
            is_csuite=0,
            notional_usd=20_000.0,
        )
        self.assertIsNone(classify(f))

    def test_nvda_cfo_meaningful_buy(self):
        # CFO open-market buy $100k → GREEN_WEAK (above 50k csuite weak threshold)
        f = _filing(
            ticker="NVDA",
            insider_title="Chief Financial Officer",
            role_weight=3.0,
            transaction_code="P",
            acquired_disposed="A",
            notional_usd=100_000.0,
        )
        signal, _ = classify(f)
        self.assertEqual(signal, GREEN_WEAK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
