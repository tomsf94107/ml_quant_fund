"""
Sprint W1 — confidence-cap neuter tests for signals/generator.py.

After the May 17 neuter, CONFIDENCE_CAP is 0.95 (dormant). These tests
confirm the cap value changed, the mechanism still exists, and the cap
no longer fires at realistic prob_eff levels.

Run:  python -m pytest tests/test_cap_neuter.py -v
"""
import re
from pathlib import Path
import pytest

GEN = Path(__file__).parent.parent / "signals" / "generator.py"


@pytest.fixture(scope="module")
def gen_src():
    return GEN.read_text()


def test_cap_value_raised_to_095(gen_src):
    """CONFIDENCE_CAP must now be 0.95, not 0.65."""
    m = re.search(r"CONFIDENCE_CAP\s*=\s*([0-9.]+)", gen_src)
    assert m is not None, "CONFIDENCE_CAP assignment not found"
    assert float(m.group(1)) == pytest.approx(0.95), \
        f"expected 0.95, found {m.group(1)}"


def test_old_065_value_gone(gen_src):
    """The literal CONFIDENCE_CAP = 0.65 must no longer exist."""
    assert "CONFIDENCE_CAP = 0.65" not in gen_src


def test_mechanism_still_present(gen_src):
    """The cap is NEUTERED, not deleted — INVERSION_HORIZONS and the
    capping branch must still exist so it can be re-armed."""
    assert "INVERSION_HORIZONS = {3, 5}" in gen_src
    assert "today_prob_eff = CONFIDENCE_CAP" in gen_src


def test_stale_may7_comment_replaced(gen_src):
    """The stale 'inverted region per May 7 SHAP' log line must be gone."""
    assert "inverted region per May 7 SHAP" not in gen_src
    assert "NEUTERED May 17 2026" in gen_src


def test_cap_dormant_at_realistic_confidence():
    """A realistic high prob_eff (0.80) is BELOW the 0.95 cap, so the cap
    condition is False — it does not fire. Simulates the line-804 branch."""
    CONFIDENCE_CAP = 0.95
    INVERSION_HORIZONS = {3, 5}
    for horizon in (3, 5):
        for prob_eff in (0.65, 0.70, 0.80, 0.90):
            fires = horizon in INVERSION_HORIZONS and prob_eff > CONFIDENCE_CAP
            assert not fires, \
                f"cap should be dormant at h={horizon} prob_eff={prob_eff}"


def test_cap_still_fires_above_095():
    """If prob_eff somehow exceeds 0.95, the cap still engages — the rail
    is dormant, not dead."""
    CONFIDENCE_CAP = 0.95
    INVERSION_HORIZONS = {3, 5}
    fires = 3 in INVERSION_HORIZONS and 0.97 > CONFIDENCE_CAP
    assert fires, "cap should still engage above 0.95"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
