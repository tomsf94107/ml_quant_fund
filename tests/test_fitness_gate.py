"""
Task D — per-horizon fitness gate tests.

Validates _check_fitness_filter() after the May 17 change from a global
`fitness < 0` cut to per-horizon thresholds _FITNESS_GATE_THRESHOLDS.

The function reads accuracy.db.fitness_scores. To test in isolation we
monkeypatch a temporary DB into place via the Path the function builds
(parent.parent / "accuracy.db"). Rather than fight that path construction,
these tests insert known rows into the REAL fitness_scores schema in a
temp DB and point the function at it by monkeypatching sqlite3.connect.

Run:  python -m pytest tests/test_fitness_gate.py -v
"""
import sqlite3
import pytest
import signals.generator as gen


# ── Threshold constant sanity ─────────────────────────────────────────────

def test_thresholds_are_per_horizon():
    """_FITNESS_GATE_THRESHOLDS must define h=1, h=3, h=5 with the
    conservative Task D values."""
    t = gen._FITNESS_GATE_THRESHOLDS
    assert t == {1: 1.0, 3: 3.0, 5: 3.0}, f"unexpected thresholds: {t}"


def test_h1_threshold_lower_than_h3():
    """h=1 cut must be looser than h=3 — the Task D finding is that h=1
    winners and losers overlap, so h=1 is deliberately lenient."""
    t = gen._FITNESS_GATE_THRESHOLDS
    assert t[1] < t[3], "h=1 threshold should be below h=3"


# ── _check_fitness_filter behaviour ───────────────────────────────────────

@pytest.fixture
def fitness_db(tmp_path, monkeypatch):
    """Build a temp accuracy.db with a fitness_scores table, and make
    _check_fitness_filter connect to it."""
    db = tmp_path / "accuracy.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE fitness_scores (
            ticker TEXT, horizon INTEGER, n_obs INTEGER, win_rate REAL,
            avg_return_per_bar REAL, annualized_return REAL,
            annualized_vol REAL, sharpe REAL, turnover REAL, fitness REAL
        )
    """)
    # Rows mirroring real Task D Stage 3 data.
    rows = [
        # ticker, horizon, fitness  (other cols 0 for brevity)
        ("ORIC", 3, -10.79),   # clear loser, h=3
        ("ORIC", 1,   0.36),   # loser, h=1, below 1.0
        ("XLV",  3,  -0.15),   # loser, h=3
        ("XLV",  1,   0.17),   # loser, h=1, below 1.0
        ("GLD",  3,   1.10),   # loser, h=3, below 3.0
        ("GLD",  1,   5.24),   # loser h=1 but ABOVE 1.0 — deliberately passes
        ("HUM",  3,  48.22),   # strong winner
        ("BA",   1,   1.55),   # marginal winner h=1, above 1.0 — passes
        ("CRWD", 1,   2.35),   # winner h=1
    ]
    for tk, h, fit in rows:
        conn.execute(
            "INSERT INTO fitness_scores "
            "(ticker,horizon,n_obs,win_rate,avg_return_per_bar,"
            "annualized_return,annualized_vol,sharpe,turnover,fitness) "
            "VALUES (?,?,30,0,0,0,0,0,0,?)", (tk, h, fit))
    conn.commit()
    conn.close()

    # Redirect the function's DB lookup to our temp DB.
    real_connect = sqlite3.connect

    def fake_connect(path, *a, **k):
        # The function builds parent.parent/"accuracy.db"; intercept any
        # path ending in accuracy.db and swap in our temp DB.
        if str(path).endswith("accuracy.db"):
            return real_connect(str(db), *a, **k)
        return real_connect(path, *a, **k)

    monkeypatch.setattr(gen.sqlite3, "connect", fake_connect)
    # Also ensure the existence check passes: patch Path.exists via the
    # function's own construction is awkward; instead rely on the temp DB
    # actually existing at a real path is enough IF the function's _db.exists()
    # points elsewhere. To be safe, patch Path.exists to True.
    import pathlib
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
    return db


def test_suppresses_clear_loser_h3(fitness_db):
    """ORIC h=3 fitness -10.79 is far below the 3.0 cut — must suppress."""
    result = gen._check_fitness_filter("ORIC", 3)
    assert result is not None
    assert result == pytest.approx(-10.79)


def test_suppresses_xlv_h3(fitness_db):
    """XLV h=3 fitness -0.15 < 3.0 — suppress."""
    assert gen._check_fitness_filter("XLV", 3) == pytest.approx(-0.15)


def test_suppresses_gld_h3_below_threshold(fitness_db):
    """GLD h=3 fitness 1.10 < 3.0 — suppress (positive but sub-threshold;
    the old `fitness < 0` rule would have WRONGLY allowed this)."""
    assert gen._check_fitness_filter("GLD", 3) == pytest.approx(1.10)


def test_allows_gld_h1_above_threshold(fitness_db):
    """GLD h=1 fitness 5.24 >= 1.0 — deliberately ALLOWED. Catching it would
    require a threshold that also kills BA/CRWD/AAPL h=1 winners."""
    assert gen._check_fitness_filter("GLD", 1) is None


def test_suppresses_loser_h1_below_threshold(fitness_db):
    """ORIC h=1 (0.36) and XLV h=1 (0.17) are below the 1.0 h=1 cut."""
    assert gen._check_fitness_filter("ORIC", 1) == pytest.approx(0.36)
    assert gen._check_fitness_filter("XLV", 1) == pytest.approx(0.17)


def test_allows_marginal_winner_h1(fitness_db):
    """BA h=1 fitness 1.55 >= 1.0 — a marginal winner must pass."""
    assert gen._check_fitness_filter("BA", 1) is None


def test_allows_strong_winner_h3(fitness_db):
    """HUM h=3 fitness 48.22 — obviously passes."""
    assert gen._check_fitness_filter("HUM", 3) is None


def test_unscored_ticker_passes(fitness_db):
    """A ticker absent from fitness_scores (e.g. EME, STX, ETN — all
    <MIN_OBS=30) must return None — un-scored names are NOT suppressed.
    This is the deliberate Task D choice protecting un-scored winners."""
    assert gen._check_fitness_filter("EME", 3) is None
    assert gen._check_fitness_filter("ETN", 1) is None


def test_unscored_horizon_passes(fitness_db):
    """HUM has an h=3 row but no h=5 row — h=5 lookup returns None."""
    assert gen._check_fitness_filter("HUM", 5) is None


def test_fails_open_on_db_error(monkeypatch):
    """Any DB exception → None (allow BUY). The gate must never suppress
    because of an infrastructure problem."""
    def boom(*a, **k):
        raise sqlite3.OperationalError("simulated DB failure")
    monkeypatch.setattr(gen.sqlite3, "connect", boom)
    import pathlib
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)
    assert gen._check_fitness_filter("ORIC", 3) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
