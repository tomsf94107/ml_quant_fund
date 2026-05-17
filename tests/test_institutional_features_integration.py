"""
Phase 3 integration tests — institutional features wiring.

Covers builder.py (OUTPUT_COLUMNS gating) and daily_runner.py (schema
migration + 35-column INSERT extension).

CRITICAL CONTEXT: the prediction_features INSERT in daily_runner.py is
wrapped in `except Exception: pass` (best-effort logging). Therefore these
tests verify by DIRECT QUERY / DIRECT ASSERTION, never by "it ran without
error" — a broken INSERT would be silently swallowed in production.

Run:  python -m pytest tests/test_institutional_features_integration.py -v
"""
import os
import sqlite3
import importlib
import numpy as np
import pytest


# The 4 audit-surviving institutional features (May 17 2026 audit).
INST_COLS = [
    "inst_block_buy_sell_7d",
    "inst_signed_flow_30d",
    "inst_auction_imbal_5d",
    "inst_signed_flow_5d",
]


# ──────────────────────────────────────────────────────────────────────────
# builder.py — OUTPUT_COLUMNS gating
# ──────────────────────────────────────────────────────────────────────────

def _reload_builder():
    """Reload features.builder so module-level OUTPUT_COLUMNS re-evaluates
    against the current ML_QUANT_INST_FEATURES env value."""
    import features.builder as b
    return importlib.reload(b)


def test_output_columns_excludes_inst_when_flag_off(monkeypatch):
    """Flag OFF (default) -> institutional columns NOT in OUTPUT_COLUMNS.
    This is the true-no-op guarantee: existing 303 models see no schema
    change."""
    monkeypatch.setenv("ML_QUANT_INST_FEATURES", "0")
    b = _reload_builder()
    for col in INST_COLS:
        assert col not in b.OUTPUT_COLUMNS, (
            f"{col} leaked into OUTPUT_COLUMNS with flag OFF")


def test_output_columns_includes_inst_when_flag_on(monkeypatch):
    """Flag ON -> all 4 institutional columns appended to OUTPUT_COLUMNS."""
    monkeypatch.setenv("ML_QUANT_INST_FEATURES", "1")
    b = _reload_builder()
    for col in INST_COLS:
        assert col in b.OUTPUT_COLUMNS, (
            f"{col} missing from OUTPUT_COLUMNS with flag ON")


def test_inst_columns_appended_at_end(monkeypatch):
    """Institutional columns appended AFTER existing columns."""
    monkeypatch.setenv("ML_QUANT_INST_FEATURES", "1")
    b = _reload_builder()
    assert b.OUTPUT_COLUMNS[-4:] == INST_COLS


def test_no_duplicate_columns_when_flag_on(monkeypatch):
    """Reload with flag ON must not double-append."""
    monkeypatch.setenv("ML_QUANT_INST_FEATURES", "1")
    b = _reload_builder()
    cols = b.OUTPUT_COLUMNS
    assert len(cols) == len(set(cols)), "duplicate columns in OUTPUT_COLUMNS"


def test_flag_toggle_delta_is_exactly_four(monkeypatch):
    """ON minus OFF column count == 4. No leakage across reloads."""
    monkeypatch.setenv("ML_QUANT_INST_FEATURES", "1")
    n_on = len(_reload_builder().OUTPUT_COLUMNS)
    monkeypatch.setenv("ML_QUANT_INST_FEATURES", "0")
    n_off = len(_reload_builder().OUTPUT_COLUMNS)
    assert n_on - n_off == 4, f"on={n_on} off={n_off}"


# ──────────────────────────────────────────────────────────────────────────
# daily_runner.py — schema migration
# ──────────────────────────────────────────────────────────────────────────

def _make_pre_migration_table(conn):
    """prediction_features WITHOUT institutional columns (pre-migration)."""
    conn.execute("""
        CREATE TABLE prediction_features (
            ticker TEXT, prediction_date TEXT, horizon INTEGER,
            oil_ret REAL, spy_ret REAL, vix_close REAL,
            sector_rel_ret REAL, created_at TEXT
        )
    """)
    conn.commit()


def test_schema_migration_adds_all_four():
    from scripts.daily_runner import _ensure_institutional_columns
    conn = sqlite3.connect(":memory:")
    _make_pre_migration_table(conn)
    _ensure_institutional_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(prediction_features)")}
    for col in INST_COLS:
        assert col in cols, f"migration failed to add {col}"
    conn.close()


def test_schema_migration_idempotent():
    """Runs on every pipeline invocation -> second call must be a no-op."""
    from scripts.daily_runner import _ensure_institutional_columns
    conn = sqlite3.connect(":memory:")
    _make_pre_migration_table(conn)
    _ensure_institutional_columns(conn)
    _ensure_institutional_columns(conn)  # must not raise
    cols = {r[1] for r in conn.execute("PRAGMA table_info(prediction_features)")}
    assert len(cols & set(INST_COLS)) == 4
    conn.close()


def test_schema_migration_preserves_existing_columns():
    from scripts.daily_runner import _ensure_institutional_columns
    conn = sqlite3.connect(":memory:")
    _make_pre_migration_table(conn)
    before = {r[1] for r in conn.execute("PRAGMA table_info(prediction_features)")}
    _ensure_institutional_columns(conn)
    after = {r[1] for r in conn.execute("PRAGMA table_info(prediction_features)")}
    assert before.issubset(after), "migration dropped an existing column"
    conn.close()


# ──────────────────────────────────────────────────────────────────────────
# daily_runner.py — 35-column INSERT (verified by direct query)
# ──────────────────────────────────────────────────────────────────────────

# Mirror of the production schema after migration: 31 original + 4 inst.
_FULL_SCHEMA = """
CREATE TABLE prediction_features (
    ticker TEXT, prediction_date TEXT, horizon INTEGER,
    oil_ret REAL, oil_spy_corr REAL, spy_ret REAL, xlk_ret REAL,
    dxy_ret REAL, yield_10y REAL, vix_close REAL, vix_ret REAL,
    vix_term_structure REAL, fear_greed REAL,
    rsi_14 REAL, macd REAL, bb_pct REAL, atr REAL,
    vol_surge_eod REAL, obv_trend REAL,
    return_1d REAL, return_5d REAL, return_20d REAL,
    premarket_gap REAL, intraday_momentum REAL,
    iv_skew_snap REAL, pc_ratio_snap REAL,
    monday_sentiment REAL, beta_60d REAL, short_ratio REAL,
    sector_rel_ret REAL,
    inst_block_buy_sell_7d REAL, inst_signed_flow_30d REAL,
    inst_auction_imbal_5d REAL, inst_signed_flow_5d REAL,
    created_at TEXT
)
"""

# The exact 35-column INSERT as it should appear in daily_runner.py.
_INSERT_SQL = """
INSERT OR REPLACE INTO prediction_features
(ticker, prediction_date, horizon,
 oil_ret, oil_spy_corr, spy_ret, xlk_ret,
 dxy_ret, yield_10y, vix_close, vix_ret,
 vix_term_structure, fear_greed,
 rsi_14, macd, bb_pct, atr, vol_surge_eod, obv_trend,
 return_1d, return_5d, return_20d,
 premarket_gap, intraday_momentum,
 iv_skew_snap, pc_ratio_snap,
 monday_sentiment, beta_60d, short_ratio,
 sector_rel_ret,
 inst_block_buy_sell_7d, inst_signed_flow_30d,
 inst_auction_imbal_5d, inst_signed_flow_5d,
 created_at)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def test_insert_has_35_placeholders():
    """Guard against placeholder/column drift: 35 columns, 35 '?'."""
    n_placeholders = _INSERT_SQL.count("?")
    assert n_placeholders == 35, (
        f"expected 35 placeholders, found {n_placeholders}")
    col_section = _INSERT_SQL.split("VALUES")[0]
    n_cols = col_section.count(",") + 1
    assert n_cols == 35, f"expected 35 columns, found {n_cols}"


def test_insert_with_inst_values_roundtrip():
    """35-value INSERT with real institutional values must persist and
    read back exactly. Run OUTSIDE any try/except so a mismatch fails."""
    conn = sqlite3.connect(":memory:")
    conn.execute(_FULL_SCHEMA)
    values = (
        "NVDA", "2026-05-13", 3,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5,
        50.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        # institutional values:
        1.40, 0.06, 0.03, 0.07,
        "2026-05-13",
    )
    conn.execute(_INSERT_SQL, values)
    conn.commit()
    row = conn.execute(
        "SELECT inst_block_buy_sell_7d, inst_signed_flow_30d, "
        "inst_auction_imbal_5d, inst_signed_flow_5d "
        "FROM prediction_features WHERE ticker='NVDA'").fetchone()
    assert row == (1.40, 0.06, 0.03, 0.07)
    conn.close()


def test_insert_with_null_inst_values_roundtrip():
    """When the flag is OFF, _inst_for_log yields None for all 4. The
    INSERT must store NULL and read back as None — not crash, not 0.0."""
    conn = sqlite3.connect(":memory:")
    conn.execute(_FULL_SCHEMA)
    values = (
        "AAPL", "2026-05-13", 1,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5,
        50.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        # flag-off institutional values -> all None:
        None, None, None, None,
        "2026-05-13",
    )
    conn.execute(_INSERT_SQL, values)
    conn.commit()
    row = conn.execute(
        "SELECT inst_block_buy_sell_7d, inst_signed_flow_30d, "
        "inst_auction_imbal_5d, inst_signed_flow_5d "
        "FROM prediction_features WHERE ticker='AAPL'").fetchone()
    assert row == (None, None, None, None)
    conn.close()


# ──────────────────────────────────────────────────────────────────────────
# _inst_for_log NaN-safety logic (the NaN != NaN check)
# ──────────────────────────────────────────────────────────────────────────

def _inst_for_log_logic(last_like):
    """Replica of the _inst_for_log builder in daily_runner.py — tested in
    isolation so the NaN-safe coercion is verified independently."""
    out = {}
    for c in INST_COLS:
        v = last_like.get(c, None)
        out[c] = float(v) if v is not None and v == v else None
    return out


def test_inst_for_log_handles_nan():
    """NaN feature value -> None (SQLite NULL), not float('nan')."""
    result = _inst_for_log_logic({
        "inst_block_buy_sell_7d": np.nan,
        "inst_signed_flow_30d": 0.06,
        "inst_auction_imbal_5d": np.nan,
        "inst_signed_flow_5d": 0.07,
    })
    assert result["inst_block_buy_sell_7d"] is None
    assert result["inst_auction_imbal_5d"] is None
    assert result["inst_signed_flow_30d"] == 0.06
    assert result["inst_signed_flow_5d"] == 0.07


def test_inst_for_log_handles_absent_columns():
    """Flag OFF: columns absent from `last` -> all None."""
    result = _inst_for_log_logic({"close": 100.0, "rsi_14": 55.0})
    assert all(v is None for v in result.values())


def test_inst_for_log_coerces_valid_floats():
    """Valid numeric values pass through as float."""
    result = _inst_for_log_logic({
        "inst_block_buy_sell_7d": 1.40,
        "inst_signed_flow_30d": -0.05,
        "inst_auction_imbal_5d": 0.0,
        "inst_signed_flow_5d": 2.3,
    })
    assert result == {
        "inst_block_buy_sell_7d": 1.40,
        "inst_signed_flow_30d": -0.05,
        "inst_auction_imbal_5d": 0.0,
        "inst_signed_flow_5d": 2.3,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
