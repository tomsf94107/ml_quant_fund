"""
Smoke test for recession.db schema v1.

Verifies:
  - schema applies cleanly to a fresh DB
  - registries seed correctly (counts match spec)
  - T3 and T4c are flagged exploratory; T1/T2/T4a/T4b are not
  - vintage-aware uniqueness constraints work (same month + different vintage = OK)
  - same (target, horizon, model, sample, prediction_date, run_id) is unique
  - foreign keys are enforced
  - the v_features_latest view returns only the latest vintage
  - the v_predictions_latest view filters out failed runs
  - re-running migrate is idempotent (no duplicate rows, no errors)

Run:
    python -m recession.tests.test_schema
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make the recession package importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from recession.db.migrate import (                                  # noqa: E402
    FEATURES_REGISTRY_SEED,
    TARGETS_REGISTRY_SEED,
    TRIGGERS_REGISTRY_SEED,
    run_migration,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_passed: list[str] = []
_failed: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        _passed.append(name)
        print(f"  PASS  {name}")
    else:
        _failed.append((name, detail))
        print(f"  FAIL  {name}  ({detail})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fresh_migration(db_path: Path) -> None:
    print("\n[1] Fresh migration applies cleanly")
    run_migration(db_path)
    conn = _connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        check("schema_version_is_1", version == 1, f"got {version}")

        # Tables exist
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        expected_tables = {
            "schema_migrations", "runs",
            "features_registry", "targets_registry", "triggers_registry",
            "features_monthly", "targets_monthly", "triggers_monthly",
            "predictions", "metrics", "model_comparisons", "model_artifacts",
        }
        missing = expected_tables - tables
        check("all_expected_tables_exist", not missing, f"missing: {missing}")

        # Views exist
        views = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            )
        }
        expected_views = {
            "v_features_latest", "v_targets_latest",
            "v_predictions_latest", "v_triggers_current",
        }
        missing_v = expected_views - views
        check("all_expected_views_exist", not missing_v, f"missing: {missing_v}")
    finally:
        conn.close()


def test_registry_seeds(db_path: Path) -> None:
    print("\n[2] Registry seeds")
    conn = _connect(db_path)
    try:
        # Counts
        n_feat = conn.execute("SELECT COUNT(*) FROM features_registry").fetchone()[0]
        check("features_count_matches_seed",
              n_feat == len(FEATURES_REGISTRY_SEED),
              f"got {n_feat}, expected {len(FEATURES_REGISTRY_SEED)}")

        n_targ = conn.execute("SELECT COUNT(*) FROM targets_registry").fetchone()[0]
        check("targets_count_matches_seed",
              n_targ == len(TARGETS_REGISTRY_SEED),
              f"got {n_targ}, expected {len(TARGETS_REGISTRY_SEED)}")

        n_trig = conn.execute("SELECT COUNT(*) FROM triggers_registry").fetchone()[0]
        check("triggers_count_matches_seed",
              n_trig == len(TRIGGERS_REGISTRY_SEED),
              f"got {n_trig}, expected {len(TRIGGERS_REGISTRY_SEED)}")

        # Spec-mandated count: 21 features (3+3+5+3+1+1+3+2 across 8 tiers)
        check("exactly_21_features", n_feat == 21, f"got {n_feat}")

        # Spec: 6 targets (T1, T2, T3, T4a, T4b, T4c)
        targets = {row["target_id"]
                   for row in conn.execute("SELECT target_id FROM targets_registry")}
        check("targets_are_T1_T2_T3_T4abc",
              targets == {"T1", "T2", "T3", "T4a", "T4b", "T4c"},
              f"got {targets}")

        # Spec: 5 triggers, ids 1..5
        trigger_ids = {row["trigger_id"]
                       for row in conn.execute("SELECT trigger_id FROM triggers_registry")}
        check("triggers_are_1_through_5", trigger_ids == {1, 2, 3, 4, 5},
              f"got {trigger_ids}")

        # Tier coverage: every feature is in tiers 1-8
        tiers = {row["tier"]
                 for row in conn.execute("SELECT DISTINCT tier FROM features_registry")}
        check("feature_tiers_within_1_to_8",
              tiers.issubset(set(range(1, 9))), f"got {tiers}")
        check("all_8_tiers_have_features",
              tiers == set(range(1, 9)), f"missing tiers: {set(range(1,9)) - tiers}")
    finally:
        conn.close()


def test_exploratory_flags(db_path: Path) -> None:
    print("\n[3] Exploratory flags on T3 and T4c per spec")
    conn = _connect(db_path)
    try:
        rows = {
            row["target_id"]: dict(row)
            for row in conn.execute(
                "SELECT target_id, is_exploratory, exploratory_caveat FROM targets_registry"
            )
        }

        # T3 must be exploratory with non-empty caveat
        check("T3_is_exploratory", rows["T3"]["is_exploratory"] == 1)
        check("T3_has_caveat",
              rows["T3"]["exploratory_caveat"] is not None
              and len(rows["T3"]["exploratory_caveat"]) > 50,
              "caveat too short or missing")
        check("T3_caveat_mentions_EXPLORATORY",
              "EXPLORATORY" in (rows["T3"]["exploratory_caveat"] or ""))

        # T4c is also exploratory (warning indicator only)
        check("T4c_is_exploratory", rows["T4c"]["is_exploratory"] == 1)
        check("T4c_caveat_mentions_WARNING",
              "WARNING" in (rows["T4c"]["exploratory_caveat"] or ""))

        # T1, T2, T4a, T4b must NOT be exploratory
        for tid in ("T1", "T2", "T4a", "T4b"):
            check(f"{tid}_is_NOT_exploratory", rows[tid]["is_exploratory"] == 0)
    finally:
        conn.close()


def test_vintage_uniqueness(db_path: Path) -> None:
    """
    Two vintages of the same (feature, month) is the WHOLE POINT of the schema.
    Same (feature, month, vintage_date) twice should fail.
    """
    print("\n[4] Vintage-aware uniqueness")
    conn = _connect(db_path)
    try:
        # Insert a feature value for May 2008, vintage 2008-06-06 (initial release)
        conn.execute(
            """INSERT INTO features_monthly
               (feature_name, observation_month, vintage_date, value, source_pull_date)
               VALUES (?, ?, ?, ?, ?)""",
            ("USSLIND", "2008-05-01", "2008-06-06", -0.5, "2026-04-27"),
        )
        # Insert a revision (different vintage_date) — must succeed
        try:
            conn.execute(
                """INSERT INTO features_monthly
                   (feature_name, observation_month, vintage_date, value, source_pull_date)
                   VALUES (?, ?, ?, ?, ?)""",
                ("USSLIND", "2008-05-01", "2008-09-04", -0.7, "2026-04-27"),
            )
            conn.commit()
            check("revision_with_different_vintage_succeeds", True)
        except sqlite3.IntegrityError as e:
            check("revision_with_different_vintage_succeeds", False, str(e))

        # Duplicate (feature, month, vintage_date) — must fail
        try:
            conn.execute(
                """INSERT INTO features_monthly
                   (feature_name, observation_month, vintage_date, value, source_pull_date)
                   VALUES (?, ?, ?, ?, ?)""",
                ("USSLIND", "2008-05-01", "2008-06-06", -0.6, "2026-04-27"),
            )
            check("duplicate_vintage_rejected", False, "duplicate insert succeeded")
        except sqlite3.IntegrityError:
            check("duplicate_vintage_rejected", True)

        # The view should return the LATEST vintage only
        latest = conn.execute(
            """SELECT value FROM v_features_latest
               WHERE feature_name='USSLIND' AND observation_month='2008-05-01'"""
        ).fetchone()
        check("v_features_latest_returns_max_vintage",
              latest is not None and abs(latest["value"] - (-0.7)) < 1e-9,
              f"got {latest['value'] if latest else None}")
    finally:
        conn.close()


def test_predictions_uniqueness_and_fk(db_path: Path) -> None:
    print("\n[5] Predictions uniqueness + FK enforcement")
    conn = _connect(db_path)
    try:
        # Need a run_id row first (FK)
        run_id = "test_2026-04-27_smoke"
        conn.execute(
            """INSERT INTO runs (run_id, run_timestamp, run_type, status)
               VALUES (?, ?, ?, ?)""",
            (run_id, datetime.now().isoformat(), "smoke_test", "success"),
        )

        # Valid prediction insert
        conn.execute(
            """INSERT INTO predictions
               (run_id, target_id, horizon_months, model_id, fit_sample,
                prediction_date, target_month, probability, n_features_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, "T1", 12, "M1", "1990_2019",
             "2026-04-30", "2027-04-01", 0.18, 20),
        )
        conn.commit()
        check("valid_prediction_insert_succeeds", True)

        # Same (run, target, horizon, model, sample, prediction_date) → fail
        try:
            conn.execute(
                """INSERT INTO predictions
                   (run_id, target_id, horizon_months, model_id, fit_sample,
                    prediction_date, target_month, probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, "T1", 12, "M1", "1990_2019",
                 "2026-04-30", "2027-04-01", 0.21),
            )
            check("duplicate_prediction_rejected", False)
        except sqlite3.IntegrityError:
            check("duplicate_prediction_rejected", True)

        # Probability out of range → fail
        try:
            conn.execute(
                """INSERT INTO predictions
                   (run_id, target_id, horizon_months, model_id, fit_sample,
                    prediction_date, target_month, probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, "T2", 12, "M1", "1990_2019",
                 "2026-04-30", "2027-04-01", 1.5),
            )
            check("probability_above_1_rejected", False)
        except sqlite3.IntegrityError:
            check("probability_above_1_rejected", True)

        # Bad horizon → fail
        try:
            conn.execute(
                """INSERT INTO predictions
                   (run_id, target_id, horizon_months, model_id, fit_sample,
                    prediction_date, target_month, probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, "T2", 9, "M1", "1990_2019",
                 "2026-04-30", "2027-04-01", 0.3),
            )
            check("invalid_horizon_rejected", False)
        except sqlite3.IntegrityError:
            check("invalid_horizon_rejected", True)

        # FK: nonexistent run_id → fail
        try:
            conn.execute(
                """INSERT INTO predictions
                   (run_id, target_id, horizon_months, model_id, fit_sample,
                    prediction_date, target_month, probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("BOGUS_RUN", "T1", 12, "M1", "1990_2019",
                 "2026-04-30", "2027-04-01", 0.18),
            )
            check("fk_violation_on_bad_run_id_rejected", False)
        except sqlite3.IntegrityError:
            check("fk_violation_on_bad_run_id_rejected", True)

        # FK: nonexistent target_id → fail
        try:
            conn.execute(
                """INSERT INTO predictions
                   (run_id, target_id, horizon_months, model_id, fit_sample,
                    prediction_date, target_month, probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, "T99", 12, "M1", "1990_2019",
                 "2026-04-30", "2027-04-01", 0.18),
            )
            check("fk_violation_on_bad_target_id_rejected", False)
        except sqlite3.IntegrityError:
            check("fk_violation_on_bad_target_id_rejected", True)
    finally:
        conn.close()


def test_predictions_latest_view_filters_failed_runs(db_path: Path) -> None:
    print("\n[6] v_predictions_latest excludes failed runs")
    conn = _connect(db_path)
    try:
        # Add a failed run with a more recent prediction
        bad_run = "test_failed_run"
        conn.execute(
            """INSERT INTO runs (run_id, run_timestamp, run_type, status)
               VALUES (?, ?, ?, ?)""",
            (bad_run, datetime.now().isoformat(), "smoke_test", "failed"),
        )
        conn.execute(
            """INSERT INTO predictions
               (run_id, target_id, horizon_months, model_id, fit_sample,
                prediction_date, target_month, probability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (bad_run, "T1", 12, "M1", "1990_2019",
             "2026-05-31", "2027-05-01", 0.99),
        )
        conn.commit()

        # The latest view should NOT show the failed-run prediction;
        # it should still show the previous successful one.
        rows = list(conn.execute(
            """SELECT probability, run_id FROM v_predictions_latest
               WHERE target_id='T1' AND horizon_months=12
               AND model_id='M1' AND fit_sample='1990_2019'"""
        ))
        check("v_predictions_latest_returns_one_row", len(rows) == 1, f"got {len(rows)}")
        if rows:
            check("v_predictions_latest_excludes_failed_run",
                  rows[0]["run_id"] != bad_run,
                  f"got run_id={rows[0]['run_id']}")
            check("v_predictions_latest_value_is_from_success_run",
                  abs(rows[0]["probability"] - 0.18) < 1e-9,
                  f"got {rows[0]['probability']}")
    finally:
        conn.close()


def test_idempotent_migration(db_path: Path) -> None:
    print("\n[7] Re-running migration is idempotent")
    conn_before = _connect(db_path)
    counts_before = {
        "features": conn_before.execute("SELECT COUNT(*) FROM features_registry").fetchone()[0],
        "targets":  conn_before.execute("SELECT COUNT(*) FROM targets_registry").fetchone()[0],
        "triggers": conn_before.execute("SELECT COUNT(*) FROM triggers_registry").fetchone()[0],
        "features_monthly": conn_before.execute("SELECT COUNT(*) FROM features_monthly").fetchone()[0],
        "predictions": conn_before.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
    }
    conn_before.close()

    # Run migration again
    run_migration(db_path)

    conn_after = _connect(db_path)
    try:
        counts_after = {
            "features": conn_after.execute("SELECT COUNT(*) FROM features_registry").fetchone()[0],
            "targets":  conn_after.execute("SELECT COUNT(*) FROM targets_registry").fetchone()[0],
            "triggers": conn_after.execute("SELECT COUNT(*) FROM triggers_registry").fetchone()[0],
            "features_monthly": conn_after.execute("SELECT COUNT(*) FROM features_monthly").fetchone()[0],
            "predictions": conn_after.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
        }
        check("registries_unchanged_after_re_migrate",
              counts_before == counts_after,
              f"before={counts_before}, after={counts_after}")
    finally:
        conn_after.close()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "recession_test.db"
        print(f"Test DB: {db_path}")
        print("=" * 70)

        test_fresh_migration(db_path)
        test_registry_seeds(db_path)
        test_exploratory_flags(db_path)
        test_vintage_uniqueness(db_path)
        test_predictions_uniqueness_and_fk(db_path)
        test_predictions_latest_view_filters_failed_runs(db_path)
        test_idempotent_migration(db_path)

    print("\n" + "=" * 70)
    print(f"RESULTS: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("\nFAILURES:")
        for name, detail in _failed:
            print(f"  - {name}: {detail}")
        return 1
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
