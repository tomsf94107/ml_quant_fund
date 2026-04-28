# `recession/` — Recession Prediction Research Module

**Standalone research project inside the `ml_quant_fund` repo.**
**Spec:** `Recession_Model_Spec_v1.md` (repo root).
**Status:** Phase 1, Step 1 complete (DB schema + migrations + smoke tests).

---

## What this is

A multi-target recession/regime-prediction system that:
- Predicts 4 targets × 4 horizons × 4 model families (+ 2 forecast combinations)
- Reads macro data from FRED/ALFRED and external sources
- Writes to its own SQLite DB (`recession.db`)
- Exposes results via a single Streamlit page (Page 14, when shipped)

This is **not** part of the ML Quant Fund equity-prediction pipeline. It runs on its own monthly cron, has its own database, and has its own evaluation framework. The two systems may be integrated later (post May 1 SELL-signal validation gate, see spec §10), but only behind explicit, gated, validated merges.

---

## Isolation rules — DO NOT VIOLATE

These rules are what make "same repo, separate concerns" work. Breaking any of them collapses the isolation that the architecture relies on.

### Rule 1: One-way data flow only
- ✅ `recession/` MAY import from `utils/timezone.py` (read-only, shared utility)
- ❌ `recession/` MUST NOT import from `features/`, `signals/`, `models/`, `predictions.py`, or any equity-pipeline module
- ❌ Equity-pipeline code MUST NOT import from `recession/`
- ❌ `recession/` MUST NOT read from or write to `accuracy.db`
- ❌ Equity pipeline MUST NOT read from or write to `recession.db`

### Rule 2: Separate cron, separate run loop
- ✅ `recession/` runs on its own monthly cron (day 7 Vietnam time, 8 AM)
- ❌ Do NOT chain it into `run_pipeline.sh` — it has independent failure modes and cadence
- ✅ If a manual refit is needed, use the standalone `python -m recession.runner` (when implemented)

### Rule 3: Separate dashboard page
- ✅ Page 14 (`ui/pages/14_Recession_Regime.py`) reads ONLY from `recession.db`
- ❌ It MUST NOT query `accuracy.db`, `predictions`, `intraday_predictions`, `outcomes`, or `prediction_features`
- ❌ It MUST NOT modify any other dashboard page or shared Streamlit state used by other pages
- ✅ Reusing the `📦 Run Strategy` / `🔄 Refresh Live` cache pattern from existing pages is encouraged — that's a UI convention, not a coupling

### Rule 4: Schema version discipline
- ✅ Every schema change increments `PRAGMA user_version` and adds a migration step in `db/migrate.py`
- ❌ NEVER `ALTER TABLE` directly on the live `recession.db`
- ✅ Run `python -m recession.db.migrate` to apply migrations; it's idempotent

### Rule 5: Vintage discipline
- ✅ All revisable features (employment, GDP, LEI, ISM, etc.) MUST be stored with `vintage_date`
- ✅ Backtest queries MUST use `vintage_date <= prediction_date` filters
- ❌ NEVER use the latest revised value in a backtest — it inflates accuracy by including future information

### Rule 6: T3 (AI Kill-Switch) is exploratory
- ✅ The `is_exploratory=1` flag and `exploratory_caveat` text are stored in `targets_registry`
- ✅ Page 14 MUST surface the caveat wherever T3 or T4c results are displayed
- ❌ T3 outputs MUST NOT be used as a production sizing input — directional reading only

---

## Quick start

```bash
# From repo root (where ml_quant_310 venv is activated)

# 1. Apply schema + seed registries (idempotent)
python -m recession.db.migrate

# 2. Run smoke tests (33 assertions, all must pass)
python -m recession.tests.test_schema

# 3. Inspect the seeded registries
sqlite3 recession.db "SELECT tier, tier_label, COUNT(*) FROM features_registry GROUP BY tier;"
sqlite3 recession.db "SELECT target_id, target_name, is_exploratory FROM targets_registry;"
```

---

## Directory layout

```
recession/
├── __init__.py
├── data/                       # (Step 2+) FRED/ALFRED ingestion, feature pipeline
├── db/
│   ├── __init__.py
│   ├── schema.sql              # full DDL: 12 tables, 4 views, FK on, WAL mode
│   └── migrate.py              # idempotent migration runner + registry seeders
├── models/                     # (Step 5+) M1 probit, M2 dynamic probit, M3 MS, M4 XGBoost
├── eval/                       # (Step 10+) walk-forward, block bootstrap, DM, RC
├── runner.py                   # (Step 12+) monthly cron entry point
└── tests/
    ├── __init__.py
    └── test_schema.py          # 33-assertion smoke test
```

---

## Database tables (schema v1)

Reference tables (seeded once, edited rarely):
- `features_registry` — 21 features across 8 tiers
- `targets_registry` — 6 targets (T1, T2, T3, T4a, T4b, T4c)
- `triggers_registry` — 5 AI kill-switch triggers

Data tables (vintage-aware, append-only):
- `features_monthly` — `(feature, month, vintage_date) → value`
- `targets_monthly` — `(target, month, announcement_date) → label`
- `triggers_monthly` — `(trigger, month, vintage_date) → fired/value`

Run-stamped tables (one row per run × per item):
- `runs` — pipeline execution log
- `predictions` — `(run, target, horizon, model, sample, prediction_date) → probability`
- `metrics` — `(run, target, horizon, model, sample, metric) → value + CI`
- `model_comparisons` — Diebold-Mariano / Reality Check pairwise tests
- `model_artifacts` — fitted coefficients, weights, SHAP values

Convenience views:
- `v_features_latest` — latest vintage per (feature, month)
- `v_targets_latest` — latest announcement per (target, month)
- `v_predictions_latest` — latest prediction per (target, horizon, model, sample), excluding failed runs
- `v_triggers_current` — current month trigger status

---

## Cross-references

| Concern | Where to look |
|---|---|
| Why these targets / horizons / models | `Recession_Model_Spec_v1.md` §2, §3, §6 |
| Why these features | Spec §4 |
| Real-time data discipline (ALFRED, vintages) | Spec §5 |
| Validation methodology | Spec §7 |
| Build sequence and timeline | Spec §10 |
| Future work backlog | Spec §12 |
| Definition of "done" for v1 | Spec §13 |

---

## Versioning

This module is versioned independently of the parent repo:
- `recession.__version__` — code version
- `recession.__spec_version__` — spec doc version it implements
- `PRAGMA user_version` on `recession.db` — DB schema version

Bumping any of these requires a migration entry and (for spec changes) a re-spec doc.

---

*Last updated: 2026-04-27 — Phase 1 Step 1 complete.*
