-- =============================================================================
-- recession.db — schema v1
-- =============================================================================
--
-- Standalone SQLite database for the recession-prediction research project.
-- ZERO coupling with accuracy.db. Lives at: <repo>/recession.db
--
-- Spec reference: Recession_Model_Spec_v1.md
--
-- Design principles:
-- 1. Vintage-aware: every observation that could be revised stores (value,
--    vintage_date) tuples so backtests can query "what was known on date D".
-- 2. Long-format features: easier to add new features without ALTER TABLE.
-- 3. Run-stamped: every prediction and metric is tagged with a run_id, so
--    multiple model runs can coexist and the latest is queryable.
-- 4. T3 (AI kill-switch) is flagged exploratory in the targets registry; the
--    flag propagates to the dashboard so users see the disclaimer.
-- 5. Foreign keys ON; integrity enforced at write time, not application time.
--
-- Schema version is managed via PRAGMA user_version (set at end of file).
-- =============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- -----------------------------------------------------------------------------
-- META TABLES
-- -----------------------------------------------------------------------------

-- Schema migration history. Every migration appends a row.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version         INTEGER PRIMARY KEY,           -- monotonically increasing
    applied_at      TEXT    NOT NULL,              -- ISO 8601, Vietnam tz
    description     TEXT    NOT NULL,
    code_sha        TEXT                           -- git SHA at apply time
);

-- Pipeline runs. Every monthly cron, manual refit, or backfill creates a row.
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT    PRIMARY KEY,       -- e.g. '2026-04-07T08:00+07:00_monthly'
    run_timestamp       TEXT    NOT NULL,          -- ISO 8601 with tz
    run_type            TEXT    NOT NULL CHECK (run_type IN
                                ('monthly_predict', 'quarterly_refit',
                                 'manual', 'backfill', 'smoke_test')),
    code_sha            TEXT,                      -- git SHA
    spec_version        TEXT,                      -- e.g. 'v1.0'
    n_features          INTEGER,
    n_observations      INTEGER,
    n_predictions       INTEGER,
    n_metrics           INTEGER,
    status              TEXT    NOT NULL DEFAULT 'in_progress'
                                CHECK (status IN ('in_progress','success','failed','partial')),
    error_message       TEXT,
    notes               TEXT,
    duration_seconds    REAL
);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp     ON runs(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_type_status   ON runs(run_type, status);

-- -----------------------------------------------------------------------------
-- FEATURE & TARGET REGISTRIES
-- -----------------------------------------------------------------------------
-- These are reference tables. They define the universe of features and
-- targets, with metadata describing each one. Feature/target *values* live
-- in features_monthly and targets_monthly.

CREATE TABLE IF NOT EXISTS features_registry (
    feature_name        TEXT    PRIMARY KEY,       -- e.g. 'T10Y3M', 'EBP'
    tier                INTEGER NOT NULL,          -- 1..8 per spec §4
    tier_label          TEXT    NOT NULL,          -- 'yield_credit', 'labor', etc.
    description         TEXT    NOT NULL,
    source              TEXT    NOT NULL,          -- 'FRED', 'ALFRED', 'TRENDFORCE', etc.
    fred_series_id      TEXT,                      -- if applicable
    revisable           INTEGER NOT NULL CHECK (revisable IN (0,1)),  -- needs ALFRED?
    publication_lag_days INTEGER NOT NULL,         -- typical days from period-end to release
    detrend_method      TEXT    CHECK (detrend_method IN
                                ('none','hamilton2018','first_diff','log_diff','yoy_pct')),
    available_from      TEXT,                      -- earliest YYYY-MM available
    notes               TEXT,
    is_active           INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1))
);

CREATE TABLE IF NOT EXISTS targets_registry (
    target_id           TEXT    PRIMARY KEY,       -- 'T1', 'T2', 'T3', 'T4a', 'T4b', 'T4c'
    target_name         TEXT    NOT NULL,          -- 'NBER recession', 'SPX 15% drawdown', ...
    target_type         TEXT    NOT NULL CHECK (target_type IN
                                ('binary_atomic','binary_composite','combination')),
    description         TEXT    NOT NULL,
    is_exploratory      INTEGER NOT NULL DEFAULT 0 CHECK (is_exploratory IN (0,1)),
    exploratory_caveat  TEXT,                      -- shown on dashboard if exploratory=1
    available_from      TEXT,                      -- earliest YYYY-MM with usable label
    n_events_full_sample INTEGER,                  -- approx event count (for sanity checks)
    notes               TEXT
);

-- -----------------------------------------------------------------------------
-- FEATURES — VINTAGE-AWARE LONG FORMAT
-- -----------------------------------------------------------------------------
-- One row per (observation_month, feature, vintage_date). Vintages let us
-- reconstruct "what was known on date D" for honest backtesting.
--
-- For non-revisable features (yields, prices), we typically have ONE vintage
-- per observation — vintage_date = the publication date.
-- For revisable features (employment, GDP, LEI), we have MULTIPLE vintages:
-- the initial release, plus any subsequent revisions.

CREATE TABLE IF NOT EXISTS features_monthly (
    feature_name        TEXT    NOT NULL,
    observation_month   TEXT    NOT NULL,           -- 'YYYY-MM-01' (first of month)
    vintage_date        TEXT    NOT NULL,           -- 'YYYY-MM-DD' when this value was published
    value               REAL,                       -- nullable: data may be NA
    source_pull_date    TEXT    NOT NULL,           -- when WE retrieved the vintage
    PRIMARY KEY (feature_name, observation_month, vintage_date),
    FOREIGN KEY (feature_name) REFERENCES features_registry(feature_name)
);

-- Most common access pattern: "give me feature X for all months, latest vintage"
CREATE INDEX IF NOT EXISTS idx_features_name_obsmonth
    ON features_monthly(feature_name, observation_month);

-- Backtest pattern: "give me feature X for month M as known on date D"
CREATE INDEX IF NOT EXISTS idx_features_vintage
    ON features_monthly(feature_name, observation_month, vintage_date);

-- Bulk pull pattern: "give me all features for date range"
CREATE INDEX IF NOT EXISTS idx_features_obsmonth
    ON features_monthly(observation_month);

-- -----------------------------------------------------------------------------
-- TARGETS — VINTAGE-AWARE BINARY LABELS
-- -----------------------------------------------------------------------------
-- NBER recession dating is announced retrospectively. Labeling month M as
-- recession=1 should only happen as of NBER's announcement date, not the
-- actual recession start. announcement_date captures this.
--
-- For T2 (drawdown) and T3 (AI kill-switch), announcement_date = month-end
-- of the observation month, since they are computed from observable data.

CREATE TABLE IF NOT EXISTS targets_monthly (
    target_id           TEXT    NOT NULL,
    observation_month   TEXT    NOT NULL,           -- 'YYYY-MM-01'
    announcement_date   TEXT    NOT NULL,           -- 'YYYY-MM-DD' when label became known
    label               INTEGER NOT NULL CHECK (label IN (0,1)),
    notes               TEXT,                       -- e.g. "NBER announced 2008-12-01"
    PRIMARY KEY (target_id, observation_month, announcement_date),
    FOREIGN KEY (target_id) REFERENCES targets_registry(target_id)
);

CREATE INDEX IF NOT EXISTS idx_targets_id_month
    ON targets_monthly(target_id, observation_month);

-- -----------------------------------------------------------------------------
-- AI KILL-SWITCH TRIGGERS — UNDERLYING DATA FOR T3
-- -----------------------------------------------------------------------------
-- T3 is composed of 5 triggers (per playbook §12). We store each trigger's
-- monthly evaluation: did it fire, what was the underlying value, what's
-- the threshold. T3's label is derived from this table (≥2 triggers fired).

CREATE TABLE IF NOT EXISTS triggers_registry (
    trigger_id          INTEGER PRIMARY KEY,        -- 1..5
    trigger_name        TEXT    NOT NULL,
    description         TEXT    NOT NULL,
    threshold_rule      TEXT    NOT NULL,           -- human-readable
    threshold_value     REAL,                       -- numeric threshold if applicable
    available_from      TEXT,                       -- earliest YYYY-MM with real data
    proxy_used_before   TEXT,                       -- proxy series used pre-availability
    is_proxied_in_v1    INTEGER NOT NULL DEFAULT 0 CHECK (is_proxied_in_v1 IN (0,1))
);

CREATE TABLE IF NOT EXISTS triggers_monthly (
    trigger_id          INTEGER NOT NULL,
    observation_month   TEXT    NOT NULL,           -- 'YYYY-MM-01'
    fired               INTEGER NOT NULL CHECK (fired IN (0,1)),
    underlying_value    REAL,
    threshold_value     REAL,
    is_proxy            INTEGER NOT NULL DEFAULT 0 CHECK (is_proxy IN (0,1)),
    vintage_date        TEXT    NOT NULL,
    notes               TEXT,
    PRIMARY KEY (trigger_id, observation_month, vintage_date),
    FOREIGN KEY (trigger_id) REFERENCES triggers_registry(trigger_id)
);

CREATE INDEX IF NOT EXISTS idx_triggers_month
    ON triggers_monthly(observation_month);

-- -----------------------------------------------------------------------------
-- PREDICTIONS
-- -----------------------------------------------------------------------------
-- Every (target, horizon, model, sample, prediction_date) is one row.
-- prediction_date = end of the month at which the prediction was made.
-- target_month = the month being predicted (= prediction_date + horizon months).

CREATE TABLE IF NOT EXISTS predictions (
    run_id              TEXT    NOT NULL,
    target_id           TEXT    NOT NULL,           -- 'T1', 'T2', 'T3', 'T4a', 'T4b', 'T4c'
    horizon_months      INTEGER NOT NULL CHECK (horizon_months IN (1,3,6,12)),
    model_id            TEXT    NOT NULL,           -- 'M1','M2','M3','M4','C1','C2'
    fit_sample          TEXT    NOT NULL CHECK (fit_sample IN
                                ('pre1990','1990_2019','2020plus','full','oos_walkforward')),
    prediction_date     TEXT    NOT NULL,           -- 'YYYY-MM-DD' (end of prediction month)
    target_month        TEXT    NOT NULL,           -- 'YYYY-MM-01' (month being predicted)
    probability         REAL    NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
    ci_lower_95         REAL    CHECK (ci_lower_95 IS NULL OR (ci_lower_95 >= 0.0 AND ci_lower_95 <= 1.0)),
    ci_upper_95         REAL    CHECK (ci_upper_95 IS NULL OR (ci_upper_95 >= 0.0 AND ci_upper_95 <= 1.0)),
    n_features_used     INTEGER,
    is_realtime         INTEGER NOT NULL DEFAULT 1 CHECK (is_realtime IN (0,1)),
                        -- 1 = used vintage data only; 0 = used revised data (backfill)
    PRIMARY KEY (run_id, target_id, horizon_months, model_id, fit_sample, prediction_date),
    FOREIGN KEY (run_id)    REFERENCES runs(run_id),
    FOREIGN KEY (target_id) REFERENCES targets_registry(target_id)
);

-- Most common dashboard query: latest predictions for a target/horizon/model
CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions(target_id, horizon_months, model_id, fit_sample, prediction_date);

-- Time-series view: how did probability evolve over prediction dates
CREATE INDEX IF NOT EXISTS idx_predictions_timeseries
    ON predictions(target_id, horizon_months, model_id, prediction_date);

-- -----------------------------------------------------------------------------
-- METRICS
-- -----------------------------------------------------------------------------
-- Out-of-sample evaluation results. One row per
-- (run, target, horizon, model, sample, metric_name).

CREATE TABLE IF NOT EXISTS metrics (
    run_id              TEXT    NOT NULL,
    target_id           TEXT    NOT NULL,
    horizon_months      INTEGER NOT NULL CHECK (horizon_months IN (1,3,6,12)),
    model_id            TEXT    NOT NULL,
    fit_sample          TEXT    NOT NULL,
    eval_sample         TEXT    NOT NULL,           -- evaluation period: same enum as fit_sample
    metric_name         TEXT    NOT NULL CHECK (metric_name IN
                                ('auc','hit_rate','sharpe_uplift','brier',
                                 'brier_reliability','brier_resolution','brier_uncertainty',
                                 'log_loss','calibration_slope','calibration_intercept')),
    value               REAL    NOT NULL,
    ci_lower_95         REAL,
    ci_upper_95         REAL,
    n_obs               INTEGER NOT NULL,
    bootstrap_method    TEXT,                       -- 'block_24m' or 'iid' or NULL
    p_value             REAL,                       -- e.g. for Diebold-Mariano comparisons
    PRIMARY KEY (run_id, target_id, horizon_months, model_id, fit_sample, eval_sample, metric_name),
    FOREIGN KEY (run_id)    REFERENCES runs(run_id),
    FOREIGN KEY (target_id) REFERENCES targets_registry(target_id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON metrics(target_id, horizon_months, model_id, eval_sample, metric_name);

-- -----------------------------------------------------------------------------
-- MODEL COMPARISONS (Diebold-Mariano test results)
-- -----------------------------------------------------------------------------
-- Pairwise comparisons of models. "Does M2 statistically beat M1 at this
-- target/horizon?"

CREATE TABLE IF NOT EXISTS model_comparisons (
    run_id              TEXT    NOT NULL,
    target_id           TEXT    NOT NULL,
    horizon_months      INTEGER NOT NULL,
    model_a             TEXT    NOT NULL,
    model_b             TEXT    NOT NULL,
    eval_sample         TEXT    NOT NULL,
    test_name           TEXT    NOT NULL CHECK (test_name IN
                                ('diebold_mariano','reality_check','spa')),
    test_statistic      REAL    NOT NULL,
    p_value             REAL    NOT NULL,
    a_better_than_b     INTEGER CHECK (a_better_than_b IN (0,1,NULL)),
                                -- 1 if A statistically beats B at 5% level
    n_obs               INTEGER NOT NULL,
    notes               TEXT,
    PRIMARY KEY (run_id, target_id, horizon_months, model_a, model_b, eval_sample, test_name),
    FOREIGN KEY (run_id)    REFERENCES runs(run_id),
    FOREIGN KEY (target_id) REFERENCES targets_registry(target_id)
);

-- -----------------------------------------------------------------------------
-- MODEL ARTIFACTS — fitted parameters, for inspection / reproducibility
-- -----------------------------------------------------------------------------
-- Stores fitted coefficients (M1, M2), Markov state probabilities (M3),
-- feature importances (M4), and combination weights (C1, C2).

CREATE TABLE IF NOT EXISTS model_artifacts (
    run_id              TEXT    NOT NULL,
    target_id           TEXT    NOT NULL,
    horizon_months      INTEGER NOT NULL,
    model_id            TEXT    NOT NULL,
    fit_sample          TEXT    NOT NULL,
    artifact_name       TEXT    NOT NULL,           -- e.g. 'coef_T10Y3M', 'shap_NFCI', 'weight_M1'
    artifact_value      REAL    NOT NULL,
    artifact_se         REAL,                       -- standard error if applicable
    PRIMARY KEY (run_id, target_id, horizon_months, model_id, fit_sample, artifact_name),
    FOREIGN KEY (run_id)    REFERENCES runs(run_id),
    FOREIGN KEY (target_id) REFERENCES targets_registry(target_id)
);

-- -----------------------------------------------------------------------------
-- VIEWS — convenience for dashboard queries
-- -----------------------------------------------------------------------------

-- Latest vintage of each feature for each month. The "current best estimate".
CREATE VIEW IF NOT EXISTS v_features_latest AS
SELECT
    f.feature_name,
    f.observation_month,
    f.value,
    f.vintage_date,
    f.source_pull_date
FROM features_monthly f
INNER JOIN (
    SELECT feature_name, observation_month, MAX(vintage_date) AS max_vintage
    FROM features_monthly
    GROUP BY feature_name, observation_month
) latest
    ON f.feature_name = latest.feature_name
    AND f.observation_month = latest.observation_month
    AND f.vintage_date = latest.max_vintage;

-- Latest target labels (post-announcement view)
CREATE VIEW IF NOT EXISTS v_targets_latest AS
SELECT
    t.target_id,
    t.observation_month,
    t.label,
    t.announcement_date
FROM targets_monthly t
INNER JOIN (
    SELECT target_id, observation_month, MAX(announcement_date) AS max_announce
    FROM targets_monthly
    GROUP BY target_id, observation_month
) latest
    ON t.target_id = latest.target_id
    AND t.observation_month = latest.observation_month
    AND t.announcement_date = latest.max_announce;

-- Latest predictions per (target, horizon, model, fit_sample), for at-a-glance
CREATE VIEW IF NOT EXISTS v_predictions_latest AS
SELECT
    p.target_id,
    p.horizon_months,
    p.model_id,
    p.fit_sample,
    p.prediction_date,
    p.target_month,
    p.probability,
    p.ci_lower_95,
    p.ci_upper_95,
    p.run_id,
    r.run_timestamp
FROM predictions p
INNER JOIN runs r ON p.run_id = r.run_id
INNER JOIN (
    SELECT target_id, horizon_months, model_id, fit_sample, MAX(prediction_date) AS max_pred_date
    FROM predictions
    WHERE run_id IN (SELECT run_id FROM runs WHERE status = 'success')
    GROUP BY target_id, horizon_months, model_id, fit_sample
) latest
    ON p.target_id = latest.target_id
    AND p.horizon_months = latest.horizon_months
    AND p.model_id = latest.model_id
    AND p.fit_sample = latest.fit_sample
    AND p.prediction_date = latest.max_pred_date
WHERE r.status = 'success';

-- Trigger status this month, for the kill-switch panel
CREATE VIEW IF NOT EXISTS v_triggers_current AS
SELECT
    t.trigger_id,
    tr.trigger_name,
    tr.description,
    t.observation_month,
    t.fired,
    t.underlying_value,
    t.threshold_value,
    t.is_proxy
FROM triggers_monthly t
INNER JOIN triggers_registry tr ON t.trigger_id = tr.trigger_id
INNER JOIN (
    SELECT trigger_id, MAX(observation_month) AS max_month
    FROM triggers_monthly
    GROUP BY trigger_id
) latest
    ON t.trigger_id = latest.trigger_id
    AND t.observation_month = latest.max_month;

-- -----------------------------------------------------------------------------
-- SCHEMA VERSION
-- -----------------------------------------------------------------------------
PRAGMA user_version = 1;
