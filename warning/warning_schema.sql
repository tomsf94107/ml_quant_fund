-- warning_schema.sql — Crash Early-Warning System, SQLite DDL
-- ============================================================================
-- RECONSTRUCTED 2026-08-28. The canonical warning_schema.sql referenced by the
-- brief was NOT provided and is NOT embedded in the report. This file is rebuilt
-- from AUTHORITATIVE sources only, tagged per table:
--   [code-pinned]   columns fixed by the exact SQL in uw_archiver.py /
--                   fetch_free_history.py — these MUST NOT drift or the provided
--                   scripts break.
--   [engine-pinned] columns fixed by warning_engine.py dataclasses
--                   (DayResult, SignalReading, alert tuples) — the persistence
--                   contract for what step() produces.
--   [report-spec]   columns fixed by the report Part V/VI/VIII prose.
--   [reconstructed] minimal, extensible; no canonical source pins the columns.
-- If the original warning_schema.sql surfaces, DIFF against it — treat the
-- [code-pinned] and [engine-pinned] tables as correct, reconcile the rest.
--
-- Apply:  sqlite3 warning.db < warning_schema.sql
-- Conventions: snake_case, plain tables, no ORM (matches accuracy.db). Dates are
-- ISO 'YYYY-MM-DD' TEXT. Booleans are INTEGER 0/1.
-- ============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- 1. data_vintages  [code-pinned + non-negotiable #1: POINT-IN-TIME]
--    fetch_free_history.py: INSERT OR IGNORE INTO data_vintages
--        (series_id, obs_date, pub_date, value, source) VALUES (?,?,?,?,?)
--    PK includes pub_date so ALFRED serves MANY vintages per obs_date and every
--    historical join keys on publication date, never reference date. Nightly
--    re-pull of the last ~90 days lands as new (obs_date, pub_date) rows; older
--    pub_dates are never overwritten — that IS the vintage history.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_vintages (
    series_id   TEXT NOT NULL,
    obs_date    TEXT NOT NULL,          -- the date the observation refers to
    pub_date    TEXT NOT NULL,          -- the date the value was published/pulled
    value       REAL NOT NULL,
    source      TEXT NOT NULL,          -- 'FRED' | 'ALFRED' | 'Cboe' | 'CFE' | ...
    pulled_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (series_id, obs_date, pub_date)
);
CREATE INDEX IF NOT EXISTS ix_vintages_series_pub ON data_vintages (series_id, pub_date);
CREATE INDEX IF NOT EXISTS ix_vintages_series_obs ON data_vintages (series_id, obs_date);

-- ---------------------------------------------------------------------------
-- 2. uw_archive  [code-pinned + non-negotiable #8: APPEND-ONLY]
--    Verbatim from uw_archiver.py's own CREATE TABLE so the two never diverge.
--    Raw JSON stored as-is; parsing is a separate re-runnable pass. Never DELETE,
--    never UPDATE payload_json. This is your only options history past the ~44d cap.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uw_archive (
    endpoint      TEXT NOT NULL,
    query_params  TEXT NOT NULL,        -- canonical sorted JSON of params
    snapshot_date TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    pulled_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (endpoint, query_params, snapshot_date)
);

-- ---------------------------------------------------------------------------
-- 3. signal_values  [engine-pinned by SignalReading + registry columns]
--    One row per (as-of date, signal). The builder computes raw_value + zscore
--    from data_vintages/uw_archive per signal_registry.csv, maps to a G/Y/O/R/B
--    state via the frozen registry thresholds, and hands SignalReading to the
--    engine. sub_score = STATE_SCORE[state]; persistence_days = consecutive days
--    at `state`; effective_state = the persistence-filtered state step() used.
--    zscore + sub_score are what non-negotiable #9 decomposes a probability into.
--    NOTE: report prose (line 753) calls this table `signals`; the brief calls it
--    `signal_values`. Using `signal_values` (brief is build authority); view
--    v_signals below aliases it for any code written to the report's name.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signal_values (
    asof_date        TEXT NOT NULL,
    signal_id        TEXT NOT NULL,     -- 'S1'..'S15','F1'..'F12','X1'..'X5'
    layer            TEXT NOT NULL,     -- 'L1'|'L2'|'L3'|'L4'
    raw_value        REAL,              -- the computed statistic (NULL if NA)
    zscore           REAL,              -- standardized; feeds M1 logit
    state            TEXT NOT NULL,     -- 'G'|'Y'|'O'|'R'|'B'|'NA'
    sub_score        REAL,              -- STATE_SCORE[state]; NULL for NA
    stale            INTEGER NOT NULL DEFAULT 0,
    persistence_days INTEGER NOT NULL DEFAULT 1,
    effective_state  TEXT,              -- post-persistence state used by engine
    source_asof      TEXT,              -- max pub_date of the vintages used (PIT audit)
    registry_version TEXT,              -- signal_registry.version at compute time
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (asof_date, signal_id)
);
CREATE INDEX IF NOT EXISTS ix_sigval_signal ON signal_values (signal_id, asof_date);

-- ---------------------------------------------------------------------------
-- 4. composite_scores  [engine-pinned by DayResult + report Part VI/VIII]
--    One row per as-of date = the full daily engine state. composite/action_gross
--    are NULL on INSUFFICIENT_DATA (band frozen, hedge='freeze').
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS composite_scores (
    asof_date         TEXT PRIMARY KEY,
    composite         REAL,             -- 0..100, NULL if insufficient
    band              TEXT NOT NULL,    -- NORMAL|WATCH|ELEVATED|DEFENSIVE|CRISIS|INSUFFICIENT_DATA
    path              TEXT NOT NULL,    -- SPECULATIVE|CREDIT|UNCLASSIFIED
    do_nothing        INTEGER NOT NULL DEFAULT 0,
    l4_override       INTEGER NOT NULL DEFAULT 0,
    insufficient_data INTEGER NOT NULL DEFAULT 0,
    l1_score REAL, l2_score REAL, l3_score REAL, l4_score REAL,   -- NULL = layer NA
    l1_cov   REAL, l2_cov   REAL, l3_cov   REAL, l4_cov   REAL,
    na_layers         TEXT,             -- JSON list e.g. '["L3"]'
    action_gross      REAL,             -- NULL on freeze
    action_hedge      TEXT,
    action_carry_bps  INTEGER,
    candidate_band    TEXT,             -- in-flight hysteresis candidate
    candidate_days    INTEGER NOT NULL DEFAULT 0,
    registry_version  TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- 5. alerts  [engine-pinned by step()'s alert tuples + report line 609]
--    Every band change / LAYER_NA / INSUFFICIENT_DATA / DO_NOTHING writes a row,
--    with the signals that triggered it (kill-switch-audit convention).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asof_date       TEXT NOT NULL,
    alert_type      TEXT NOT NULL,      -- BAND_UP|BAND_DOWN|LAYER_NA|INSUFFICIENT_DATA|DO_NOTHING_CONFLICT
    from_state      TEXT,               -- prior band/layer
    to_state        TEXT,               -- new band/layer
    reason          TEXT,               -- e.g. 'persisted 21d','L4 propagation override'
    trigger_signals TEXT,               -- JSON list of signal_ids that fired
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_alerts_asof ON alerts (asof_date);

-- ---------------------------------------------------------------------------
-- 6. hedge_book  [report Part VIII lines 541, 636-649; OPEN ITEM 4]
--    The Part VIII action tickets. `instrument` (SPX vs SPY) and account/broker
--    conventions are OPEN ITEM 4 — left free-text until you confirm. carry_budget
--    is the per-band bleed cap (breach forces structure review, not a band change).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hedge_book (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    asof_date            TEXT NOT NULL,
    band                 TEXT NOT NULL,
    structure            TEXT,           -- e.g. 'put_spread_3m_5_15','puts_6m_10otm'
    instrument           TEXT,           -- SPX|SPY|... (OPEN ITEM 4)
    tenor_months         INTEGER,
    otm_pct              REAL,
    delta_long           REAL,
    notional_pct_gross   REAL,
    carry_budget_bps_mo  INTEGER,
    roll_after_days      INTEGER,
    status               TEXT NOT NULL DEFAULT 'proposed',  -- proposed|open|rolled|closed
    opened_date          TEXT,
    closed_date          TEXT,
    notes                TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_hedge_status ON hedge_book (status, asof_date);

-- ---------------------------------------------------------------------------
-- 7. carry_paths  [report Part V lines 541-542; Phase 4]
--    Carry-cost P&L path for the put-hedge template. Three named historical runs
--    are mandatory: 'hedge_1998_09' (right-thesis-3y-early, the ruinous case),
--    'hedge_2007_09_18' (favorable), 'hedge_2021_12' (2022 validation); plus
--    'live'. Pre-2004 rows MUST carry an IV/skew assumption (flagged in output).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS carry_paths (
    run_id             TEXT NOT NULL,   -- hedge_1998_09|hedge_2007_09_18|hedge_2021_12|live
    asof_date          TEXT NOT NULL,
    months_from_entry  REAL,
    gross              REAL,
    hedge_notional_pct REAL,
    monthly_bleed_bps  REAL,
    cum_bleed_bps      REAL,
    payoff             REAL,            -- option payoff to date
    net_pnl            REAL,
    iv_assumption      TEXT,            -- BS/VIX-anchored skew note (mandatory pre-2004)
    assumptions        TEXT,
    PRIMARY KEY (run_id, asof_date)
);

-- ---------------------------------------------------------------------------
-- 8. eval_predictions + eval_metrics  [report Part V lines 512-549; Phase 4]
--    Expanding-window walk-forward. Targets = P(SPX peak-to-trough dd >=10/20/30%)
--    within 60/120/252 td. Models M1..M5 + baselines B0 (base rate), B1 (price/vol
--    logit). N_eff ~ N/h; block bootstrap CIs (block=h); Newey-West (lag=h).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    model       TEXT NOT NULL,          -- M1..M5|B0|B1
    asof_date   TEXT NOT NULL,
    target      TEXT NOT NULL,          -- dd10|dd20|dd30
    horizon_td  INTEGER NOT NULL,       -- 60|120|252 (20 = quarantined)
    prob        REAL,
    realized    INTEGER,                -- 0/1, NULL until the horizon resolves
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_evalpred ON eval_predictions (run_id, model, target, horizon_td, asof_date);

CREATE TABLE IF NOT EXISTS eval_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    model       TEXT,                   -- M1..M5|B0|B1 (NULL for per-signal rows)
    signal_id   TEXT,                   -- set for per-signal lead/hit/FPR rows
    target      TEXT,                   -- dd10|dd20|dd30
    horizon_td  INTEGER,                -- 60|120|252
    metric      TEXT NOT NULL,          -- pr_auc|roc_auc|brier|log_loss|lead_days|
                                        -- hit_rate|fpr|precision|recall|rank_ic|
                                        -- delta_brier_vs_b0|delta_brier_vs_b1|...
    value       REAL,
    ci_low      REAL,
    ci_high     REAL,
    n_eff       REAL,
    decade      TEXT,                   -- for stability-by-decade rows
    notes       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_evalmetrics ON eval_metrics (run_id, metric);

-- ---------------------------------------------------------------------------
-- 9. schema_meta  [non-negotiable #3: frozen thresholds; version-bump discipline]
--    Records the applied schema version and the signal_registry.version in force.
--    Thresholds change ONLY with a registry version bump at annual review.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR REPLACE INTO schema_meta (key, value) VALUES
    ('schema_version',   '0.1-reconstructed-2026-08-28'),
    ('registry_version', 'UNSET — set to signal_registry.version on first load'),
    ('applied_at',       datetime('now'));

-- ---------------------------------------------------------------------------
-- Views  [reconstructed convenience — the operator dashboard reads these]
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_signals AS SELECT * FROM signal_values;  -- report-name alias

CREATE VIEW IF NOT EXISTS v_latest_composite AS
    SELECT * FROM composite_scores ORDER BY asof_date DESC LIMIT 1;

CREATE VIEW IF NOT EXISTS v_latest_signal_state AS
    SELECT sv.* FROM signal_values sv
    JOIN (SELECT MAX(asof_date) md FROM signal_values) m ON sv.asof_date = m.md;

CREATE VIEW IF NOT EXISTS v_active_hedges AS
    SELECT * FROM hedge_book WHERE status IN ('open','rolled');

-- Per-signal contribution to the latest composite (non-negotiable #9 audit).
CREATE VIEW IF NOT EXISTS v_signal_contributions AS
    SELECT sv.asof_date, sv.signal_id, sv.layer, sv.state, sv.effective_state,
           sv.zscore, sv.sub_score, sv.stale
    FROM signal_values sv
    JOIN (SELECT MAX(asof_date) md FROM signal_values) m ON sv.asof_date = m.md
    ORDER BY sv.layer, sv.signal_id;
