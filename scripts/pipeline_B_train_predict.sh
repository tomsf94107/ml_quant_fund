#!/bin/bash
# scripts/pipeline_B_train_predict.sh
# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + PREDICT PIPELINE
# Runs 20:00 ET Mon-Fri (was 07:00 VN before ET migration)
# Uses DBs populated by Pipeline A
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 1: Dependency check — Pipeline A must have completed today
#   Stage 2: Retrain all models (models/train_all)
#   Stage 3: Run daily predictions (daily_runner.run_daily)
#   Stage 4: Daily validator (sanity-check prediction log)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
DATE_TAG=$(date +%Y%m%d)
LOGDIR=$ROOT/logs/pipeline_B_$DATE_TAG
mkdir -p "$LOGDIR"

cd $ROOT
# DISABLED launchd 78 (zshrc unbound-var under set -u): source /Users/atomnguyen/.zshrc 2>/dev/null || true

# Load .env file (feature flags, API keys, etc) - added 2026-05-21
# Without this, env vars set in .env are not available to Python subprocess
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"
}

fail() {
    log "FAILED at $1"
    osascript -e "display notification \"Pipeline B failed at $1\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
}

log "=== PIPELINE B START ==="

# ── Stage 1: Dependency check ────────────────────────────────────────────────
log "Stage 1: Checking Pipeline A completed today"
MARKER=$ROOT/logs/.pipeline_A_done_$DATE_TAG
if [ ! -f "$MARKER" ]; then
    log "FATAL: Pipeline A did not complete today. Marker not found: $MARKER"
    log "Refusing to retrain/predict with potentially stale data."
    osascript -e "display notification \"Pipeline B skipped: A did not complete\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
fi
log "Stage 1 OK (Pipeline A completed at $(stat -f '%Sm' "$MARKER"))"

# Stage 1.5: Reconcile outcomes so retrain trains on freshest labels.
# Session closed by retrain time; matured predictions now have realized returns.
# Non-fatal: reconcile failure -> retrain proceeds on existing labels, not abort.
# Added Jul 7 2026.
log "Stage 1.5: Reconciling outcomes (fresh labels for retrain)"
if $PYTHON -c "import sys; sys.path.insert(0,'.'); from accuracy.sink import reconcile_outcomes; print('reconciled', reconcile_outcomes(), 'outcomes')" >> "$LOGDIR/015_reconcile.log" 2>&1; then
    log "Stage 1.5 OK ($(tail -1 "$LOGDIR/015_reconcile.log"))"
else
    log "WARN Stage 1.5 reconcile failed (non-fatal, retraining on existing labels - see 015_reconcile.log)"
fi

# ── Stage 2: Retrain ─────────────────────────────────────────────────────────
# NOTE: If you move to weekly retrain, wrap this in a day-of-week check:
#   if [ "$(date +%u)" -eq 7 ]; then ... fi   # only on Sundays
# For now, keeps your daily retrain behavior but with proper sequencing.
log "Stage 2: Retrain all models"
$PYTHON -m models.train_all_batched \
    > "$LOGDIR/02_train_all.log" 2>&1 || fail "Stage 2 (train_all)"
log "Stage 2 OK"

# ── Stage 2.5: signal sanity guard (Jun 17 2026) ─────────────────────────────
# Moved here from Pipeline C when C's daily_runner was removed. B is now the
# SOLE signal-publish path, so the inverted-direction guard must live here,
# BEFORE Stage 3 publishes. Defense-in-depth alongside the ML_QUANT_DISABLE_BUY
# kill switch (which forces every BUY->HOLD at generation) and the Stage 4
# post-publish validator. On RED: abort the publish only if the direction model
# is LIVE; if BUYs are disabled, WARN + continue (nothing broken can publish).
log "Stage 2.5: signal sanity guard (direction / BUY-hit / rank-IC)"
if $PYTHON analysis/signal_sanity_guard.py --db accuracy.db --days 90 \
    > "$LOGDIR/025_sanity_guard.log" 2>&1; then
    log "Stage 2.5 OK — sanity guard GREEN, proceeding to publish"
else
    if [ "${ML_QUANT_DISABLE_BUY:-1}" = "1" ]; then
        log "Stage 2.5 RED — but DIRECTION MODEL DISABLED (ML_QUANT_DISABLE_BUY=1)."
        log "  WARNING only; CONTINUING (kill switch forces every BUY->HOLD)."
        cat "$LOGDIR/025_sanity_guard.log" | tee -a "$LOGDIR/pipeline.log"
    else
        log "Stage 2.5 RED — direction model ENABLED; ABORTING publish (broken signal direction)"
        osascript -e "display notification \"Pipeline B ABORTED: signal sanity guard RED\" with title \"ML Quant Fund\"" 2>/dev/null || true
        cat "$LOGDIR/025_sanity_guard.log" | tee -a "$LOGDIR/pipeline.log"
        exit 2
    fi
fi

# ── Stage 2.7: GLOBAL retrain (ensemble + ranker) — non-blended (Jul 9 2026) ──
# GLOBAL is a cross-sectional pooled model (all tickers), stored in predictions
# as prob_up_global / prob_up_global_ranker for COMPARISON only. It is NOT in the
# live signal formula (prob_eff = per-ticker prob x multipliers). We retrain it
# nightly so it stays fresh for monitoring (the h=5 agreement signal). Non-fatal:
# a GLOBAL failure must never abort the per-ticker pipeline.
log "Stage 2.7: GLOBAL retrain (cross-sectional ensemble + ranker, non-blended)"
if ML_QUANT_INST_FEATURES=1 $PYTHON -m models.train_cross_sectional --horizons 1 3 5 \
    > "$LOGDIR/027_global_ensemble.log" 2>&1; then
    log "Stage 2.7a OK — GLOBAL ensemble retrained"
else
    log "Stage 2.7a WARN — GLOBAL ensemble retrain failed (non-fatal, see 027_global_ensemble.log)"
fi
# NOTE Jul 9 2026: GLOBAL ranker retrain REMOVED — the ranker answers the wrong
# question (relative rank, not direction). prob_eff is the validated signal. Ensemble
# retrain above stays (feeds Prob Global comparison column). Ranker permanently disabled.

# ── Stage 3: Daily predictions ───────────────────────────────────────────────
log "Stage 3: Daily runner (generates today's signals)"
$PYTHON -m scripts.daily_runner_batched \
    > "$LOGDIR/03_daily_runner.log" 2>&1 || fail "Stage 3 (daily_runner)"
log "Stage 3 OK"

# ── Stage 4: Daily validator ─────────────────────────────────────────────────
log "Stage 4: Daily validator (checks recent predictions for anomalies)"
# Jun 26 2026: --fix re-enabled. Root cause was sink writing outcomes a
# session early (weekday _add_trading_days + asof roll-back) -> 3,758
# stale 2026 rows; backfilled. Both sink.reconcile_outcomes and
# validator.compute_return now target REAL sessions by bar position and
# skip not-yet-posted targets, so --fix repairs without corrupting.
$PYTHON scripts/daily_validator.py --days 30 --fix \
    > "$LOGDIR/04_daily_validator.log" 2>&1 || fail "Stage 4 (daily_validator)"
log "Stage 4 OK"

# ── Stage 5: REC % A/B backfill (non-critical) ───────────────────────────────
# Auto-populates portfolio_returns_ab table as outcomes mature.
# A/B decision date: Wed Jun 24, 2026. Idempotent (INSERT OR REPLACE).
# Non-critical: failure here does NOT fail the pipeline.
log "Stage 5: REC % A/B backfill (non-critical)"
if $PYTHON scripts/backfill_rec_weight_ab.py \
    > "$LOGDIR/05_rec_ab_backfill.log" 2>&1; then
    log "Stage 5 OK"
else
    log "Stage 5 FAILED (rc=$?) — continuing anyway, A/B backfill is non-critical"
fi

# ── Stage 6: Momentum shadow 20d outcome reconcile (non-critical) ─────────────
# Writes matured 20d forward returns for momentum_shadow_predictions into
# momentum_shadow_outcomes. Point-in-time: only scores picks once 20 trading days
# have elapsed (refuses immature picks). Completes the self-running shadow->outcome
# loop so the live momentum validation accumulates without manual runs. Non-fatal.
log "Stage 6: Momentum shadow 20d outcome reconcile"
if $PYTHON scripts/reconcile_momentum_shadow.py \
    > "$LOGDIR/05_momentum_reconcile.log" 2>&1; then
    log "Stage 6 OK — momentum shadow outcomes reconciled"
else
    log "Stage 6 FAILED rc=$? (continuing — shadow reconcile is non-critical)"
fi

# ── Stage 7: Horizon health snapshot (non-critical) ──────────────────────────
# Computes high-conf + overall accuracy per horizon into the horizon_health
# table (history archive) + prints a summary here for pipecheck. Tracks whether
# h1 recovers (broke ~week 2026-24). Read-only vs predictions/outcomes. Non-fatal.
log "Stage 7: Horizon health snapshot"
if $PYTHON -m analysis.horizon_health_compute >> "$LOGDIR/pipeline.log" 2>&1; then
    log "Stage 7 OK — horizon health recorded"
else
    log "Stage 7 FAILED rc=$? (continuing — horizon health is non-critical)"
fi

log "=== PIPELINE B COMPLETE ==="
