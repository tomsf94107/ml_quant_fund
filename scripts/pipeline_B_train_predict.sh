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
source /Users/atomnguyen/.zshrc 2>/dev/null || true

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

# ── Stage 2: Retrain ─────────────────────────────────────────────────────────
# NOTE: If you move to weekly retrain, wrap this in a day-of-week check:
#   if [ "$(date +%u)" -eq 7 ]; then ... fi   # only on Sundays
# For now, keeps your daily retrain behavior but with proper sequencing.
log "Stage 2: Retrain all models"
$PYTHON -m models.train_all_batched \
    > "$LOGDIR/02_train_all.log" 2>&1 || fail "Stage 2 (train_all)"
log "Stage 2 OK"

# ── Stage 3: Daily predictions ───────────────────────────────────────────────
log "Stage 3: Daily runner (generates today's signals)"
$PYTHON -m scripts.daily_runner_batched \
    > "$LOGDIR/03_daily_runner.log" 2>&1 || fail "Stage 3 (daily_runner)"
log "Stage 3 OK"

# ── Stage 4: Daily validator ─────────────────────────────────────────────────
log "Stage 4: Daily validator (checks recent predictions for anomalies)"
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

log "=== PIPELINE B COMPLETE ==="
