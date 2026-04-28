#!/bin/bash
# scripts/pipeline_B_train_predict.sh
# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + PREDICT PIPELINE
# Runs 7 AM VN (8 PM ET prev day) Tue-Sat
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
$PYTHON -m models.train_all \
    > "$LOGDIR/02_train_all.log" 2>&1 || fail "Stage 2 (train_all)"
log "Stage 2 OK"

# ── Stage 3: Daily predictions ───────────────────────────────────────────────
log "Stage 3: Daily runner (generates today's signals)"
$PYTHON -c "import sys; sys.path.insert(0,'.'); from scripts.daily_runner import run_daily; run_daily()" \
    > "$LOGDIR/03_daily_runner.log" 2>&1 || fail "Stage 3 (daily_runner)"
log "Stage 3 OK"

# ── Stage 4: Daily validator ─────────────────────────────────────────────────
log "Stage 4: Daily validator (checks recent predictions for anomalies)"
$PYTHON scripts/daily_validator.py --days 30 --fix \
    > "$LOGDIR/04_daily_validator.log" 2>&1 || fail "Stage 4 (daily_validator)"
log "Stage 4 OK"

log "=== PIPELINE B COMPLETE ==="
