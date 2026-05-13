#!/bin/bash
# scripts/pipeline_C_preopen.sh
# ─────────────────────────────────────────────────────────────────────────────
# PRE-OPEN FRESH RUNFUND
# Runs 08:00 ET Mon-Fri — ~90 min before US market open (was 19:00 VN before ET migration)
# Chains sentiment + UW snapshot + fresh daily runner predictions
# Uses the model trained by Pipeline B that morning
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 0: Daily sentiment scoring (~5-8 min, non-critical, continues on fail)
#   Stage 1: UW full snapshot (short interest, analyst, FTDs, seasonality)
#   Stage 2: Run daily predictions again with fresh live features
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Cron compatibility: explicit PATH so homebrew tools (timeout, etc.) are found
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
DATE_TAG=$(date +%Y%m%d)
LOGDIR=$ROOT/logs/pipeline_C_$DATE_TAG
mkdir -p "$LOGDIR"

cd $ROOT
source /Users/atomnguyen/.zshrc 2>/dev/null || true

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"
}

fail() {
    log "FAILED at $1"
    osascript -e "display notification \"Pipeline C failed at $1\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
}

log "=== PIPELINE C START ==="

# ── Stage 0: Daily sentiment (non-critical, 15-min timeout) ──────────────────
log "Stage 0: Daily sentiment scoring"
SENT_START=$(date +%s)
if timeout 900 $PYTHON scripts/daily_sentiment.py \
    > "$LOGDIR/00_sentiment.log" 2>&1; then
    SENT_DUR=$(($(date +%s) - SENT_START))
    log "Stage 0 OK (${SENT_DUR}s)"
else
    SENT_RC=$?
    SENT_DUR=$(($(date +%s) - SENT_START))
    if [ $SENT_RC -eq 124 ]; then
        log "Stage 0 TIMEOUT after 15min (continuing anyway)"
    else
        log "Stage 0 FAILED rc=$SENT_RC (continuing anyway, sentiment is non-critical)"
    fi
    osascript -e "display notification \"Pipeline C: sentiment failed (continuing)\" with title \"ML Quant Fund\"" 2>/dev/null || true
fi

# ── Stage 1: UW full snapshot ────────────────────────────────────────────────
log "Stage 1: UW full snapshot (short interest, analyst, FTDs, seasonality)"
$PYTHON scripts/daily_uw_snapshot.py --mode full \
    > "$LOGDIR/01_uw_snap.log" 2>&1 || fail "Stage 1 (uw_snapshot)"
log "Stage 1 OK"

# ── Stage 2: Fresh runfund ───────────────────────────────────────────────────
log "Stage 2: Daily runner (fresh signals with live UW data)"
$PYTHON -m scripts.daily_runner_batched \
    > "$LOGDIR/02_daily_runner.log" 2>&1 || fail "Stage 2 (daily_runner)"
log "Stage 2 OK"

log "=== PIPELINE C COMPLETE ==="
