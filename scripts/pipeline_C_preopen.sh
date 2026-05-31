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

# ── STEP 1 GUARD (May 31 2026): block publish if signal direction is inverted ──
# The direction model has been near-coin-flip and inverted at the extremes. This
# guard runs the standing sanity check BEFORE any signal work; on RED (direction
# decile spread significantly negative) it ABORTS the pipeline so no broken signals
# get published. Defense-in-depth alongside the ML_QUANT_DISABLE_BUY kill switch.
log "Stage -1: signal sanity guard (direction / BUY-hit / rank-IC)"
if $PYTHON analysis/signal_sanity_guard.py --db accuracy.db --days 90 \
    > "$LOGDIR/neg1_sanity_guard.log" 2>&1; then
    log "Stage -1 OK — sanity guard GREEN, proceeding"
else
    log "Stage -1 RED — sanity guard FAILED; ABORTING publish (broken signal direction)"
    osascript -e "display notification \"Pipeline C ABORTED: signal sanity guard RED\" with title \"ML Quant Fund\"" 2>/dev/null || true
    cat "$LOGDIR/neg1_sanity_guard.log" | tee -a "$LOGDIR/pipeline.log"
    exit 2
fi


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

# ── Stage 3: Momentum shadow signal (VALIDATED Step 2b, SHADOW MODE) ──────────
# Cross-sectional momentum (passed purged-WF 4/4 OOS folds). Runs ONCE over the
# full universe (needs all names at once). Writes to momentum_shadow_predictions
# ONLY — NOT live predictions, NOT live BUYs. Builds the live track record that
# will gate promotion to the real signal path. Non-critical: never breaks publish.
log "Stage 3: Momentum shadow signal (cross-sectional, shadow-only)"
MOM_START=$(date +%s)
if timeout 600 $PYTHON scripts/momentum_shadow.py \
    > "$LOGDIR/03_momentum_shadow.log" 2>&1; then
    MOM_DUR=$(($(date +%s) - MOM_START))
    log "Stage 3 OK (${MOM_DUR}s) — momentum shadow picks logged"
else
    MOM_RC=$?
    if [ $MOM_RC -eq 124 ]; then
        log "Stage 3 TIMEOUT after 10min (continuing — shadow is non-critical)"
    else
        log "Stage 3 FAILED rc=$MOM_RC (continuing — shadow is non-critical)"
    fi
fi

log "=== PIPELINE C COMPLETE ==="
