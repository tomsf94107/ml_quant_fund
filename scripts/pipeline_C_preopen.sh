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

# ── Stage -2 (May 31 2026): Momentum shadow runs FIRST, INDEPENDENT of the guard ──
# Momentum is pure-price and does NOT use the broken direction model, so it must
# log every night regardless of the direction-guard verdict. Previously it sat
# after the Stage -1 guard's 'exit 2' and never ran while the guard was RED — the
# shadow loop was strangled. Moved above the guard so shadow accumulates; the guard
# still aborts the DIRECTION-MODEL publish (Stages 1-2) on RED.
log "Stage -2: Momentum shadow signal (cross-sectional, shadow-only)"
MOM_START=$(date +%s)
if timeout 600 $PYTHON -m scripts.momentum_shadow \
    > "$LOGDIR/03_momentum_shadow.log" 2>&1; then
    MOM_DUR=$(($(date +%s) - MOM_START))
    log "Stage -2 OK (${MOM_DUR}s) — momentum shadow picks logged"
else
    MOM_RC=$?
    if [ $MOM_RC -eq 124 ]; then
        log "Stage -2 TIMEOUT after 10min (continuing — shadow is non-critical)"
    else
        log "Stage -2 FAILED rc=$MOM_RC (continuing — shadow is non-critical)"
    fi
fi

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
    # ── GUARD-RED HANDLING (Jun 2 2026 fix) ──────────────────────────────────
    # The guard exists to stop publishing a BROKEN DIRECTION signal. But the abort
    # only matters if the direction model is LIVE. While ML_QUANT_DISABLE_BUY=1
    # (BUYs forced to HOLD in generator.py), a RED guard cannot publish anything
    # broken — so aborting only freezes the PRICE / momentum / sentiment refresh
    # (Stage 2 daily_runner never runs -> signals_cache.json keeps prior-session
    # prices). That is exactly the Jun 1 2026 bug: guard went RED (h=5 dir inverted),
    # pipeline aborted, NVDA showed the May 29 close (211.14) all of Jun 1.
    # FIX: if BUYs are disabled, WARN + CONTINUE (prices/momentum still refresh,
    # kill switch keeps every BUY->HOLD). Only ABORT when the model is truly live.
    if [ "${ML_QUANT_DISABLE_BUY:-1}" = "1" ]; then
        log "Stage -1 RED — but DIRECTION MODEL DISABLED (ML_QUANT_DISABLE_BUY=1)."
        log "  WARNING only; CONTINUING (prices/momentum/sentiment refresh; kill switch forces BUY->HOLD)."
        cat "$LOGDIR/neg1_sanity_guard.log" | tee -a "$LOGDIR/pipeline.log"
    else
        log "Stage -1 RED — direction model ENABLED; ABORTING publish (broken signal direction)"
        osascript -e "display notification \"Pipeline C ABORTED: signal sanity guard RED\" with title \"ML Quant Fund\"" 2>/dev/null || true
        cat "$LOGDIR/neg1_sanity_guard.log" | tee -a "$LOGDIR/pipeline.log"
        exit 2
    fi
fi


# ── Stage 0: Daily sentiment (non-critical, 45-min timeout (394 names; was 15min at 149)) ──────────────────
log "Stage 0: Daily sentiment scoring"
SENT_START=$(date +%s)
if timeout 2700 $PYTHON -m scripts.daily_sentiment \
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
$PYTHON -m scripts.daily_uw_snapshot --mode full \
    > "$LOGDIR/01_uw_snap.log" 2>&1 || fail "Stage 1 (uw_snapshot)"
log "Stage 1 OK"

# ── Stage 2: Fresh runfund ───────────────────────────────────────────────────
log "Stage 2: Daily runner (fresh signals with live UW data)"
$PYTHON -m scripts.daily_runner_batched \
    > "$LOGDIR/02_daily_runner.log" 2>&1 || fail "Stage 2 (daily_runner)"
log "Stage 2 OK"


log "=== PIPELINE C COMPLETE ==="
