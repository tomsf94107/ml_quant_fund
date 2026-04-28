#!/bin/bash
# scripts/pipeline_A_ingest.sh
# ─────────────────────────────────────────────────────────────────────────────
# POST-CLOSE INGEST PIPELINE
# Runs 3 AM VN (4 PM ET prev day) Tue-Sat
# Populates DB tables that Pipeline B's retrain/predict reads from
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 1: Insider ETL (incremental, 7-day window — fast)
#   Stage 2: UW post-market snapshot (dark pool + skew)
#   Stage 3: Feature validator (catches any bad DB rows early)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail  # exit on any error, undefined var, or pipe failure

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
DATE_TAG=$(date +%Y%m%d)
LOGDIR=$ROOT/logs/pipeline_A_$DATE_TAG
mkdir -p "$LOGDIR"

cd $ROOT
source /Users/atomnguyen/.zshrc 2>/dev/null || true

# Marker file — Pipeline B checks for this before running
MARKER=$ROOT/logs/.pipeline_A_done_$DATE_TAG
rm -f "$MARKER"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"
}

fail() {
    log "FAILED at $1"
    log "See $LOGDIR/ for details"
    # Optional: osascript for macOS desktop notification
    osascript -e "display notification \"Pipeline A failed at $1\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
}

log "=== PIPELINE A START ==="

# ── Stage 1: Insider ETL (incremental) ───────────────────────────────────────
log "Stage 1: Insider ETL (7-day incremental)"
$PYTHON -c "
from pathlib import Path
from data.etl_insider import run_insider_etl
tickers = [t.strip().upper() for t in Path('tickers.txt').read_text().splitlines()
           if t.strip() and not t.startswith('#')]
r = run_insider_etl(tickers, days_back=7, verbose=False)
total = sum(r.values())
print(f'insider ETL: {total} rows across {len(tickers)} tickers')
" > "$LOGDIR/01_insider.log" 2>&1 || fail "Stage 1 (etl_insider)"
log "Stage 1 OK"

# ── Stage 2: UW post-market snapshot ─────────────────────────────────────────
log "Stage 2: UW post-market snapshot"
$PYTHON scripts/daily_uw_snapshot.py --mode post_market \
    > "$LOGDIR/02_uw_snap.log" 2>&1 || fail "Stage 2 (uw_snapshot)"
log "Stage 2 OK"

# ── Stage 3: Feature validator ───────────────────────────────────────────────
log "Stage 3: Feature validator"
$PYTHON scripts/feature_validator.py --fix \
    > "$LOGDIR/03_feature_validator.log" 2>&1 || fail "Stage 3 (feature_validator)"
log "Stage 3 OK"

# Mark success so Pipeline B can run
touch "$MARKER"
log "=== PIPELINE A COMPLETE ==="
