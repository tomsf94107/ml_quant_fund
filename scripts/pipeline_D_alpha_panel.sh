#!/bin/bash
# scripts/pipeline_D_alpha_panel.sh
# ─────────────────────────────────────────────────────────────────────────────
# ALPHA PANEL BUILDER PIPELINE (Sprint 2 Stage 2)
# Runs 17:00 ET Mon-Fri
# Depends on: Pipeline A (16:00 ET, marker file)
# Feeds into: Pipeline B (20:00 ET, optional consumer)
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 1: Dependency check — Pipeline A marker must exist
#   Stage 2: Build alpha panel via analysis.build_alpha_panel
#   Stage 3: Verify parquet write succeeded
#   Stage 4: Write marker for downstream consumers
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
DATE_TAG=$(date +%Y%m%d)
LOGDIR=$ROOT/logs/pipeline_D_$DATE_TAG
mkdir -p "$LOGDIR"

cd $ROOT
source /Users/atomnguyen/.zshrc 2>/dev/null || true

MARKER=$ROOT/logs/.pipeline_D_done_$DATE_TAG
rm -f "$MARKER"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"
}

fail() {
    log "FAILED at $1"
    osascript -e "display notification \"Pipeline D failed at $1\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
}

log "=== PIPELINE D START (alpha panel) ==="

# ── Stage 1: Dependency check ────────────────────────────────────────────────
log "Stage 1: Checking Pipeline A completed today"
A_MARKER=$ROOT/logs/.pipeline_A_done_$DATE_TAG
if [ ! -f "$A_MARKER" ]; then
    log "WARNING: Pipeline A marker not found — continuing anyway (D doesn't strictly require it)"
fi

# ── Stage 2: Build alpha panel ───────────────────────────────────────────────
log "Stage 2: Building alpha panel via analysis.build_alpha_panel"
$PYTHON -c "
import sys, logging, time
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from analysis.build_alpha_panel import build_alpha_panel

t0 = time.time()
summary = build_alpha_panel(
    tickers=None,           # load all from tickers.txt (125 tickers)
    start_date='2024-01-01',
    target_dates=None,      # write all dates available
    verbose=True,
)
print(f'Elapsed: {time.time()-t0:.1f}s')
print(f'Dates: {summary[\"dates_written\"]}, Alphas: {summary[\"alphas_written\"]}')
if summary['dates_written'] == 0:
    sys.exit('Pipeline D produced 0 dates — failing loudly')
" >> "$LOGDIR/build_alpha_panel.log" 2>&1 || fail "Stage 2 (build_alpha_panel)"

# ── Stage 3: Verify ──────────────────────────────────────────────────────────
log "Stage 3: Verifying parquet output"
# Find the most recent parquet file in alpha_panel/
LATEST_PARQUET=$(ls -t $ROOT/data/alpha_panel/*.parquet 2>/dev/null | head -1)
if [ -z "$LATEST_PARQUET" ]; then
    fail "Stage 3 — no parquet files written"
fi
LATEST_DATE=$(basename "$LATEST_PARQUET" .parquet)
SIZE_KB=$(ls -l "$LATEST_PARQUET" | awk '{print int($5/1024)}')

# Warn if latest date is older than 1 trading day (would suggest data not fresh)
TODAY=$(TZ=America/New_York date +%Y-%m-%d)
YESTERDAY_TS=$(TZ=America/New_York date -v-1d +%Y-%m-%d)
if [ "$LATEST_DATE" = "$TODAY" ]; then
    log "  ✅ $LATEST_DATE.parquet present (${SIZE_KB} KB) — today's data"
elif [ "$LATEST_DATE" = "$YESTERDAY_TS" ]; then
    log "  ✅ $LATEST_DATE.parquet present (${SIZE_KB} KB) — yesterday's data (likely ran before US close)"
else
    log "  ⚠️  $LATEST_DATE.parquet present (${SIZE_KB} KB) — older than expected"
fi

# ── Stage 4: Marker ──────────────────────────────────────────────────────────
touch "$MARKER"
log "=== PIPELINE D DONE ==="
