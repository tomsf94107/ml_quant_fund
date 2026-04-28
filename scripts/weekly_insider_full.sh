#!/bin/bash
# scripts/weekly_insider_full.sh
# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY FULL INSIDER ETL
# Runs Sundays 4 AM VN. Does a full 365-day Form 4 rebuild.
# Daily incremental (Pipeline A) catches new filings; weekly full rebuild
# catches late corrections and backfills anything missed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
DATE_TAG=$(date +%Y%m%d)
LOGDIR=$ROOT/logs/weekly_insider_$DATE_TAG
mkdir -p "$LOGDIR"

cd $ROOT
source /Users/atomnguyen/.zshrc 2>/dev/null || true

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/pipeline.log"
}

log "=== WEEKLY INSIDER ETL START ==="

# Compute days_back based on actual gap in DB so we only fetch what's missing.
# Daily Pipeline A normally keeps the gap small (1-3 days). After a missed run
# (e.g. Mac asleep), gap could be larger. +2 day buffer catches filing latency.
# Floor at 7 ensures we always re-check the last week for late amendments.
DAYS_BACK=$($PYTHON -c "
import sqlite3
from datetime import date
conn = sqlite3.connect('insider_trades.db')
row = conn.execute('SELECT MAX(date) FROM insider_flows').fetchone()
conn.close()
if row and row[0]:
    gap = (date.today() - date.fromisoformat(row[0])).days
    print(max(gap + 2, 7))
else:
    print(365)
")
log "Computed days_back=$DAYS_BACK based on DB gap"

$PYTHON -m data.etl_insider --days-back $DAYS_BACK > "$LOGDIR/etl_insider.log" 2>&1 || {
    log "FAILED: etl_insider (days_back=$DAYS_BACK)"
    osascript -e "display notification \"Weekly insider ETL failed\" with title \"ML Quant Fund\"" 2>/dev/null || true
    exit 1
}

log "=== WEEKLY INSIDER ETL COMPLETE (days_back=$DAYS_BACK) ==="
