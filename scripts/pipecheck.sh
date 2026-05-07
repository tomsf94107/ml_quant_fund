#!/bin/bash
# Check status of Pipelines A, B, C
DATE=$(date +%Y%m%d)
NOW_HOUR=$(date +%H)

echo "=== $(date '+%Y-%m-%d %H:%M %Z') ==="
echo ""

check_pipeline() {
    local P=$1
    local SCHED=$2
    local LOGDIR=~/Desktop/ML_Quant_Fund/logs/pipeline_${P}_$DATE
    
    echo "── Pipeline $P (scheduled $SCHED VN) ──"
    
    if [ -d "$LOGDIR" ] && [ -f "$LOGDIR/pipeline.log" ]; then
        tail -8 "$LOGDIR/pipeline.log"
        ERR=$(grep -lE "FAILED|ERROR|Traceback" $LOGDIR/*.log 2>/dev/null | wc -l | tr -d ' ')
        if [ "$ERR" -gt 0 ]; then
            echo "⚠️ $ERR log file(s) with errors"
        fi
    else
        local SCHED_HOUR=$(echo $SCHED | cut -d: -f1 | sed 's/^0//')
        local NOW_H=$(echo $NOW_HOUR | sed 's/^0//')
        if [ "${NOW_H:-0}" -lt "${SCHED_HOUR:-0}" ]; then
            echo "⏳ Pending (will run at $SCHED VN)"
        else
            echo "❌ MISSING — should have run by now"
        fi
    fi
    echo ""
}

check_pipeline A "03:00"
check_pipeline B "07:00"
check_pipeline C "19:00"

echo "── DB predictions today ──"
TODAY=$(~/.pyenv/versions/ml_quant_310/bin/python -c 'from utils.timezone import today_et; print(today_et())')
sqlite3 ~/Desktop/ML_Quant_Fund/accuracy.db <<SQL
.headers on
.mode column
SELECT 
    COUNT(DISTINCT ticker) as tickers,
    COUNT(*) as total_rows
FROM predictions 
WHERE prediction_date = '$TODAY';
SQL

echo ""
echo "── Cache ──"
~/.pyenv/versions/ml_quant_310/bin/python -c "
import json, os
p = os.path.expanduser('~/Desktop/ML_Quant_Fund/data/signals_cache.json')
try:
    d = json.load(open(p))
    sigs = d['signals']
    print(f'Cache date:      {d[\"date\"]}')
    print(f'Last generated: {d[\"generated_at\"]}')
    print(f'Tickers:        {len(set(s[\"ticker\"] for s in sigs))}')
    print(f'Signals:        {len(sigs)}')
except Exception as e:
    print(f'Cache read error: {e}')
"
