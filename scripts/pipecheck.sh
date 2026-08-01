#!/bin/bash
# scripts/pipecheck.sh
# ─────────────────────────────────────────────────────────────────────────────
# Status check for Pipelines A, B, C, D
# All times displayed in ET (system tz is ET via crontab TZ=America/New_York)
# ─────────────────────────────────────────────────────────────────────────────
DATE=$(date +%Y%m%d)
NOW_HOUR=$(TZ=America/New_York date +%H)

echo "=== $(TZ=America/New_York date '+%Y-%m-%d %H:%M %Z') ==="
echo ""

check_pipeline() {
    local P=$1
    local SCHED_HOUR=$2   # hour in ET (0-23)
    local SCHED_LABEL=$3  # human-readable e.g. "16:00 ET"
    local LOGDIR=~/Desktop/ML_Quant_Fund/logs/pipeline_${P}_$DATE

    echo "── Pipeline $P (scheduled $SCHED_LABEL) ──"

    if [ -d "$LOGDIR" ] && [ -f "$LOGDIR/pipeline.log" ]; then
        tail -8 "$LOGDIR/pipeline.log"
        ERR=$(grep -lE "FAILED|ERROR|Traceback" $LOGDIR/*.log 2>/dev/null | wc -l | tr -d ' ')
        if [ "$ERR" -gt 0 ]; then
            echo "⚠️ $ERR log file(s) with errors"
        fi
        # Check marker if pipeline writes one
        local MARKER=~/Desktop/ML_Quant_Fund/logs/.pipeline_${P}_done_$DATE
        if [ -f "$MARKER" ]; then
            echo "✅ Marker present"
        fi
    else
        local NOW_H=$(echo $NOW_HOUR | sed 's/^0//')
        if [ "${NOW_H:-0}" -lt "${SCHED_HOUR:-0}" ]; then
            echo "⏳ Pending (will run at $SCHED_LABEL)"
        else
            echo "❌ MISSING — should have run by now"
        fi
    fi
    echo ""
}

#                  P   ET_HOUR  LABEL
check_pipeline    A    16       "16:00 ET (Mon-Fri)"
check_pipeline    D    17       "17:00 ET (Mon-Fri) — alpha panel"
check_pipeline    B    20       "20:00 ET (Mon-Fri)"
check_pipeline    C    8        "08:00 ET (Mon-Fri) — pre-open"

echo "── Health check ──"
~/.pyenv/versions/ml_quant_310/bin/python - <<'PYEOF'
import json, os, datetime as dt
p = os.path.expanduser('~/Desktop/ML_Quant_Fund/logs/health_status.json')
try:
    d = json.load(open(p))
    ts = dt.datetime.fromisoformat(d['checked_at'])
    mins = (dt.datetime.now() - ts).total_seconds() / 60
    age = f"{mins:.0f}m ago" if mins < 120 else f"{mins/60:.1f}h ago"
    if mins > 30 * 60:
        print(f"⚠️  STALE — last health check {age} ({ts:%Y-%m-%d %H:%M}). "
              f"Is the 13:00 VN cron running?")
    if d.get('status') == 'ok':
        print(f"✅ all checks passed   (checked {ts:%H:%M}, {age}, for {d.get('last_date')})")
    else:
        print(f"❌ {', '.join(d.get('failures') or ['unknown'])}   "
              f"(checked {ts:%H:%M}, {age}, for {d.get('last_date')})")
except FileNotFoundError:
    print("○ no health_status.json yet — run scripts/health_check.py once")
except Exception as e:
    print(f"health status read error: {e}")
PYEOF
echo ""

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
echo "── Alpha panel today ──"
PARQUET=~/Desktop/ML_Quant_Fund/data/alpha_panel/$TODAY.parquet
if [ -f "$PARQUET" ]; then
    SIZE_KB=$(ls -l "$PARQUET" | awk '{print int($5/1024)}')
    echo "✅ $TODAY.parquet (${SIZE_KB} KB)"
else
    echo "○ no parquet for $TODAY yet"
fi

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
