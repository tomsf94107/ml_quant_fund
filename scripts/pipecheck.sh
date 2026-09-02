#!/bin/bash
# scripts/pipecheck.sh
# ─────────────────────────────────────────────────────────────────────────────
# Status check for Pipelines A, B, C, D
# Times displayed in ET. NOTE (Aug 1 2026): the old header claimed the crontab
# sets TZ=America/New_York -- it does NOT. macOS BSD cron ignores TZ entirely,
# which is why scripts/crontab_VN_anchored.txt encodes VN local times. The chain
# (A->D->B) is owned by launchd com.atom.pipeline-chain at 04:00 VN = 17:00 ET;
# Pipeline C by com.atom.pipeline-c at 17:00 VN = 06:00 ET. See
# docs/SCHEDULER_INVENTORY.md (generated from live state).
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
        tail -14 "$LOGDIR/pipeline.log"
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

# ET hours below match ACTUAL launchd schedules, not the pre-migration plan.
# Chain starts 04:00 VN = 17:00 ET: A ~17:00-17:45, D ~17:45-19:50, B ~19:50-21:50.
# C starts 17:00 VN = 06:00 ET and finishes ~07:30 ET (kept ahead of the 09:30
# open; the old 08:00 ET start would have landed at the bell at current runtime).
#                  P   ET_HOUR  LABEL
check_pipeline    A    17       "17:00 ET (Tue-Sat VN 04:00) — ingest"
check_pipeline    D    18       "~17:45 ET (chained after A) — alpha panel"
check_pipeline    B    20       "~19:50 ET (chained after D) — train+predict"
check_pipeline    C    6        "06:00 ET (Mon-Fri VN 17:00) — pre-open"

echo "── Health check ──"
~/.pyenv/versions/ml_quant_310/bin/python - <<'PYEOF'
import json, os, datetime as dt
p = os.path.expanduser('~/Desktop/ML_Quant_Fund/logs/health_status.json')
try:
    d = json.load(open(p))
    ts = dt.datetime.fromisoformat(d['checked_at'])
    mins = (dt.datetime.now() - ts).total_seconds() / 60
    age = f"{mins:.0f}m ago" if mins < 120 else f"{mins/60:.1f}h ago"
    # SCHEDULE-AWARE (Aug 3 2026). health_check runs `0 13 * * 2-6` = Tue-Sat,
    # so a Sat 13:00 -> Tue 13:00 gap is 72h and completely normal. The flat 30h
    # threshold cried STALE every Sunday and Monday -- the exact false-alarm
    # class this display exists to end. Compare against the most recent
    # SCHEDULED run instead of a fixed age.
    _now = dt.datetime.now()
    _exp, _probe = None, _now.replace(hour=13, minute=0, second=0, microsecond=0)
    for _ in range(9):
        if _probe <= _now and _probe.weekday() in (1, 2, 3, 4, 5):  # Tue..Sat
            _exp = _probe
            break
        _probe -= dt.timedelta(days=1)
    if _exp and ts < _exp - dt.timedelta(hours=1):
        print(f"⚠️  STALE — last health check {age} ({ts:%Y-%m-%d %H:%M}); "
              f"expected a run at {_exp:%a %Y-%m-%d 13:00}. Is the cron firing?")
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
