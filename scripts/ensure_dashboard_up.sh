#!/bin/bash
# scripts/ensure_dashboard_up.sh
# Ensures Streamlit dashboard is up + Mac stays awake 24/7, reachable over
# Tailscale (http://<tailscale-ip>:8501) anytime. Idempotent (cron self-heal).
# NOTE: 24/7 awake needs the Mac PLUGGED IN (macOS sleeps on battery regardless).
ROOT="/Users/atomnguyen/Desktop/ML_Quant_Fund"
STREAMLIT="/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/streamlit"
APP="ui/1_Dashboard.py"
PORT=8501
LOGDIR="$ROOT/logs"
LOG="$LOGDIR/dashboard_ensure.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p "$LOGDIR"

# ABSOLUTE PATH (Aug 1 2026). lsof lives in /usr/sbin, which is NOT on cron's
# PATH (/usr/bin:/bin) -- verified: `env -i /bin/bash -c 'lsof -v'` exits 127.
# Unqualified lsof -> command-not-found -> non-zero -> the else branch fired and
# spawned a doomed duplicate streamlit EVERY 15 MINUTES: 2,854 false "dashboard
# DOWN" lines vs 3 correct "already up", while pid 56574 held :8501 since Jul 14.
LSOF=/usr/sbin/lsof
[ -x "$LSOF" ] || LSOF=$(command -v lsof || echo /usr/sbin/lsof)
if "$LSOF" -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[$(ts)] dashboard already up on :$PORT" >> "$LOG"
else
    echo "[$(ts)] dashboard DOWN — starting on :$PORT" >> "$LOG"
    cd "$ROOT" || exit 1
    nohup "$STREAMLIT" run "$APP" \
        --server.address 0.0.0.0 --server.port $PORT --server.headless true \
        >> "$LOGDIR/dashboard_streamlit.log" 2>&1 &
    echo "[$(ts)] started streamlit pid=$!" >> "$LOG"
fi

if ! pgrep -x caffeinate >/dev/null 2>&1; then
    nohup caffeinate -i -s >> "$LOGDIR/caffeinate.log" 2>&1 &
    echo "[$(ts)] caffeinate armed (continuous 24/7)" >> "$LOG"
fi
