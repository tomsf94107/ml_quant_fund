#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
# Sequential pipeline chain: A (ingest) -> D (alpha panel) -> B (train+predict).
# Replaces fixed-time entries (03:00 A / 04:00 D / 07:00 B) that raced when A
# overran (Jun 12: first 394-name ingest took 4.6h, B fired into missing marker).
# B's own marker guard remains as failure protection; this handles timing.
LOG=/Users/atomnguyen/Desktop/ML_Quant_Fund/logs/pipeline_chain.log

# ── MUTUAL EXCLUSION (Aug 1 2026) ────────────────────────────────────────────
# The chain fired TWICE daily: cron 0 3 * * 2-6 AND launchd com.atom.pipeline-
# chain 04:00 (a Jun-30 cron->launchd migration that added agents but never
# removed the cron lines). Two full A->D->B runs = double UW spend, double
# retrain, CONCURRENT WRITERS on the same SQLite files -- a better explanation
# for the Jul-17 and Jul-25 "database is locked" incidents than iCloud.
# Removing a scheduler is not enough: a long A-run can overlap the next day's,
# a manual run can collide with a scheduled one, and a launchd job suspended in
# DarkWake can resume into a fresh one. mkdir is atomic on POSIX (flock is
# util-linux and NOT present on this Mac -- verified); the PID check recovers
# from kill -9, which is the one thing flock would do better.
LOCKDIR=/tmp/ml_quant_pipeline_chain.lock
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    OLDPID=$(cat "$LOCKDIR/pid" 2>/dev/null || echo "")
    if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: chain already running (pid $OLDPID)" >> "$LOG"
        exit 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] stale lock (pid '${OLDPID:-none}' dead) -- reclaiming" >> "$LOG"
    rm -rf "$LOCKDIR"
    mkdir "$LOCKDIR" || { echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: lock race lost" >> "$LOG"; exit 0; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
# ─────────────────────────────────────────────────────────────────────────────

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === CHAIN START (pid $$) ===" >> "$LOG"
/Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/pipeline_A_ingest.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] A exited $? -> starting D" >> "$LOG"
/Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/pipeline_D_alpha_panel.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] D exited $? -> starting B" >> "$LOG"
/Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/pipeline_B_train_predict.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] B exited $? === CHAIN END ===" >> "$LOG"
