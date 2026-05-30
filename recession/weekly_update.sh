#!/usr/bin/env bash
#
# recession/weekly_update.sh
#
# Weekly cron wrapper for the recession model. Does the two steps that
# keep the model live, IN ORDER:
#
#   1. INGEST  — python -m recession.data.ingest --update --refresh
#                Pulls the latest FRED data into recession.db.
#                --update  : incremental (latest months only, fast)
#                --refresh : bypass the local cache — this is the flag
#                            that was missing when INDPRO sat stale at a
#                            March vintage while April was already
#                            published. Without --refresh the ingest
#                            replays a stale cached response and writes
#                            nothing new.
#   2. REFRESH — python -m recession.refresh
#                Re-fits M1 on the (now updated) DB and appends the
#                reading to recession/refresh_log.txt.
#
# Step 2 runs ONLY if step 1 succeeds — re-evaluating the model on
# stale data would be pointless and misleading.
#
# WHAT THIS DOES AND DOES NOT FIX
#   Fixes: stale-cache lag. New FRED data is pulled automatically every
#          week with no manual step.
#   Does NOT fix: source publication lag. The ingest pulls whatever
#          FRED's API has at run time. Series FRED itself posts late
#          (e.g. MICH / Michigan inflation expectations, on a
#          contractual delay) only appear once FRED posts them — the
#          cron picks them up on the first run AFTER they land.
#
# CRON LINE (Vietnam-anchored, consistent with the project convention —
# macOS BSD cron uses system localtime and the box is on VN time).
# Ingest at 06:00 Sunday, two hours BEFORE refresh.py's 08:00 Sunday:
#
#   # recession model — weekly data ingest + model refresh, Sunday
#   0 6 * * 0  /Users/atomnguyen/Desktop/ML_Quant_Fund/recession/weekly_update.sh >> /Users/atomnguyen/Desktop/ML_Quant_Fund/recession/cron.log 2>&1
#
# (Remove refresh.py's own separate 08:00 cron line if it exists — this
# wrapper now runs refresh itself, so a separate refresh cron would just
# run it twice.)
#
# Exit code: 0 if both steps succeed, non-zero otherwise (for cron /
# log monitoring to detect).

set -u  # error on unset variables; we do NOT use -e (we handle errors)

# --- configuration -----------------------------------------------------
# Project root: the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python from the ml_quant_310 environment. Adjust if the env moves.
# Falls back to plain `python` if the explicit path is not found.
PYTHON="${ML_QUANT_PYTHON:-$HOME/miniconda3/envs/ml_quant_310/bin/python}"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="python"
fi

cd "${PROJECT_ROOT}" || {
    echo "[weekly_update] FATAL: cannot cd to ${PROJECT_ROOT}"
    exit 1
}

TS() { date "+%Y-%m-%d %H:%M:%S"; }

echo "=================================================================="
echo "[weekly_update] $(TS)  starting weekly recession-model update"
echo "[weekly_update] project root : ${PROJECT_ROOT}"
echo "[weekly_update] python        : ${PYTHON}"
echo "=================================================================="

# --- step 1: ingest ----------------------------------------------------
echo "[weekly_update] $(TS)  STEP 1/2  FRED ingest (--update --refresh)"
"${PYTHON}" -m recession.data.ingest --update --refresh
INGEST_RC=$?

if [ "${INGEST_RC}" -ne 0 ]; then
    echo "[weekly_update] $(TS)  INGEST FAILED (exit ${INGEST_RC})."
    echo "[weekly_update] skipping the model refresh — will not"
    echo "[weekly_update] re-evaluate the model on un-updated data."
    echo "[weekly_update] $(TS)  weekly update FAILED"
    exit "${INGEST_RC}"
fi
echo "[weekly_update] $(TS)  ingest OK"

# --- step 2: refresh ---------------------------------------------------
echo "[weekly_update] $(TS)  STEP 2/2  model refresh (recession.refresh)"
"${PYTHON}" -m recession.refresh
REFRESH_RC=$?

if [ "${REFRESH_RC}" -ne 0 ]; then
    echo "[weekly_update] $(TS)  REFRESH FAILED (exit ${REFRESH_RC})."
    echo "[weekly_update] note: ingest succeeded — the DB IS updated;"
    echo "[weekly_update] only the model re-evaluation failed."
    echo "[weekly_update] $(TS)  weekly update FAILED"
    exit "${REFRESH_RC}"
fi
echo "[weekly_update] $(TS)  refresh OK"

echo "=================================================================="
echo "[weekly_update] $(TS)  weekly update COMPLETE — both steps OK"
echo "=================================================================="
exit 0
