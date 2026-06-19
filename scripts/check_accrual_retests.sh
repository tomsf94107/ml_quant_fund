#!/bin/bash
# Pings when accrual-gated candidate signals have enough history to re-validate.
# Runs monthly; notifies via macOS notification + log.
ROOT="/Users/atomnguyen/Desktop/ML_Quant_Fund"
cd "$ROOT" || exit 1
N=$(sqlite3 accuracy.db "SELECT COUNT(DISTINCT date) FROM options_skew_history;" 2>/dev/null)
if [ "${N:-0}" -ge 150 ]; then
    MSG="skew_change re-test READY: options_skew_history has $N dates (>=150). Run validate_skew_change.py"
    osascript -e "display notification \"$MSG\" with title \"ML Quant — Re-test Ready\"" 2>/dev/null || true
    echo "[$(date '+%F %T')] $MSG" >> logs/accrual_retests.log
else
    echo "[$(date '+%F %T')] skew_change not ready: $N/150 dates" >> logs/accrual_retests.log
fi
