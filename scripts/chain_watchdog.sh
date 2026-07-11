#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
LOG="$ROOT/logs/pipeline_chain.log"
WLOG="$ROOT/logs/chain_watchdog.log"
TODAY=$(date '+%Y-%m-%d')
TS=$(date '+%Y-%m-%d %H:%M:%S')
if grep -q "\[$TODAY .* CHAIN START" "$LOG" 2>/dev/null; then
    echo "[$TS] OK chain fired today ($TODAY)" >> "$WLOG"
    exit 0
fi
echo "[$TS] ALERT chain did NOT fire today ($TODAY) - attempting catch-up" >> "$WLOG"
osascript -e 'display notification "ADB chain did not run. Check pipecheck / fire catch-up." with title "ML Quant WATCHDOG"' 2>/dev/null
nohup "$ROOT/scripts/pipeline_chain_ADB.sh" >> "$ROOT/logs/watchdog_catchup_$TODAY.log" 2>&1 &
echo "[$TS] catch-up launched pid $!" >> "$WLOG"
