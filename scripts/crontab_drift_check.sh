#!/bin/bash
# Detects divergence between the LIVE crontab and its source-of-truth file.
#
# WHY: the file went stale because jobs were added with `crontab -e` and never
# written back, which is what made the Jul-26 install destructive. Runs under
# LAUNCHD, deliberately: a crontab accident must not be able to disable the
# thing that detects crontab accidents.
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
FILE="$ROOT/scripts/crontab_VN_anchored.txt"
LOG="$ROOT/logs/crontab_drift.log"
TS=$(date '+%Y-%m-%d %H:%M:%S')
act() { grep -vE '^\s*#|^\s*$' | grep -E '^[0-9*]' | sort; }
LIVE=$(crontab -l 2>/dev/null | act)
SRC=$(act < "$FILE")
if [ "$LIVE" = "$SRC" ]; then
    echo "[$TS] OK crontab matches file ($(echo "$LIVE" | grep -c .) jobs)" >> "$LOG"
    exit 0
fi
ONLY_LIVE=$(comm -23 <(echo "$LIVE") <(echo "$SRC") | wc -l | tr -d ' ')
ONLY_FILE=$(comm -13 <(echo "$LIVE") <(echo "$SRC") | wc -l | tr -d ' ')
echo "[$TS] DRIFT: $ONLY_LIVE job(s) live-only, $ONLY_FILE file-only" >> "$LOG"
comm -23 <(echo "$LIVE") <(echo "$SRC") | sed "s/^/[$TS]   LIVE-ONLY: /" >> "$LOG"
comm -13 <(echo "$LIVE") <(echo "$SRC") | sed "s/^/[$TS]   FILE-ONLY: /" >> "$LOG"
/usr/bin/osascript -e "display notification \"crontab drift: $ONLY_LIVE live-only, $ONLY_FILE file-only. See logs/crontab_drift.log\" with title \"ML Quant SCHEDULER\"" 2>/dev/null
exit 1
