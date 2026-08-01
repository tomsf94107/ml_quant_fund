#!/bin/bash
# Safe crontab installer. NEVER install without diffing first.
#
# WHY (2026-07-26 incident): `crontab scripts/crontab_VN_anchored.txt` was run
# on a file exported after the May-13 VN revert. Every job added to the LIVE
# crontab since May existed only there -- the install destroyed 11 of them
# (sync_prices_from_rawbars, refresh_greeks, backfill_vix, si_fetch_v2,
# si_cycle, borrow_fetch, analyst_revisions, research_reminders, 2 void
# monitors, and feed_freshness_check -- the canary itself) and stripped
# `set -a && . ./.env && set +a` from ~15 others, running them keyless.
# Undetected for 6 days because the detector died with them.
#
# This refuses any install that would LOSE an active job unless --force.
set -euo pipefail
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
FILE="${1:-$ROOT/scripts/crontab_VN_anchored.txt}"
FORCE="${2:-}"
TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$HOME/Desktop/crontab_backup_$TS.txt"

[ -f "$FILE" ] || { echo "FATAL: $FILE not found"; exit 1; }
crontab -l > "$BACKUP" 2>/dev/null || echo "" > "$BACKUP"
echo "current crontab backed up -> $BACKUP"

active() { grep -vE '^\s*#|^\s*$' "$1" | grep -E '^[0-9*]' | sort; }
LOST=$(comm -23 <(active "$BACKUP") <(active "$FILE") || true)
GAINED=$(comm -13 <(active "$BACKUP") <(active "$FILE") || true)

if [ -n "$LOST" ]; then
    echo ""
    echo "!!! THESE ACTIVE JOBS WOULD BE DESTROYED:"
    echo "$LOST" | sed 's/^/    /'
    if [ "$FORCE" != "--force" ]; then
        echo ""
        echo "REFUSING. Either add them to $FILE, or re-run with --force if"
        echo "removal is intended. Backup: $BACKUP"
        exit 2
    fi
    echo "--force given; proceeding with removal."
fi
[ -n "$GAINED" ] && { echo ""; echo "jobs being ADDED:"; echo "$GAINED" | sed 's/^/    /'; }

crontab "$FILE"
echo ""
echo "installed. active job count: $(crontab -l | grep -cE '^[0-9*]')"
diff <(active "$FILE") <(crontab -l | grep -vE '^\s*#|^\s*$' | grep -E '^[0-9*]' | sort) \
    && echo "VERIFIED: installed crontab matches file"
