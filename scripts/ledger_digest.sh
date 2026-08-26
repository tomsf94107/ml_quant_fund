#!/bin/bash
# Paste-ready digest of docs/DEFECT_LEDGER.md for report threads that cannot
# read the repo. Prints OPEN-class rows only, with a provenance stamp so a
# report can cite exactly which ledger state it rendered from.
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
L="$ROOT/docs/DEFECT_LEDGER.md"
cd "$ROOT" || exit 1
CLOSED='\| *(FIXED|DEAD|REMOVED|DONE|NOTED|RETRACTED|NOT A DEFECT|POLICY|CORRECTION) *\|'
echo "LEDGER DIGEST -- $(date '+%Y-%m-%d %H:%M %Z') -- repo $(git rev-parse --short HEAD)"
echo "Source: docs/DEFECT_LEDGER.md. Render section 4 from THIS. Carry ONLY the rows below."
echo ""
# Collapse duplicate items: the ledger is append-only, so a status change adds a
# ROW rather than amending one. Three items carried contradictory statuses as of
# 2026-08-26 (Peer-relative DORMANT: OPEN + CLOSED; Aggressor-tilt: BLOCKED +
# UNBLOCKED; Intraday session H/L: OPEN + FIXED) and section 4 rendered BOTH halves.
# Last row wins; the row prints at the position of its FIRST appearance.
# MUST run BEFORE the $CLOSED filter -- filtering first would strip the resolving
# row and leave the stale one, which is the defect this fixes.
_collapse() { awk -F'|' '/^\|/{k=$2;gsub(/^ +| +$/,"",k); if(!(k in seen))order[++n]=k; seen[k]=$0} END{for(i=1;i<=n;i++)print seen[order[i]]}' "$L"; }

_collapse | grep -viE "$CLOSED" | grep -vE '^\|-+' | grep -viE '^\| *Item *\|'
echo ""
echo "open-class rows: $(_collapse | grep -viE "$CLOSED" | grep -vcE '^\|-+|^\| *Item *\|')"
echo "total rows:      $(grep -cE '^\|' "$L")  ($(_collapse | grep -cE '^\|') after collapse)"
