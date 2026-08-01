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
grep -E '^\|' "$L" | grep -viE "$CLOSED" | grep -vE '^\|-+' | grep -viE '^\| *Item *\|'
echo ""
echo "open-class rows: $(grep -E '^\|' "$L" | grep -viE "$CLOSED" | grep -vcE '^\|-+|^\| *Item *\|')"
echo "total rows:      $(grep -cE '^\|' "$L")"
