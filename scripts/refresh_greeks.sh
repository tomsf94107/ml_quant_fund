#!/bin/bash
# Daily GEX pull. UW's /api/stock/{t}/greek-exposure returns a rolling ~250 days
# and CANNOT be backfilled further -- history only accrues by pulling it. Every day
# this doesn't run is a day of data that is gone permanently. That is why it is
# cron'd immediately, before the signal is even proven.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
cd "$ROOT"
source /Users/atomnguyen/.zshrc 2>/dev/null || true
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi
: "${UW_API_KEY:?FATAL: UW_API_KEY not set}"
/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python backfill_greeks.py
