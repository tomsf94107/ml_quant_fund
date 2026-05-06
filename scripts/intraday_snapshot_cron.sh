#!/bin/bash
# scripts/intraday_snapshot_cron.sh
# Cron wrapper for log_intraday_snapshot() — loads env vars so UW/Massive/etc.
# API keys are available in cron's minimal environment.
#
# Why this exists:
#   - Cron has bare-bones environment (no .env loaded, no .zshrc)
#   - Direct `python -c` calls from cron miss UW_API_KEY → 401 errors
#   - This wrapper loads .env first, then calls Python

set -e

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python

# Explicit PATH so homebrew tools are found (matches pipeline_C_preopen.sh)
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Load .env (UW_API_KEY, MASSIVE_API_KEY, ANTHROPIC_API_KEY, FRED_API_KEY)
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

# Also try .zshrc as fallback (covers anything exported there)
source ~/.zshrc 2>/dev/null || true

cd "$ROOT"
$PYTHON -c 'import sys; sys.path.insert(0,"."); from scripts.daily_runner import log_intraday_snapshot; log_intraday_snapshot()'
