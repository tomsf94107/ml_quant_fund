#!/bin/bash
# scripts/intraday_kill_switch_cron.sh
# Cron wrapper for intraday_kill_switch.py — loads env vars so UW_API_KEY
# is available in cron's minimal environment.

set -e

ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python

# Explicit PATH
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Load .env
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

source ~/.zshrc 2>/dev/null || true

cd "$ROOT"
$PYTHON scripts/intraday_kill_switch.py
