#!/bin/bash
# SI CYCLE — the only live-money book in this system, and the only loop that was
# still manual. Three jobs:
#   1. FETCH   new FINRA settlements (published ~8 business days after settlement)
#   2. ALERT   when a new settlement lands -> time to enter a new cohort
#   3. ALERT   when an open cohort hits 40 trading days -> EXIT DUE
#
# It does NOT trade. It tells you when to act. That is the right boundary for
# real money, and it mirrors the kill-switch philosophy used elsewhere.
#
# WHY THIS EXISTS: on 2026-07-12 the audit found six silently-dead jobs (six crons
# without credentials, the walk-forward dead 13 days, the VIX feed dead for months,
# momentum_shadow frozen 2 weeks). Every one had the same shape: it runs, writes
# nothing, exits 0, nobody is told. The SI book was the LAST manual loop -- and the
# only one where a missed step costs money directly.
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
PY=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
cd "$ROOT"

source /Users/atomnguyen/.finra_creds 2>/dev/null || true
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi
: "${FINRA_CLIENT_ID:?FATAL: FINRA_CLIENT_ID not set (source ~/.finra_creds)}"
: "${FINRA_SECRET:?FATAL: FINRA_SECRET not set}"

notify() { osascript -e "display notification \"$2\" with title \"$1\"" 2>/dev/null || true; }
log()    { echo "[$(date '+%F %T')] $*"; }

# ── 1. what settlement do we have BEFORE the fetch? ───────────────────────
BEFORE=$(sqlite3 "$ROOT/short_interest.db" "SELECT MAX(settlement_date) FROM short_interest;")
log "settlement before fetch: $BEFORE"

# ── 2. FETCH ──────────────────────────────────────────────────────────────
$PY si_fetch_v2.py --root . --months-back 2 2>&1 | tail -4

AFTER=$(sqlite3 "$ROOT/short_interest.db" "SELECT MAX(settlement_date) FROM short_interest;")
log "settlement after fetch:  $AFTER"

# ── 3. NEW SETTLEMENT -> rebalance alert ──────────────────────────────────
if [ "$AFTER" != "$BEFORE" ]; then
    log "*** NEW SETTLEMENT: $AFTER ***"
    notify "SI BOOK — NEW SETTLEMENT" "$AFTER published. Regenerate the book: si_positions_live.py"
    echo ""
    echo "  Run:  python3 si_positions_live.py --root . --capital 150000 --short-frac 0 --log-ledger"
    echo ""
    echo "  🔴 SIZING: settlements come every ~15 days, the hold is 40 TRADING days,"
    echo "     so ~2.7 cohorts overlap. Each cohort must be sized at ~1/2.7 of capital,"
    echo "     NOT 100%. Deploying full capital per settlement = ~2.7x LEVERED = the"
    echo "     -84% drawdown in si_book_diagnostic. This is the single most dangerous"
    echo "     misunderstanding available in this strategy."
else
    log "no new settlement (still $AFTER)"
fi

# ── 4. EXIT DUE? ──────────────────────────────────────────────────────────
TRACK=$($PY si_track.py --root . 2>&1)
echo "$TRACK" | tail -8
if echo "$TRACK" | grep -qi "EXIT DUE"; then
    N=$(echo "$TRACK" | grep -ci "EXIT DUE")
    log "*** EXIT DUE on $N cohort(s) ***"
    notify "SI BOOK — EXIT DUE" "$N cohort(s) hit 40 trading days. Close them."
fi
