#!/bin/zsh
# scripts/warning_chain.sh
#
# CEWS daily chain. Runs AFTER pipeline_chain_ADB.sh (launchd, 04:00 VN) because
# warning.db's price/breadth rows are copied out of prices.db, which that chain writes.
# Fixed cron minutes race a chain of variable duration -- hence one sequenced wrapper.
#
# INSTALL ONE SCHEME ONLY. If the fixed-clock lines from the old ledger (05:15 / 05:20 /
# 05:30) are already in crontab, remove them before installing this, or the ingests
# double-run.
#
# UNVERIFIED: every CLI flag below is carried from the defect ledger, not from --help.
# Run `python warning/ingest_spx.py --help` etc. and correct before first install.

set -euo pipefail

cd /Users/atomnguyen/Desktop/ML_Quant_Fund
set -a; . ./.env; set +a

PY=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
LOG=logs/warning_chain.log
DONE="logs/pipeline_ADB.$(date +%F).done"

ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }
step() { echo "[$(ts)] STEP $*" >> "$LOG"; }
fail() { echo "[$(ts)] FAIL $*" >> "$LOG"; exit 1; }

echo "[$(ts)] === warning_chain start ===" >> "$LOG"

# ---------------------------------------------------------------------------
# 1. Gate on the upstream chain's completion marker.
#
#    Requires this as the LAST line of scripts/pipeline_chain_ADB.sh:
#        touch logs/pipeline_ADB.$(date +%F).done
#
#    A done-file is used instead of grepping a log for today's date: a date string
#    can appear in a FAILURE line, so a grep gate can green-light a broken chain.
#    Waits up to 60 min, then gives up rather than running on stale prices.
# ---------------------------------------------------------------------------
step "waiting for $DONE"
for i in {1..60}; do
  [[ -f "$DONE" ]] && break
  sleep 60
done
[[ -f "$DONE" ]] || fail "pipeline_ADB done-file absent after 60 min -- prices.db not refreshed; aborting rather than ingesting stale rows"
step "upstream chain complete"

# ---------------------------------------------------------------------------
# 2. Massive-derived closes and breadth: prices.db -> warning.db
#    Fixes: S4, S5-S8, S14, L4A  (last rows 2026-08-29 = last manual run)
# ---------------------------------------------------------------------------
step "ingest_spx"
$PY warning/ingest_spx.py --db warning.db                             >> "$LOG" 2>&1 \
  || fail "ingest_spx"

step "ingest_spx RSP (not in prices.db -- must come from Massive)"
$PY warning/ingest_spx.py --db warning.db --ticker RSP --from-massive >> "$LOG" 2>&1 \
  || fail "ingest_spx RSP"

step "ingest_breadth"
$PY warning/ingest_breadth.py --db warning.db                         >> "$LOG" 2>&1 \
  || fail "ingest_breadth"

# ---------------------------------------------------------------------------
# 3. Free sources.
#    fred : F2/F3 (VIXCLS, VXVCLS), S2 (BAMLH0A0HYM2), L4B, F10 inputs
#    cboe : COR1M/3M, SKEW, VVIX, VIX9D -- no FRED equivalent -> L4C
#
#    The weekly Sunday `--only fred` job timed out on ALL 18 series including IORB
#    (1.8k rows), so it is a connection stall, not payload size. Fix retry + per-series
#    commit INSIDE fetch_free_history.py; this wrapper only changes when it runs.
#    Until a --series filter exists, the daily pull is the full leg.
# ---------------------------------------------------------------------------
step "fetch_free_history fred,cboe"
$PY scripts/fetch_free_history.py --db warning.db --out data/raw --only fred,cboe \
                                                                      >> "$LOG" 2>&1 \
  || fail "fetch_free_history"

# ---------------------------------------------------------------------------
# 4. UW snapshot BEFORE the driver.
#    Current crontab has the driver at 06:00 and the archiver at 06:30 -- the driver
#    reads yesterday's snapshot. Harmless while no UW-fed signal is live; wrong the
#    moment Phase 5 lands. Sequencing removes the ordering bug permanently.
# ---------------------------------------------------------------------------
step "uw_archiver"
$PY scripts/uw_archiver.py --db warning.db                            >> "$LOG" 2>&1 \
  || fail "uw_archiver"

# ---------------------------------------------------------------------------
# 5. Engine step.
#
#    PRECONDITION -- do not enable this line until B8 has landed. The driver currently
#    labels asof_date with the wall-clock day, so a run at 06:00 VN Tue (= 19:00 ET Mon)
#    writes a point-in-time violation into signal_values on every execution. Until then,
#    run the ingests only and step the driver by hand.
# ---------------------------------------------------------------------------
step "daily_driver"
$PY warning/daily_driver.py --db warning.db                 >> logs/warning_daily.log 2>&1 \
  || fail "daily_driver"

# ---------------------------------------------------------------------------
# 6. Dashboard export (C3) -- uncomment once the script exists.
# ---------------------------------------------------------------------------
# step "export_dashboard_json"
# $PY scripts/export_dashboard_json.py --db warning.db --out exports/cews_latest.json \
#                                                                     >> "$LOG" 2>&1 \
#   || fail "export_dashboard_json"

echo "[$(ts)] === warning_chain OK ===" >> "$LOG"


# ===========================================================================
# CRONTAB (VN local; macOS BSD cron ignores TZ= and uses system localtime).
# Tue-Sat because it processes the prior US session. DST needs a manual edit
# in Mar/Nov, same as the fund's other VN-anchored lines.
#
#   chmod +x scripts/warning_chain.sh
#
# 45 5 * * 2-6 /opt/homebrew/bin/timeout 3600 /Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/warning_chain.sh >> /Users/atomnguyen/Desktop/ML_Quant_Fund/logs/warning_chain_cron.log 2>&1
#
# THEN DELETE the standalone 06:00 daily_driver and 06:30 uw_archiver lines, and any
# 05:15 / 05:20 / 05:30 ingest lines. Two schemes = double ingest.
# ===========================================================================
