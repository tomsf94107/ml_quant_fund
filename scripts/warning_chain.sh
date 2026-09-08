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
# Flags verified against --help 2026-09-08. ingest_spx --table defaults to raw_bars
# (unadjusted); daily_prices would give a total-return series that reaches new highs
# early and biases a 52-week-high test. Passed explicitly so a default change is loud.

set -euo pipefail

cd /Users/atomnguyen/Desktop/ML_Quant_Fund
set -a; . ./.env; set +a

PY=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
LOG=logs/warning_chain.log
DONE="logs/pipeline_A.$(date +%F).done"   # A writes prices.db and ends ~04:55; the full A->D->B chain runs to ~08:24

ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }
step() { echo "[$(ts)] STEP $*" >> "$LOG"; }
fail() { echo "[$(ts)] FAIL $*" >> "$LOG"; exit 1; }

echo "[$(ts)] === warning_chain start ===" >> "$LOG"

# ---------------------------------------------------------------------------
# 1. Gate on the upstream chain's completion marker.
#
#    Requires this as the LAST line of scripts/pipeline_chain_ADB.sh:
#        touch logs/pipeline_A.$(date +%F).done
#
#    A done-file is used instead of grepping a log for today's date: a date string
#    can appear in a FAILURE line, so a grep gate can green-light a broken chain.
#    Waits up to 50 min. Pipeline A ended 04:55 on its last run (04:00 start), so a
#    05:45 launch normally finds the marker immediately. The cap sits under the cron
#    timeout of 3600s -- a longer wait would be killed mid-sleep and never log the
#    reason it gave up.
# ---------------------------------------------------------------------------
step "waiting for $DONE"
for i in {1..50}; do
  [[ -f "$DONE" ]] && break
  sleep 60
done
[[ -f "$DONE" ]] || fail "pipeline_A done-file absent after 50 min -- prices.db not refreshed; aborting rather than ingesting stale rows"
step "upstream chain complete"

# ---------------------------------------------------------------------------
# 2. Massive-derived closes and breadth: prices.db -> warning.db
#    Fixes: S4, S5-S8, S14, L4A. These two scripts have never been scheduled; every
#    row they wrote came from a manual run, which is why SPY_CLOSE sat at 2026-08-29
#    until a hand backfill on 2026-09-07.
# ---------------------------------------------------------------------------
step "ingest_spx"
$PY warning/ingest_spx.py --db warning.db --table raw_bars                            >> "$LOG" 2>&1 \
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
#    The weekly Sunday job failed on all 18 series for ten weeks. Cause was the
#    User-Agent, not the network: FRED sits behind Akamai, which stalls unknown UAs
#    until the socket times out. Fixed in commit cf796020; all 18 now land in 33.9s.
#    Until a --series filter exists, the daily pull is the full leg -- cheap enough.
# ---------------------------------------------------------------------------
step "fetch_free_history fred,cboe"
$PY scripts/fetch_free_history.py --db warning.db --out data/raw --only fred,cboe \
                                                                      >> "$LOG" 2>&1 \
  || fail "fetch_free_history"

# ---------------------------------------------------------------------------
# 3b. Parse the Cboe CSVs into data_vintages.
#
#     fetch_free_history --only cboe DOWNLOADS the files and stops; parse_cboe.py
#     is a separate CLI and nothing invoked it. That is why every CBOE_* series
#     sat at 2026-08-27 while COR1M_History.csv on disk was current. L4C was not
#     reading NA -- it was reading G off week-old correlation data, which is
#     worse. Parsing took L4 coverage from 40% to 60%.
# ---------------------------------------------------------------------------
step "parse_cboe"
$PY warning/parse_cboe.py --dir data/raw/cboe --db warning.db          >> "$LOG" 2>&1 \
  || fail "parse_cboe"

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
#    B8 landed in commit aab085b7: asof_date now derives from
#    utils.market_calendar.last_completed_session(), FEED_STALE exits 2 rather than
#    labelling a session whose SPY_CLOSE is not ingested, and unfilled gaps exit 3
#    with re-step commands (D24).
#
#    STAGED: commented for the first night so this run exercises only the gate and the
#    ingests, which are additive and idempotent. The standalone 06:00 cron still steps
#    the driver meanwhile. Uncomment once logs/warning_chain.log shows a clean pass,
#    and delete the 06:00 line in the same edit.
#
#    Exit 3 (unfilled gap) is a REFUSAL, not a failure -- but `fail` treats any
#    non-zero status as fatal and stops the chain. That is the intended behaviour:
#    a gap needs a human.
# ---------------------------------------------------------------------------
# step "daily_driver"
# $PY warning/daily_driver.py --db warning.db               >> logs/warning_daily.log 2>&1 \
#   || fail "daily_driver"

# ---------------------------------------------------------------------------
# 6. Dashboard export (C3). The script already exists and already runs on its own
#    cron line at 06:05 VN -- the ledger's claim that this was manual was wrong.
#    Enable here in the SAME edit that deletes that line, and only after the driver
#    step above is live: exporting before the engine steps publishes yesterday.
# ---------------------------------------------------------------------------
# step "cews_engine_json"
# $PY exports/cews_engine_json.py --db warning.db --out exports/engine-data.json \
#                                                                     >> "$LOG" 2>&1 \
#   || fail "cews_engine_json"

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
# On install, DELETE the standalone 06:30 uw_archiver line -- step 4 replaces it.
# KEEP the 06:00 daily_driver and 06:05 cews_engine_json lines until steps 5 and 6 are
# uncommented, then delete both in that same edit. Two schemes = double ingest.
# There are no 05:15 / 05:20 / 05:30 lines; that scheme was never installed.
# ===========================================================================
