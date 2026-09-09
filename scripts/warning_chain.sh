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

# ---------------------------------------------------------------------------
# 2b. Sector ETFs. S7 reads XLP/XLU/XLV against SPY; S8 ranks all eleven.
#
#     These were NOT in the original wrapper -- ingest_spx defaults to SPY
#     alone -- so they stopped advancing when the manual backfill did and sat
#     at 2026-08-28 while SPY_CLOSE moved to 09-08. S7 and S8 then computed
#     relative strength between an 11-day-old numerator and a current
#     denominator, both reading G with stale=False because stale_days of 7 is
#     inside their limit. That is worse than a stale reading: it is a
#     comparison across mismatched dates. Same class as the L4C failure, where
#     a signal read G off week-old correlation data.
# ---------------------------------------------------------------------------
step "ingest_spx sectors"
$PY warning/ingest_spx.py --db warning.db --table raw_bars \
    --ticker XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY                 >> "$LOG" 2>&1 \
  || fail "ingest_spx sectors"

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
$PY scripts/fetch_free_history.py --db warning.db --out data/raw_live --only fred,cboe \
                                                                      >> "$LOG" 2>&1 \
  || fail "fetch_free_history"

# ---------------------------------------------------------------------------
# 3b. Parse the Cboe CSVs into data_vintages.
#
#     Fetch writes to data/raw_live, which is gitignored. data/raw/cboe stays
#     as the COMMITTED point-in-time archive from 2026-08-28 and is not
#     overwritten by cron. warning.db is gitignored and its only backup is on
#     this same disk, so those 16 files are the only off-machine copy of the
#     Cboe history. Today's Shiller lesson: the Yale mirror is frozen at
#     2023-08 and still serves a valid file, so a CDN that works today is not
#     insurance. The commit that added them said re-fetch needs a VPN because
#     the ISP SNI-filters cboe.com -- both halves are now false, cdn.cboe.com
#     resolves normally and the wrapper pulls all 16 without one, but the
#     archive is worth keeping regardless.
#
#     fetch_free_history --only cboe DOWNLOADS the files and stops; parse_cboe.py
#     is a separate CLI and nothing invoked it. That is why every CBOE_* series
#     sat at 2026-08-27 while COR1M_History.csv on disk was current. L4C was not
#     reading NA -- it was reading G off week-old correlation data, which is
#     worse. Parsing took L4 coverage from 40% to 60%.
# ---------------------------------------------------------------------------
step "parse_cboe"
$PY warning/parse_cboe.py --dir data/raw_live/cboe --db warning.db          >> "$LOG" 2>&1 \
  || fail "parse_cboe"

# ---------------------------------------------------------------------------
# 3c. Shiller CAPE -> data_vintages (S13).
#
#     Monthly series, so this is a no-op most days; running it daily costs
#     nothing and means a new release lands the day it appears. The xls itself
#     is NOT downloaded here: the live file is a GoDaddy CDN blob linked from
#     shillerdata.com with a ?ver= parameter that will change, and the Yale
#     mirror at econ.yale.edu is FROZEN at 2023-08 while still downloading and
#     parsing cleanly. ingest_shiller.py refuses a file whose newest complete
#     month is over 120 days old rather than ingesting a dead mirror.
# ---------------------------------------------------------------------------
step "ingest_shiller"
$PY warning/ingest_shiller.py --xls data/raw/shiller/ie_data.xls --db warning.db \
                                                                      >> "$LOG" 2>&1 \
  || fail "ingest_shiller"

# ---------------------------------------------------------------------------
# 3d. FINRA margin statistics -> data_vintages (S10, and L4D when built).
#
#     Monthly, published in the third week of the month FOLLOWING the reference
#     month, so this is a no-op most days. Unlike Shiller, the URL is stable
#     with no version parameter, and reachable from Vietnam without a VPN as of
#     2026-09-08 -- contrary to a note elsewhere in this crontab. The download
#     is still not automated here: the ingest refuses a file whose newest
#     observation is over 75 days old, and a silent re-download would defeat
#     that check.
# ---------------------------------------------------------------------------
step "ingest_finra_margin"
$PY warning/ingest_finra_margin.py --xlsx data/raw/finra/margin-statistics.xlsx \
                                   --db warning.db                  >> "$LOG" 2>&1 \
  || fail "ingest_finra_margin"

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
step "daily_driver"
$PY warning/daily_driver.py --db warning.db                 >> logs/warning_daily.log 2>&1 \
  || fail "daily_driver"

# ---------------------------------------------------------------------------
# 6. Dashboard export (C3). The script already exists and already runs on its own
#    cron line at 06:05 VN -- the ledger's claim that this was manual was wrong.
#    Enable here in the SAME edit that deletes that line, and only after the driver
#    step above is live: exporting before the engine steps publishes yesterday.
# ---------------------------------------------------------------------------
step "cews_engine_json"
$PY exports/cews_engine_json.py --db warning.db --out exports/engine-data.json \
                                                                    >> "$LOG" 2>&1 \
  || fail "cews_engine_json"

# ---------------------------------------------------------------------------
# 6b. Push the export to Drive, where the Crash Odds Desk artifact reads it.
#
#     Was a standalone cron line at 07:07 VN -- 81 minutes after the wrapper
#     writes the file, on a fixed clock. It worked, but if the wrapper ever ran
#     long the upload would carry the PREVIOUS day's file, and the artifact
#     would show stale data while reporting a fresh fetch. Sequenced here so
#     the upload cannot precede the write.
# ---------------------------------------------------------------------------
step "rclone -> gdrive:CEWS"
/opt/homebrew/bin/rclone copy exports/engine-data.json gdrive:CEWS/  >> "$LOG" 2>&1 \
  || fail "rclone copy"

# ---------------------------------------------------------------------------
# 7. Weekly core dump to iCloud, Saturdays only.
#
#    warning.db is gitignored and untracked. The full file is 266MB but 70% of
#    that is Ken French portfolio data that re-downloads freely; the core
#    tables gzip to about 10MB. What is genuinely hard to reacquire is small:
#    the Shiller CAPE history, whose live URL is a CDN blob with a version
#    parameter and whose Yale mirror is frozen at 2023-08, the Cboe archive,
#    and the engine's own state.
#
#    Keeps the four most recent dumps. VACUUM INTO would give a consistent
#    binary copy but at 266MB; the SQL dump is smaller and restores with
#    gzcat FILE | sqlite3 new.db.
# ---------------------------------------------------------------------------
if [[ "$(date +%u)" == "6" ]]; then
  BK="$HOME/Library/Mobile Documents/com~apple~CloudDocs/ml_quant_backups"
  mkdir -p "$BK"
  step "weekly core dump"
  /usr/bin/sqlite3 warning.db ".dump data_vintages signal_values composite_scores alerts schema_meta runs" \
    | gzip > "$BK/warning_core_$(date +%F).sql.gz" 2>> "$LOG" \
    || fail "core dump"
  ls -1t "$BK"/warning_core_*.sql.gz | tail -n +5 | while read -r f; do rm -f "$f"; done
fi

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
