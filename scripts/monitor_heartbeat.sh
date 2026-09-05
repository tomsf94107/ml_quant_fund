#!/bin/bash
# monitor_heartbeat.sh — do the monitors themselves still run?
#
# WHY THIS EXISTS
#   scripts/feed_freshness_check.py's own header records the failure this
#   guards against, in its own words:
#
#     "prediction_features / portfolio_returns_ab had date_col='date'. Neither
#      table HAS a 'date' column ... so both ERRORed every run and were
#      effectively UNWATCHED. An entry that ERRORs every run is not 'watched';
#      it is unwatched with extra steps."
#
#   The monitor built to catch stale feeds had itself gone blind on two entries.
#   Nothing noticed, because nothing watches the watchers.
#
#   On 2026-09-05 the same shape appeared four more times: eightk_items frozen
#   at 2026-05-27 and sentiment_scores at 2026-05-28, both three months
#   unnoticed; vix_term_structure pinned to the literal 1.0 for ten weeks; and
#   24,741 PCT7 shadow predictions written and never scored for fourteen weeks.
#   Every one had the same signature -- it runs, writes nothing useful, exits 0,
#   and nobody is told.
#
#   There are now eleven monitors writing to eleven logs. This checks that each
#   log is RECENT and NON-EMPTY. It cannot tell whether a monitor's verdict is
#   correct; it can tell whether the monitor is producing one at all, which is
#   the failure mode none of the eleven can see in itself.
#
# EXIT CODES
#   0  every monitor has written within its budget
#   1  at least one log is stale, empty, or missing
set -uo pipefail
cd /Users/atomnguyen/Desktop/ML_Quant_Fund || exit 1
LOGS=logs
NOW=$(date +%s)
FAIL=0

# name|log|max age in HOURS. Budgets follow each monitor's own cadence plus
# roughly one period of slack, so a single missed run does not cry wolf.
CHECKS=(
  "feature health|feature_health.log|192"
  "fund_ep verdict|fund_ep_verdict.log|192"
  "SI period split|si_period_split.log|840"
  "SI staleness|si_staleness.log|192"
  "SI cycle|si_cycle.log|48"
  "break audit|break_audit.log|192"
  "pipeline audit|pipeline_audit.log|192"
  "h40 shadow|h40_shadow.log|48"
  "feed freshness|feed_freshness.log|48"
  "8-K weekly|eightk_weekly.log|192"
  "dark-pool extra|dp_extra_daily.log|48"
)

printf "%-20s %-28s %8s %10s  %s\n" MONITOR LOG AGE_H SIZE STATUS
for row in "${CHECKS[@]}"; do
  IFS='|' read -r name log budget <<< "$row"
  path="$LOGS/$log"
  if [ ! -f "$path" ]; then
    printf "%-20s %-28s %8s %10s  %s\n" "$name" "$log" "-" "-" "MISSING"
    FAIL=1
    continue
  fi
  mt=$(stat -f %m "$path" 2>/dev/null || echo 0)
  age=$(( (NOW - mt) / 3600 ))
  sz=$(stat -f %z "$path" 2>/dev/null || echo 0)
  st="ok"
  if [ "$sz" -eq 0 ]; then st="EMPTY"; FAIL=1
  elif [ "$age" -gt "$budget" ]; then st="STALE (budget ${budget}h)"; FAIL=1
  fi
  printf "%-20s %-28s %8s %10s  %s\n" "$name" "$log" "$age" "$sz" "$st"
done

echo
if [ "$FAIL" -eq 1 ]; then
  echo "!! One or more monitors are not producing output."
  echo "   A monitor that errors every run is not watching anything."
  echo "   Check: crontab -l | grep <script>, then run it by hand and read the"
  echo "   error. A silent monitor is worse than no monitor, because it buys"
  echo "   confidence it has not earned."
  exit 1
fi
echo "All monitors have written within budget."
echo
echo "NOTE: this checks that logs are recent and non-empty. It does NOT check"
echo "that a verdict is correct. A monitor can run, write, and still be blind --"
echo "feed_freshness_check watched two tables by a column name that does not"
echo "exist in either, and errored on both every run while looking healthy."
