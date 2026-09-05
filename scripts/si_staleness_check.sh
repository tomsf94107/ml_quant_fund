#!/bin/bash
# SI staleness. FINRA settles mid-month and month-end and publishes ~8 business
# days later, so the newest settlement is legitimately up to ~26 days old just
# before a publication lands. Older than 28 means one was missed.
#
# This exists because si_cycle.log cannot distinguish failure from a quiet day:
# a VPN outage, an expired OAuth token, a FINRA schema change and a window with
# genuinely nothing new all print the same "[AUTH ERROR]" line. The settlement
# date can tell them apart. FINRA fetches require VPN -- the ISP does SNI
# filtering -- and nothing in si_cycle.sh checks for it.
cd /Users/atomnguyen/Desktop/ML_Quant_Fund || exit 1
MAXD=$(sqlite3 short_interest.db "SELECT MAX(settlement_date) FROM short_interest;")
AGE=$(sqlite3 short_interest.db "SELECT CAST(julianday('now') - julianday(MAX(settlement_date)) AS INT) FROM short_interest;")
if [ "$AGE" -gt 28 ]; then
  echo "!! SI STALE: newest settlement $MAXD is ${AGE}d old (expect <=28)."
  echo "   A publication was missed. Check in this order:"
  echo "     1. VPN up? FINRA fetches fail without it and log [AUTH ERROR]."
  echo "     2. FINRA OAuth token still valid?"
  echo "     3. dateRangeFilters still accepted? (compareFilters returns 400)"
  exit 1
fi
echo "SI fresh: newest settlement $MAXD, ${AGE}d old"
