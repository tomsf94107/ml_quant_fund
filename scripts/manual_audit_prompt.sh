#!/bin/bash
# manual_audit_prompt.sh — monthly prompt to do what no script can.
#
# The automated monitors catch KNOWN failure shapes. Everything found on
# 2026-09-05 was found by a human running an audit that had never been run,
# prompted by reading someone else's research methodology. Monitors catch
# repeats; they do not catch first instances.
#
# This prints a checklist. It checks nothing itself, deliberately -- a script
# that pretended to do this would buy confidence it has not earned.
cd /Users/atomnguyen/Desktop/ML_Quant_Fund || exit 1
echo "================================================================"
echo "  MANUAL AUDIT DUE — $(date +%Y-%m-%d)"
echo "================================================================"
echo
echo "  1. READ THE COMMIT LOG since the last audit."
echo "     Distinguishes a deliberate change from a silent death. In April,"
echo "     macd_signal and es_overnight were culled on purpose (commit"
echo "     8da49533, collinearity measured). In June, vix_term_structure died"
echo "     by accident. Both look identical in the feature tables."
echo "       git log --oneline --since='1 month ago' -- features/ models/"
echo
echo "  2. TRACE ONE FEATURE END TO END."
echo "     source table -> builder assignment -> OUTPUT_COLUMNS ->"
echo "     FEATURE_COLUMNS -> importance history. A break at any step is"
echo "     silent. df = df[OUTPUT_COLUMNS] drops unlisted columns with no"
echo "     error, which cost six wrong diagnoses on 2026-09-05."
echo
echo "  3. RE-MEASURE ONE VALIDATED SIGNAL."
echo "     The SI brick's recorded IC drifted 28% between July and September"
echo "     with nobody checking -- recorded -0.054, measured -0.037. Not"
echo "     decay: the record was overstated and never re-derived."
echo
echo "  4. READ ONE EXTERNAL METHODOLOGY."
echo "     The break audit and the leave-one-out test both came from reading"
echo "     a research report, not from introspection. Between them they found"
echo "     four frozen feeds and reversed four conclusions reached the same"
echo "     day from importance rankings alone."
echo
echo "  4b. QUARTERLY: re-run the dark-pool feature A/B."
echo "     analysis/darkpool_feature_ab.py --seeds 3 --tickers 80"
echo "     dp_volume_share was non-NaN on 1% of a 2016-start panel when first"
echo "     tested (2026-09-07): the feed begins 2026-03-19, so the A/B could"
echo "     not have found a positive even if one existed. Coverage grows ~1.4pp"
echo "     a month. Re-run each quarter; the moment the diff turns positive"
echo "     with seed agreement, the columns have started earning their place."
echo "     The Boulton EXCLUSION FILTER is separate and already works: -7.1pp"
echo "     in the high-DTC/high-share cell measured on populated dates only."
echo
echo "  5. CHECK WHAT THE MONITORS CANNOT SEE."
echo "     docs/coverage_gaps.md lists the known holes in cost order."
echo
echo "  Log the findings. An audit whose result is not written down has to be"
echo "  repeated from scratch next time."
