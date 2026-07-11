#!/bin/bash
cd ~/Desktop/ML_Quant_Fund
echo "===================================================================="
echo " GLOBAL vs PER-TICKER — accuracy from real outcomes"
echo "===================================================================="
echo ""
echo "=== [1a] PER-TICKER accuracy by horizon (all-time) ==="
sqlite3 -column -header accuracy.db "
SELECT o.horizon AS h, COUNT(*) AS n,
  ROUND(100.0*SUM(((p.prob_up>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS perticker_pct
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.prob_up IS NOT NULL GROUP BY o.horizon;"
echo ""
echo "=== [1b] GLOBAL ENSEMBLE accuracy (its own non-null rows) ==="
sqlite3 -column -header accuracy.db "
SELECT o.horizon AS h, COUNT(*) AS n,
  ROUND(100.0*SUM(((p.prob_up_global>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS global_pct
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.prob_up_global IS NOT NULL GROUP BY o.horizon;"
echo ""
echo "=== [2] LAST 7 DAYS — per-ticker vs global ensemble (fresh era) ==="
sqlite3 -column -header accuracy.db "
SELECT o.horizon AS h, COUNT(*) AS n,
  ROUND(100.0*SUM(((p.prob_up>=0.5)=(o.actual_up=1)))/COUNT(*),1)        AS perticker_pct,
  ROUND(100.0*SUM(((p.prob_up_global>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS global_pct
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.prob_up_global IS NOT NULL AND o.prediction_date >= date('now','-7 days')
GROUP BY o.horizon;"
echo ""
echo "=== [2b] LAST 7 DAYS — ranker (its own filter) ==="
sqlite3 -column -header accuracy.db "
SELECT o.horizon AS h, COUNT(*) AS n,
  ROUND(100.0*SUM(((p.prob_up_global_ranker>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS ranker_pct
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.prob_up_global_ranker IS NOT NULL AND o.prediction_date >= date('now','-7 days')
GROUP BY o.horizon;"
echo ""
echo "=== [3] STALE split — GLOBAL ensemble before/after freeze (2026-05-26) ==="
sqlite3 -column -header accuracy.db "
SELECT o.horizon AS h,
  CASE WHEN o.prediction_date < '2026-05-26' THEN 'pre-freeze' ELSE 'post(stale)' END AS era,
  COUNT(*) AS n,
  ROUND(100.0*SUM(((p.prob_up_global>=0.5)=(o.actual_up=1)))/COUNT(*),1) AS global_pct
FROM predictions p JOIN outcomes o
  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
WHERE p.prob_up_global IS NOT NULL
GROUP BY o.horizon, era ORDER BY o.horizon, era;"
echo ""
echo "=== [4] TODAY's biggest per-ticker vs GLOBAL ensemble disagreements (h=1) ==="
sqlite3 -column -header accuracy.db "
SELECT ticker,
  ROUND(prob_up,3) AS perticker, ROUND(prob_up_global,3) AS global,
  ROUND(prob_up - prob_up_global,3) AS delta,
  CASE WHEN (prob_up>=0.5)=(prob_up_global>=0.5) THEN 'agree' ELSE 'DISAGREE' END AS dir
FROM predictions
WHERE prediction_date=(SELECT MAX(prediction_date) FROM predictions)
  AND prob_up_global IS NOT NULL
ORDER BY ABS(prob_up - prob_up_global) DESC LIMIT 15;"
echo ""
echo "===================================================================="
echo " NOTE: ranker has fewer rows (added recently) — its 54% is promising"
echo " but small-sample. Ensemble is weak (46-49%). Watch ranker as it grows."
echo "===================================================================="
