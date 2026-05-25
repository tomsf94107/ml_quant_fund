# Monitoring Phase epsilon — what to check Friday May 29

Phase epsilon shipped May 25 2026 logs prob_pct7 alongside production.
First h=5 outcomes arrive Friday May 29 (5 business days after Monday).

## Checks Friday after market close

### 1. Outcomes exist

Run: sqlite3 accuracy.db "SELECT COUNT(*) FROM outcomes WHERE prediction_date='2026-05-25';"

Expected: ~127 rows (one per h=5 prediction).

### 2. Run the monitor script

Run: PYTHONPATH=. python scripts/monitor_pct7_ab.py

Output:
- prob_up hit rate (baseline)
- prob_pct7 hit rate at 0.20 threshold
- Calibration buckets

### 3. What good looks like (success criteria)

| Metric | Target | Meaning |
|---|---|---|
| prob_pct7 base rate | ~13% | Matches training distribution |
| prob_pct7 BUY precision (>0.20) | >20% | Better than random for high-conviction |
| Calibration bucket [0.20, 0.30) | actual ~25% | Roughly calibrated |
| Calibration bucket [0.30, 1.00] | actual >30% | Tail-end well-calibrated |

### 4. What bad looks like

- prob_pct7 BUY precision < 13% (worse than base rate) -> model is anti-signal
- Calibration buckets show no monotonic relationship -> uncalibrated
- All prob_pct7 cluster in 0.05-0.15 (no separation) -> model not differentiating

### 5. Cross-comparison: prob_up vs prob_pct7

For tickers where per-ticker said BUY but prob_pct7 was low, did they
underperform? That validates the overlay filter hypothesis (Phase 2 H).

SQL: SELECT COUNT(*) bad_BUYs_caught_by_overlay FROM predictions p
JOIN outcomes o USING(ticker,prediction_date,horizon)
WHERE p.signal='BUY' AND p.prob_pct7 < 0.10 AND o.actual_return < 0
  AND p.prediction_date='2026-05-25' AND p.horizon=5;

If most BUYs that had prob_pct7 < 0.10 failed -> overlay is useful.

## Schedule

| Date | Action |
|---|---|
| Fri May 29 | First outcomes batch (127 rows). Run monitor. |
| Mon Jun 1 | Outcomes for Tue May 26 predictions arrive. Run monitor again. |
| Thru Fri Jun 5 | One week of A/B data. Decision point. |
| Fri Jun 5 EOD | Review. If signal real, proceed to Phase 2 A. If not, root-cause. |

## Decision tree after 1 week

- IF prob_pct7 BUY precision > prob_up BUY precision:
    Phase epsilon confirmed -> start Phase 2 A (A8 prob as feature in main model)
- ELIF prob_pct7 BUY precision == prob_up BUY precision:
    Equivalent -> keep shadow-logging but no urgency
- ELSE:
    prob_pct7 underperforms -> root-cause, possibly retrain or change target
