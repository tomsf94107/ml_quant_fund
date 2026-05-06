# Accuracy Reconciliation — 2026-04-30 (Thursday)

**Run:** `python3 -m accuracy.sink --reconcile --cache`
**Exit code:** 0
**Cache `computed_at` after run:** `2026-04-30T05:29:27` ET

## Summary

- **New outcomes recorded this run:** **0**
- **Cache rows refreshed:** 382 (130 tickers × {1, 3, 5}-day horizons, 90-day window)
- **Reconcile + cache succeeded.** No tickers reported price-fetch errors.

### Why 0 new outcomes?

This is **expected, not a pipeline issue.** All predictions whose outcome dates have already passed have already been reconciled. The only pending predictions are 125 horizon-5 entries from `prediction_date = 2026-04-24`, whose trading-day outcome date is **2026-05-01** (tomorrow). They will reconcile on the next run.

Confirmed by the outcomes table:

| horizon | most recent prediction_date with outcome | total rows |
|---|---|---|
| 1 | 2026-04-29 | 3,612 |
| 3 | 2026-04-27 | 2,266 |
| 5 | 2026-04-23 | 2,016 |

Outcomes whose **outcome_date = 2026-04-30** (today): **377 rows** — already in the table. An earlier scheduled run today (around 05:29 ET) caught them.

## Accuracy Metrics — 90-day rolling window

### Aggregate by horizon

| horizon | tickers | avg accuracy | avg ROC-AUC | total predictions |
|---|---|---|---|---|
| 1 | 130 | 0.4924 | 0.5027 | 3,608 |
| 3 | 126 | 0.5026 | 0.4699 | 2,266 |
| 5 | 126 | 0.5060 | 0.4719 | 2,016 |

Horizon-1 ROC-AUC sits right at chance (0.50). Horizons 3 and 5 are slightly below chance on the 90-day window — worth flagging but not unusual for a recent macro-regime shift.

### Top 10 tickers (avg ROC-AUC across horizons, n ≥ 10 each)

| ticker | avg accuracy | avg ROC-AUC | total n |
|---|---|---|---|
| SMCI | 0.6730 | 0.7375 | 79 |
| AMD | 0.7728 | 0.7202 | 79 |
| QQQ | 0.6497 | 0.7198 | 67 |
| ROKU | 0.6180 | 0.6826 | 72 |
| COIN | 0.6566 | 0.6730 | 67 |
| PYPL | 0.5047 | 0.6430 | 79 |
| PLTR | 0.6644 | 0.6420 | 79 |
| ADSK | 0.6586 | 0.6352 | 72 |
| NET | 0.6022 | 0.6331 | 72 |
| TEAM | 0.5726 | 0.5999 | 72 |

### Bottom 10 tickers (worst avg ROC-AUC, n ≥ 10 each)

| ticker | avg accuracy | avg ROC-AUC | total n |
|---|---|---|---|
| ONTO | 0.2696 | 0.1987 | 72 |
| NVMI | 0.4504 | 0.2088 | 72 |
| RZLV | 0.3000 | 0.2600 | 10 |
| INTC | 0.4575 | 0.2782 | 72 |
| ROST | 0.3957 | 0.2891 | 72 |
| ASAN | 0.3869 | 0.2961 | 72 |
| LLY | 0.3151 | 0.3107 | 72 |
| XLF | 0.4769 | 0.3107 | 67 |
| SLV | 0.4863 | 0.3171 | 79 |
| AAPL | 0.5646 | 0.3255 | 79 |

ONTO, NVMI, INTC, ROST, ASAN, LLY all sub-0.31 AUC — consider de-weighting in tactical sizing or routing them through the recession overlay.

## Data-thin / problematic rows

- **WCC** has only 1 prediction at every horizon → ROC-AUC is `NaN`. Cause: ticker added very recently. Will resolve as more predictions accumulate.
- **EME** and **MRVL** at horizon 5: 4 predictions but `actual_up` is constant (all up or all down) → ROC-AUC undefined.
- 24 other tickers have only n=4 at horizon 5 (recently added). Metrics are noisy until n grows.

No ticker reported a price-fetch failure during reconcile.

## Notes for the operator

- The full per-ticker × per-horizon table (382 rows) was printed by `update_accuracy_cache` and is now persisted in `accuracy.db.accuracy_cache`. The dashboard should pick it up automatically.
- The next reconcile run will pick up the 125 horizon-5 predictions from 2026-04-24 once 2026-05-01 closes.
- A handful of ROC-AUC = 1.00 / 0.00 entries at horizon=5 are an n=4 artifact, not a model success/failure.

---

**Run executed by:** scheduled task `ml-accuracy-reconcile`
**Database:** `accuracy.db` (33.1 MB)
