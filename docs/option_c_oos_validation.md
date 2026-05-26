# Option C OOS Validation — May 26, 2026

LightGBMRanker on 137,468 train rows (2020-01-30 to 2024-12-31), tested on
42,778 rows (2025-01-02 to 2026-05-15).

## Results

| Predicted Quintile | Avg 5-day fwd return | Hit rate (>0) | N |
|---|---|---|---|
| Q1 (bottom)        | +0.287% | 54.0% | 8,508 |
| Q2                 | +0.408% | 53.9% | 8,575 |
| Q3                 | +0.502% | 51.9% | 8,545 |
| Q4                 | +0.821% | 53.1% | 8,575 |
| Q5 (top)           | +1.847% | 53.4% | 8,575 |

## Key findings

- **Average return is monotonic Q1 → Q5**: 0.29 → 1.85 (6.4x ratio)
- **Spread Q5-Q1 = +1.56pp over 5 days**
- Hit rate flat ~53% across quintiles (not great at direction)
- Ranker captures relative outperformance, not absolute direction

## Comparison to per-ticker classifier

| Metric | Per-ticker classifier | GLOBAL ranker |
|---|---|---|
| Output type | binary prob | continuous rank |
| Variance | high (range 0.1-0.9) | high (-1.69 to +1.35) |
| OOS hit rate | 52-64% per bucket | 51-54% per quintile |
| Q5-Q1 spread | unmeasurable | +1.56pp |

The ranker is better at relative ranking, per-ticker is better at direction.
Combining them via meta-labeling could be powerful:
  - per-ticker says BUY at prob >= 0.65
  - AND ranker puts in Q4 or Q5
  → only take both-agree BUYs

## Next steps

1. Integrate ranker into daily_runner Path A logging
2. Save trained ranker as GLOBAL_ranker_5d.joblib
3. Compute rank-aware BUY filter (Phase 2 H style)
4. Measure: do "per-ticker BUY AND ranker Q4+" outperform plain per-ticker BUY?
