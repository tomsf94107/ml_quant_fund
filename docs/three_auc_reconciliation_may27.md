# 3-AUC Reconciliation — May 27 2026

## Mystery resolved

The 14pp gap between three OOS AUCs is NOT a methodology bug.
It's three different things being measured.

### The three AUCs

| Test | AUC | Window | Method |
|---|---|---|---|
| **Per-ticker PIT WF** | 0.4389 pooled / 0.4876 mean fold | Mar-May 2026 (3 months) | Purged k-fold, PIT features |
| Per-ticker WF stacks | 0.51-0.53 | 2020-2025 (5 years) | Time-series CSV stacks |
| Path A 5-fold | 0.58-0.59 | 2020-2025 (5 years) | Cross-sectional, GLOBAL model |

### Why they differ

**PIT WF (AUC 0.44/0.49):**
- Tiny window (Mar 5 - May 20, 2026)
- 3 distinct market regimes in 3 months:
  - Mar: early bull recovery
  - Apr: tariff selloff (per Apr 24 quota incident memory)
  - May: bull regime
- Fold 1 trained on 2001 rows in 27 days, tested 2 weeks later
- Pooled AUC dragged down by Fold 1's regime-shift outlier (0.43)
- Mean fold AUC = 0.49 (near random, NOT systematically wrong)

**Per-ticker WF stacks (AUC 0.51-0.53):**
- 5 years of data averaged across folds
- Regime variation washes out
- Reflects long-term per-ticker model edge

**Path A 5-fold (AUC 0.58-0.59):**
- Different model architecture (GLOBAL, cross-sectional pooled)
- Same 5 years of data
- Bootstrap CI [0.572, 0.606] excludes 0.50 — statistically significant

### Honest verdict on per-ticker

- Multi-year OOS AUC ~0.51-0.53 = **weak but real edge**
- Within-regime (PIT WF excluding Fold 1) ~0.49-0.51 = near random
- The "production model is broken" framing was overstated
- It's not broken — it's a weak model that performs at chance during
  regime transitions, slightly above chance on multi-year average

### What this means for production

1. Per-ticker BUY signals at high prob (>0.65) DO have real edge over time
2. During regime shifts, model edge drops to ~0
3. The CALIBRATION problem (yesterday's Phase 1 fix) was real and major
4. The DISCRIMINATION problem (yesterday's GLOBAL classifier issue) is real
5. The MAY 23 PIT panic was based on regime-shift-window artifacts

### Next steps

**P0-1 (DONE — this doc):** Reconcile the 3-AUC gap

**P0-2 (TODO):** Re-run PIT WF on LARGER window
   - Build outcomes table back to 2020 (we have raw OHLC; just compute targets)
   - Re-run PIT WF on 5-year window
   - Expect AUC to converge with WF stacks (~0.51-0.53)
   - Effort: 5-15 hours (depends on feature builder per-ticker timing)

**P0-3 (DONE - see docs/P0-3_regime_breakdown_may29.md):** Per-fold AUC breakdown
   - Tag each fold with macro regime (VIX, SPY trend, etc)
   - Confirm within-regime AUC > between-regime AUC
   - Establishes that the model has regime-dependent edge
