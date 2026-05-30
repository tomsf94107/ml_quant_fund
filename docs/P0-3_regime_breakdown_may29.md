# P0-3 — Per-fold regime breakdown (May 29 2026)

Closes P0-3 in three_auc_reconciliation_may27.md.
Source: reports/w2_pit_production_20260523_folds.csv (the 0.44 run). No model re-run.

## Folds mapped to regime (by TEST window)

| Fold | Test window | Regime | test_auc | train_auc | gap | n_train |
|---|---|---|---|---|---|---|
| 1 | Apr 6-17  | tariff selloff              | 0.433 | 0.801 | 0.368 | 2001 |
| 2 | Apr 17-29 | selloff->recovery (transition) | 0.514 | 0.768 | 0.254 | 4021 |
| 3 | Apr 29-May 8 | May bull                 | 0.516 | 0.735 | 0.218 | 7039 |
| 4 | May 8-20  | May bull                    | 0.487 | 0.712 | 0.225 | 9310 |

No fold TESTS March (March is training-only). We only observe selloff, transition, bull.

## Result — regime-edge claim NOT supported

- Within-bull (folds 3,4): mean test_auc = 0.50. Chance.
- Selloff (fold 1): 0.43, below chance — but also smallest train set (2001 rows). Regime vs small-sample is confounded at n=4.
- NO regime shows edge. Best regime = chance.
- The doc's verdict ("within-regime edge exists, regime shifts kill it") is unsupported on this window. Truth on this 3-mo window: OOS ~= chance regardless of regime.

## Real signal in this table: uniform overfit

- train_auc 0.80->0.71, test_auc ~0.43-0.52 every fold. Overfit gap 0.22-0.37 in ALL folds.
- Gap shrinks as train grows (0.37->0.22) but test PLATEAUS at ~0.50. More data cuts overfit; it does not lift OOS above chance on this window.

## Verdict

- n=4 folds / 3 months / one regime transition is too small to conclude edge OR no-edge.
- Does NOT contradict the 5-yr WF-stacks 0.51-0.53 — means that number cannot be validated on 3 months.
- The 0.44-vs-0.53 gap stays unreconciled until tested on equal footing.

## Escalation -> P0-2 (now critical path)

Build outcomes back to 2020, re-run PIT WF on 5-yr window. Question: does within-regime edge appear at scale, or does OOS stay ~0.50 across more regimes? That is the actual reconciliation. Until P0-2, no P1/P3 sizing work is legitimate. P3 roadmap stays BLOCKED.
