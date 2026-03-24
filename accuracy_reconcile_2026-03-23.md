# ML Quant Fund — Daily Accuracy Reconciliation
**Run date:** 2026-03-23 (Monday)
**Status:** ⚠️ Partial — `yfinance` unavailable in sandbox; cache read from DB directly

---

## Execution Summary

The reconciliation script (`python3 -m accuracy.sink --reconcile --cache`) failed at the `reconcile_outcomes()` step with:

```
ModuleNotFoundError: No module named 'yfinance'
```

The sandbox VM does not have internet access (proxy returns 403), so `yfinance` could not be installed and no new outcomes were fetched. **0 new outcomes were recorded in this run.**

---

## Database State (as of this run)

| Table | Rows |
|---|---|
| predictions | 789 |
| outcomes | 591 |
| accuracy_cache | 96 (96 tickers, all horizon=1d) |
| intraday_predictions | 198 |
| intraday_outcomes | 198 |

**Outcomes coverage:** 2026-03-05 → 2026-03-20 (most recent outcome date)
**Accuracy cache last computed:** 2026-03-22

---

## Pending Predictions (198 total — no outcome recorded)

| Prediction Date | Count | Notes |
|---|---|---|
| 2026-03-06 | 1 (NVO) | ⚠️ Overdue — outcome date ~Mar 7, well past |
| 2026-03-09 | 1 (RZLV) | ⚠️ Overdue — outcome date ~Mar 10, well past |
| 2026-03-18 | 1 (GM) | ⚠️ Overdue — outcome date ~Mar 19, well past |
| 2026-03-20 | 94 | ⚠️ Outcome date = Mar 21 (Fri) — should have been reconciled |
| 2026-03-23 | 101 | ✅ Expected — today's predictions, outcome date = Mar 24 |

**Root cause:** The Mar 20 batch (94 predictions) and the 3 older outliers should have been reconciled by the Mar 21–22 scheduled runs but were not — likely due to the same `yfinance`/network issue in the sandbox.

---

## Current Accuracy Cache (96 tickers, horizon=1d, 90-day window)

> Cache last computed: **2026-03-22**. Reflects outcomes through 2026-03-20.

| Ticker | Accuracy | ROC-AUC | Brier | N Preds |
|---|---|---|---|---|
| AAPL | 63.6% | 0.429 | 0.2653 | 11 |
| ABNB | 50.0% | 0.333 | 0.2742 | 4 |
| ABT | 75.0% | 0.750 | 0.2425 | 4 |
| ADSK | 100.0% | 1.000 | 0.2020 | 4 |
| AI | 50.0% | 1.000 | 0.2005 | 4 |
| ALK | 50.0% | 0.250 | 0.2590 | 4 |
| AMD | 54.5% | 0.667 | 0.2431 | 11 |
| AMPX | 25.0% | 0.333 | 0.3009 | 4 |
| AMZN | 50.0% | 0.333 | 0.2516 | 4 |
| APLD | 75.0% | 1.000 | 0.2222 | 4 |
| ARM | 0.0% | N/A | 0.3659 | 4 |
| ASAN | 50.0% | 0.750 | 0.2348 | 4 |
| ASTS | 25.0% | 0.333 | 0.2746 | 4 |
| AXP | 45.5% | 0.517 | 0.2726 | 11 |
| AZN | 50.0% | 0.250 | 0.2575 | 4 |
| BA | 50.0% | 0.333 | 0.2351 | 4 |
| BETR | 50.0% | 0.250 | 0.2691 | 4 |
| BNED | 50.0% | 0.500 | 0.2754 | 4 |
| BRKR | 75.0% | 0.500 | 0.2101 | 4 |
| BSX | 63.6% | 0.800 | 0.2306 | 11 |
| CAVA | 75.0% | 0.667 | 0.2531 | 4 |
| CI | 75.0% | 1.000 | 0.2198 | 4 |
| CNC | 45.5% | 0.433 | 0.2585 | 11 |
| COST | 75.0% | N/A | 0.2044 | 4 |
| CRCL | 42.9% | 0.750 | 0.2753 | 7 |
| CRM | 50.0% | 0.500 | 0.3486 | 4 |
| CRWD | 54.5% | 0.567 | 0.2574 | 11 |
| DDOG | 27.3% | 0.562 | 0.3010 | 11 |
| DNA | 75.0% | 0.500 | 0.2873 | 4 |
| DUOL | 50.0% | 0.480 | 0.2996 | 10 |
| ETSY | 25.0% | N/A | 0.3006 | 4 |
| FIVN | 50.0% | 0.500 | 0.2489 | 4 |
| FSLY | 50.0% | 0.500 | 0.2534 | 4 |
| FTNT | 25.0% | 0.375 | 0.3092 | 4 |
| GM | 66.7% | 0.500 | 0.2776 | 3 |
| GME | 50.0% | 0.500 | 0.2541 | 4 |
| GOOG | 27.3% | 0.214 | 0.2776 | 11 |
| HUM | 75.0% | 0.500 | 0.2669 | 4 |
| HY | 75.0% | 1.000 | 0.1942 | 4 |
| INSM | 50.0% | 0.000 | 0.2781 | 4 |
| INTC | 25.0% | 0.000 | 0.3129 | 4 |
| IREN | 50.0% | 0.750 | 0.2566 | 4 |
| JNJ | 27.3% | 0.100 | 0.3123 | 11 |
| KVUE | 50.0% | 0.250 | 0.2619 | 4 |
| LLY | 50.0% | 0.333 | 0.2468 | 4 |
| LULU | 50.0% | 0.667 | 0.2726 | 4 |
| LYFT | 25.0% | 0.667 | 0.2960 | 4 |
| META | 54.5% | 0.667 | 0.2526 | 11 |
| MP | 45.5% | 0.467 | 0.2622 | 11 |
| MRNA | 63.6% | 0.517 | 0.2501 | 11 |
| MSFT | 72.7% | 0.792 | 0.2061 | 11 |
| MU | 50.0% | 0.500 | 0.2554 | 4 |
| NET | 75.0% | 0.500 | 0.2297 | 4 |
| NFLX | 72.7% | 0.571 | 0.2220 | 11 |
| NIO | 25.0% | 0.667 | 0.2740 | 4 |
| NOK | 25.0% | 1.000 | 0.2470 | 4 |
| NVDA | 54.5% | 0.567 | 0.2696 | 11 |
| NVMI | 50.0% | 0.500 | 0.2550 | 4 |
| NVO | 20.0% | 0.208 | 0.2934 | 10 |
| OKLO | 25.0% | N/A | 0.2549 | 4 |
| ONTO | 50.0% | 0.250 | 0.2800 | 4 |
| OPEN | 63.6% | 0.714 | 0.2594 | 11 |
| ORIC | 75.0% | 0.000 | 0.2362 | 4 |
| PFE | 45.5% | 0.536 | 0.2863 | 11 |
| PL | 100.0% | 1.000 | 0.1663 | 4 |
| PLTR | 45.5% | 0.411 | 0.2635 | 11 |
| PUBM | 0.0% | 0.000 | 0.3013 | 4 |
| PYPL | 63.6% | 0.867 | 0.2057 | 11 |
| QS | 25.0% | 1.000 | 0.2785 | 4 |
| QUBT | 50.0% | 0.500 | 0.2679 | 4 |
| QURE | 25.0% | 1.000 | 0.2902 | 4 |
| ROKU | 50.0% | 0.667 | 0.2490 | 4 |
| ROST | 25.0% | 0.250 | 0.2717 | 4 |
| RZLV | 33.3% | 0.325 | 0.2529 | 9 |
| S | 50.0% | 0.667 | 0.3151 | 4 |
| SENS | 25.0% | 0.333 | 0.2871 | 4 |
| SHOP | 40.0% | 0.286 | 0.2671 | 10 |
| SLV | 54.5% | 0.375 | 0.2941 | 11 |
| SMCI | 81.8% | 0.792 | 0.1676 | 11 |
| SMMT | 50.0% | 0.500 | 0.2679 | 4 |
| SNOW | 54.5% | 0.483 | 0.2562 | 11 |
| TEAM | 75.0% | 0.667 | 0.2147 | 4 |
| TGT | 75.0% | N/A | 0.2436 | 4 |
| TJX | 75.0% | 0.000 | 0.2484 | 4 |
| TPR | 75.0% | 0.500 | 0.2298 | 4 |
| TSLA | 63.6% | 0.607 | 0.2460 | 11 |
| TSM | 63.6% | 0.679 | 0.2501 | 11 |
| UAL | 25.0% | 0.333 | 0.3139 | 4 |
| UNH | 54.5% | 0.679 | 0.2317 | 11 |
| USAR | 100.0% | 1.000 | 0.1466 | 4 |
| V | 50.0% | 0.000 | 0.2739 | 4 |
| VKTX | 50.0% | 0.500 | 0.2552 | 4 |
| VZ | 100.0% | N/A | 0.1986 | 4 |
| WMT | 100.0% | N/A | 0.2048 | 4 |
| XYZ | 25.0% | 0.333 | 0.2523 | 4 |
| ZM | 36.4% | 0.467 | 0.2619 | 11 |

**Portfolio-wide averages:** Accuracy = 52.1% | ROC-AUC = 0.520 (mean across 89 tickers with data)

---

## Tickers Requiring Attention

| Ticker | Issue |
|---|---|
| ARM | Accuracy = 0.0% (4 predictions, all wrong) |
| PUBM | Accuracy = 0.0% (4 predictions, all wrong) |
| NVO | Pending outcome from **2026-03-06** — stale for 2+ weeks |
| RZLV | Pending outcome from **2026-03-09** — stale for ~2 weeks |
| GM | Pending outcome from **2026-03-18** — stale for 5 days |
| 94 tickers | Pending outcomes from 2026-03-20 (outcome date was Mar 21) |

---

## Action Required

To resolve the reconciliation backlog, run the script from the **host machine** (not the sandbox) where `yfinance` is installed in the `ml_quant_310` conda environment:

```bash
cd /Users/atomnguyen/Desktop/ML_Quant_Fund
conda run -n ml_quant_310 python3 -m accuracy.sink --reconcile --cache
```

This will fetch the ~95 overdue outcomes (Mar 6, 9, 18, 20 predictions) and update the accuracy cache with an additional trading day's data.
