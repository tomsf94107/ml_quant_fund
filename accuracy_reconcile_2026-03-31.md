# Daily Accuracy Reconciliation — 2026-03-31

**Run timestamp:** 2026-03-31 (automated scheduled task)
**Status:** ✅ Completed with partial data gaps (5 tickers)

---

## Summary

| Metric | Value |
|--------|-------|
| New outcomes recorded | **0** |
| Predictions pending (future horizons) | 505 |
| Predictions pending (due but failed) | 7 |
| Accuracy cache rows refreshed | 303 |
| Tickers with errors | 5 |

**Why 0 new outcomes?**
Today (2026-03-31, Tuesday) is a regular trading day, but the 7 predictions that were technically "due" all belong to the 5 failed tickers (BRKR, KVUE, NIO, NVO, RZLV). The remaining 505 pending predictions are for horizons whose outcome dates fall on 2026-04-01 or later — they are **not yet due** and this is expected behavior.

The majority of pending predictions break down as:
- 101 predictions from 2026-03-30 → outcome dates 2026-04-02 (h=3) and 2026-04-06 (h=5)
- 101 predictions from 2026-03-31 (today) → outcome dates 2026-04-01 (h=1), 2026-04-03 (h=3), 2026-04-07 (h=5)

---

## Tickers with Errors (5)

| Ticker | Error Type | Notes |
|--------|-----------|-------|
| BRKR | Proxy/network 403 | Bruker Corp — valid ticker, likely yfinance proxy block in sandbox |
| KVUE | Proxy/network 403 | Kenvue Inc — valid ticker, same proxy issue |
| NIO | Possibly delisted / proxy 403 | NIO Inc ADR — Chinese EV company; may have delisting/data issues |
| NVO | Possibly delisted / proxy 403 | Novo Nordisk — valid company; data fetch blocked in sandbox |
| RZLV | Possibly delisted / proxy 403 | Ticker may be invalid or delisted |

These 5 tickers represent 7 stale prediction rows across dates 2026-03-06, 2026-03-09, 2026-03-24, and 2026-03-26. **Recommendation:** Review RZLV for delisting, and consider adding to an exclusion list if the data source consistently fails.

---

## Accuracy Cache — Full Table (90-day window, sorted by ROC-AUC)

| Ticker | Accuracy | ROC-AUC | N Predictions |
|--------|----------|---------|--------------|
| XLE | 0.833 | 1.000 | 8 |
| USAR | 0.273 | 1.000 | 13 |
| COIN | 0.667 | 1.000 | 8 |
| AI | 0.879 | 0.857 | 13 |
| WMT | 0.909 | 0.821 | 13 |
| BRKR | 0.267 | 0.800 | 12 |
| PYPL | 0.870 | 0.778 | 20 |
| ABNB | 0.879 | 0.767 | 13 |
| SMCI | 0.574 | 0.764 | 20 |
| ADSK | 0.939 | 0.750 | 13 |
| CRCL | 0.429 | 0.750 | 7 |
| TSLA | 0.556 | 0.750 | 20 |
| META | 0.537 | 0.733 | 20 |
| MSFT | 0.222 | 0.733 | 20 |
| APLD | 0.182 | 0.733 | 13 |
| LLY | 0.152 | 0.722 | 13 |
| BSX | 0.204 | 0.700 | 20 |
| SENS | 0.400 | 0.667 | 5 |
| NOK | 0.515 | 0.667 | 13 |
| TPR | 0.485 | 0.643 | 13 |
| NFLX | 0.852 | 0.642 | 20 |
| QURE | 0.455 | 0.633 | 13 |
| NET | 0.606 | 0.625 | 13 |
| TSM | 0.889 | 0.625 | 20 |
| MU | 0.394 | 0.611 | 13 |
| UNH | 0.537 | 0.611 | 20 |
| OPEN | 0.167 | 0.605 | 20 |
| IREN | 0.242 | 0.604 | 13 |
| FIVN | 0.849 | 0.600 | 13 |
| XLI | 0.722 | 0.600 | 8 |
| CNC | 0.519 | 0.597 | 20 |
| CRWD | 0.500 | 0.593 | 20 |
| CI | 0.485 | 0.589 | 13 |
| PL | 0.576 | 0.571 | 13 |
| GM | 0.212 | 0.571 | 13 |
| AXP | 0.167 | 0.569 | 20 |
| PLTR | 0.537 | 0.569 | 20 |
| VKTX | 0.182 | 0.567 | 13 |
| GLD | 0.778 | 0.563 | 8 |
| PFE | 0.815 | 0.558 | 20 |
| ZM | 0.148 | 0.558 | 20 |
| MRNA | 0.519 | 0.556 | 20 |
| ABT | 0.242 | 0.542 | 13 |
| MP | 0.852 | 0.538 | 20 |
| VZ | 0.212 | 0.536 | 13 |
| AMPX | 0.182 | 0.536 | 13 |
| AZN | 0.879 | 0.533 | 13 |
| SMMT | 0.879 | 0.533 | 13 |
| S | 0.818 | 0.533 | 13 |
| SNOW | 0.500 | 0.525 | 20 |
| NVDA | 0.833 | 0.519 | 20 |
| AMD | 0.500 | 0.506 | 20 |
| INSM | 0.545 | 0.500 | 13 |
| ORIC | 0.515 | 0.500 | 13 |
| BNED | 0.600 | 0.500 | 5 |
| FTNT | 0.121 | 0.467 | 13 |
| DNA | 0.515 | 0.467 | 13 |
| HY | 0.545 | 0.467 | 13 |
| CAVA | 0.212 | 0.467 | 13 |
| HUM | 0.849 | 0.467 | 13 |
| AMZN | 0.152 | 0.467 | 13 |
| TEAM | 0.849 | 0.467 | 13 |
| QS | 0.879 | 0.464 | 13 |
| NIO | 0.546 | 0.464 | 11 |
| SLV | 0.852 | 0.444 | 20 |
| ASAN | 0.818 | 0.433 | 13 |
| QUBT | 0.182 | 0.433 | 13 |
| LULU | 0.515 | 0.417 | 13 |
| NVMI | 0.121 | 0.417 | 13 |
| DUOL | 0.824 | 0.414 | 19 |
| GME | 0.515 | 0.400 | 13 |
| ONTO | 0.515 | 0.400 | 13 |
| SPY | 0.722 | 0.400 | 8 |
| AAPL | 0.185 | 0.377 | 20 |
| XLF | 0.111 | 0.375 | 8 |
| BA | 0.818 | 0.375 | 13 |
| DDOG | 0.463 | 0.363 | 20 |
| FSLY | 0.212 | 0.357 | 13 |
| GOOG | 0.148 | 0.338 | 20 |
| UAL | 0.152 | 0.333 | 13 |
| RZLV | 0.333 | 0.325 | 9 |
| ROKU | 0.121 | 0.321 | 13 |
| COST | 0.182 | 0.304 | 13 |
| BETR | 0.485 | 0.300 | 13 |
| OKLO | 0.788 | 0.300 | 13 |
| NVO | 0.431 | 0.286 | 19 |
| TGT | 0.182 | 0.286 | 13 |
| TJX | 0.485 | 0.286 | 13 |
| ALK | 0.212 | 0.286 | 13 |
| JNJ | 0.833 | 0.273 | 20 |
| SHOP | 0.843 | 0.269 | 19 |
| CRM | 0.121 | 0.267 | 13 |
| XYZ | 0.788 | 0.267 | 13 |
| XLV | 0.778 | 0.250 | 8 |
| PUBM | 0.152 | 0.233 | 13 |
| XLU | 0.778 | 0.222 | 8 |
| INTC | 0.091 | 0.214 | 13 |
| ETSY | 0.455 | 0.200 | 13 |
| V | 0.485 | 0.200 | 13 |
| ARM | 0.121 | 0.200 | 13 |
| KVUE | 0.455 | 0.179 | 11 |
| ASTS | 0.727 | 0.179 | 13 |
| LYFT | 0.727 | 0.143 | 13 |
| ROST | 0.455 | 0.133 | 13 |
| QQQ | 0.000 | N/A | 8 |

**Overall mean accuracy:** 0.500 | **Overall mean ROC-AUC:** 0.499

---

## Notes

- The accuracy cache was refreshed using all outcomes within the 90-day rolling window.
- Multiple `ROC AUC undefined` warnings were logged for tickers with only one outcome class (all up or all down) — these are displayed as N/A in the table.
- **Pipeline health:** Normal — 0 new outcomes is expected behavior today since most pending outcomes mature between 2026-04-01 and 2026-04-07.
- **Action required:** Review RZLV (possibly delisted). BRKR, KVUE, NIO, NVO persistently fail in the sandbox environment — if these remain unresolved across multiple runs, consider adding a yfinance fallback or alternative data source for these tickers.
