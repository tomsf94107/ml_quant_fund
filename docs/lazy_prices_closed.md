# Hunt #7 Lazy Prices: CLOSED — tested, KILLED (June 3 2026)

## Status: pipeline FULLY BUILT (Jun 1), VALIDATED + KILLED (Jun 3). Do not re-attempt.

The Lazy Prices pipeline is complete and was run:

- data/sec_section_parser.py: extract_10k_sections() FULLY IMPLEMENTED (not a stub —
  Item 1/1A/7 MULTILINE boundary regex, longest-valid-span, min-length drops)
- data/etl_10k_lazy_prices.py: fetch_all + compute_similarity (cosine + jaccard)
- sec_filings.db: sec_10k_sections (1,960 sections, 120 tickers, 2019-2026) +
  sec_10k_similarity (1,607 rows, 115 tickers)
- (10-Q / S-1 / DEF 14A parsers remain NotImplementedError “Session B/D” — those are
  OTHER filing types, NOT needed for Lazy Prices. Irrelevant to #7.)

## The validation (analysis/lazy_prices_validate.py) — KILL

CMN 2020 hypothesis: high-similarity (stable) 10-K filers OUTPERFORM low-similarity
(big-change) filers, so long-stable / short-changed spread should be POSITIVE.

Result on 115-ticker universe (tercile L/S within each filing-year cohort, net 10bps):
business      63d: -8.2%  126d: -13.4%   rank-IC ~ -0.02/+0.01
risk_factors  63d: -2.8%  126d: -7.7%    rank-IC ~ +0.04
mda           63d: -9.7%  126d: -17.0%   rank-IC ~ -0.04/+0.00

- Spread is NEGATIVE everywhere (signal slightly BACKWARDS, not just absent).
- rank-IC ~ZERO across all sections/horizons (no monotone relationship).
- Per-year breakdown = NOISE, not a consistent effect: e.g. business 126d goes
  2021:-22.6%, 2022:+8.3%, 2023:+6.5%, 2024:-31.7%, 2025:-27.5% — wild sign-flipping
  driven by a few names. No stable effect in EITHER direction.

## Why it failed (same as the other killed hunts)

CMN found this on THOUSANDS of firms over decades. At 115 tickers x ~5 annual
cross-sections the breadth is far too low (Fundamental Law: IR = IC x sqrt(breadth)).
Annual signal = ~5 independent periods. The anomaly does not replicate at this scale.

## Fetch-bug note (NOT worth fixing)

39 universe tickers missing from sections. Most legit: ETFs (SPY/QQQ/XL*), foreign
20-F filers (ASML/AZN/TSM/NVO/SHOP/STLA/NIO/ARM/NBIS), recent/small (RZLV/SANA/ALT).
BUT ~10 real US filers wrongly dropped (AMZN, INTC, COST, NOW, CEG, VZ, NOK, GFS,
CYBR). Checked: get_cik() resolves ALL of them fine (AMZN 0001018724, INTC 0000050863,
etc.) — so NOT a shared-infra CIK bug; 8-K sentiment pipeline’s CIK path is unaffected.
The drop is a downstream 10-K fetch/parse quirk specific to those filings. NOT FIXED:
(a) Lazy Prices is dead so +10 names won’t revive it; (b) shared CIK infra is fine.

## The meta-finding (the real takeaway)

FIVE return-signal hunts now killed at 115-159 ticker scale: per-ticker direction,
global direction, reversal (cs-demean + PCA-residual), pairs/cointegration, Lazy Prices.
All fail the same way: no edge at this universe size. THE BOTTLENECK IS BREADTH, NOT
SIGNAL CHOICE. Per the Research Report Part 6: the levers are (1) EXPAND the universe
(more names = more breadth), (2) lengthen horizons, or (3) accept momentum as the one
signal. The next move is NOT a 6th signal hunt — it is confronting breadth.

## Standing instruction

Lazy Prices CLOSED. If it recurs: point here. Pipeline is built but the signal is dead
on this universe. Only worth revisiting if the universe expands to many hundreds of names.