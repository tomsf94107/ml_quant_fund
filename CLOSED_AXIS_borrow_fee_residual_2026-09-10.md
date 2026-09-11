# AXIS CLOSED — borrow-fee residual (HTB beyond short interest)
**Closed 2026-09-10.** Sixth closed axis. The fund still has one validated brick.

The battery was written for this candidate and never run. It has now been run
and the candidate fails two of its four pre-registered criteria.

---

## The claim under test

On each FINRA settlement date, residualize `fee_gt_5pct` on `days_to_cover`
across stocks (OLS, keep the residual), then IC that residual against h=40
forward return. The residual is "hard to borrow beyond what short interest
explains."

DIAG 2 had reported permutation p=0.000, z=+6.26 on the residual, while the raw
standalone IC failed `|t|>=2`. So it was a conditional/interaction candidate
from the start, and `validate_borrow_battery.py` was written to decide "real 3rd
brick, or residualization mirage?"

## Result

Baseline, full sample: **IC +0.0273, NW-t +2.30, 62% of dates positive**, 121
settlement dates, average 381 stocks, 2021-06-15 .. 2026-06-15.

| # | test | bar | result | |
|---|---|---|---|---|
| 1 | OOS cold split at 2024-01-01 | sign holds, \|t\| >= 1.5 | +0.0270, **t=+1.25** | **FAIL** |
| 2 | sector-neutral | retains > 40% of IC | +0.0092, **34%**, t=+0.92 | **FAIL** |
| 3 | year-by-year sign | >= 80% stable | 83% | pass |
| 4 | float-bucket terciles | — | **0 dates, skipped** | not run |
| 5 | outside index-event windows | survives | t=+1.79 | pass |

## Why it closes

**Two thirds of the effect is sector.** Demeaning feature and return within
`bucket` per date cuts the IC from +0.0273 to +0.0092 at t=+0.92 — 34% retained
against a 40% bar. Most of "hard to borrow beyond short interest" is "this
sector is hard to borrow", and sector exposure is available far more cheaply
than through a borrow-fee signal.

**What survives does not hold out of sample.** Post-2024 carries the same sign
at t=+1.25, below the battery's own 1.5 and far below the fund's t>3.0 brick
bar.

**The two failures compound.** The OOS half is measured on the un-neutralized
signal, so the 34% that is stock-level is a subset of an already-weak +1.25.

**The headline z was a different statistic.** DIAG 2's z=+6.26 is a permutation
result against shuffled returns, not a Newey-West t on overlapping 40-day
windows. The full-sample NW-t is +2.30. Both are correct answers to different
questions, and the second is the one that governs deployment.

**Year stability is thinner than 83% suggests.** 2022 at +0.0010 and 2024 at
-0.0009 are effectively zero; the mean is carried by 2021 (+0.052), 2023
(+0.040) and 2025 (+0.063). Three years of six do the work.

## Comparison that settles it

| | SI brick | this |
|---|---|---|
| IC | -0.037 | +0.027 |
| NW-t | -4.46 | +2.30 |
| year signs | correct every year 2021-26 | near-zero in 2 of 6 |
| sector-neutral | survives | 34% retained |
| OOS | survives cold holdout | t=+1.25 |

Different class of result.

## Test 4 was not run, and this is why

All three float terciles returned `0 dates (<6)`. No float data source resolves
in the battery — the mechanism question it was written to answer, "is this only
small-float scarcity?", is unanswered.

It is recorded as a gap rather than chased. Tests 1 and 2 settle the verdict on
their own, so knowing which mechanism produced a result being closed anyway is
nice to have, not decisive. **Anyone reopening this should implement Test 4
first**: if the effect concentrates in small-float names, that explains both the
sector concentration and the OOS decay, and the axis stays closed for a better
reason than "it failed two tests."

## What stays

**The features stay wired; no standalone borrow signal exists.** Same disposition
as the insider axis closed 2026-09-07 after seven constructions. `borrow.db`
holds `borrow_fees` (224,774 rows, 1,920 tickers, 2021-06-15 onward),
`borrow_features` (fee changes over 1 and 3 settlements, availability change,
cross-sectional fee z-score) and `borrow_live`. Those may contribute inside
model interactions; none leads a trade.

**Registry F12** — "Hard-to-borrow breadth: % of universe on special /
days-to-cover z" — remains listed experimental and is untouched by this. It is a
breadth aggregate, not the cross-sectional residual tested here.

## Data notes worth keeping

`borrow_fees` is keyed to FINRA settlement dates, not to live stock-loan quotes:
125 rows per ticker over five years is semi-monthly, and every series ends
2026-08-14 — the same settlement as short interest, for the same reason. **There
is no timing advantage in this table over the FINRA release.**

`borrow_live` is the only daily source, from `UW:/api/shorts/{t}/data`, and
covers 74 tickers because `squeeze_radar.py` probes on demand — the same names
carried in the `squeeze_*.db` files. Coverage is what has been asked for, not a
vendor limit, so it could be widened if a live-borrow signal were ever worth
pursuing. On this evidence it is not.

`borrow_fetch.py` runs Sundays and last completed 2026-09-06: 1,942 tickers,
1,918 ok, 24 empty, 0 errors, 2,305 seconds.

## Files
- `validate_borrow_battery.py` — the five-test battery, criteria pre-registered
- `validate_borrow_diag.py`, `validate_borrow.py` — earlier diagnostics
- `borrow_fetch.py`, `borrow_features.py` — ingest and derived features

## Status
**CLOSED.** Fund still has one validated brick (SI / days-to-cover).
