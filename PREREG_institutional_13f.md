# PRE-REGISTRATION — institutional 13F tracker
**Written 2026-09-20, before the backfill completed.** The bars below are fixed.
Anything that fails them is closed, not retuned.

## Why this document exists

The congressional trading axis took a full day and ended in eight nulls across
eight angles. Every one of those angles was plausible before it was measured,
and three of them looked positive until they were benchmarked, split by year,
or computed at the right grain. The party gap read +2.34pp fanned and +0.74pp
at the return grain, with two thirds of what remained sitting in 2020.

The same day, a market-level earnings-season hypothesis produced a +1.07pp
excess that survived until three placebo anchors — one deliberately 45 days
away from any earnings — each produced the same +0.79pp. The effect was a bad
unconditional baseline, and it took a placebo to see it.

So the bars are written here first.

## The prior, stated honestly

The literature is not encouraging for the obvious version of this product.

Fama & French (2010), 1984-2006: in aggregate, active managers have **zero
alpha before costs and negative after**, and an investor buying a portfolio of
funds from the **top three percent can expect an alpha of zero**. The
value-weighted fund portfolio's alpha rises to only 0.1pp per year when
expenses are added back, and market return alone explains 99% of its variance.

The copycat literature (Frank et al. 2004; Verbeek & Wang 2013) found copycats
match their targets net of costs, with a +0.05%/month improvement after
disclosure went quarterly in 2004. But that edge is **the fee saving** — the
copycat matches gross return and skips the 1% expense ratio. We do not pay
their fee, so that edge does not exist here.

**Expected outcome: T1 fails.** It is registered anyway, because "institutions
in aggregate have no edge" measured on our own data is worth more than the same
claim inherited from a paper.

---

## T1 — Does consensus buying predict returns?

**Claim.** Tickers bought by the most tracked managers in a quarter outperform
in the following quarter.

**Construction.** Rank tickers by the count of tracked managers with
`units_change > 0` in quarter Q. Top decile against bottom decile, measured
from the **filing date**, not the quarter end.

**Bar.** Excess over SPY greater than 1pp per quarter, positive in at least 70%
of quarters, and the sign holds in both halves of the sample.

**Fails if** the spread straddles zero, or lives in fewer than 7 of 10 years.

## T2 — Does crowding predict drawdown?

**Claim.** Names held by an unusually high number of concentrated managers fall
harder when they fall. This is a RISK claim, not an alpha claim, and it is the
one most likely to survive.

**Construction.** Per ticker per quarter, count tracked holders. Compare the
worst 20-day drawdown over the following quarter for the top-crowding quintile
against the rest, controlling for sector and market cap.

**Bar.** Top quintile drawdown at least 2pp worse, holding in both halves.

**Why this differs from T1.** T1 asks whether managers pick well. T2 asks what
happens when many of them own the same thing and the exit is narrow. A null on
T1 says nothing about T2.

## T3 — Does a thesis shift precede sector moves?

**Claim.** A manager who liquidates their book and rebuilds into a different
sector is making a dated, high-conviction statement, and that sector
outperforms over the following two to four quarters.

**Construction.** Flag manager-quarters where (a) position count fell by more
than 70% from the prior quarter, and (b) within four quarters the book was
rebuilt with at least 50% weight in a sector previously under 10%. Measure the
new sector's ETF against SPY from the rebuild's filing date, over 1, 2 and 4
quarters.

**Bar.** At least 20 events across the sample, mean excess above 2pp at the
2-quarter horizon, positive in at least 60% of events.

**Fails if** fewer than 20 events exist. Thiel Macro's 2025-Q4 liquidation and
2026-Q2 energy rebuild is one instance; one instance is an anecdote.

---

## Confounds that must be handled, not noted

**Survivorship.** The UW institution list is TODAY's list. Managers who closed,
blew up or fell below the filing threshold are absent. A backtest of "follow
the smart money" run on survivors is the oldest trap in this literature and it
would make every test above look better than it is. Where a manager's history
starts mid-sample, that is a reporting-threshold artifact, not an entry.

**Truncation.** The vendor caps holdings responses at 500 rows. Citadel,
Millennium and ARK all hit it, so their position counts, sector weights and
concentration changes are artifacts of the cap. The roster is therefore
restricted to managers whose entire book fits — 127 of 500 probed. This biases
"held by N managers" toward smaller books, and that bias runs THROUGH every
test above. It is a limitation, not a bug, and any count-based statistic must
be read as "among concentrated managers".

**Keying.** Every measurement starts at `filing_date`, never `report_date`. The
gap is up to 45 days. Keying on quarter end puts a month and a half of
look-ahead into the result, which is exactly the error the congress ingest was
built to avoid.

**Grain.** One row per (manager, quarter, ticker). Never pool line items or a
manager filing many positions outweighs one filing few. The congress party
split moved from +2.34pp to +0.74pp on this alone.

**What 13F omits.** Long US equity only. No shorts, no most derivatives, no
cash, no bonds, nothing private. A manager showing zero positions holds no
reportable long equity — Thiel Macro showed zero for two quarters of 2025 —
which is not the same as holding nothing.

---

## What gets built

**If T2 survives and T1/T3 fail** — the likely outcome — the product is a
crowding and exposure map, not a signal. It answers "who else owns what I own
and how narrow is the exit", which is a position-sizing input. That is worth
building and it is honest.

**If T1 survives** it would contradict Fama-French on our own data, which
demands a second look at survivorship before anything is traded.

**If everything fails**, the page reports that, the ingest keeps running, and
the answer changes if the data ever does. The congressional page is the
template: a null reported plainly is a finished piece of work.

**No page is built before the tests run.** A dashboard built first will find a
way to look useful.
