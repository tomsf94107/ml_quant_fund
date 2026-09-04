# CORRECTION — PCT7 is NOT a validated brick

**2026-09-05, written hours after `BRICK_2_PCT7_2026-09-05.md`.**
That document is **retracted**. Read this instead.

---

## What the brick document claimed

That PCT7 passed the gauntlet at threshold 0.30 and was the fund's second
validated signal, on the basis of:

- per-date lift +8.83pp, Newey-West t = +3.34, null t = +1.37
- positive every month: +2.8 / +5.6 / +14.3 / +7.9pp
- beat its own session's base rate on 48 of 65 days
- **selected names returned +1.46% against a universe mean of +0.34%**, holding
  net of 40bps

## What was never checked

**Whether that +1.46% is investable at $105k.**

Every statistic above is a *pooled* figure over all 1,105 selections. The
deployment check asked what happens when you can only hold a subset — which is
the actual situation at this capital — and the answer reverses the conclusion.

## The measurement that retracts it

Cap positions per day, taking the highest `prob_pct7` first:

| cap | n | mean return |
|---|---|---|
| top 3 | 189 | **−2.38%** |
| top 5 | 306 | −2.25% |
| top 8 | 445 | −2.00% |
| top 10 | 520 | −1.63% |
| top 15 | 684 | −0.71% |
| **all** | **1,105** | **+1.46%** |

Monotone in how many you take, and negative for every subset.

Taking the LOWEST-probability names instead is no better — −2.12% at cap 3,
−1.47% at cap 5. **So the ordering is not inverted; there is no usable ordering
within the selected set at all.** Any small subset loses money regardless of how
it is chosen.

## Where the +1.46% actually comes from

Grouping the 64 dates by how many names fired that day:

| names/day | days | selections | mean return |
|---|---|---|---|
| 1–5 | 11 | 41 | **−3.96%** |
| 6–15 | 27 | 253 | **−5.17%** |
| 16–40 | 20 | 469 | +0.88% |
| **41+** | **6** | **342** | **+6.55%** |

**38 of 64 days lose money. Six days carry the entire result.**

The pooled mean weights each NAME equally; per-date it weights each DAY equally.
Daily counts run from 1 to 81, so the two differ enormously — pooled +1.46%
against a per-date mean of −1.97%, beating the universe on only 20 of 64 days.

Both numbers are correctly computed. They answer different questions, and only
the second describes a book you could hold.

## What is still true

The within-date selection edge is real and was not an artifact:

- beats its own session's base rate on **48 of 65 days**
- hit-rate lift **monotone in threshold**: +12.0pp at 0.20 rising to +46.8pp
  at 0.50
- calibration monotone through all six buckets
- the whole record is genuinely out-of-sample: the model was trained 2026-05-25
  and never retrained

PCT7 picks names more likely to move +7% than the average name on the same day.
That is a real property. It just does not convert into money at a size where you
must choose among its picks.

## Capacity is NOT the constraint — corrected

An earlier draft blamed capital. That was wrong. Hold every selection each day at
a $500 position floor and vary the capital:

| capital | max positions | mean cohort return | negative days |
|---|---|---|---|
| $105,000 | 210 | **−1.97%** | 41/64 |
| $200,000 | 400 | **−1.97%** | 41/64 |
| $500,000 | 1,000 | **−1.97%** | 41/64 |

Identical at every level. At $105k the book already holds 210 positions and the
busiest day fires 81, so the cap never binds. Liquidity does not bind either —
the median position is 0.001% of 20-day dollar ADV.

**The real cause is n-weighting.** The +1.46% is a NAME-weighted average over all
1,105 selections: the return of a portfolio placing equal dollars in each
selection regardless of which day it fired. Running that would require knowing
the future distribution of daily counts in order to size correctly. It is not a
strategy.

Weight each DAY equally — what trading it actually means — and the result is
−1.97% per cohort, 41 of 64 days negative.

Both figures are correct from the same data. They describe different portfolios,
and only the day-weighted one can be held.

## Verdict

**PCT7 is a statistically real signal that is not investable at $105k.**

That is the same shape as the direction model's verdict earlier the same day:
measurable, and economically absent. It is not a failed signal — it is a signal
whose edge lives in breadth this fund cannot buy.

## The methodological error, recorded

The gauntlet tested significance, nulls, regime stability, month-by-month
consistency and a cost ladder. Every one passed. **It never asked whether the
measured quantity was capturable.**

A pooled mean over all selections is the return of a portfolio that holds all
selections. If you cannot hold them all, that number is not your return, and no
amount of statistical rigour on it changes that.

**Capacity is not a footnote to a validation. It is part of one.** The gauntlet
should end with a capped-subset test, and `analysis/pct7_gauntlet.py` should be
amended to include it so this is not repeated on the next candidate.

## What would change the verdict

- **More capital.** The edge is real at breadth; it needs roughly $250k+ to hold
  a peak day.
- **A sub-signal that orders within the selected set.** Currently nothing does —
  neither high nor low `prob_pct7`. If a second feature could rank the 1,105,
  the top slice might become investable.
- **Understanding the six days.** They are 342 of 1,105 selections at +6.55%.
  If what makes those days different is identifiable in advance, that is a
  market-timing signal worth having on its own — and a different product from
  stock selection.

## Status changes

- `BRICK_2_PCT7_2026-09-05.md` — **RETRACTED**, superseded by this document
- PCT7 remains in **shadow mode**; nothing was deployed, so no capital was at
  risk at any point
- `analysis/pct7_status.py` stays crond Mondays 11:00 — the evidence keeps
  accruing and the within-date edge is worth tracking
- The fund still has **one** validated brick: short interest, low days-to-cover,
  h=40
