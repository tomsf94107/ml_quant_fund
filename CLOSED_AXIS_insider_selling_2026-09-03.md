# CLOSED AXIS — insider selling as a short-horizon predictor

**2026-09-03.** Tested and closed. Reproducible via
`analysis/insider_construction_test.py`.

---

## What prompted it

CRWV scored **9.1% accuracy on 11 high-confidence h=5 predictions** — the model
was confidently long while:

- Magnetar Financial LLC, a **30.14% holder (107,962,916 shares)**, distributed
  roughly 70% of its stake — ~75.0M shares sold, ~32.9M remaining;
- the officer complex (Intrator/Omnadora, McBee's trusts, Venturo/West Clay,
  Agrawal, Goldberg, McVeety) sold **~7.79M shares for ~$730M in one quarter**;
- the stock fell from ~$106 to ~$84.

The natural inference was that the model needed insider data. It already had it.

---

## What was already there

`features/builder.py` carries five insider features —
`insider_net_shares`, `insider_7d`, `insider_21d`, `insider_60d`,
`insider_90d` — constructed at lines 1086–1089 as `net.rolling(N).sum()`.

`insider_90d` had the highest importance of the five (**4.53**, mid-pack among
~100 features), and CRWV had **170 rows in `insider_flows`**, the second-highest
of any ticker. The model saw the selling and predicted 0.78 up regardless.

---

## The hypothesis, and why it was plausible

Two defects in the existing construction looked like they explained the failure:

1. **Not normalised.** A raw share count makes 200,000 shares identical for a
   small float and for Apple.
2. **No timing content.** A rolling 10b5-1 distribution produces a steadily
   negative number every day for months — the same value on the day the stock
   rises and the day it falls. A level cannot pick weeks.

**Field & Hanka (*Journal of Finance*, 2001)** supported a fix. On 1,510 Nasdaq
IPOs they found lockup expiration produces a **~2% negative abnormal return**,
measured on **3-day and 5-day event windows** — horizons compatible with this
model, unlike Cohen-Malloy-Pomorski's monthly result. The effect is
**permanent**, not a temporary price-pressure bounce, and is **more negative
when volume is abnormally high**. Their mechanism: *"When insiders sell their
shares, the public is asked to hold a greater number of shares… the share price
will fall, permanently."*

That argued for normalising by absorption capacity rather than raw shares.

---

## The test

309,205 insider filings, 379 tickers, 2021-01-01 to 2026-08, 285 evaluation
dates.

**Disciplines applied**, each from an error that has cost this project before:

- **Keyed on `filing_date`, not `trade_date`.** The market cannot know about a
  transaction until it is filed, and Form 4 allows two business days. Using
  trade_date leaks up to two days of hindsight into every observation — the same
  class of error that voided the earlier PEAD work, where `report_date` turned
  out to be fiscal-period-end rather than announcement.
- **Per-date IC with Newey-West at the horizon lag.** Pooled stock-date rows are
  not independent — a market-wide move correlates every stock on a date — and
  treating them as independent has inflated t-statistics 10–20× here before.
- **Shuffle null on every construction**, reported beside the result.

## The result

| construction | mean IC | NW t | null IC | null t |
|---|---|---|---|---|
| `net_shares_90d` (current) | +0.0015 | +0.20 | +0.0013 | +0.36 |
| `days_of_volume` | +0.0028 | +0.42 | −0.0043 | −1.23 |
| `accel` | −0.0020 | −0.38 | −0.0027 | −0.70 |
| `sell_only_dov` | +0.0028 | +0.42 | −0.0041 | −1.08 |
| `net_dov` | +0.0025 | +0.36 | +0.0056 | +1.51 |

**Every IC is under 0.003. Every t is under 0.5. The nulls are the same size as
the signals, and larger in two cases.**

On this sample size that is not an underpowered test returning nothing — it is a
well-powered test returning zero.

---

## Why the reasoning failed

**Field & Hanka measure a discrete EVENT.** A lockup expires on a known date and
float jumps at once. Their 3–5 day window is an *event* window around that date.

**CRWV is continuous distribution** — the opposite. No event, no date, spread
across months through rolling 10b5-1 tranches. Every Form 4 was public within
two business days, so there was never a moment when the information arrived.

And the magnitude does not clear the noise: a ~20% decline over four months is
roughly **0.3% per week** in a name with ±5% daily swings.

The mechanism is real. It simply does not produce a 5-day signal.

---

## Consequences

- **`insider_90d` is not underperforming a better construction.** There is no
  better construction. Its mid-pack importance of 4.53 reflects what the
  information supports. **No change to `features/builder.py`.**
- **Nothing is added to `prediction_features`.** Logging insider columns was
  proposed so the next CRWV would be diagnosable; with no predictive content
  there is nothing to diagnose.
- **CRWV was not a fixable miss.** The model cannot see supply distribution, and
  neither can any construction of the data available. That is a measured limit,
  not a defect.

## A separate defect found on the way, NOT fixed here

`insider_flows` — the aggregate table `builder.py` reads at lines 1171/1178 —
holds **5 rows before 2025** and 5,512 after, while `insider_filings_raw` holds
**383,355 filings back to 2019**. So the model trains on a panel where insider
features are near-zero across roughly half the effective training window
(which begins ~2021-07).

This does not change the verdict above — the test used the raw table and found
nothing at any construction. But it means the feature the model *does* use is
built on a fraction of the available data, and if the axis is ever reopened,
this must be fixed first.

## Reopen conditions

- The horizon extends to **monthly**. Cohen, Malloy & Pomorski (*JF* 2012) found
  opportunistic trades earn 82bps/month value-weighted — but at a monthly
  horizon, and their routine/opportunistic split would classify CRWV's 10b5-1
  selling as *routine* and uninformative.
- A **discrete lockup-expiry event feature** is built — a known date, not a
  continuous level. That is the construction Field & Hanka actually tested, and
  it was not tested here.
- `insider_flows` is backfilled from `insider_filings_raw` and the test is rerun
  on the feature the model actually consumes.

---

**Sixth closed axis**, alongside Lazy Prices, idiosyncratic vol, vol-gate, gross
profitability and peer pre-announcement.
