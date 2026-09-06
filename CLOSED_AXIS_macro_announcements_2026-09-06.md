# CLOSED AXIS — macro announcement effects

**2026-09-06.** Two independent tests find nothing. The published effects are
real and well documented; they died around 2015, and this fund's data begins in
2016.

No features to be built from the economic calendar. The calendar ETL is retained
for the discretionary sleeve and for future event studies.

---

## What was tested and why

The proposal was to feed macro-event proximity into the model — `days_to_fomc`,
`days_to_cpi` and similar — on the reasoning that scheduled events move markets
at h=1/3/5.

**A market-wide countdown is identical for every ticker on a given date.** A
cross-sectional model that ranks 1,920 names against each other cannot use a
constant. The only route in is an interaction with something stock-specific, and
the literature names it: Savor & Wilson (2014) find the security market line has
a significantly positive slope on announcement days, estimated at **6.81 basis
points**, against roughly flat otherwise. High-beta names earn more specifically
then.

So the whole case rested on that interaction existing in this universe over this
period. It was tested before anything was built.

## Test 1 — cross-sectional: does beta pay on announcement days?

`analysis/announcement_beta_test.py`, 300 tickers, 2016-08 onward, per-date
Spearman(beta_60d, forward return), Newey-West on the per-date series.

| horizon | announcement IC | other IC |
|---|---|---|
| h=1 | **−0.0097** | +0.0057 |
| h=3 | −0.0028 | +0.0085 |
| h=5 | −0.0036 | +0.0128 |
| h=40 | +0.0233 | +0.0297 |

**Announcement days are WORSE than ordinary days at every horizon.** FOMC — where
Lucca & Moench say the effect concentrates — is the most negative at h=1:
IC −0.0335, top-minus-bottom beta quintile −0.078pp.

Nothing clears NW t = 1.5 except NFP at h=40 (t = +2.29, 115 dates), which is
below the t > 3.0 bar this fund applies to new claims per Harvey, Liu & Zhu.

## Test 2 — intraday: is there a pre-FOMC drift?

`analysis/fomc_intraday_test.py`, SPY 1-minute bars, 77 FOMC sessions from 2017,
each matched against a control weekday seven days earlier.

| window | FOMC | control | diff | NW t |
|---|---|---|---|---|
| pre_2h (12:00–13:45) | −0.041% | +0.009% | −0.050% | −1.44 |
| **pre_30m (13:30–13:45)** | **−0.001%** | −0.017% | +0.017% | −0.08 |
| ann_30m (14:00–14:30) | −0.044% | −0.003% | −0.042% | −1.16 |
| post_2h (14:30–16:00) | −0.115% | −0.003% | −0.112% | −1.10 |
| full_day | −0.153% | +0.033% | −0.187% | −1.42 |

**`pre_30m` is −0.001%.** Lucca & Moench (2015) measured **+49 basis points**
over the 24 hours before the announcement for September 1994 to March 2011,
roughly 80% of the annual equity premium. It is now zero.

Nothing clears t = 1.5. By era, 2017–2019 pre_2h is −0.029% and 2020–2026 is
−0.046% — no revival in the earlier block either.

## This matches the literature exactly

The disappearance is documented, and this sample begins after it:

- **Kurov, Wolfe & Gilbert (2020)**, *The Disappearing Pre-FOMC Announcement
  Drift*: extending to December 2019, the drift "essentially disappeared after
  2015" in announcements both with and without a press conference. Proposed
  cause: **reduced uncertainty** as the Fed became more transparent.
- **Kurov & Gu (2016)** put the break at 2011.
- **Ben Dor & Rosa (2019)** find no evidence from 2011 to 2017.
- **Boguth et al. (2019)** find it limited to press-conference announcements
  through 2017; **Kurov et al. (2021)** find it gone without press conferences
  and weakened with them through 2019.

**So the null result is the expected result, not a data failure.** It is also a
clean instance of post-publication decay — the McLean-Pontiff pattern this
fund's notes record at a ~26% base rate. This one decayed to zero.

## On the single observation that prompted this

A Warsh speech was observed with the index rising roughly 30 minutes before and
falling 15–30 minutes into it.

The `ann_30m` average across 77 meetings is **−0.044%** — the same sign as the
second half of that observation, roughly twenty times too small to trade, and
not distinguishable from zero at t = −1.16.

One instance matching a remembered pattern is what randomness looks like. The
check was worth running; the answer is that it was a draw.

## What is retained

`analysis/etl_macro_calendar.py` and `accuracy.db.macro_events` — **892 dated
events across FOMC, CPI, NFP, PPI, GDP, IP and PCE, 2016 to 2026**, sourced from
FRED release dates and the Fed's published FOMC calendar.

The dates are PIT-honest: BLS and BEA publish schedules a year ahead and the Fed
publishes FOMC dates the summer before. Useful for the discretionary sleeve, for
event-window studies, and for any future test. Just not as model features.

Note the FOMC list is **hardcoded and must be extended each year**, or it will go
stale silently — the failure shape this codebase keeps finding.

## What would reopen this

- A **surprise** measure rather than a date. `economic_calendar` carries
  `forecast` and `previous`, so actual-minus-consensus is constructible going
  forward — but only from 2026-05 and only knowable after release, so it must be
  lagged like `eps_surprise`.
- Evidence the effect returns. It is tied to uncertainty, and a less transparent
  Fed could revive it. That is a reason to re-run in a few years, not to build
  now.
