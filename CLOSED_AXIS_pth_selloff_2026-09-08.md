# AXIS CLOSED — PTH × past-winner in selloffs
**Closed 2026-09-08.** Supersedes `OPEN_AXIS_pth_selloff_2026-09-08.md` and
`HANDOFF_pth_selloff_2026-09-07.md`, both retained as the evidence trail.
Fifth axis to die at the beta control.

---

## Verdict
**Not a signal.** The beta-neutral test failed its pre-registered bar. The
residual keeps its sign in every seed and never clears NW t > 3.0.

## The remaining test, and its result
Ranking PTH **within beta quintiles** — neutral by construction rather than by
post-hoc regression. `analysis/pth_beta_quintile.py`, 3 seeds, same panel and
same filters as the gauntlet so the numbers are comparable.

| | value | seeds same sign |
|---|---|---|
| raw IC (retained dates) | −0.0957 | 3/3 |
| post-hoc beta strip | −0.0523 | 3/3 |
| **WITHIN-BETA IC** | **−0.0487** | 3/3 |
| within-beta book | +1.278pp | 3/3 |
| within-beta null | +0.004 to +0.028 | clean |

Pre-registered acceptance, decided before the run:

| criterion | result |
|---|---|
| NW t > 3.0 in any seed | **NO** — max \|t\| 2.43 |
| same sign every seed | YES — 3/3 |
| book clears 20bp round-trip | YES — +1.278pp, but at t +0.78 and +1.62 |

Two of three. **SIGNED BUT THIN** — the third outcome, and a close rather than
a shadow book. A consistently-signed sub-bar residual is what a partially
removed beta exposure looks like.

Bucket betas separated cleanly (Q1 0.30 · Q2 0.70 · Q3 0.95 · Q4 1.17 ·
Q5 1.63), so the construction did what it claimed.

---

## The finding worth keeping: the beta strip is trustworthy

Within-beta **−0.0487** against post-hoc strip **−0.0523**. Ranking inside beta
buckets and orthogonalising afterwards agree to within 7%.

That matters beyond this axis. Post-hoc orthogonalisation removes only the
linear component, so the standing worry was that it might *over*-remove and kill
real signal. It does not. **Three prior discoveries in this fund died at the
beta strip; this confirms they died honestly.** The cheap control can be trusted
in future tests, and a candidate that fails it does not need this more expensive
version to confirm the verdict.

## Caveat, recorded rather than acted on
Bucketing needs more names per date, so 39–50% of selloff dates dropped out. On
the retained dates the **raw** IC reaches only t −2.03 and −1.78, against −3.64
on the full set. The subsample is weaker before beta is touched at all, so part
of the shortfall is lost power rather than removed beta.

`--buckets 3` would retain more dates and more power. **Not run.** That is a
second look at a hypothesis that has just failed its pre-registered test, and
trying variants until one passes is precisely what the bar exists to prevent.
Anyone reopening this must have a new reason, not a new parameter.

## What was banked on the way — keep these regardless

**State windows must be lagged clear of the outcome window.** The first version
classified market state from SPY's trailing 21 sessions ending **on** the
formation date, so at h=40 the "selloff" label partly described the drop the
forward return then recorded. Lagging the state window 45 sessions before
formation made them disjoint and cut the effect by a third: IC −0.1394 →
−0.0877, t −4.73 → −3.44. Not a look-ahead in the strict sense — only past
prices were used — but state and outcome were not independent. **This
generalises to every state-conditioned test in the fund.**

**Formation windows need the 21-session skip.** Jegadeesh (1990): the prior
month reverses at roughly 2.49%/month. Without the skip the winner/loser split
measures last month's bounce rather than momentum.

**Post-hoc beta orthogonalisation is a sufficient control.** See above.

## Why the sign was opposite to George & Hwang
G&H (2004) found nearness to the 52-week high predicts high returns on a
1963–2001 equal-weighted sample that implicitly overweights micro- and
small-caps. This ran the 400 most liquid US names, where an Australian
out-of-sample study found the strategy fails to produce significant dollar
profits after short-sale restrictions, costs and liquidity constraints. The
conditional mechanism from *Momentum Crashes and the 52-Week High* — stocks far
from their highs draw speculative demand on perceived room to run and surge when
the market is not in a normal state — predicts a negative PTH IC in abnormal
states, which is what the split found.

**The provenance was good and the mechanism held. Only the magnitude failed**,
and what magnitude there was turned out to be about half beta.

## Files
- `analysis/pth_winner_test.py` — univariate and market-state split
- `analysis/pth_winner_test_v2.py` — same, state window lagged 45 sessions
- `analysis/pth_selloff_gauntlet.py` — multi-seed, null, beta strip, economic
  test, threshold sweep
- `analysis/pth_beta_quintile.py` — the beta-neutral test that closed it

## Status
**CLOSED.** Fund still has one validated brick (SI / days-to-cover).
