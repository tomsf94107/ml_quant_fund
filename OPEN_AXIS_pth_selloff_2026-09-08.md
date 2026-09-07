# AXIS — PTH × past-winner in selloffs
**Status: OPEN · pending beta-neutral test · logged 2026-09-08**
Supersedes `HANDOFF_pth_selloff_2026-09-07.md` (retained as the evidence memo).
Sits alongside the closed axes; is **not** one of them.

---

## Claim
When SPY has been falling in a window ending well before setup, **past winners near their
52-week high subsequently underperform past winners far from theirs**, over the following
40 sessions.

Provenance: George & Hwang (2004) on 52-week-high proximity; Chen et al. (2026) sharpening it
to an interaction concentrated among past winners. Direction was **predicted before the split
was run** — better provenance than a swept result.

## Verdict
**Not a signal. Not closed.** Passes four of five controls. **~60% of it is beta.**
One test from a verdict either way.

---

## Evidence

**Winner × PTH interaction, h=40** `[fact — pth_winner_test_v2.py]`

| Split | N | IC | NW-t |
|---|---|---|---|
| past winners | 886 | −0.0316 | −2.41 |
| past losers | 886 | −0.0285 | −1.90 |
| winners / rebound | 107 | −0.0445 | −1.49 |
| winners / normal | 595 | −0.0119 | −0.85 |
| **winners / selloff** | 184 | **−0.0877** | **−3.44** |

**Gauntlet, 3 seeds** `[fact — pth_selloff_gauntlet.py]`

| Control | Result | Verdict |
|---|---|---|
| Raw IC | −0.0991, 3/3 same sign, seed-3 NW-t −3.64 | pass |
| Shuffle null | +0.0038 | pass — pipeline sound |
| Economic test | low-PTH book **+2.284pp**, 3/3 seeds, ~60% of dates positive, t +2.5 to +2.8 | under bar |
| Threshold robustness | smooth, no knife edge | pass |
| **Beta strip** | IC −0.0991 → **−0.0389** (−61%); per-seed t −2.33 / −1.60 | **FAIL** |

Threshold sweep (seed 1, IC in selloffs): −1% → −0.0677 t −2.74 · −2% → −0.0952 t −3.64 ·
−3% → −0.0927 t −3.09 · −5% → −0.0911 t −2.46.

## Why it fails
High-PTH names are the crowded winners. Crowded winners are high-beta. In a selloff, high-beta
falls harder. **Roughly 60% of this is a beta bet wearing a 52-week-high label** — the same
control where three prior discoveries in this fund died. This is the fourth.

The residual 40% keeps its sign in 3/3 seeds. That is not nothing, and it is not significant.

---

## Two corrections banked on the way (keep these regardless of the outcome)

1. **State window shared calendar with the outcome.** v1 classified market state from SPY's
   trailing 21 sessions **ending on the formation date**, so at h=40 the "selloff" label partly
   described the drop the forward return then recorded. Lagging the state window **45 sessions
   before formation** made them disjoint and **cut the effect by a third**: IC −0.1394 → −0.0877,
   t −4.73 → −3.44. Not a look-ahead in the strict sense (past prices only), but state and
   outcome were not independent.
   → **Generalise:** any state-conditioned test must lag its state window clear of the outcome window.

2. **Formation uses a 21-session skip.** Jegadeesh (1990): the prior month reverses at roughly
   2.49%/month. Without the skip, the winner/loser split measures last month's bounce, not momentum.

## Why the sign is opposite to G&H
- **Wrong universe.** G&H's 1963–2001 sample is equal-weighted and implicitly overweights micro-
  and small-caps. This test runs the **400 most liquid US names**. An Australian out-of-sample
  study found the 52-week-high strategy on liquid stocks fails to produce significant dollar
  profits once short-sale restrictions, transaction costs and liquidity constraints are counted.
- **Conditional mechanism.** *Momentum Crashes and the 52-Week High*: stocks far from their highs
  attract speculative demand on perceived room-to-run and surge when the market is not in a normal
  state → negative PTH IC exactly in abnormal states, which is what the split found.

Both the mechanism and the conditional structure were predicted before the test. **Only the
magnitude failed.**

---

## The one remaining test
**Construct beta-neutral from the start** rather than stripping beta afterwards: rank PTH
**within beta quintiles**, so the comparison is high-PTH vs low-PTH at the same beta.
Post-hoc orthogonalisation removes the linear component; ranking within buckets removes it by
construction.

### Pre-registered acceptance — write this down before running
Three outcomes, not two:

| Outcome | Rule |
|---|---|
| **Real** | Beta-neutral IC at **NW-t > 3.0**, 3/3 seeds same sign, **and** a book that clears cost after the haircut → shadow book |
| **Beta all along** | Effect collapses → **close the axis** |
| **Signed but thin** | Sign holds (already 3/3) but t < 3.0 and/or the book does not clear cost → **close the axis** |

The third outcome is the likely one and the source memo does not name it. Raw was ~60% beta, so
the neutral book is plausibly ~half of +2.284pp or less `[inf — rough scaling from the 61% beta
share]`, at a t that is already marginal at +2.5–2.8. **A consistently-signed residual is not a
promotion.** Do not let "3/3 same sign" launder a sub-bar result into a shadow book.

Secondary checks if it does survive: sector-neutral rerun (financials/REITs carry structurally
high beta that is not crowding), turnover and cost at 20 bps, and whether the low-PTH long leg
is separable from the SI brick's low-DTC long leg — they may be holding the same names.

---

## Files
- `analysis/pth_winner_test.py` — univariate + market-state split
- `analysis/pth_winner_test_v2.py` — same, state window lagged 45 sessions
- `analysis/pth_selloff_gauntlet.py` — multi-seed, null, beta strip, economic test, threshold sweep

## Next action
Write the beta-quintile variant against `pth_selloff_gauntlet.py`'s existing IC / beta-strip /
book / null machinery. **Blocked:** needs the gauntlet source to match its constructions and DB
schema exactly. A rewrite from scratch would not be comparable to the numbers above.

## ASSUMPTIONS
- Beta strip = regression of forward returns on a beta factor, IC recomputed on the residual;
  exact construction not read from source. `[unconfirmed]`
- "3 seeds" = ticker-sample seeds, as in the fund's other multi-seed tests. `[unconfirmed]`
- Bar is NW-t > 3.0 per Harvey-Liu-Zhu, consistent with the fund's other axes. `[confirmed —
  stated in the scripts' own headers]`
