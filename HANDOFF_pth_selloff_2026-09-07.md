# HANDOFF — PTH × past-winner in selloffs

**2026-09-07.** Not a brick. Passes four of five controls; **60% of it is beta.**
The residual is consistently signed but not significant. Recorded because the
provenance is unusually good and there is one specific test left that would
settle it.

---

## The claim

When SPY has been falling in a window ending well before setup, **past winners
NEAR their 52-week high subsequently underperform past winners FAR from theirs**,
over the following 40 sessions.

## What passed

| control | result |
|---|---|
| **Raw IC** | −0.0991 across 3 seeds, **3/3 same sign**, seed 3 at NW t −3.64 |
| **Shuffle null** | +0.0038 — clean, pipeline sound |
| **Economic test** | low-PTH book **+2.284pp**, 3/3 seeds, ~60% of dates positive |
| **Threshold robustness** | smooth degradation, no knife edge: |

```
SPY 21d < -1%   220 dates   IC -0.0677   NW t -2.74
SPY 21d < -2%   184 dates   IC -0.0952   NW t -3.64
SPY 21d < -3%   146 dates   IC -0.0927   NW t -3.09
SPY 21d < -5%    83 dates   IC -0.0911   NW t -2.46
```

A knife-edge that works at one threshold only is a fitted parameter. This is not
that.

## What failed

**The beta strip cuts it by 61%** — IC −0.0991 → **−0.0389**, and the per-seed
t-stats fall to −2.33 and −1.60, below the t > 3.0 bar.

High-PTH names are the crowded winners. Crowded winners are high-beta. In a
selloff, high-beta falls harder. **Roughly 60% of this is a beta bet wearing a
52-week-high label** — the same control where three prior "discoveries" in this
fund died, and the one `validate_gex.py`'s header flags as the one that matters.

The residual 40% keeps its sign in 3/3 seeds. That is not nothing, and it is not
significant.

## Two corrections made on the way, both worth keeping

**The state window originally shared calendar with the outcome.** A first version
classified market state from SPY's trailing 21 sessions **ending on the formation
date**, so at h=40 a "selloff" label partly described the drop the forward return
then recorded. Lagging the state window **45 sessions before formation** made the
two disjoint and **cut the effect by a third** — IC −0.1394 → −0.0877, t −4.73 →
−3.44. Not a look-ahead in the strict sense, since only past prices were used,
but state and outcome were not independent.

**The formation window uses a 21-session skip.** Jegadeesh (1990) finds the prior
month reverses at roughly 2.49%/month, so without a skip the winner/loser split
measures last month's bounce rather than momentum.

## Why the sign is opposite to the published finding

George & Hwang (2004) found nearness to the 52-week high **predicts high** future
returns. This measures the reverse. The literature explains both parts:

- **Wrong universe.** Their 1963–2001 sample is equal-weighted and *"implicitly
  overweights micro- and small-cap stocks."* This test runs the **400 most liquid
  US names**. An Australian out-of-sample study found *"the 52-week high strategy
  comprising liquid stocks fails to produce significant dollar profits once
  short-sale restrictions, transaction costs and liquidity constraints are
  accounted for."*
- **The conditional mechanism.** *Momentum Crashes and the 52-Week High*: stocks
  far from their highs *"are prone to speculative demands as investors perceive
  these stocks to have large 'room-to-run'"* and surge when the market is not in
  a normal state. That produces a negative PTH IC exactly in abnormal states —
  which is what the split found.

The mechanism was predicted before the split was run. The conditional structure
was predicted by Chen et al. (2026), who sharpen the effect to an interaction
concentrated among past winners. **Both predictions held; only the magnitude
failed.**

## The one test that would settle it

**Construct it beta-neutral from the start** rather than stripping beta
afterwards: rank PTH *within* beta quintiles, so the comparison is high-PTH
against low-PTH at the same beta. Post-hoc orthogonalisation removes the linear
component; ranking within buckets removes it by construction.

If the effect survives that, it is real and worth a shadow book. If it collapses,
it was beta all along and the axis closes.

Nothing else is needed first. The data is local, the test is cheap, and it is the
only remaining question.

## Files

- `analysis/pth_winner_test.py` — the univariate and market-state split
- `analysis/pth_winner_test_v2.py` — same, state window lagged 45 sessions
- `analysis/pth_selloff_gauntlet.py` — multi-seed, null, beta strip, economic
  test, threshold sweep

## Status

**NOT a signal. NOT closed.** One test from a verdict either way.
