# FINDING — the h=40 estimate is unstable in the TRAINING panel
**2026-09-13.** `analysis/h40_book_test.py`, four arms, 3 seeds each, h=40, cap-3,
2021-06-01 onward, 564 rebalances per seed in every arm.

Supersedes nothing. Extends `FINDING_h40_dispersion_is_draw_size_2026-09-13.md`
with the arm that was missing from it, and the answer is not the one that test
was set up to find.

**The expansion of the universe is contraindicated in both forms — traded and
training — and the reason for the second is new.**

---

## The four arms

| arm | model trains on | ranking restricted to | cap-3 | seed spread |
|---|---|---|---|---|
| A | 400 drawn from `tickers.txt` (422) | the same 400 | **+1.543pp** | **0.58pp** |
| B | 400 drawn from `tickers_expanded.txt` (1,924) | the same 400 | +4.437pp | 1.61pp |
| C | 400 drawn from expanded | ~87 (the traded names in that draw) | +2.715pp | 3.04pp |
| D | 800: **411 pinned traded names** + 389 random expanded | the same 411, every seed | +3.305pp | **4.17pp** |

Row counts confirm the panels built as intended: A ~479k rows/seed, B and C
462-468k, D 939-947k — roughly double A, as an 800-ticker panel should be.

## Arm D per seed

| seed | cap-3 | NW t | cap-5 | cap-10 | turnover |
|---|---|---|---|---|---|
| 1 | +2.241pp | +3.14 | +1.477pp | +0.693pp | 37% |
| 2 | +1.752pp | +2.68 | +1.851pp | +1.960pp | 41% |
| 3 | **+5.923pp** | +5.88 | +4.043pp | +3.011pp | 29% |
| agg | +3.305pp | | +2.457pp | +1.888pp | |

## The finding

**Arm D scored the SAME 411 names in every seed.** The pinned set is identical
across seeds by construction; only the 389 random training companions varied.

That produced a **4.17pp spread** — wider than any arm in which the scored names
themselves changed.

| what varied between seeds | spread |
|---|---|
| A — which 400 of 422 are trained AND scored | 0.58pp |
| B — which 400 of 1,924 are trained AND scored | 1.61pp |
| C — which 400 are trained; ~87 scored | 3.04pp |
| **D — ONLY which 389 extra names are trained on** | **4.17pp** |

**The model's measured edge on a fixed book depends materially on which arbitrary
other companies it was fitted alongside.** Seed 3 says +5.923pp at t +5.88; seed 1
says +2.241pp at t +3.14. Both are "significant" and they differ by a factor of
2.6 on identical picks over an identical period.

This is a stronger statement than the estimation claim being unsupported. The
expansion case argued a wider panel would SHARPEN the estimate. Measured, a wider
panel made the estimate depend on its own composition.

## Why arm C could not answer it, and why arm D needed pinning

422 traded names in a 1,924 pool is 22%, so a 400-name draw from the expanded
file contains only about 87 of them — measured 86, 94, 82 for seeds 1-3. Arm C
therefore ranked cap-3 from ~87 candidates (3.4% selectivity) against arm A's 400
(0.75%). A lower score in C would have been selectivity, not the training panel.

Arm C's own dispersion confirms the mechanism the earlier finding identified:
seed 1 +0.703pp at t +0.94 — not significant — against +3.745 and +3.698 at t
+4.83 and +5.70. **Ranking from ~87 names reintroduced exactly the noise that
moving from 80-name to 400-name draws had removed.** Consistent with the earlier
result that the noise tracks how many names are being ranked, and the reason
`--pin-score` was added.

## What this does and does not change

**Does not change:** the h=40 edge exists. Every arm is positive, 3/3 seeds, and
the daily book, the shadow book and live predictions are untouched. No model was
modified; these are throwaway fits.

**Does change: the confidence attached to any single backtest number from this
construction.** A model whose measured excess moves 4pp on training-set
composition should have its point estimates read as one draw from a wide
distribution, not as a measurement.

**Settles the universe question in both directions:**

- **Traded universe.** Arm B's +4.437pp is the illiquidity slice, not a better
  estimate. `FINDING_survivorship_bound_2026-09-09.md` measured the same edge
  monotone in ADV — mega +1.721pp to small +4.932pp — and 1,512 of the expanded
  set's 1,924 names are mid and small. Arm B lands in the small band. That gap is
  roughly thirteen times the upper peer-reviewed survivorship estimate, and the
  same literature finds ML edge in small names collapses under trading costs.
- **Training panel.** Arm D is the clean test — same scored names, same
  selectivity, only the training companions differ — and it shows instability,
  not improvement. There is no measured case for training wide either.

## What the traded book actually earns

From arm A, the only arm that both trains and scores on the live universe:
+1.543pp gross per 40-day period, ~27% turnover, one round trip per rebalance.

| bps/leg | net | approx annualised at ~6.3 rebalances |
|---|---|---|
| 0 | +1.543pp | ~+9.7% |
| 20 | +1.143pp | ~+7.2% |
| **40** | **+0.743pp** | **~+4.7%** |
| 100 | -0.457pp | negative |

With 50% of rebalances negative, -11.0% max drawdown, and only one of three seeds
clearing the fund's own t>3.0 bar (+2.03, +2.07, +3.04).

## An open question this raises

In arm A, cap-5 (+2.202pp, t +3.03 to +4.94) and cap-10 (+2.186pp, t +4.23 to
+5.01) beat cap-3 on excess, t-stat AND turnover. In arm D the order reverses and
cap-3 wins — but that reversal is carried by seed 3, the outlier.

**The shadow book runs cap-3.** Whether that is the right selection is unresolved
and is now entangled with the instability above: the arms disagree about cap
ordering, and the disagreement tracks the training draw. Changing the shadow
book's selection would restart its clock, so this is a decision rather than a fix.

## Method note

`--score-universe` and `--pin-score` were added to `h40_book_test.py` for arms C
and D. Both touch the TEST keys only — `ktr` is unchanged, so nothing the model
learns changes; only which names compete for the cap slots. Default behaviour is
byte-identical, verified against a 60-ticker single-seed run before use.

Arm D briefly overlapped arm C for about 30 seconds before being killed. The
lock-contention failure mode is silent — `except Exception: continue` swallows a
failed build — so row counts were checked against arms A and B afterward and
matched, confirming no seed was corrupted.

## Arm D's full path, and one correction it forces

| bps/leg | D net | A net |
|---|---|---|
| 0 | +3.305pp | +1.543pp |
| 20 | +2.905pp | +1.143pp |
| 40 | +2.505pp | +0.743pp |
| 100 | +1.305pp | **-0.457pp** |

D path: max drawdown **-8.5%**, **43%** of rebalances negative, top-5 tickers 29%
of picks. Against A's -11.0%, 50% and 27%.

**D survives 100bps/leg where A does not.** On the cost ladder alone the wide
panel looks better on every rung, with a shallower drawdown and fewer losing
rebalances.

**That does not rescue the expansion case, and the reason is the whole point of
this note.** D's aggregate is one number drawn from +2.241 / +1.752 / +5.923. At
seed 2's +1.752pp the 40bps net is roughly +0.95pp -- no better than arm A -- and
at 100bps it is close to nothing. The aggregate looks robust only because seed 3
carries it. A cost ladder computed on an unstable mean inherits the instability
and hides it behind smooth-looking arithmetic.

**The threshold rows say the same thing louder.** `prob>=0.5` is **-0.043pp and
positive in only 1 of 3 seeds**; `prob>=0.55` is +0.055pp at 2/3. The model's
probability calibration on the wide panel is close to useless below 0.6 --
selecting 342 names a day at prob>=0.5 earns nothing. Only `prob>=0.7` (+3.950pp,
3/3, 49 names/day) and the caps work at all.

So the wide panel does not produce a better-calibrated model. It produces a model
whose usable output is confined to the extreme tail, with a mean that depends on
which companies happened to share its training draw.

## Files
- `analysis/h40_book_test.py` — all four arms; `--universe`, `--score-universe`, `--pin-score`
- `logs/h40_C400.log`, `logs/h40_D800.log`
- `FINDING_h40_dispersion_is_draw_size_2026-09-13.md` — arms A and B
- `FINDING_survivorship_bound_2026-09-09.md` — the ADV slice table
- `FINDING_horizon_h40_2026-09-05.md` — the original +1.70pp
