# FINDING — the h=40 dispersion was DRAW size, not UNIVERSE size
**2026-09-13.** `analysis/h40_book_test.py`, 3 seeds, h=40, cap-3, 2021-06-01 onward.
Two arms, 35 and ~30 minutes, ~465-480k rows and 564 rebalances per seed in both.

This revises `FINDING_horizon_h40_2026-09-05.md` and removes the stated basis for
expanding the traded universe. It does not settle whether a wider TRAINING panel
helps — that test still has not been run, and the reason is recorded below.

---

## What was claimed

`analysis/universe_expand.py` opens with the expansion case:

> "The cross-sectional signal found at h=40 on 2026-09-05 shows wide dispersion
> across ticker samples — prob>=0.70 ranged +2.14pp to +5.46pp over three draws
> of 80 names. That is what a thin cross-section looks like."

and concludes that "universe size is plausibly a bigger lever than any estimator
change."

## What the arms measured

**ARM A — `tickers.txt` (422 names), draws of 400, 3 seeds**

| seed | cap-3 | NW t | drawdown | neg rebalances | top-5 share | distinct |
|---|---|---|---|---|---|---|
| 1 | +1.435pp | +2.03 | -10.6% | 51% | 27% | 122 |
| 2 | +1.307pp | +2.07 | -12.2% | 48% | 27% | 121 |
| 3 | +1.887pp | +3.04 | -10.2% | 50% | 26% | 127 |
| **agg** | **+1.543pp** | | -11.0% | 50% | 27% | |

**ARM B — `tickers_expanded.txt` (1,924 names), draws of 400, 3 seeds**

| seed | cap-3 | NW t | drawdown | neg rebalances | top-5 share | distinct |
|---|---|---|---|---|---|---|
| 1 | +5.449pp | +7.41 | -6.6% | 36% | 42% | 92 |
| 2 | +3.839pp | +5.11 | -6.4% | 35% | 41% | 88 |
| 3 | +4.024pp | +5.73 | — | — | — | — |
| **agg** | **+4.437pp** | | -6.9% | 36% | 38% | |

## Finding 1 — the dispersion is draw size, and the claim misread it

| | draw | cap-3 across seeds | spread |
|---|---|---|---|
| 2026-09-05 | 80 names | +2.14 to +5.46pp | **3.32pp** |
| ARM A | 400 names | +1.307 to +1.887pp | **0.58pp** |

Same universe file, same model, same horizon, same period. **The only change is
how many names each seed draws, and the seed spread fell about six-fold.**

Arm A draws 400 from 422, so its three seeds share roughly 95% of their names and
their agreement is partly mechanical. That does not rescue the original claim,
because the direction is what matters: the noise tracks SAMPLE size within one
pool, and the 80-name draws came from that same 422-name pool. Nothing in the
2026-09-05 dispersion was evidence that 422 names is too thin a UNIVERSE. It was
evidence that 80 is too thin a SAMPLE.

**The expansion case rested on reading one for the other.**

## Finding 2 — the h=40 headline was carried by a noisy seed

+1.70pp was the aggregate of a range 3.32pp wide. On stable draws the same
construction gives **+1.543pp**, and the +5.46pp seed that made it look strong is
inside the noise the draw size produced.

The mean was roughly right. The range was meaningless, and it was the range that
motivated a plan to expand the universe.

## Finding 3 — cap-3 is marginal after costs, and cap-5/cap-10 dominate it

Cost ladder on arm A's cap-3, +1.543pp gross at ~27% turnover, one round trip
per rebalance:

| bps/leg | net | approx annualised at ~6.3 rebalances |
|---|---|---|
| 0 | +1.543pp | ~+9.7% |
| 10 | +1.343pp | ~+8.5% |
| 20 | +1.143pp | ~+7.2% |
| **40** | **+0.743pp** | **~+4.7%** |
| 100 | -0.457pp | negative |

With 50% of rebalances negative and an -11.0% max drawdown. **Only one of three
seeds clears the fund's own t>3.0 bar** (+2.03, +2.07, +3.04).

And within arm A, wider caps beat cap-3 on every axis:

| selection | excess | NW t (seed range) | turnover |
|---|---|---|---|
| cap 1 | +0.559pp | +0.41 to +0.89 | ~41% |
| cap 3 | +1.543pp | +2.03 to +3.04 | ~33% |
| **cap 5** | **+2.202pp** | **+3.03 to +4.94** | ~31% |
| **cap 10** | **+2.186pp** | **+4.23 to +5.01** | ~27% |

The script's own note says a selection matching cap-3's excess with MORE names is
the better book — same edge, less idiosyncratic risk. Cap-5 and cap-10 exceed it
with better t-stats and lower turnover. **The shadow book runs cap-3.** Nobody
has acted on this and it predates tonight.

## Finding 4 — arm B measures the SLICE, not the panel

Arm B is 2.9x arm A with better t-stats and a shallower drawdown, and it is not
the estimation win it resembles.

`FINDING_survivorship_bound_2026-09-09.md` measured the h=40 edge as monotone in
illiquidity: mega $2,139M ADV **+1.721pp**, large +1.972pp, mid +2.998pp, small
$53M ADV **+4.932pp**. Arm B lands at **+4.437pp** — in the small band, because
1,512 of `tickers_expanded.txt`'s 1,924 names are mid and small.

**Arm A samples large caps. Arm B samples mostly small ones. The difference is
which names, not how many**, and that difference was already measured and already
argued against trading, by the same bound test: the +20.2pp annualised gap between
slices is roughly thirteen times the upper peer-reviewed survivorship estimate,
and the same literature finds ML edge in small names collapses under costs.

**The concentration columns confirm it.** Arm B picks from 88-92 distinct names
against arm A's 121-127, with top-5 names taking **38%** of all picks against 27%.
And `cap 1` swings +9.705pp / +2.944pp / +5.578pp across seeds — a 3.3x spread on
the single highest-conviction pick. A handful of small names carry the result.

## The test that still has not been run

Train on 1,924, **predict and score on the same 415 traded names.** Same scoring
universe both arms, so any difference is the panel rather than the slice. That is
the only form that tests the estimation claim, and it is what the expansion case
actually asserts.

`h40_book_test.py` cannot do it: `--universe` feeds one list, the sampler shuffles
by seed and takes `[:args.tickers]`, and the same draw is both fitted and scored.
A train/score split is a small change — the model fits whatever panel it is given
and ranks whatever names it is asked to rank — but it is an edit to a backtest
script, which is where look-ahead enters.

**Until that runs, there is no measured support for expanding the traded universe
and none for expanding the training panel either.** What exists is a measurement
that the original motivation was a sampling artifact.

## Unchanged

The h=40 shadow book stays frozen on the current 415 names, sha
`9ef0cdd954dfd6ad`, clock from 2026-09-05. `h40_book_test.py`'s `--universe` flag
exists precisely so the expanded set can be tested without swapping `tickers.txt`,
which every cron job reads. Neither arm touched it.

## Files
- `analysis/h40_book_test.py` — both arms, `--universe` and `--tickers`
- `logs/h40_B400.log` — arm B full output
- `FINDING_horizon_h40_2026-09-05.md` — revised by findings 1-3
- `FINDING_survivorship_bound_2026-09-09.md` — the slice table finding 4 rests on
