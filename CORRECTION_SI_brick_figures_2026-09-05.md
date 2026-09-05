# CORRECTION — SI brick figures overstated in the record

**2026-09-05.** The short-interest brick is **real, tradeable, and not decaying**.
Its recorded numbers are too high. Reproducible via
`si_dissemination_lag_test.py`, `si_leg_decomp.py`, and
`analysis/si_period_split.py`.

---

## What the record says vs what it measures

| metric | recorded | measured 2026-09-05 |
|---|---|---|
| per-date IC | **−0.054** | **−0.0407** (lag 0) / **−0.0368** (lag 8) |
| NW-t | **−4.46** | **−3.27** (lag 0) / **−3.12** (lag 8) |
| long-leg Sharpe | **~1.25** | **0.83** |
| null rejection | 8.3σ | not re-run |
| long/short split | 71% / 29% | 79% / 21% |

`si_dissemination_lag_test.py`'s own docstring cites IC −0.053 / NW-t −4.73, so
the record and the script's header agree with each other and both disagree with
the live run. The gap is roughly **28% of the IC**.

## It is not decay

That was the first hypothesis, and it was wrong. Per-date IC by half-sample,
same construction as the lag test:

| period | lag 0 | lag 8 |
|---|---|---|
| first half (63 dates) | −0.0395, t −3.25 | −0.0361, t −2.63 |
| second half (63 dates) | **−0.0419**, t −1.94 | **−0.0375**, t −1.99 |

The second half is marginally **stronger**. By year, at lag 0:

| year | dates | mean IC | right-sign |
|---|---|---|---|
| 2021 | 18 | −0.0349 | 72% |
| 2022 | 24 | −0.0312 | 62% |
| 2023 | 24 | −0.0497 | 71% |
| 2024 | 24 | −0.0388 | 75% |
| 2025 | 24 | −0.0779 | 79% |
| 2026 | 12 | **+0.0201** | **42%** |

Stable and correctly signed for five years. **The edge has always been about
−0.038; the recorded −0.054 came from a shorter or luckier window.**

## 2026 is the thing to watch, and it is not yet a finding

2026 shows IC **+0.0201** — the wrong sign — with right-sign at 42%. But that is
12 rebalances at t=+0.30. At hold=40 a year holds only ~6 independent
observations, so this is noise. It becomes meaningful only if the sign stays
positive through 2027, and it should be checked each quarter rather than
reacted to now.

## What still holds

- **Publication lag is survivable.** FINRA disseminates ~8 business days after
  settlement, so entry at settlement is not tradeable. At lag 8, IC −0.0368 and
  NW-t −3.12 — about **90% of the edge survives**, confirmed independently in
  July (commit 4c805bd3) and again today.
- **Survivorship is nearly moot.** 79% of the edge is in the LOW days-to-cover
  long leg, and low-DTC names rarely delist. The survivor-tilted price history
  barely touches them.
- **Right-sign 66% at the tradeable lag**, across 125 rebalances.

## A methodological note, because it cost an hour

A first attempt at the period split reimplemented the decomposition and got
long/reb **+0.57%** against `si_leg_decomp.py`'s **+1.28%** — less than half.
Three differences: it read `raw_bars.close` instead of `daily_prices.adj_close`,
it entered on the first bar **strictly after** settlement rather than **at**
settlement (`d + 0..5`), and it applied an arbitrary `abs(r) < 3.0` outlier
filter.

Its year-by-year output looked plausible and was meaningless.

**A reimplementation that does not reproduce the baseline cannot answer a
question about the baseline.** The FULL row must match before any split below it
is read. `analysis/si_period_split.py` reproduces the lag test to within
0.003 IC and is the version to use.

## What should change

- **The record.** IC −0.037, NW-t −3.12, long-leg Sharpe 0.83 at the tradeable
  lag. Still significant, still the fund's one validated edge, smaller than
  documented.
- **Nothing about the book.** No decay, no leak, no reason to stop.
- **Add a quarterly re-run.** These figures drifted 28% between July and
  September with nobody checking. `analysis/si_period_split.py` should run on a
  schedule alongside the other Monday monitors.
