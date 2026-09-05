# CANDIDATE — h=40 cross-sectional book

**2026-09-05.** XGBoost on the existing feature set, ranked cross-sectionally,
top 3 names per rebalance, 40-day hold. **+1.70pp excess per period** over the
same day's universe, 2021–2025, positive in four of five years.

**Status: candidate, not a brick.** It has no out-of-sample record and its
survivorship exposure is unsized. Both are addressable and neither is done.

Reproducible via `analysis/horizon_sweep_test.py`, `analysis/h40_book_test.py`,
`analysis/h40_yearly_test.py`, `analysis/si_overlap_test.py`.

---

## The headline shrank by two thirds, and every reduction was earned

| stage | cap-3 | what the check removed |
|---|---|---|
| first pass, 2021–2026 | **+5.13pp** | — |
| widened to 2016–2026 | **+2.91pp** | one regime |
| excluding 2020 and 2026 | **+1.70pp** | COVID, and a partial year whose 40-day windows almost fully overlap |

The first figure was a bull-market sample. The second included a COVID year
where a model trained through 2019 and tested in March 2020 produces extremes in
both directions, and a 2026 stub where 246 rebalances hold perhaps three
independent outcomes.

**+1.70pp is the number.** Anything larger is an artifact of the window.

## What it is

Rank the universe each rebalance by the model's probability, take the top 3,
hold 40 days, measure against the mean return of that same day's universe.

| | |
|---|---|
| excess per period | **+1.70pp** (median +0.57pp) |
| rebalances positive | **52%** |
| periods per year | ~6.3 |
| annualised excess | **~+10–11%** before costs |
| turnover | 27% |
| net at 20bps/leg | ~+9% |
| max drawdown | −11.4% |
| distinct names used | 53–74 per seed |
| top-5 tickers | 28% of picks |

**The mean is not carried by outliers.** Median +0.57pp against a mean of
+1.70pp is skewed, but the median is POSITIVE — more than half of rebalances
make money. That is the opposite of PCT7, retracted the same day, where 38 of 64
periods lost and six days held the entire result.

## Why cap-3 and not the probability gate

`prob>=0.70` looked stronger at first — +3.86pp against cap-3's +2.90pp on the
full sample — and it has the appeal of being the EXISTING production gate.

**It fires on nothing in 34% of rebalances.** So a book gated that way sits in
cash a third of the time, and its +1.65pp is conditional on firing. Unconditional
that is roughly +1.09pp per period, BELOW cap-3's +1.70pp, which always trades.

When it does fire it selects a median of 7 names (max 77), so it is also a
lumpier book.

**Cap-3 is the better instrument once idle time is counted.** That reverses the
reading from the per-period table alone, and it is the kind of thing a
conditional average hides.

## By year, pooled over three seeds

| year | n | cap-3 | positive |
|---|---|---|---|
| 2020 | 696 | +5.26pp | 59% |
| **2021** | 756 | **−0.77pp** | **40%** |
| 2022 | 753 | +1.96pp | 57% |
| 2023 | 750 | +1.99pp | 53% |
| 2024 | 756 | +1.74pp | 54% |
| 2025 | 750 | +3.59pp | 59% |
| 2026 | 246 | +8.92pp | 59% |

**Four of five clean years positive. 2021 negative in 0 of 3 seeds** — the only
unambiguous failure, and it was the meme-stock and retail-flow year, when
price-based signals broke down broadly. That reads as a known conditionality
rather than a flaw, but it is a real one: this should be expected to fail in
speculative retail-driven markets.

## The horizon is the finding, not the model

| horizon | cap-3 per day | turnover | AUC |
|---|---|---|---|
| 3 | 0.023pp | 61% | 0.4796 |
| 5 | 0.008pp | 51% | 0.4778 |
| 20 | 0.071pp | 29% | 0.5153 |
| **40** | **0.130pp** | **27%** | **0.5772** |

XGBoost is BELOW random at h=3/5 and 0.577 at h=40 — its only above-random AUC
in the sweep. Linear is the reverse, best at short horizons and negative at
h=20/40.

That is consistent with the mechanism. Gu, Kelly & Xiu (2020) trace tree gains
to nonlinear predictor interactions; interactions need signal to estimate, and
at daily SNR (~0.8%, return sd 3.49% against a mean 0.028%) there is not enough
— which is why an L2 sweep the same day improved monotonically as coefficients
were shrunk toward zero. At h=40 there is enough.

**So the twenty configurations that failed on 2026-09-05 — six insider
constructions, four label definitions, two estimator classes, ensembles,
consensus voting — were all tested at the wrong horizon.** Several deserve a
rerun at h=40 before staying closed.

## It is not the SI brick relabelled

The fund's one validated edge also runs at h=40, so this had to be ruled out.

| | |
|---|---|
| overlap of top-3 with the lowest-DTC quintile | 32.2% (random gives 19.5%) |
| Spearman(model probability, days_to_cover) | −0.060 |
| excess excluding every SI long-leg name | +3.784pp, 3/3 seeds |
| **excess retrained with short_ratio and ALL its derivatives removed** | **+5.376pp, 3/3 seeds** |

Removing short interest entirely leaves it slightly STRONGER. The 32% overlap is
a mild tilt toward low-DTC names, not the mechanism. This matters because
`short_ratio` IS days-to-cover in this feature set and was revived from a
broadcast constant to a live PIT join on 2026-08-26, going from 0.000 to 5.119
importance — so the two could easily have been one signal.

## What is NOT established

- **No out-of-sample record.** Everything here is a backtest. PCT7's accidental
  fourteen unscored weeks on a frozen model turned out to be the strongest
  evidence in this system, precisely because nobody could touch it.
- **Survivorship is unsized.** `prices.db` holds the names that exist today;
  everything that delisted 2016–2026 is absent. `si_leg_decomp.py` sized this for
  the SI brick and found it near-moot there (79% of that edge in low-DTC names,
  which rarely delist). A model picking six high-conviction names could be
  selecting exactly the volatile ones where delisting bites.
- **~30 independent periods** across five clean years. Seed dispersion is wide:
  prob≥0.70 ranged +2.14pp to +5.46pp across three ticker draws.
- **80-ticker samples.** The live universe is 420. Breadth was not tested.
- **Costs are a flat ladder**, not modelled per name.

## Next, in order

1. **Shadow-log it, frozen.** Train once, write daily picks, never retrain,
   score nothing for at least six months. At h=40 three months gives only ~3
   independent outcomes, so the wait is long by construction — start it now.
2. **Size survivorship**, the treatment `si_leg_decomp.py` gave the SI brick.
3. **Rerun the closed axes at h=40** — the four label definitions especially,
   which were only ever compared at h=5.
4. **Widen the universe.** `analysis/universe_expand.py` shows 427 of 452
   ingested tickers pass a liquidity screen; the target is ~2,000 and the
   binding constraint is the fetch, not the screen.

**The kill switch stays on.** A backtested candidate is not a reason to re-enable
BUYs, and the two unestablished items above are exactly what turned PCT7 from
"brick #2" into a retraction earlier the same day.
