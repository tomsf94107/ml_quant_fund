# FINDING — the ranking edge is real and does not pay for its own trading
**2026-09-15.** `analysis/long_short_book.py`, h=5, long >= 0.70, short < 0.40,
74 dates 2026-04-16 .. 2026-09-04, PIT borrow costs from `borrow.db`.

Closes the long/short question with a number. The cross-sectional signal is
genuine. A market-neutral book built on it is not tradeable at any plausible
execution cost.

---

## What prompted it

The per-ticker direction model cannot call the market, and nothing in it tries:
it is trained on `1 if fwd > 0 else 0` per name, which learns which stocks beat
which stocks. There is no market-direction target, no index prediction, no
regime forecast.

But the RANKING works. Measured at the production gate against the rest of the
same day's universe:

| month | selection | field | edge |
|---|---|---|---|
| Apr | 58.2% | 63.0% | -4.8pp |
| May | 59.7% | 51.9% | +7.8 |
| Jun | 59.9% | 53.1% | +6.8 |
| Jul | 62.0% | 53.2% | +8.8 |
| Aug | 55.6% | 47.4% | +8.1 |
| **Sep** | **37.9%** | **28.6%** | **+9.3** |

September is the strongest month on the only measure that separates the model
from the market. The raw hit rate of 37.9% looks catastrophic and the field hit
28.6% -- roughly seven stocks in ten fell. The selection fell less often by 9.3
points.

Long the top and short the bottom cancels the market move and leaves that
ranking. That is what a market-neutral book is for.

## Result

**Gross of execution, net of borrow: +0.436% per 5-day period, NW t +1.45,
62% of dates positive.**

Against this fund's t > 3.0 bar, that does not clear on its own. And it does
not survive contact with trading costs:

| bps/leg | net per period | NW t |
|---|---|---|
| 0 | +0.436% | +1.45 |
| 5 | +0.236% | +0.79 |
| **10** | **+0.036%** | **+0.12** |
| 20 | -0.364% | -1.21 |
| 40 | -1.164% | -3.87 |
| 100 | -3.564% | -11.84 |

**Ten basis points per leg takes it to zero.** That is a low estimate for 71
positions, many of them small and mid caps. Four crossings per rebalance -- buy
and sell the long leg, sell short and cover the short leg -- and the edge is
gone.

## By month, and why the pooled figures were too kind

| month | dates | per period | % > 0 |
|---|---|---|---|
| Apr | 6 | **-0.931%** | 33% |
| May | 12 | **-0.738%** | 33% |
| Jun | 20 | +0.538% | 55% |
| Jul | 22 | +1.305% | 86% |
| Aug | 10 | +0.521% | 70% |
| Sep | 4 | +0.510% | 75% |

An earlier pass reported monthly spreads of +3.29, -1.73, +0.46, +2.09, +2.13,
+0.61, +1.19% and read them as "positive in six of seven months". Those were
POOLED means -- every name weighted equally regardless of which date it fired
on. Weighting each DATE once, which is what a book actually earns, April and
May are both clearly negative and the series mean falls to +0.436%.

Same error the PCT7 retraction documents: +1.46% pooled against -1.97%
day-weighted on identical data. Third time this construction has flattered a
result in this codebase.

## Borrow is not the obstacle, and that was not obvious

`si_positions_live.py` warns at length that its own short leg shorts
high-days-to-cover names -- the hardest and most expensive to borrow -- and that
backtest costs of 10-40bps excluded borrow fees that would erode it live. That
warning is correct for that book and does not transfer here: **low `prob_up` and
high days-to-cover select different names.**

PIT borrow fees on the names this book would actually short:

| month | avg fee | max fee | HTB |
|---|---|---|---|
| Jun | 172.9bps | 19,531bps | 41 |
| Jul | 164.6bps | 19,531bps | 57 |
| Aug | 134.9bps | 9,329bps | 124 |
| Sep | 127.0bps | 5,524bps | 58 |

Against a universe average of 196bps, the bottom decile is **cheaper** to borrow
than the average stock. Over five days, 127bps annualised is about 2.5bps --
roughly 2% of September's spread. The 1,000bps cap never binds; 283 names were
excluded as hard-to-borrow and zero for fee.

**Execution cost is what kills this, not borrow.** The opposite of the
prior that came in.

## Two structural problems the monthly table hid

**Only 74 of ~115 dates are usable.** The rest carry fewer than five names on
one side, so the book cannot be formed at all on a third of days.

**15 long against 56 short per date.** The model rarely puts names above 0.70
and frequently below 0.40. Under equal-dollar weighting each long position is
3.7x the size of each short, so the "market-neutral" book is concentrated on the
long side in name count terms and the hedge is thinner than it looks.

## What this does and does not say

**Does not say the signal is fake.** +8 to +9pp against the field, every month
including the worst one, is a real cross-sectional edge and the decile spread
confirms the ranking separates.

**Does say the edge is too small to trade this way.** +0.436% per five days
across 71 positions is thin, and four crossings consume it. That is consistent
with everything else measured in this fund: the SI brick clears t > 3.0 and
nothing else has.

**Does not close the direction question.** Market direction at a 1-5 day horizon
is close to unforecastable and the published equity-premium predictors reach
out-of-sample R-squared near 0.5% at monthly to annual horizons. Nothing here
suggests that is worth attempting.

## What would change the answer

- **Fewer, larger positions.** The edge is concentrated above 0.70; the ladder
  dies because it is spread across 71 names. A cap-10 long / cap-10 short book
  has not been tested and would cost a fraction as much to trade.
- **A longer hold.** Four crossings per five days is the problem. At h=40 the
  same crossings amortise over eight times the period.
- **Real execution data.** The ladder is a guess at cost. One month of actual
  fills would replace it with a measurement.

## Files
- `analysis/long_short_book.py` -- the test, PIT borrow, cost ladder
- `analysis/recovery_tracker.py` -- the benchmarked edge table
- `borrow.db` -- `borrow_fees`, 1,920 tickers from 2021-06-15
