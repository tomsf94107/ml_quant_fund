# System Audit · TODO Reconciliation · Research Plan

*ML Quant Fund · 2026-08-21 (rev 2) · verified against the repo, not the docs*

> **rev 2 supersedes rev 1.** §3 of rev 1 claimed "three measurements, one
> mechanism: the tails invert." That was **wrong as stated** and is replaced by
> §3 below. The correction is documented rather than quietly edited, because the
> way it was reached is itself the most useful thing in this document — see §7.

---

## 0. The headline

Three facts define the system's state. None appear in
`Gap_Check_and_Roadmap.md` (Apr 29) or `The_Complete_Quant_Build_Menu` — both
materially stale.

| # | Finding | Evidence |
|---|---|---|
| 1 | **The direction model is OFF.** 26,571 predictions in 30 days, **every one HOLD, zero BUY** | `ML_QUANT_DISABLE_BUY` defaults `1`; not set in `.env` |
| 2 | **Momentum breached its KILL LINE.** Live edge **−10.95pp** vs 99% bound −6.68pp — "INCONSISTENT with the 18yr backtest, not merely unlucky" | `momentum_promotion_check.py` |
| 3 | **The decile spread FLIPPED SIGN in June 2026** — and three things changed at once, so it cannot be attributed | §3 |

**The system currently trades one thing: the SI brick.** Everything else is
shadow, disabled, or killed. That is a defensible state — it is what honest
validation produces — but it should be stated plainly.

---

## 1. What the stale docs get wrong

`The_Complete_Quant_Build_Menu`'s top-10 build order is largely **already built
and tested**, including both of its starred priorities:

| Menu priority | Menu says | Reality |
|---|---|---|
| #1 Learning-to-rank | "the most important untested model" | **Killed 2026-07-09** — purged-WF FAIL 5d −0.52; the +5.6 Sharpe was an in-sample leak |
| #2 Alpha combiner / HRP | "the highest-value untested thing in your entire system" | **Built** (`c1_combiner.py`), **ran**, verdict **NO-SHIP** |
| #3 Linear/ridge baseline | untested | `models/linear_baseline.py` exists — **referenced by 0 files** |
| #4 PCA residual reversal | "possibly the best reframe" | Built; crude version "failed full-history validation (net −0.43)" |
| B3/B4 Value + Quality | "needs build" 🔨 | **Built and validated** — `qv_validate.py`: gp, op, bm, ep |
| B10 Lazy Prices | "needs build" 🔨 | Built Jun 1, **validated and KILLED Jun 3** |

**`docs/MASTER_TODO_LIST.md` is the plan of record** — its own document map says
so, and that if a satellite conflicts, MASTER_TODO wins. The two files in
`/mnt/project/` are satellites that have drifted.

**Action:** archive or re-date both. This session began by treating LTR and the
combiner as open work; both were resolved months ago.

---

## 2. Why C1 failed — the finding that still stands

```
       mom    gp    op    ep
mom   1.00  0.59  0.61  0.57
gp    0.59  1.00  0.81  0.70
op    0.61  0.81  1.00  0.86
ep    0.57  0.70  0.86  1.00
```

The four books correlate **0.57–0.86**. Grinold's IR = IC × √breadth × TC needs
**decorrelated** bets; at ρ ≈ 0.8 you hold one bet in three costumes.
Combination is mathematically incapable of helping.

Backtest, walk-forward weights, pre-registered gate:

```
mom_only  Sharpe +1.53  maxDD -29.9%   ew  +1.40   ivol +1.31   hrp +1.22
VERDICT: NO-SHIP — momentum-alone stays the system
```

**The combiner is not broken. The inputs are redundant.** That reframes the alpha
hunt: the constraint is not "test more signals," it is **"find signals
decorrelated from momentum."** Every family tested so far — momentum, quality,
value, profitability — is a variant of the same price/fundamental complex.

**This is the single most actionable finding in the document.**

---

## 3. The decile spread — what the data actually shows

*(replaces rev 1 §3, which claimed a universal tail inversion)*

Equal-weighted per date, h=5, watchlist excluded, 365-day window:

| Era | Dates | D10 − D1 | NW-t (lag 5) |
|---|---|---|---|
| Narrow universe (≤ May 2026) | 44 | **−1.5633%/5d** | **−2.31** |
| Wide universe (Jun 2026 +) | 51 | **+1.3318%/5d** | +1.82 |
| Full sample | 95 | −0.0968% | −0.19 |

**The full-sample zero is two large opposite effects cancelling — not a flat
signal.** The extremes *were* inverted pre-June at t = −2.31, consistent with
MASTER_TODO §1.2 ("inverted at the confident extremes"). Post-June they are
positive but do not clear the t > 3 bar.

**Monotonicity is positive in both windows** (`prob_up` +0.273, `prob_raw`
+0.733), so the earlier "tails invert everywhere" reading was wrong.

### 3.1 Why the flip cannot be attributed

Three things changed simultaneously in June 2026:

1. **Universe** 149 → 394 names
2. **Training depth** 2022 → 2016 (commit `75ae24a9`: folds 9 → ~30, "one regime → five")
3. **Market regime** — mean 5d return 2.80% (Mar) → 0.38% (Jun)

A sign flip coinciding with all three is **unattributable**. It could be a
genuinely better model on deeper training data, a wider cross-section giving the
ranking room to work, or inverted momentum simply ceasing to lose once the
melt-up stopped. **Do not treat the +1.33% as an edge discovery.**

### 3.2 A weighting trap worth naming

Pooling all observations instead of averaging per-date decile means gives
**+0.96%** where the equal-weighted answer is **−0.10%** — a 1.06pp swing on the
same rows. Cause: names-per-date grew 101 → 394, and the wider universe has
larger cross-sectional dispersion, so pooling loads onto the era with mechanically
bigger decile gaps.

**The equal-weighted number is the tradeable one** — you rebalance once per date
regardless of how many names exist.

This is the same error documented in `Two_Brick_Findings` §5.1 (pooling
stock-dates as independent → fake t = −20 on the SI brick). It recurred here.

---

## 4. TODO reconciliation

### 4.1 Overdue / unresolved dated commitments

| Item | Status |
|---|---|
| **Momentum verdict (Jun 29)** | **KILL LINE BREACHED**, still deferred — §4.2 |
| `risk_gate.py` live UW fetch | pending (FDA endpoint stale, Events page deleted) |
| `options_flow.py` Greeks integration | **now unblocked** — 110,321 rows loaded 2026-08-20 |
| `insider builder.py` | pending |
| Dark-pool skew monthly verdict (~Aug 12) | UNDERPOWERED, unresolved |
| VRP (Aug checkpoint) | logging fixed, accruing — verdict not run |

### 4.2 The momentum contradiction — decide it

```
!! KILL LINE: edge -10.95pp < 99% bound -6.68pp -- shadow INCONSISTENT
   with 18yr backtest, not merely unlucky
VERDICT: NOT YET -- keep momentum in shadow. Re-run weekly.
```

**Those two lines contradict each other.** A kill-line breach is terminal;
"re-run weekly" is a deferral. As written, momentum can fail its own kill test
indefinitely — and has now done so ~7 weeks past the pre-committed verdict date.

**Recommended:** make the kill line terminal in the script, then decide
explicitly — retire momentum, or document why the breach is overridden. Leaving
it ambiguous is the worst of the three options.

### 4.3 Orphaned assets (built, referenced by nothing)

| File | References |
|---|---|
| `models/linear_baseline.py` | **0** |
| `models/vol_forecast.py` | 1 |
| `models/train_panel_ranker.py` | 1 |
| `models/importance_tracker.py` | 1 |

**Correct the `vol_forecast` record:** commit `bd501f29` reports "OOS Spearman
+0.706", but the file states `log(trailing_vol_20d)` **alone** scores +0.545. The
model's lift is ~+0.16, not +0.706. Useful for sizing; not a brick. The headline
reads 4× the result.

### 4.4 From this session (13 research reports audited)

**Fixed:** sector-ETF resolution (411 tickers; `TICKER_CONFIG` covered 12) · 13F
snapshot-date display · AVB/EQR merger → VMRK · `sector_rel_ret` silent zero ·
pre-market spot anchor (two independent paths) · `dma_levels.py` built — nothing
computed per-ticker moving averages before.

**Still open:** dark-pool date bucketing (SMCI, 10 cycles) · news 404 / keyword
garbage (8 reports; self-described "longest-standing costly defect") · 10-Q body
extraction · fiscal-quarter label · earnings-date vendor conflicts.

**Four false ledger entries** — 13F "stale", block-threshold "malformed", GFS
drift "false positive", NVDA 200-DMA "stale" (0.4% off; the report compared it
against the 50-DMA). All four are one failure: **a value rendered without the
context needed to read it.**

---

## 5. Research plan — ranked by (value × evidence) ÷ cost

### R1. Test the options-Greeks axis · **1–2 days, data already on disk**
110,321 rows / 405 tickers / 13 months of dealer positioning (GEX/DEX/vanna/charm),
loaded 2026-08-20, **never tested**. §2 says the binding constraint is
decorrelation; this is the strongest candidate on disk. `backfill_greeks.py`'s own
docstring: *"Every signal in this system so far is derived from PRICE and VOLUME.
That is one information axis, and it is the most crowded axis in finance."*

Test with the existing harness — `validate_confidence_filter.py` (per-date
rank-IC, NW, block bootstrap, shuffle null, FDR, net-of-cost) plus
`decile_monotonicity.py`. **Check its correlation with momentum first** — if
ρ > 0.5, it is not the decorrelated axis and the rest is moot.

### R2. Add decile monotonicity to the alpha gate · **1 day**
`alpha_fitness` scores `rank_ic, ic_t, sharpe, turnover, fitness` — no
monotonicity. 1,222 alphas pass |t|>3 while the top 20 show IC +0.04 with Sharpe
−0.51. Nothing distinguishes tradeable IC from mid-book IC. Add `mono` as a
column and re-score; expect the survivor count to fall sharply, which is the point.

### R3. Decide momentum · **1 hour**
§4.2. Blocks C1, promotion mechanics, and the vol-sizing work behind it.

### R4. Let `ic_history` accumulate · **zero effort, already cron'd**
The monthly job logs h=1/3/5, raw and bucket-neutral, cost-aware, on the recent
window. The June+ spread (+1.33%, t=1.82) and the earlier h=5 net (+0.94%/5d,
t=2.75 on 10 rebalances) are the same effect. It needs **more dates, not more
analyses** — see §7.

### R5. Linear/ridge baseline · **hours**
`linear_baseline.py` exists, referenced by nothing. Gu-Kelly-Xiu found regularized
linear competitive with ML on low-SNR equity data, and it cannot memorise noise
the way trees do. Cheapest remaining model test.

### R6. What to STOP
- More price/fundamental families — ρ 0.57–0.86 with what you have.
- Reviving LTR (killed) or Lazy Prices (killed Jun 3).
- Building from the Build Menu's order without checking the repo first.

---

## 6. The honest position

Seven signals tested and killed, one survivor (SI), one candidate that just
breached its kill line. The direction model is off. The combiner works but its
inputs are redundant. The decile spread flipped sign at a boundary where three
things changed at once.

**This is what a correctly-run research programme looks like when the answer is
mostly "no."** The infrastructure — purged walk-forward, HLZ t>3, PBO, shuffle
nulls, block bootstrap, net-of-cost gates, pre-registered kill lines — is
genuinely strong and has done its job: it has kept bad signals away from capital.

**Two things would change the picture:** a genuinely decorrelated information
axis (Greeks, already on disk), and enough accumulated dates to resolve the
post-June window without further slicing.

---

## 7. Method note — how rev 1 got §3 wrong

The tail-inversion claim was reached by reading the same table four ways in four
messages: *tails invert* → *signal absent* → *sign flip by era* → *wide-universe
positive*. Each was stated confidently and revised by the next query.

Four slices of one dataset, stopping when a number looked significant. That is
the failure mode this system's gates exist to prevent, and it happened inside the
analysis rather than inside the pipeline.

**Two rules worth adopting for future sessions:**

1. **Pre-register the cut.** Decide the window, weighting, and horizon *before*
   querying. A result found on the fourth slice needs a multiple-testing
   correction that usually erases it.
2. **Equal-weight per date, always.** Pooling stock-date rows has now produced a
   fake result twice — t = −20 on the SI brick, +0.96% here. Per-date first,
   aggregate second, no exceptions.

Neither is new to this system. Both are already written down in
`Two_Brick_Findings`. They were not applied.

---

*Verified against: `docs/MASTER_TODO_LIST.md`, `alpha_fitness` (9,626 rows),
`momentum_promotion_check.py`, `c1_combiner.py` output, `predictions` ⋈ `outcomes`
(95 dates, h=5), pipeline B/C sources. Stale: `Gap_Check_and_Roadmap.md`,
`The_Complete_Quant_Build_Menu`.*
