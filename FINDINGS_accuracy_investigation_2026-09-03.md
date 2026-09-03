# FINDINGS — the high-confidence accuracy investigation

**2026-09-03.** What was investigated, what was eliminated, what was actually
true, and what changed as a result.

Written because this took most of a day and would otherwise be re-derived in
three months. The measurements are reproducible from the scripts named
throughout.

---

## The question as asked

> "High-confidence h=3 and h=5 accuracy has dropped significantly over the last
> 30 days — from 60%-ish to high 40s. Macro news, rates, wars, politics have
> been hitting the market. What dragged accuracy down?"

## The answer

**The market fell and the model was long into it. No skill was lost.**

Over 2026-08-21 to 08-26, h=5 daily accuracy at `prob_up ≥ 0.55` read
**25.6% / 28.1% / 30.1% / 33.0%** — four consecutive days with confidence
intervals excluding 50%, which looks like a model that has broken.

Over the same outcome windows, the up-rate across the whole 417-name universe
was **41.5% / 34.3% / 25.2% / 28.5%**, with a mean 5-day return of **−2.32%** on
the worst day. Three quarters of stocks fell.

Every prediction above the gate is a bet on UP. When a quarter of stocks rise,
roughly a quarter of long calls are right — by arithmetic, regardless of skill.
**Accuracy tracked the base rate almost exactly.**

**Raw accuracy at a long-only gate measures skill PLUS market direction, and
direction dominates.** Lift over the base rate is the skill measure. The two
had never been shown side by side, which is why a normal selloff looked like a
broken model.

---

## Six hypotheses tested and eliminated

Each was measured, not argued.

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | The model degraded | **No.** Walk-forward AUC at h=5 was 0.5347 in early May and 0.5362 at end-August, across ~400 tickers. Flat all year. |
| 2 | Macro news / market direction hurt it | **Yes for raw accuracy, no for edge.** Lift is measured over each month's own base rate, so direction cancels. And the model was leaning bearish — mean `prob_up` 0.525 → 0.463 — which should have helped. |
| 3 | Cross-sectional dispersion collapsed | **Partly, ~10%.** Robust SD of 5-day returns fell 5.00% → 4.48%. A 10% reduction in what is available to predict cannot produce a 93% reduction in what was captured. |
| 4 | A few tickers dragged it down | **No.** Per-ticker skill does not persist: Spearman ρ ≈ +0.21 to +0.26 between windows. May's top contributors pooled to 42.2% afterwards against a 51.8% base. Regression to the mean, not culprits. |
| 5 | Volatility regime | **Confounded.** The split looked clean — 50.9% in low vol vs 58.5% in mid vol, intervals not overlapping — but May was 20/20 low-vol days and *good* while August was 21/21 low-vol and *bad*. The vol bucket was month composition wearing a regime label. |
| 6 | The August sample is incomplete or biased | **No.** 78.8% scored, the 21 unscored rows are the last five sessions legitimately pending at h=5, and mean probability is 0.761 scored vs 0.764 unscored. No selection bias. |

---

## A premise that was wrong, and worth recording

The investigation began from "a four-month monotone decline: +11.7 → +6.9 →
+1.9 → +0.8pp." That is the **top-decile** cut. At the actual confidence gate
the picture is different:

| month | prob ≥ 0.70 | n | prob ≥ 0.55 |
|---|---|---|---|
| 2026-05 | +7.4pp | 134 | +2.2pp |
| 2026-06 | +6.2pp | 469 | +3.6pp |
| **2026-07** | **+8.5pp** | 332 | +1.0pp |
| 2026-08 | +0.5pp | 78 | −2.1pp |

**July was the best month at the confidence gate.** There was no four-month
decline — three good months, then August. And at that gate August's interval is
[37.9, 59.6] on n=78, which contains May, June and July. It is only at the wider
0.55 gate, with n=1,373, that August is statistically distinguishable.

Three different cuts told three different stories. Naming the cut is not
pedantry.

---

## What the model can actually do

Walk-forward AUC 0.5362 → Somers' D = 2(AUC − 0.5) = **0.0724** (exact identity
for a binary outcome). Discrimination that weak supports a top-decile lift of
roughly **2–4 percentage points**.

May's +11.7pp exceeded that by three to five times. **May was the anomaly, not
August.** Any proposal implying a return to +11pp is a red flag about the
proposal.

---

## The timing overlay: tested, not built

The natural response to "long into a selloff" is an overlay that stands down
when a decline is coming. Nine already-available series were tested against the
universe's forward 5-day up-rate (`analysis/market_timing_test.py`, 915 dates):

| predictor | rho | naive t | after ÷2.2 | quintile spread |
|---|---|---|---|---|
| HY OAS | +0.196 | +5.46 | **2.48** | +9.3pp |
| SPY 20d realised vol | +0.180 | +4.70 | 2.14 | +9.1pp |
| % above 200DMA | −0.156 | −4.07 | 1.85 | −9.2pp |
| VIX level | +0.106 | +3.23 | 1.47 | +6.4pp |
| everything else | — | \|t\| < 3 | — | < 3.5pp |

Overlapping 5-day windows on consecutive dates inflate the naive t by about
√5 ≈ 2.2×, so only HY OAS clears t=2 — marginally.

**All three leaders are one factor measured three ways**: credit stress, price
volatility, and participation. And the sign is **buy fear** — high spreads
predict *higher* forward up-rates.

**Two reasons it was not built:**

1. **The sample is one regime.** 2023–2026 is a bull market with dips, in which
   mean reversion works by construction. The test would need 2000, 2008 or 2020
   to be meaningful.
2. **That test cannot be run.** FRED serves ICE BofA OAS series on a **rolling
   three-year window only** — confirmed 2026-09-03 by requesting
   `observation_start=1900-01-01` and receiving 786 rows from 2023-09-04. The
   truncation is at the source, from ICE licensing. **The one predictor that
   cleared the bar is the one whose history cannot be obtained.**

The closest long-history substitute is `BAA10YM` (Moody's Baa minus 10-year,
monthly, 1953+), but the default-spread-predicts-returns relationship is
canonical Welch-Goyal territory with a known negative out-of-sample result.
Building it would likely re-derive a null.

**Standing recommendation:** log the predicted up-rate alongside the realised
one, forward-only. If the relationship survives a period that is not
buy-the-dip, revisit. Do not gate a small real edge behind an unproven signal —
a wrong overlay costs twice, standing down before good weeks and staying long
before bad ones.

---

## Changes made

| | |
|---|---|
| `analysis/horizon_health_compute.py` | prints **base rate and lift** beside accuracy, and Wilson intervals on every figure. The line that would have prevented this investigation. |
| `scripts/pipecheck.sh` | `tail -14` so the added lines do not evict `[highconf 30d]`; `DATE` now uses the ET clock, not VN |
| `warning.db` | DGS2 and T10Y2Y to 1976, WTI to 1986, broad dollar to 2006 — 40,536 rows |
| `analysis/market_timing_test.py` | the overlay test, re-runnable |
| `analysis/highconf_attribution.py` | the four-way decomposition, re-runnable |
| `analysis/highconf_sample_check.py` | scored-fraction and selection-bias check |

---

## What is NOT in the model, and why

`warning.db` holds 18 FRED series plus Cboe volatility history, but **none of
it reaches the direction model** — `features/builder.py` never touches that
database. The macro series feed the crash-warning system only.

**No CPI, PPI, FX or activity data exists anywhere in the repo.** Adding them
was considered and rejected on the research finding that no macro series
discriminates *between* individual US stocks at 3–5 days beyond what beta and
sector relative strength already span. CPI and PPI are monthly with a two-week
lag — structurally incompatible with the horizon.

The one cheap exception not yet built: **"days until the next CPI release"** as
an *event* feature rather than a macro level. Event-day effects are real and the
calendar is free.

---

## The lesson, stated plainly

Two reporting changes were made in two days — Wilson intervals, then base rates
— each because a number was presented without the context needed to read it.
Each cost roughly a day of investigation before being added.

The pattern is the same one this codebase keeps producing: **a plausible value
with no way to tell whether it is meaningful.** A frozen cache, a 5×
overconfident probability, a 2× price seam, an accuracy figure without its base
rate. None raised an error. All looked fine.

The operator's instinct was correct throughout — the market did hurt raw
accuracy, late August was distinctly bad, and pushing back against a
too-dismissive framing was right. What was missing was the second number that
makes the first one legible.
