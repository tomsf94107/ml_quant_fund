# Research Findings — Signal Validation Sweep

*ML Quant Fund · Alpha Research · 2026-08-15 · full universe, 406 tickers*
*Supersedes `NEGATIVE_RESULT_confidence_filter.md` — that doc is folded in below.*

---

## Bottom line

Two hypotheses tested to conclusion. **Both closed negative.** One hypothesis
instrumented and accumulating. One generated but untested.

| Hypothesis | Verdict |
|---|---|
| High-confidence filter (`prob_up ≥ threshold`) has edge | **CLOSED — no** |
| Signal has a longer information horizon than h=5 | **CLOSED — no** |
| Recent-window (post-2026-05-26) signal is real | **OPEN — accumulating**, monthly cron |
| Signal is reversal / dip-buying, not momentum | **OPEN — untested**, 3 independent supports |

No live model or position is affected. Neither tested hypothesis was ever
deployed. What changes is how Pipeline B's horizon-health panel is read (§6).

---

## 1. Confidence filter — CLOSED

**0 of 15 cells clear the bar** (NW |t|>3 AND block-bootstrap CI excluding 0 AND
BH-FDR). Horizons 1/3/5, thresholds 0.55–0.70, 365-day window, full universe.

| Horizon | dates | rank-IC | NW-t | bootstrap 95% CI |
|---|---|---|---|---|
| h=1 | 111 | +0.01389 | +1.11 | [−0.0115, +0.0382] |
| h=3 | 93 | +0.02228 | +1.37 | [−0.0124, +0.0546] |
| h=5 | 91 | +0.02018 | +1.01 | [−0.0243, +0.0569] |

### 1.1 The decisive test — group-neutralisation

| Horizon | raw IC | bucket-neutral (45 grp) | retained |
|---|---|---|---|
| h=1 | +0.01389 | +0.00749 | 54% |
| **h=3** | **+0.02228** | **+0.00241** | **11%** |
| h=5 | +0.02018 | +0.01301 | 64% |

**Benchmark at matched granularity:** the SI brick retained **80%** under
47-bucket neutralisation (NW-t −4.07). This filter retains **11%** under 48
buckets (NW-t +0.20). The association is bucket exposure, not stock selection.

*Caveat stated: finer grouping mechanically removes more variance, so bucket
(~8.5 names/group) and tier (~68/group) retentions are not comparable to each
other. The SI comparison at 47 vs 48 groups is the valid control.*

Confirms `The_Complete_Quant_Build_Menu` §C3 — neutralisation destroys the edge
because the edge *is* the group tilt — now measured full-universe with proper
inference.

### 1.2 Three inflation mechanisms, each removed in turn

Started from CRWV showing 100% accuracy (11/11) at `prob_up≥0.60`, h=3:

| Stage | Result | Artifact removed |
|---|---|---|
| Per-ticker, naive CI | 100%, CI [74, 100] | — |
| Overlap-corrected | CI [48.8, 100] | overlapping h-day windows |
| Episode clustering | 3 clusters, **one V-bottom** | consecutive-day duplication |
| Pooled hit rate | z = 4.97 | — |
| Dependence-corrected | z = **0.36** | cross-sectional dependence |
| Per-date, beta-neutral | t = 1.32 → NW ~1.0 | market drift |
| Group-neutral | 12.4% retention | bucket exposure |

**Beta, quantified:** the LOW-confidence bucket averaged **+0.672% per 3 days**
over the window. Accuracy in a rising tape is mostly drift.
**Pooling, quantified:** 22,312 rows over 93 dates overstates n by ~240× — the
structure that produced the spurious SI t = −20 (`Two_Brick_Findings` §5.1).

### 1.3 Overlay adds nothing to ranking

Identical 53 dates, identical rows:

| Column | rank-IC | NW-t |
|---|---|---|
| `prob_raw` | +0.04305 | +2.13 |
| `prob_up` (post-overlay) | +0.04304 | +2.13 |
| `prob_up_global` | +0.00119 | +0.04 |

Seven multipliers (`risk/sent/regime/options/squeeze/intraday/fg_mult`) and 1,940
`overlay_downgraded` rows change the cross-sectional ranking by **1e-5**. The
overlay is rank-preserving. *(Measures ranking only — it may still affect
BUY/HOLD thresholds and sizing.)*

`prob_up_global_ranker`: **1,701 rows over 4 days (2026-05-26 → 05-29), then
nothing.** Shadow-only by design and permanently disabled 2026-07-09 — see §5.

---

## 2. Horizon decay — CLOSED

Tested whether IC keeps rising past h=5, i.e. whether the book rebalances faster
than the signal's information horizon. Documented precedent: many cross-sectional
signals show t-stats rising monotonically toward the ~1-month mark; the SI brick
peaks at h=40.

**Common 72 dates, all horizons, raw:**

| h | horizon IC | NW-t | ICIR | hit% | lagged IC |
|---|---|---|---|---|---|
| 1 | +0.0025 | 0.16 | 0.018 | 50.0 | +0.0033 |
| 3 | −0.0064 | −0.29 | −0.045 | 52.8 | −0.0179 |
| 5 | −0.0024 | −0.10 | −0.017 | 48.6 | −0.0025 |
| 10 | −0.0115 | −0.61 | −0.095 | 47.2 | −0.0196 |
| 20 | −0.0025 | −0.16 | −0.025 | 45.8 | +0.0011 |
| **40** | **+0.0676** | **1.93** | **0.561** | **70.8** | **+0.0254** |

**No information horizon to extend into.** IC is flat-to-negative h=1→h=20, and
**lagged IC is ≈0 beyond day 1–2** — no return accrues in the later days of the
window. Hit rates at h=5/10/20 are 45–49%, below coin.

**h=40 is the only coherent cell** — positive IC, hit 70.8%, null-σ 5.17,
bootstrap excluding 0, and the only horizon where lagged IC is also positive.
It still fails: NW-t 1.93 < 3, ICIR 0.561, bucket-neutral retention **45%**
(+0.0676 → +0.0304, bootstrap then spans zero). At h=40 on daily predictions,
72 dates ≈ **1.8 independent blocks** — it cannot clear at this sample size.

*Noted, not claimed: the SI brick also peaks at h=40. Whether the model partly
re-derives positioning effects is a separate untested hypothesis.*

### 2.1 Second finding — the regime split, confirmed independently

Restricting to a common window moved it ~40 sessions earlier and the
short-horizon IC **vanished**: h=1 from +0.0122 → +0.0025; h=3 and h=5 turned
negative. The positive short-horizon IC lives entirely in the recent period.
Independent confirmation of the pre/post 2026-05-26 split found in §3.

---

## 3. Recent-window signal — OPEN, accumulating

80-day window (post-2026-05-26), `prob_up`, watchlist excluded:

| h | raw IC | bucket-neutral | retention | neutral t | dates to t>3 |
|---|---|---|---|---|---|
| 1 | +0.0201 | +0.0110 | 55% | 0.88 | ~25 mo |
| 3 | +0.0400 | +0.0212 | 53% | 1.27 | ~11 mo |
| **5** | **+0.0505** | **+0.0432** | **85%** | **2.41** | **~1.3 mo** |

h=5 bucket-neutral retention (85%) exceeds the SI brick's 80%, and its bootstrap
CI excludes zero. **Nothing clears the bar.**

**Two caveats, stated:**
1. "~1.3 months" assumes the effect size is stationary — the data contradicts
   that assumption (pre-05-26 IC was **−0.008**, a sign flip).
2. Naming h=5 "primary" because it scored best is post-hoc selection. What
   defensibly survives is that the **ordering h=5 > h=3 > h=1 replicates** across
   every window tested *and* across 16 independent walk-forward runs since May.

**Independent corroboration** — walk-forward AUC → implied IC (AUC ≈ 0.5 + IC/2):

| | h=1 | h=3 | h=5 |
|---|---|---|---|
| WF AUC → IC | +0.020 | +0.050 | +0.071 |
| per-date rank-IC | +0.020 | +0.040 | +0.051 |

Different data, method, and code path; same ordering and magnitude.

**Instrumented:** `ic_history` table + monthly cron (1st, 11:00/11:30 VN), raw
and bucket-neutral, h=1/3/5.

**Pre-committed decision rule:**

| Reading at ~3 months (h=5) | Action |
|---|---|
| NW-t > 3 AND bootstrap excludes 0 AND retention > 50% | Cold holdout + shadow book |
| t rising, retention holding, not yet 3 | Keep accumulating |
| IC reverts to ~0 or flips negative | **Regime was noise. Close permanently.** |

---

## 4. Reversal hypothesis — OPEN, untested

CRWV's 12 high-confidence h=3 calls fired in three consecutive-day runs
(Jul 15–17, Jul 27–31, Aug 6–11). **9 of 12 fired on NEGATIVE trailing 5-day
momentum**, median ≈ −8.9%. Jul 29 fired at `trail_5d = −26.4%`, close $60.82 —
the 60-day low. Three days later: +41%. CRWV fell 90 → 60.82 (Jul 8–29) then
recovered to 107.73 by Aug 12.

**Dip-buy signature, not momentum.** Eight of twelve calls are the same trade in
one drawdown; had CRWV kept falling, all would have been wrong.

Three independent supports: this episode analysis, Build Menu §B2 (negative
within-sector decile spread is a reversal signature), Build Menu §B14 (PCA
residual reversal / stat-arb as the strongest reframe). **Generated, not tested.**
Needs its own validation with the same harness before it means anything.

**Also observed — hit rate and P&L diverge:** h=1 thr≥0.70 showed accuracy
+5.1pp above base but a **negative** spread (−0.113%/1d). More winners, worse
returns. Direct evidence that accuracy is the wrong objective.

---

## 5. The ranker — correctly closed, do not reopen

`signals/generator.py` disabled `prob_up_global_ranker` on **2026-07-09**:
leaked (fit on all data, fake +5.6 Sharpe); purged-WF FAIL 5d −0.52.

Research confirms the decision. The ~3× Sharpe headline for learning-to-rank
comes from **long-short** implementations; the loss functions are explicitly
long-short motivated, and such portfolios are "more insulated against common
market moves." This book is long-only — `Research_Report.md` already notes that
long-only halves breadth and imposes a transfer coefficient well below 1.

**Stale doc:** the Build Menu still says the ranker was "never evaluated
honestly." It was, and it failed. **Correct that line** — it misled this session's
analysis into recommending a reopen against an explicit "Do NOT re-enable."

*RULE-1 gap: the −0.52 figure is not in `walk_forward_history` (that table is
per-ticker XGB: ticker/horizon/accuracy/auc/buy_hit_55). The decision rests on a
code comment, not auditable stored data.*

---

## 6. Operational consequence

Pipeline B's horizon-health panel reports high-confidence accuracy (h=5 high-conf
30d 59.4% vs 50.6% overall). Those figures are **correct as description** but must
not be read as a tradeable filter — they are pooled, beta-contaminated, and
dependence-inflated. Suggest annotating the panel with a pointer to this document.

**Walk-forward breadth is declining and nothing alerts on it:** h=5 `%AUC>0.55`
fell 38.1% (Jun 29) → 27.5% (Aug 9), while average AUC stayed flat (0.539 →
0.535). `walk_forward_history` has **no production consumer** — only parity-check
scripts read it. A threshold alert is the fix, not more logging.

---

## 7. Method

`scripts/validate_confidence_filter.py` and `scripts/horizon_decay.py`, both
mirroring `validate_si_v2.py`:

1. **Date is the unit of observation**, never the stock-date row.
2. **Rank-IC primary**, not hit rate — beta-neutral by construction.
3. **Newey-West, lag = horizon** (overlapping h-day returns are MA(h−1)).
   NW **over-rejects** under high serial correlation (den Haan & Levin 1997) →
   the reported t is an **UPPER BOUND**.
4. **Moving-block bootstrap**, block = h (block length should mirror the NW lag);
   less size-distorted than NW at small n. Matches `audit_combination.py`.
5. **Shuffle null = leakage check ONLY.** Anti-conservative, ignores regime
   variance, NOT part of the bar.
6. **Multiple testing:** Bonferroni + Benjamini-Hochberg FDR across all cells.
7. **Group-neutralisation** on any metadata column.
8. **Common dates** across prob columns and across horizons — different windows
   are different time periods, not different treatments.
9. **Per-row certification** (`horizon_decay`): each ticker-date must reproduce
   stored h=5 or is dropped at every horizon. Cause-agnostic — splits, dividends
   and bad bars are handled identically with no diagnosis required.
10. Standard IC reporting: mean, t, sd, **ICIR**, **hit rate**, n.

Bar: **NW |t| > 3 AND bootstrap CI excludes 0 AND BH-FDR survival.**

**Both tools validated in both directions before use:** pure noise + market beta
→ no edge; injected signal → detected at the injected horizon. A validator that
only ever says "no edge" is worthless.

---

## 8. Data findings

| Finding | Status |
|---|---|
| **`outcomes` is a split-adjusted PRICE return** — matches neither stored table (`adj_close` is dividend-adjusted, `raw_bars` unadjusted). This is what `price_cache` produces. | Documented |
| `daily_prices` CRWD seam: 682.80 (06-12) → 173.23 (06-15), fake −75%; actual split 07-02 (verified: 8-K record 06-25, distribution 07-01, trading adjusted 07-02; OCC ex-date 07-02) | **FIXED** |
| Full scan of 19 splits since Aug 2025: CRWD was the **only** seam | Verified |
| `audit_splits.py` labels 4 records "SPURIOUS"; **APLD is a false positive** — 1:6 ratio confirmed by SEC S-1/424B3; the tape break is the 2022-04-13 Nasdaq IPO, not a bad record. CERO/BNED/AMC are large genuine reverse-split ex-date moves (−26 to −29%). | **Soften verdict to "candidates"** |
| Certification exclusions confirm the price-return diagnosis: dividend payers (VZ, KVUE, PFE, D, TGT) fail vs `adj_close`; split names (OPEN, XLE, CRWD, KLAC, NFLX) fail vs `raw_bars` | Diagnostic |
| VXRT excluded 46× — $0.45–0.60 stock, relative-only tolerance too strict at penny prices | **FIXED** (absolute+relative) |

---

## 9. Tool defects found and fixed

| Defect | Impact | Status |
|---|---|---|
| `highconf_accuracy.py`: independence-assuming CIs, no base rate, no overlap correction, no null | Reported 100% with CI [74,100] on what was one rally | **DELETED** |
| `load_sectors()` returned `{}` with no `sector` column | `--sector-neutral` silently did not run; printed `sector_neutral=False` while reporting raw numbers | **FIXED** — fails loudly, auto-detects `bucket` |
| `--group-col` without `--sector-neutral` | Silently ignored | **FIXED** — now implies it |
| `null-sigma` shown beside the t-stat | Readable as clearing a bar (h=3 showed 3.02 vs NW-t 1.37) | **FIXED** — relabelled, removed from bar |
| Multi-column comparison on different date sets | Apples-to-oranges (72/93/53 dates) | **FIXED** — common-date intersection |
| Parity gate sampled first 4,000 rows in insertion order | Never reached a split date; silently passed the unadjusted table | **FIXED** — random sample across full window |
| Parity gate used mean \|diff\| | One split in 400 tickers barely moves the mean | **FIXED** — max-diff, then exclusion-rate |
| Bootstrap block = horizon | h=60 on <60 dates → `randrange` crash | **FIXED** — block capped at n//2 |
| Horizons measured on different windows | h=40 was an older time period, not a longer horizon | **FIXED** — common-date restriction |

**Pattern:** the majority are the **silent-fallback** class — same as
`price_cache.cached_daily()`'s empty-gap no-op that hid a dead feed for six
months. *A control that silently does not run is worse than no control, because
it produces confident output.* Worth making "does this fail loudly?" a standing
review question on any new validation code.

---

## 10. What this does NOT say

- Not "the model has no signal" — **these two hypotheses** have no demonstrated
  stock-selection content beyond group exposure.
- Not "`prob_up` is meaningless" — it remains a descriptive output.
- The reversal hypothesis (§4) is **generated, not tested**.
- The recent-window result (§3) is **unresolved**, not refuted.

---

*Reproduce:*
`python scripts/validate_confidence_filter.py --days 365 --shuffles 200 --bootstrap 2000 --sector-neutral`
`python scripts/horizon_decay.py --days 365 --horizons 1,3,5,10,20,40 --shuffles 100 --bootstrap 1000`
