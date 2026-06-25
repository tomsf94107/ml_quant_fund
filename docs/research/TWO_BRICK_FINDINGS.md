# Two Validated Bricks & the Combination Question

**ML Quant Fund — Alpha Research Note**
**Date:** 2026-06-25
**Status:** 2 confirmed bricks · combination directionally supported but underpowered · all results audited

---

## Bottom line

Two cross-sectional signals are now independently validated as real, modest alpha
sources: **post-earnings drift (PEAD)** and **short interest / days-to-cover**. Short
interest was confirmed this session on 5 years of free official FINRA data using a
corrected per-date methodology. The two are largely uncorrelated and every measurement
points toward them combining beneficially — but across three independent tests, the
combination benefit **cannot be established as statistically significant**. The binding
constraint is data sparsity (PEAD generates few independent observations), not analysis
quality.

---

## 1. Verified scorecard

Every claim was checked by an independent audit script that recomputed the number from
raw data and showed its arithmetic.

| Claim | Verdict | Evidence |
|---|---|---|
| PEAD is a real brick | **CONFIRMED** | OOS IC +0.20–+0.24 (t 3.6–4.2); survives cold holdout, beta-strip; 40-day horizon |
| Short interest is a real brick | **CONFIRMED** | Per-date IC −0.054 (NW t −4.46); negative every year 2021–2026; null control 8.3σ |
| Short interest is stock-specific, not a sector bet | **CONFIRMED** | 80% of IC survives 47-bucket sector-neutralization (NW t −4.07) |
| PEAD + SI combination beats either alone | **UNPROVEN** | All point estimates positive (Sharpe lift +0.20, IC uplift +0.012, corr +0.22) but no test reaches significance. Underpowered |
| "Combination thesis confirmed" (earlier conclusion) | **OVERTURNED** | Bootstrap lift CI [−0.87, +0.62] spans zero at n=23 months. Was a false positive, caught by audit |

**"Unproven" ≠ "disproven."** The thesis wasn't shown false — the claim that it was
*proven* was shown premature. No live model is affected; each brick remains individually
valid. Only the claim that a COMBINED book is provably better is withheld pending data.

---

## 2. Brick #1 — PEAD (carried forward)

Measured via SUE (standardized unexpected earnings, PIT-trailing denominator).

| Property | Value | Read |
|---|---|---|
| Peak horizon | 40 days | Separate ~quarterly book, not the daily system |
| OOS IC (cold holdout) | +0.20 to +0.24 | Real on unseen data |
| Realistic IC (after haircuts) | ~0.06–0.10 | Size on this, not the headline |
| Survivorship bias | Mild | ~99% universe coverage back to 2009 |

Caveats: magnitude-concentrated (depends on large-SUE events), edge concentrated in recent
years, realistic tradeable IC below the +0.125 in-sample headline.

---

## 3. Brick #2 — Short interest (NEW this session)

Days-to-cover from FINRA consolidated short interest — free, official, 5yr bi-monthly.

**Data:** FINRA consolidated short interest via public API (Rule 4560). 20,882 rows,
396 tickers, 60 settlement dates, 2021-05 to 2026-05. days-to-cover pre-computed by FINRA.

**Validation (corrected per-date method):**

| Metric | h=40 | h=20 | Read |
|---|---|---|---|
| mean per-date IC | **−0.054** | −0.030 | High short interest → lower return |
| Newey-West t | **−4.46** | −2.27 | Significant; stronger at 40d (slow signal) |
| naive t (contrast) | −4.79 | −2.67 | ≈ NW t → not an inflation artifact |
| % years correct sign | 100% | — | Negative every year 2021–2026 |
| sector-neutral retention | **80%** | — | Stock-specific, not a sector tilt |
| null control | **8.3σ** | — | Shuffled returns collapse IC to ~0; real IC far outside |

Trusted because naive and NW t-stats are close (−4.79 vs −4.46) → significance is NOT
from non-independence inflation. Null control decisive — shuffling forward returns within
each date collapses IC to zero; real IC sits 8.3σ outside.

---

## 4. The combination question — three tests, one answer

| Test | Sample | Result |
|---|---|---|
| Sharpe-level (monthly LS streams) | 23 months | Combined Sharpe 2.18 vs best single 1.98 (+10%); bootstrap lift CI [−0.87, +0.62] spans zero |
| IC-level (per-date) | 11 dates | Combined IC +0.281 > best single +0.269; combined t 6.02 exceeds both; corr −0.13. But uplift t=0.14 — not significant |
| Diversification ratio | 23 months | 1.25 (>1 = real vol reduction); return-stream correlation +0.22 |

**Consistent finding:** the combined signal looks like the best of the three (highest IC,
highest t-stat, complementary timing, real DR) yet neither test reaches significance.
Three methods agreeing the data can't resolve an effect this size. Verdict: **promising,
directionally consistent, underpowered.**

**Binding constraint:** PEAD event sparsity. ~23 monthly observations; overlap with
bi-monthly SI is only ~11 dates of ~30 stocks. Only path to a definitive verdict is more
PEAD history — a data-collection task, not analysis.

---

## 5. Methodology — two bugs caught by audit

### 5.1 t-statistic inflation bug
- **Symptom:** early SI validation reported t = −20 (absurdly strong)
- **Cause:** pooled ~20,000 stock-date rows as independent, but they're ~400 stocks ×
  ~60 dates with persistent short interest → effective n ~60, not 20,000 → t inflated ~7×
- **Fix:** per-date cross-sectional IC averaged with Newey-West SE (also corrects
  overlapping-window autocorrelation). Corrected honest t = −4.46

### 5.2 Look-ahead-benchmark bug
- **Symptom:** IC-combination "uplift" metric self-contradicted
- **Cause:** compared combined signal against the per-date MAX of the two singles — an
  unattainable oracle that picks the winner each date with hindsight. A fixed combination
  can't beat a hindsight oracle → metric structurally negative
- **Fix:** compare against the best single signal chosen once, overall (attainable).
  Re-audited against complementary, redundant, dilution cases

**Discipline:** every result re-derived by a separate audit script that printed its
arithmetic, cross-checked stats two ways, and ran a null control (shuffle outcome →
signal must vanish). Caught both bugs before they reached a conclusion.

---

## 6. Next steps

1. **Trade the two bricks independently.** Both validated on own merits. PEAD (40d,
   ~quarterly) and short interest (40d, bi-monthly) sized on haircut ICs.
2. **Resolve combination via more PEAD history (Solution 4).** Unprovable at current
   sample sizes. As events accumulate, re-run IC-level + Sharpe-level tests.
3. **Productionize short interest.** Wire FINRA fetcher to periodic refresh (settlement
   dates published on known schedule), clip 999.99 OTC placeholders, join to exchange-listed
   universe.
4. **Hunt brick #3 from a different mechanism.** Options-implied (put/call, IV skew) is
   strongest untested lead; needs historical data source (live feed = current snapshots only).
5. **Keep auditing.** Run audit scripts on every new result. Both bugs would have shipped
   without them.

---

## 7. Tooling delivered

All offline (read cached data), read-only against existing DBs, validated against synthetic
data with known structure before use.

| Script | Purpose |
|---|---|
| `finra_short_interest.py` | Fetches free FINRA short interest (5yr) → short_interest.db. OAuth/dataset/field/filter verified vs live responses |
| `validate_si_v2.py` | Per-date IC validator with Newey-West — corrected method replacing inflated pooled t |
| `validate_si_sector.py` | Sector-neutral version: demeans signal+returns within sector |
| `combine_ic_test.py` | IC-level combination test, null control (uplift metric corrected) |
| `combine_pead_si.py` | Sharpe-level combination: monthly LS streams, blended Sharpe, DR |
| `audit_combination.py` | Independently recomputes combination numbers; block-bootstrap CI on lift |
| `audit_ic.py` | Independently recomputes IC numbers; cross-checks NW two ways; shuffle null control |
| (prior) PEAD suite | fetch_and_pead, pead_sue, pead_walkforward, pead_oos, pead_survivorship |

---

## Key data-source facts (for reuse)

- **FINRA short interest:** free, official, ~5yr rolling. Dataset `otcMarket/consolidatedShortInterest`.
  Fields: `symbolCode`, `currentShortPositionQuantity`, `averageDailyVolumeQuantity`,
  `daysToCoverQuantity`, `settlementDate`. Filter syntax: `dateRangeFilters` (NOT
  `compareFilters` — returns 400). OAuth: client_id = the "API Client (user)" value, secret
  via RESET. Token endpoint `ews.fip.finra.org/fip/rest/ews/oauth2/access_token`, Bearer for
  data calls. 10GB/mo limit; full 5yr pull ≈ 0.24GB. OTC names have junk DTC=999.99 (clip them).
- **yfinance:** fine for prices, but only CURRENT options/short-interest — cannot backfill history.
- **Validator gotcha:** old `validate_signal.py` auto-searches all DBs for a feature column and
  grabbed the stale `accuracy.db.short_interest_cache` stub. Use `validate_si_v2.py` (reads only
  short_interest.db) or the `--db` flag.
