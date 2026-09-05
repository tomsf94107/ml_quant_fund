# SCOPE — firm characteristics, GKX-style

**Written 2026-09-05.** A build plan, not a build. Nothing here is implemented.

**The premise, in one line:** every model failure on 2026-09-05 was on the same
119 mostly price-derived columns. The constraint is the data, not the estimator.

---

## Why this project and not another model

On 2026-09-05, at h=3/5, the following were tested and returned near-zero
day-weighted excess return:

| tested | result |
|---|---|
| 6 insider constructions (flow, trajectory, overhang, CMP split, buy×SI, accuracy gate) | null |
| 4 label definitions (any-positive, top-decile, pct7, triple-barrier) | best is the one in production |
| XGBoost vs L2-logistic across 5 regularisation levels | linear wins on AUC, neither on return |
| XGB+LightGBM ensemble | +0.002 AUC, t=+0.63 |
| linear+tree blends, weight-optimised | no better than components |
| 3-model consensus voting | 3-of-3 minus 1-of-3 = **−0.173pp** |

Twenty-plus configurations, one feature set. The regularisation sweep is the
diagnostic that matters: performance rose monotonically as coefficients were
shrunk toward zero, plateauing at C=1e-05 where the model is nearly an
equal-weight composite.

That is the theoretically correct response to low signal-to-noise. Daily equity
returns run an SNR near 0.8% — return standard deviation ~3.49% against a mean
~0.028%, versus ~100% for image classification — and in low-SNR settings the
optimal strategy is to predict the mean. A model that wins by barely fitting is
telling you the features carry little.

**Gu, Kelly & Xiu (2020, *RFS*)** is the benchmark comparison. They find trees
and neural networks best, tracing the gains to nonlinear predictor interactions.
Their setup:

| | GKX | this fund |
|---|---|---|
| horizon | monthly | h=3/5 daily |
| stocks | ~30,000 | 420 |
| history | 60 years (1957–2016) | 5 years of features |
| predictors | **94 firm characteristics + 8 macro + industry dummies** | 119 mostly technical indicators |

Their OOS R² is ~0.4% monthly — tiny, detectable only because the panel is
~1.8M stock-months. **Their winning models will not transfer at this sample
size**, and today's sweep already showed why: flexibility loses here.

The transferable part is the **predictor set**, not the estimator.

---

## What already exists — verified, not assumed

| | |
|---|---|
| `xbrl_facts` | 624,353 facts, **9 normalised concepts**, 400 tickers, **2009→2026** |
| PIT key | `filed_date`, and the schema comment calls it "THE date that matters" |
| Writer | `data/etl_xbrl_fundamentals`, crond Sunday 07:10 VN, ~5 min, 394 tickers |
| Reader | `features/fundamental_features.py:36` |
| As-filed rule | `fundamentals_fetch.annual_by_fy` keeps the **earliest** filed value per fiscal year, so restatements cannot leak backwards |
| Built features | `fund_gp_assets`, `fund_op_equity`, `fund_ni_margin`, `fund_bm`, `fund_ep` |
| Wired to the model | only `fund_ep` and `fund_ni_margin`, both added 2026-09-05 |

The nine concepts: `equity`, `revenue`, `eps_diluted`, `net_income`,
`operating_income`, `cogs`, `total_assets`, `shares_out`, `gross_profit`.

**Seventeen years of PIT-correct fundamentals already ingested on a weekly
cron.** This project extends that, it does not start it.

---

## Phase 1 — widen the tag list

Twelve concepts from the SAME `companyfacts` API already called. No new vendor,
no new auth, no new PIT question.

| concept | us-gaap tags (fallback lists, as the existing code does) | unlocks |
|---|---|---|
| operating cash flow | `NetCashProvidedByUsedInOperatingActivities` | cash-flow-to-price, accruals, cash productivity |
| long-term debt | `LongTermDebtNoncurrent`, `LongTermDebt` | leverage, debt-to-equity, Δdebt |
| current debt | `DebtCurrent`, `ShortTermBorrowings` | same |
| cash | `CashAndCashEquivalentsAtCarryingValue` | cash holdings, net debt |
| inventory | `InventoryNet` | inventory growth |
| receivables | `AccountsReceivableNetCurrent` | receivables growth |
| capex | `PaymentsToAcquirePropertyPlantAndEquipment` | capex growth, investment |
| R&D | `ResearchAndDevelopmentExpense` | R&D intensity |
| current assets | `AssetsCurrent` | current ratio, working-capital accruals |
| current liabilities | `LiabilitiesCurrent` | same |
| PP&E | `PropertyPlantAndEquipmentNet` | tangibility |
| dividends | `PaymentsOfDividendsCommonStock` | dividend yield, payout |

**Effort:** extend the tag block in the writer, rerun the backfill for 400
tickers × 17 years. A few hours of scraping at SEC's 10 req/s.

**Gap check before starting:** confirm `data/etl_xbrl_fundamentals` uses the same
fallback-list pattern as `fundamentals_fetch.py`. If it hard-codes single tags,
coverage will be worse than expected — filers use different tags for the same
concept, which is exactly why the existing code has fallback lists.

---

## Phase 2 — construct the characteristics

From the 9 existing concepts (≈15, of which 5 exist):

`bm`, `ep`, `sp`, `gma`, `roe`, `roa`, `operprof`, `ni_margin`, `ato`,
`agr`, `sgr`, `egr`, `chcsho`, `mve`, `lev`

From the 12 new (≈25 more):

`cfp`, `acc`, `pctacc`, `cashpr`, `cash`, `chinv`, `chrec`, `invest`,
`rd_mve`, `rd_sale`, `currat`, `quick`, `salecash`, `salerec`, `saleinv`,
`tang`, `divi`, `divo`, `dy`, `de_ratio`, `chdebt`, `capex_gr`, `noa`,
`grcapx`, `pchcapx`

**Price/volume characteristics GKX also uses, which you largely have already:**
`mom1m`, `mom6m`, `mom12m`, `mom36m`, `chmom`, `maxret`, `retvol`, `idiovol`,
`beta`, `betasq`, `turn`, `std_turn`, `dolvol`, `ill`, `zerotrade`,
`baspread`, `indmom`.

**Realistic total: 60–70 of GKX's 94.** The rest need CRSP/Compustat fields not
in XBRL (analyst estimates, institutional ownership history, credit ratings).

**The step where these projects fail is PIT alignment.** Rules, non-negotiable:

1. A characteristic is knowable only from `filed_date` forward, never from
   `period_end`. This is the exact error that voided the PEAD work — `report_date`
   was fiscal-period-end and admitted the figure 14–30 days early, quantified at
   IC +0.2612, t=+30.
2. Use the **as-filed** value, not the restated one. The existing
   `annual_by_fy` already does this by keeping the earliest filed per FY.
3. Anything divided by market cap changes daily; anything from the filing does
   not. Recompute the price-linked ones daily, forward-fill the rest.
4. Every characteristic gets a **coverage report** — % non-NaN per ticker per
   year — before it is wired. `short_pct_float` sat 100% NaN in production
   killing five downstream features; that must not repeat.

---

## Phase 3 — test before wiring

**Do not wire and let importance decide.** That takes weeks and cannot separate
"no signal" from "already captured".

For each characteristic, and for the set:

- per-date IC with Newey-West at the horizon lag, plus a shuffle null
- **orthogonalised** against the existing 119 columns. `rate_beta` failed exactly
  here on 2026-09-05: raw t=+3.80 at h=20, orthogonal t=+1.52, retaining 37–41%,
  because it was derived from returns and the model already had ~100 return-based
  features. Accounting characteristics are structurally different — they cannot
  be a linear combination of momentum, RSI and beta — so high retention here
  would be meaningful and low retention would be a genuinely surprising negative.
- **multi-seed**, ≥3 ticker samples. Three single-seed results reversed today:
  a top-N book at +1.17pp → −0.81pp, a linear economic edge at +0.645pp →
  −0.021pp, and 3-of-3 consensus at −0.173pp.
- day-weighted excess return at several caps, with turnover — not AUC. AUC and
  return decoupled in every test today: `top_decile` posted AUC 0.7316 and
  delivered +0.01pp.

---

## Phase 4 — horizon

**Target h=20/40, not h=3/5.** Preliminary sweep (2026-09-05, 2 of 3 seeds in):

| horizon | cap-3 excess (xgb) | per day | turnover |
|---|---|---|---|
| 3 | +0.052pp | 0.017pp | 61% |
| 5 | −0.038pp | — | 51% |
| 20 | +1.596pp | 0.080pp | 31% |
| 40 | +5.727pp | 0.143pp | 30% |

Per-day excess rises ~8× from h=3 to h=40 while turnover halves. **Caveats:**
linear goes negative at h=20/40 while XGBoost goes strongly positive — a sign
flip between models at one horizon suggests noise — and 564 dates at h=40 with
overlapping windows is roughly 14 independent periods. Seed 3 pending.

This matters for the build because **annual characteristics are near-constant
within a year**, so they suit a 40-day horizon far better than a 5-day one. And
the fund's one validated edge, the SI brick, already runs at h=40.

---

## Effort and sequencing

| phase | effort | blocking? |
|---|---|---|
| 1 — widen tags, backfill | ~1 day incl. scraping | yes, everything depends on it |
| 2 — construct characteristics | ~1–2 days, PIT alignment is the risk | yes |
| 3 — test before wiring | ~1 day | yes |
| 4 — rerun model sweep on characteristics | ~half a day | no |

**Realistic total: 3–4 days of focused work.**

## What would make this fail, stated in advance

- **Coverage.** 400 tickers is not 30,000. Annual characteristics give ~400
  independent values per year over 17 years. That is a far smaller effective
  sample than GKX and it caps what any model can extract.
- **Survivorship.** `prices.db` is survivor-tilted. Accounting characteristics on
  surviving firms overstate quality effects. The SI brick's leg decomposition
  sized this as near-moot for a low-DTC long leg; it is NOT moot for a
  distress-linked characteristic like leverage.
- **The honest prior.** GKX's monthly R² is ~0.4%. Scaled to a 420-name universe
  over 17 years, the expected effect is small. This project is worth doing
  because it addresses the actual constraint — but it should be expected to
  produce a modest, slow signal, not a fix for the direction model.

## First command when picking this up

    sqlite3 fundamentals.db "SELECT concept, COUNT(*), COUNT(DISTINCT ticker) FROM xbrl_facts GROUP BY concept;"
    grep -n "TAGS\|fallback" data/etl_xbrl_fundamentals.py | head -20

Confirm the writer uses fallback tag lists before extending it.
