# ML Quant Fund — System Onboarding

*Prepared Jun 12, 2026 · For new research collaborator · Honest state, no marketing*

-----

## 1. What This Is

A solo-built (now two-person) systematic equity research system, ~6 months old, that went through the full retail-quant arc — built an ML direction predictor, measured it honestly, found it at the industry ceiling (coin-flip), killed it, and pivoted to what the evidence supports: **cross-sectional basket signals with pre-registered validation gates.** The system today is (a) a production data/feature/model pipeline over 394 US equities, (b) one validated signal in live shadow trading (momentum), (c) three validated-but-parked fundamental signals, (d) a graveyard of honestly-killed ideas with documented unlock conditions, and (e) a research methodology that is arguably the most valuable artifact: every signal passes OSAP published-decay pre-screen → PIT-correct backtest → pre-registered gates → adversarial attack battery, or it dies.

**Product shape:** long-only, no leverage, no shorts (retail brokerage). This constrains everything — see §8 (C1 finding) for why this matters more than any model choice.

## 2. The Honest Performance Arc (read this first)

- **Phase 1 (Jan-Apr):** XGBoost per-ticker direction classifiers, 1/3/5-day horizons. Training AUC ~0.967. Looked amazing.
- **Phase 2 (Apr 29, the reckoning):** live out-of-sample measurement: **ROC-AUC 0.510, BUY accuracy 49.3%** (n=3,483), daily alpha vs SPY −0.08%. The 0.46 AUC gap = overfitting to noise, not hidden alpha. Literature confirms the ceiling: honest 1-5d equity direction tops out at AUC 0.51-0.55 for everyone (Gu-Kelly-Xiu monthly R² 0.33-0.40% is the *landmark success*).
- **Phase 3 (May):** leak hunts (one real leak found+fixed in calibration; isotonic added), regularization to depth-2/30-trees/λ10 (honest OOS ~0.51+), labels audit (2,761 fake-zero outcomes fixed), BUY signals quarantined behind a kill-switch — **the direction model never traded and never will at days-scale.**
- **Phase 4 (May-Jun, the pivot):** direction → cross-sectional ranking. Rank-IC analysis showed the signal lives in *ordering*, not per-name sign. Momentum survived every attack; five other signal hunts died honestly. Breadth identified as the binding constraint → universe expanded 149→394 (Jun 11).

**The one-line summary for a newcomer:** per-name day-scale prediction is dead for everyone; weak-but-provable basket claims compound; this system now hunts, validates, and (soon) trades the latter.

## 3. Architecture

**Repo:** `~/Desktop/ML_Quant_Fund` (GitHub `tomsf94107/ml_quant_fund`, branch `research-track`). Python 3.10, env `ml_quant_310`. Runs on a single Mac, macOS cron (VN timezone, ET-anchored market times).

**Nightly pipelines** (event-chained as of Jun 12 — `scripts/pipeline_chain_ADB.sh`, single 03:00 VN cron):

- **A — ingest** (~2-2.5h at 394 names): insider ETL (EDGAR, 7-day incremental), UW post-market snapshot, feature validator, revenue/quarters (Massive). Writes completion marker.
- **D — alpha panel:** transformation panel build.
- **B — train + predict:** per-ticker XGB retrain + nightly predictions, 3 horizons. Guard: refuses to run if A’s marker missing (stale-data protection).
- **C — pre-open** (19:00 VN = 08:00 ET): sentiment, UW snap, daily runner (signal generation), **momentum shadow logging** (Stage -2), sanity guard.
- **Weekly (Sun):** recession model 07:00, XBRL fundamentals refresh 07:10, analyst snapshot 07:20.

**Databases:** `accuracy.db` (predictions, outcomes, momentum_shadow_*, intraday, fitness_scores, analyst_snapshots, economic_calendar), `fundamentals.db` (592K XBRL facts, 387 cos, 2009+), `insider_trades.db` (161K+ Form-4 rows, 2019+, expansion crawl in flight), `institutional_trades.db` (1.9M rows), `recession.db` (separate sub-project, `recession/`).

**Dashboards:** Streamlit `ui/` — Dashboard (signals + PSR-guarded Sharpe), Accuracy (4 tabs: Wilson CIs, Bayesian shrinkage, calibration, explorer), Forecast, Events.

## 4. Data Sources

|Source                                                |$/mo |Feeds                                                                           |Notes                                                            |
|------------------------------------------------------|-----|--------------------------------------------------------------------------------|-----------------------------------------------------------------|
|Massive Developer (ex-Polygon — never call it Polygon)|79   |OHLCV daily+intraday, revenue quarters                                          |resilient client w/ yfinance fallback; no /v3/quotes on this plan|
|Unusual Whales Basic                                  |125  |dark pool, options flow, short interest, earnings calendar                      |40K calls/day, 120/min; ~44-day darkpool history cap             |
|SEC EDGAR (free)                                      |0    |Form 4 insider (raw, per-insider), XBRL companyfacts (fundamentals)             |PIT keyed on **filed_date**, never period_end                    |
|yfinance (free)                                       |0    |fallback prices, analyst revisions (snapshot-only → weekly accrual since Jun 12)|                                                                 |
|FinBERT (`yiyanghkust/finbert-tone`)                  |0    |news sentiment                                                                  |label-order verified                                             |
|Anthropic API                                         |usage|sentiment/research assist                                                       |                                                                 |

**Feature panel:** ~100 features/ticker/day: price/vol/momentum families, options flow, dark pool, short interest, sentiment, macro regime (VIX/oil/DXY/credit-spread proxy), institutional PIT features, interaction features, and (new Jun 11) 5 PIT fundamental ratios (`fund_gp_assets, fund_op_equity, fund_ni_margin, fund_bm, fund_ep` — strict filed_date<as_of, merge_asof fast path).

## 5. Universe

394 tickers (was 149 until Jun 11). Original 149: AI/tech-heavy (semis, hyperscalers, neoclouds, memory). Expansion +245 from SP500+SP400 with **deliberate anti-AI sector quotas** (40 Financials, 40 Healthcare, 38 Industrials, …, only 12 tech) — because momentum-cap analysis showed the AI names = ~6 effective bets. `tickers_metadata.csv` maps ticker→bucket→tier. Plus a small watchlist (predictions-only, excluded from accuracy). **Rule: no comment lines in tickers.txt** (production readers don’t filter them — caused a Jun 12 outage).

## 6. Models (current state)

- **Per-ticker XGB direction ensemble:** 394 tickers × 3 horizons. Depth 2, 30 trees, λ=10, isotonic-calibrated. **Status: SHADOW ONLY** — kill-switch forces BUY→HOLD; dashboard shows both kill-on/kill-off views. Honest OOS ~0.51 AUC. Kept because: infrastructure, feature-importance research bed, and h=5 shows the strongest fitness gradient.
- **Fitness scorer** (`analysis/fitness_scorer.py`): per-(ticker,horizon) Sharpe/turnover/fitness on clean labels. Jun 11: 378 models, median fitness h=1 0.53 / h=3 0.96 / h=5 1.27 — **monotone: longer horizons = more signal.** Re-scores: h=3 ~Jun 18, h=5 ~Jun 25.
- **Hysteresis** (`signals/generator.py`): asymmetric entry/exit per horizon (ENTRY 0.80/0.60/0.60, EXIT 0.65/0.50/0.50), stateful.
- **D6 event gate:** earnings-proximity risk gates (Tier A ≤7d & IV≥8% etc.).
- **Intraday model: broken, deferred** (do not trust; XGB retrain queued Jun-Jul).
- **Sink-level integrity guard:** `_validate_buy_signal()` mirrors generator hysteresis, downgrades invalid BUYs pre-INSERT (defense vs the May 12 unknown-writer incident).

## 7. Validated Alphas

**Momentum (THE live candidate).** Cross-sectional 6-1 and 12-1, top-decile (rank≥0.90), 20-trading-day basket holds, inverse-vol weights, bucket cap 3. Validation: +3.99pp edge vs universe, net Sharpe +1.24 walk-forward, survived purged WF + attack battery. OSAP back-check: published Mom12m still +16%/yr since 2015 (the survivor of the survivors). **In live shadow since May 29** — nightly logging (~39 BUY candidates at 394 names). **Promotion gate Jun 26-29:** ≥20 resolved picks, edge >+2pp vs field, positive mean, ≥60% weeks positive (`scripts/momentum_promotion_check.py`). Caveat on record: backtest panel was survivorship-tilted; the shadow is forward and immune; Jul re-validation will use listing-aware panels.

**Quality/Value (validated-parked, Jun 10).** gp (gross-profit/assets), op (oper-income/equity), ep (earnings yield) — monthly top-decile books on PIT fundamentals. Residual alpha vs SPY: +9.6/+8.5/+8.0%/yr (IR 0.68-0.75), residual corr vs momentum 0.25/0.17/0.07, both eras positive. **Methodological key:** raw corr between long-only books is dominated by shared beta-1; the correct distinctness gate is **residual (beta-stripped) correlation.** bm killed (residual 0.43). Parked because: not monetizable in current product shape (see C1).

## 8. The Graveyard (killed, with reasons + unlock conditions)

|Idea                                 |Verdict           |Why                                                                                                            |Unlock                                        |
|-------------------------------------|------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------|
|1-5d direction trading               |KILLED (permanent)|AUC ceiling is real; 0.51 honest                                                                               |none at days-scale                            |
|SELL signal (prob_up<0.30)           |NO-SHIP, degraded |Jun 12: n=608, acc 52.6%, avg_ret **+0.837%** = counter-signal                                                 |Jul 12 re-eval decides closing cadence        |
|C1 combiner (long-only)              |CLOSED            |4 books correlate 0.57-0.86 (shared beta) despite residual decorrelation; all combos lose to mom-alone +1.53   |more streams (6+) or higher IRs               |
|C1 combiner (beta-hedged)            |CLOSED            |hedge works (corr→0.07-0.58, DD halves) but costs Sharpe 1.53→1.02; combo adds only +0.04                      |same + shorts capability                      |
|Insider events (mega-cap)            |FAIL              |2,182 buys/7yr = sample-starved; +7% mean is skew mirage (median +0.6%); positive years = crash rebounds (beta)|**SP400 data (crawl in flight) → Jul re-test**|
|TSMOM family (10 ETFs)               |CLOSED            |v2 passed gates but failed attack A2 (58% P&L from SPY+QQQ = equity creep); window-kindness confirmed          |futures + shorts infrastructure               |
|value_bm                             |KILLED            |residual corr 0.43 vs momentum — book-value overlaps momentum picks here                                       |different universe                            |
|M16 macro regime gate                |NO-SHIP           |DD improvement 6% < pre-registered bar; Sharpe cost 0.14                                                       |none stated                                   |
|PEAD                                 |KILLED            |OSAP: EarningsSurprise +0.4%/yr post-2015 — dead for everyone                                                  |none                                          |
|Lazy Prices, ST-reversal, LT-reversal|KILLED            |published decay / cost mirage / negative                                                                       |none                                          |
|A8 pooled blend                      |KILLED (leak era) |bugged labels + blend artifact; confidence-cap finding retracted on clean data                                 |n/a                                           |
|Sector/dollar neutralization         |REJECTED          |kills 50-90% of edge; the edge IS directional sector beta                                                      |product change                                |
|Intraday signals                     |BROKEN/DEFERRED   |DOWN h=1 39% acc                                                                                               |Jun-Jul retrain plan                          |

**Cross-cutting strategic finding (the most important sentence in this doc):** five independent hunts all died the same way at 115-159 tickers — **breadth, not signal choice, is the binding constraint** (IR = IC·√breadth). That’s why the universe expanded and why the July re-test battery exists.

## 9. Methodology (the actual moat)

1. **OSAP pre-screen first** (`scripts/osap_prescreen.py`, Chen-Zimmermann 212-predictor library): if the published version decayed to zero, don’t build it.
1. **PIT discipline everywhere:** filed_date (never period_end), filing-date entries for events, no same-day leaks (allow_exact_matches=False), walk-forward weights only.
1. **Pre-registered gates before running:** sample floors, accuracy/IR bars, era-stability — written down, then the test runs once.
1. **Attack batteries after passing:** concentration attacks (is it 2 tickers?), era splits (window-kindness?), skew checks (mean vs median), beta-stripping (is it just the market?).
1. **Kill with unlock conditions:** nothing is “maybe later” — it’s dead until a *named, measurable* condition changes.
1. **Rule #1 done-definition:** a feature isn’t done until committed + tested + imported in builder + visible in panel + source on cron + nonzero Pipeline B importance + documented.
1. **Honest metrics:** Wilson CIs, Bayesian shrinkage, PSR-guarded Sharpe display, deflated expectations (expect 26%+ OOS haircut per McLean-Pontiff).

## 10. Current In-Flight (as of Jun 12)

- Pipeline B manual rerun: minting first models for the 245 new tickers (multi-hour)
- Insider expansion crawl: backfilling Form-4 2019+ for new names (~60% done)
- Analyst revision snapshots: accruing weekly since Jun 12 (signal testable ~Sep-Dec)
- Momentum shadow: 9 nights logged, 302+ BUY candidates, zero resolved yet (first resolutions ~Jun 26)

## 11. Roadmap (dated + event-gated)

|When              |What                                                                                                                         |Decision                                          |
|------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
|Jun 18 / Jun 25   |Fitness re-score h=3 / h=5                                                                                                   |model-quality tracking                            |
|**Jun 26-29**     |**Momentum promotion verdict** (first resolved 20d cohorts)                                                                  |**GO → first live signal; NO → record + diagnose**|
|Jul 12            |SELL signal re-eval                                                                                                          |close the cadence if still counter-signal         |
|~Jul (event-gated)|**Post-expansion re-test battery:** insider on SP400, momentum at 394 (survivorship-aware), qv at 394, C1 re-open if IRs lift|the breadth thesis put to the test                |
|Aug               |VRP (variance risk premium) stream build                                                                                     |candidate 5th stream                              |
|Sep 15            |Inst + fundamental feature importance check                                                                                  |nonzero or drop                                   |
|Sep-Dec           |Analyst-revisions signal first testable                                                                                      |candidate 6th stream                              |
|Jun-Jul           |Intraday XGB retrain plan (A-D steps)                                                                                        |fix or kill intraday                              |

## 12. Where You Can Take This to the Next Level

Ranked by (value × evidence) ÷ cost, from the standing build menu:

1. **Learning-to-rank honest eval** — `train_global_ranker.py` (LightGBM lambdarank) exists but was never honestly evaluated (purged WF, rank-IC). Literature: ~3× Sharpe vs classification (Poh et al. 2021). Days of work, file exists.
1. **GKX-style monthly pooled model** — 394 names finally makes the monthly cross-section viable; the fundamental features are the input layer, built. The natural academic-grade upgrade.
1. **C1 re-open at scale** — if the Jul re-test lifts stream IRs and/or insider validates as stream 5 (VRP as 6), the combiner math changes; HRP machinery already written (`analysis/c1_combiner.py`, `c1_hedged.py`).
1. **Stat-arb / PCA residual reversal** — the within-sector negative decile spread is a mean-reversion signature; market-neutral framing sidesteps the long-only ceiling *if* the product can ever short.
1. **Meta-labeling + fractional Kelly sizing** — untested Stage D; converts low-precision signals into sized trades.
1. **Conditional autoencoder / IPCA** — the documented state-of-the-art for cross-sectional returns; biggest build, highest ceiling; honest caveat: returns halve ex-microcaps (Avramov).
1. **Hygiene queue:** TTM normalization for fundamentals; survivorship-aware panels for all re-validations; qv production fast-path; regime note in promotion verdicts.

**Reading list to internalize the system’s worldview:** Lopez de Prado (10 Reasons ML Funds Fail; AFML), Grinold-Kahn (Fundamental Law), Gu-Kelly-Xiu 2020, McLean-Pontiff 2016, Kakushadze 101 Alphas, Chen-Velikov (costs), Chen-Zimmermann (OSAP), Poh et al. 2021 (ranking). The repo docs that matter: `docs/strategic_review_jun9.md`, `docs/models_by_horizon_jun9.md`, `docs/MASTER_TODO_LIST.md` (living), `Research_Report.md`, `The_Complete_Quant_Build_Menu.md`.

-----

*Working norms: terse, evidence-first, pre-register before testing, kill honestly, commit everything, no deferred work that can run now. Welcome aboard.*