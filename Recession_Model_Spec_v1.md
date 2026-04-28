# Recession Prediction Model — Locked Specification v1.0

*Prepared April 27, 2026 · Standalone research project · No coupling to ML Quant Fund production pipeline*

---

## 0. Document Purpose

This is the locked spec for a multi-target recession-prediction model. Every methodological choice has been deliberated and committed. New decisions during build go in §11 (Open Questions) and require explicit revision of this doc, not silent drift.

**This is a separate project from the ML Quant Fund.** It reads no production tables, writes to no production tables, and lives in its own SQLite database (`recession.db`). Any future integration with the equity model is gated behind a separate validation step (§10).

---

## 1. Objectives

Build a recession/regime-prediction system that:

1. Outputs probabilities for **4 distinct targets** at **4 horizons** (16 probability series total)
2. Reports performance via **3 evaluation metrics** per target × horizon (48 evaluation cells)
3. Surfaces a **single composite probability** for at-a-glance use, plus drill-down per-target detail
4. Is methodologically defensible to a Fed/BIS-grade economist (real-time data handling, proper validation, calibration, parameter-stability tests)
5. Is operationally simple: one new Streamlit page, one new SQLite DB, one new monthly cron

**Non-objectives** (explicitly out of scope for v1):
- Real-time intraday predictions (recessions are not an intraday event)
- Direct trading signals (this is a regime classifier, not a strategy)
- Modification of any existing ML Quant Fund component

---

## 2. Targets — what we're predicting

Four targets, fitted independently. Each target answers a different question.

### 2.1 NBER Recession (T1)
- **Definition:** FRED `USREC` series. Binary, 1 = NBER-dated recession month.
- **Source:** NBER Business Cycle Dating Committee, retrieved via FRED.
- **Events post-1960:** ~10.
- **Question answered:** Is the US in an officially-dated recession?
- **Lag concern:** NBER declares retrospectively. We label months as recession only as of the *announcement date*, not retroactively. Critical for honest backtesting.

### 2.2 Market Drawdown (T2)
- **Definition:** S&P 500 closing price ≥15% below its 12-month rolling maximum, monthly observation.
- **Source:** FRED `SP500` (or Yahoo for longer history).
- **Events post-1960:** ~12.
- **Question answered:** Is the equity market in a meaningful drawdown regime?
- **Why 15% not 20%:** at 20% threshold there are only ~6 events post-1960; insufficient to fit a model with more than 1–2 features without overfitting. 15% gives ~12 events while still capturing material market stress.

### 2.3 AI Kill-Switch Composite (T3)
- **Definition:** Binary, 1 if **2 or more** of the 5 macro triggers from `AI_Investment_Thesis_and_Playbook.md §12` fire in a given month:
  1. Top-4 hyperscaler 2027 capex guide ≤ 2026 (post-2020 only — pre-2020 use proxy: tech sector capex YoY ≤ 0)
  2. NAND or DRAM contract prices declined QoQ for 2 consecutive quarters (post-2010; pre-2010 use semi industry sales YoY)
  3. ≥2 enterprise AI productivity studies report null/negative in 90 days (post-2023; before that, label as 0)
  4. NVDA top-4 customer concentration > 70% (post-2015; before, label as 0)
  5. >10 GW announced data-center cancellations in 90 days (post-2018; before, label as 0)
- **Backtest reality:** Three of five triggers don't have historical data extending to 1960. We backfill with the proxies above and treat the AI kill-switch target as **post-2010 only** for primary evaluation, with a longer-history secondary evaluation using only triggers 1, 2 (proxied).
- **Events:** TBD on data pull; expect 4–8 in 2010–2025 window.
- **Question answered:** Is the AI capital cycle plausibly breaking down in a way that historically preceded equity stress?

### 2.4 Composite (T4)
Three views computed from T1–T3 probabilities, all displayed:

- **T4a Equal-weighted:** `(P(T1) + P(T2) + P(T3)) / 3`. Headline number for the at-a-glance box.
- **T4b Constrained data-driven:** weights estimated by regressing forward equity stress on the 3 component probabilities, subject to `w1 + w2 + w3 = 1` and each `wi ∈ [0.1, 0.7]`. Floor prevents any model being dropped; ceiling prevents overfitting to one.
- **T4c Max:** `max(P(T1), P(T2), P(T3))`. Used as the "any-target-firing" warning indicator; not a calibrated probability.

---

## 3. Horizons

Four prediction horizons, all forecast monthly:

- **H1: 1 month ahead** — coincident-to-near-term
- **H2: 3 months ahead** — short-term
- **H3: 6 months ahead** — medium-term
- **H4: 12 months ahead** — primary horizon (matches NY Fed standard)

For each (target, horizon) pair: at month *t*, predict `P(target=1 in month t+h)` for h ∈ {1, 3, 6, 12}.

---

## 4. Feature Set

21 features, organized in 8 tiers. Designed to capture all major recession transmission channels identified in academic and central-bank literature.

### 4.1 Yield curve & credit (Tier 1)
- `T10Y3M` — 10y minus 3m Treasury spread (NY Fed probit input)
- `BAA10Y` — ICE BofA HY OAS
- `EBP` — Excess Bond Premium (Gilchrist-Zakrajšek 2012). Strongest single non-yield-curve recession predictor in published literature.

### 4.2 Labor market (Tier 2)
- `SAHMREALTIME` — Sahm rule indicator (real-time vintage)
- `JTSQUR` — quits rate (JOLTS); proxies labor market tightness
- `ICSA` — initial claims, 4-week moving average

### 4.3 Real activity (Tier 3)
- `CFNAI` — Conference Board LEI 6-month change
- `ISRATIO` — manufacturing & trade inventory-to-sales ratio
- `PERMIT` — building permits, YoY % change
- `INDPRO` — industrial production, 6-month change
- `NAPMPI` — ISM PMI (binary <50 dummy + level)

### 4.4 Financial conditions (Tier 4)
- `NFCI` — Chicago Fed National Financial Conditions Index
- `SP500` — 6-month return
- `DTWEXBGS` — broad trade-weighted dollar index, 12m change

### 4.5 Monetary stance (Tier 5)
- **Real Fed funds gap:** `DFF − inflation_expectations − r*_HLW`, where `r*_HLW` is the Holston-Laubach-Williams real neutral rate estimate. Captures whether monetary policy is restrictive vs equilibrium.

### 4.6 Credit supply (Tier 6)
- `DRTSCILM` — SLOOS net % of banks tightening C&I lending standards

### 4.7 Global (Tier 7)
- China credit impulse (BIS or Bloomberg derivation; leads global cycle ~9 months)
- Copper/gold ratio (risk-on/risk-off proxy)
- Oil 12-month change (`DCOILWTICO`)

### 4.8 Sector — AI cycle (Tier 8, post-2010 only)
- Hyperscaler capex YoY (Mag 7 aggregate)
- Memory contract price index (TrendForce or DRAMeXchange)

### 4.9 Detrending and stationarity

For non-stationary series (`INDPRO`, `SP500` levels, etc.), apply **Hamilton (2018) regression filter** rather than HP filter:

> `y(t+h) = β0 + β1·y(t) + β2·y(t-1) + β3·y(t-2) + β4·y(t-3) + ε(t+h)`, with h=24 for monthly data. Cycle = ε.

Hamilton's filter: purely backward-looking (no end-point bias from real-time use), no spurious cycles (HP creates business-cycle oscillations from random walks per Cogley-Nason 1995), no arbitrary smoothing parameter. HP is rejected because end-point bias would silently inflate backtest accuracy.

### 4.10 Dimensionality reduction

With 21 features and ~720 monthly observations (with ~10–12 events for some targets), raw-feature regressions risk overfitting. Standard practice:

- **Baseline 1:** raw features with L1/L2 regularization (Lasso/Ridge).
- **Baseline 2:** PCA on the feature matrix → 3 principal components → probit on PCs.
- Report both. The PCA approach is robust to multicollinearity (yields, spreads, credit are correlated); the regularized approach is interpretable.

---

## 5. Real-Time Data Discipline

### 5.1 ALFRED vintages
Required for all **survey/index/national-accounts data** that gets revised:
- `USREC`, employment series, GDP, `CFNAI`, `ISRATIO`, `INDPRO`, ISM, `JTSQUR`, `ICSA`, retail sales, building permits, consumer sentiment

Not required for **transactionally-recorded market data** (no revisions exist):
- Yields, spreads, equity prices, oil, FX, copper, gold

Pre-1990 data is largely unavailable in ALFRED. For that period, current-vintage data is used with an explicit caveat in model documentation. **Backtest performance pre-1990 is overstated by an unknown but real amount and should not be used to claim model accuracy.**

### 5.2 Publication-lag handling
Each feature has a `feature_available_date(t)` function. When predicting at month *t*, day *d*:
- Employment for month *t-1*: available ~day 5 of month *t*
- GDP for quarter *q-1*: available ~day 30 of first month of quarter *q*
- Yields, spreads: available T+0
- ISM: available ~day 1 of month *t* (for month *t-1* data)
- SLOOS: quarterly, ~6 weeks after quarter end
- ALFRED queries return `(value, vintage_date)` tuples; we only use observations where `vintage_date ≤ feature_available_date(t)`.

Predictions are stamped at month-end. Use only data available by month-end.

### 5.3 NBER labeling discipline
- Label month *m* as recession (=1) only if NBER announced by date *d*.
- This means early-recession months stay labeled 0 in the training set until NBER's announcement.
- Removes the look-ahead bias that contaminates most published recession backtests.

---

## 6. Models

Three model families per (target, horizon), plus forecast combination. Three is the right number for this sample size — equal-weighted forecast combination across diverse model families consistently beats any single tuned model on small macro samples (Stock-Watson 2004; Timmermann 2006).

### 6.1 M1: Static Probit
- Specification: `P(y_{t+h}=1) = Φ(α + β'X_t)`
- The NY Fed / Estrella-Mishkin standard.
- Fit via maximum likelihood.
- Fully transparent; coefficients are interpretable.

### 6.2 M2: Dynamic Probit (Kauppi-Saikkonen)
- Specification: `P(y_{t+h}=1) = Φ(α + β'X_t + γ·y_{t+h-1} + δ·P_{t+h-1})`
- Adds lagged binary outcome and lagged probability.
- Outperforms static probit at multi-month horizons (h=6, h=12).
- Per Kauppi-Saikkonen (2008).

### 6.3 M3: Filardo TVTP Markov-Switching
- Hamilton (1989) regime-switching with time-varying transition probabilities (Filardo 1994).
- Two latent states (expansion, contraction).
- Transition probabilities depend on covariates: `P(state_t=R | state_{t-1}=E, X_t) = Φ(α_E + β'_E X_t)`.
- Theoretically aligned with how recessions actually work (regime change conditional on observables).
- Implementation: `statsmodels` `MarkovRegression` with TVTP extension or custom Kalman-EM.

### 6.4 M4: XGBoost
- Captures non-linearities and interactions the linear models miss.
- Heavily regularized (max_depth ≤ 4, min_child_weight ≥ 5, subsample 0.7, L2 ≥ 1.0).
- Class-weighted loss (no SMOTE — synthetic recession months are economically meaningless).
- Calibrated via Platt scaling on a held-out fold.

### 6.5 Forecast combination (per target × horizon)

Two combinations reported:
- **C1: Equal-weighted** — `(M1 + M2 + M3 + M4) / 4`. The "forecast combination puzzle" finding (Smith-Wallis 2009): equal weights consistently beat estimated optimal weights on small macro samples. This is the primary combination.
- **C2: Constrained data-driven** — Granger-Ramanathan weights with floor/ceiling constraints (each `w_i ∈ [0.1, 0.5]`, sum to 1). Reported alongside C1 to show whether data-driven weights agree with equal weights. Divergence is itself a signal of overfitting.

For each (target, horizon) we publish 6 probability series: M1, M2, M3, M4, C1, C2. C1 is the canonical published number.

---

## 7. Validation Protocol

Central-bank-grade evaluation. Designed to be defensible at a Fed seminar.

### 7.1 Sample splits
Three sub-samples, fit independently, reported side-by-side:
- **Pre-1990:** 1960–1989. Pre-Greenspan-Bernanke regime; gold-standard exit through Volcker disinflation. Limited ALFRED coverage.
- **1990–2019:** Post-Volcker, pre-COVID. Most relevant pre-pandemic regime.
- **2020+:** Post-COVID. With explicit COVID dummy. May only have 1–2 cycles.

Showing parameter instability across these periods is a feature, not a bug. If the model is unstable (likely), the dashboard surfaces it.

### 7.2 Out-of-sample evaluation
- **Recursive expanding-window evaluation** with retraining at each step. Matches production behavior: at month *t*, refit on data up to *t-1*, predict *t+h*. Standard in Pesaran-Timmermann (2009).
- **Purged walk-forward** (López de Prado 2018) where target horizon overlaps training: drop training observations whose forward outcome falls in the test window. 12-month embargo at each fold boundary.

### 7.3 Inference
- **Block bootstrap** (Politis-Romano 1994) — resample contiguous 24-month blocks to preserve serial correlation in the recession indicator. Used for confidence intervals on out-of-sample metrics.
- **Diebold-Mariano test** (1995) — formal pairwise comparison of model accuracy. Reports whether M2 beats M1 statistically, etc.
- **Reality Check** (White 2000) — controls for false discoveries when comparing many model variants. Required because we test 4 models × 4 targets × 4 horizons = 64 comparisons.

### 7.4 Metrics — three per (target × horizon × model)

**Primary: AUC**
- Threshold-free, uses full probability distribution.
- Reported with block-bootstrap 95% CI.
- Model selection driver.

**Secondary: Hit rate**
- At threshold = sample base rate (e.g., 13% for NBER).
- Easily interpretable.

**Tertiary: Sharpe uplift on gated strategy**
- Baseline: top-decile-BUY equal-weighted strategy on a generic monthly equity panel (any standard 1m-momentum cross-sectional rank). Held 3 days, monthly rebalance. **This is a synthetic baseline; it has zero connection to the ML Quant Fund signal generator.**
- Treatment: same strategy, but BUY signals suppressed when recession_prob > threshold (threshold optimized on training data, validated out-of-sample).
- Metric: Δ Sharpe (treatment − baseline), with block-bootstrap CI.

**Calibration diagnostics (always reported):**
- **Brier score** decomposed into reliability + resolution + uncertainty (Murphy 1973).
- **Calibration plot** — predicted vs observed frequency in 10 bins.
- **PIT (probability integral transform)** for density forecast evaluation.

### 7.5 Output matrix

The dashboard's evaluation tab displays a matrix:

| Target | Horizon | Model | AUC (95% CI) | Hit Rate | ΔSharpe | Brier |
|---|---|---|---|---|---|---|
| T1 NBER | 1m | M1 | … | … | … | … |
| T1 NBER | 1m | M2 | … | … | … | … |
| ... (6 models × 4 horizons × 4 targets = 96 rows; 4 cells × 6 metrics ≈ 576 numbers) |

Plus per-sub-sample versions (×3 for pre-1990 / 1990-2019 / 2020+).

---

## 8. Outputs and Storage

### 8.1 New SQLite database: `recession.db`

Tables:

- **`features_monthly`** — `(month, feature_name, value, vintage_date)`. Long format for ALFRED-aware queries.
- **`predictions`** — `(prediction_date, target, horizon, model, probability, lower_ci, upper_ci, fit_sample, run_id)`.
- **`triggers`** — `(month, trigger_id, trigger_name, fired, value)`. The 5 AI kill-switch triggers, evaluated monthly.
- **`metrics`** — `(target, horizon, model, sample, metric_name, value, ci_lower, ci_upper, n_obs, run_id)`.
- **`runs`** — `(run_id, run_timestamp, code_version, n_features, n_obs, notes)`.

### 8.2 Code structure (new repo subdirectory)
```
ml_quant_fund/
├── recession/                 # NEW, isolated subtree
│   ├── __init__.py
│   ├── data/
│   │   ├── alfred_client.py        # ALFRED vintage queries, cached
│   │   ├── feature_pipeline.py     # 21-feature builder, with availability rules
│   │   └── targets.py              # T1, T2, T3 label generation
│   ├── models/
│   │   ├── m1_static_probit.py
│   │   ├── m2_dynamic_probit.py
│   │   ├── m3_markov_switching.py
│   │   ├── m4_xgboost.py
│   │   └── combine.py              # C1 equal, C2 constrained
│   ├── eval/
│   │   ├── validation.py           # walk-forward, purge, embargo
│   │   ├── bootstrap.py            # block bootstrap
│   │   └── tests.py                # Diebold-Mariano, Reality Check
│   ├── runner.py                   # monthly cron entry
│   └── tests/
└── ui/pages/
    └── 14_Recession_Regime.py      # NEW Streamlit page
```

No imports from existing ML Quant Fund modules. No writes to `accuracy.db`. Pure isolation.

### 8.3 Cron schedule
- Runs monthly, day 7 (Vietnam time), 8 AM. Most macro releases (employment for prior month, ISM, ALFRED revisions) are out by then.
- Triggered separately from the existing 7 AM main pipeline; doesn't share state.
- One run produces: feature pull → all 4 models × 4 horizons × 3 targets refit → predictions written → metrics updated.
- Quarterly retrain (full refit); monthly is just feature update + new prediction.

---

## 9. Dashboard — Page 14 (or whatever number you slot)

### 9.1 At-a-glance box (top of page)

```
┌────────────────────────────────────────────────────────────────┐
│  COMPOSITE REGIME PROBABILITY (12-month horizon, equal-wtd)    │
│                                                                │
│         ┌──────┐                                               │
│         │ 24%  │  ← large number, color-graded                 │
│         └──────┘                                               │
│                                                                │
│   Target          1mo    3mo    6mo    12mo                    │
│   T1 NBER          2%     5%    11%    19%                     │
│   T2 Drawdown      8%    14%    22%    31%                     │
│   T3 AI K-switch   4%     9%    16%    24%                     │
│   T4a Composite    5%    10%    17%    25%                     │
│                                                                │
│   Nearest trigger: HY OAS widening (+28bp/30d)                 │
│   Disagreement (max - min): 12pp ← model uncertainty signal    │
└────────────────────────────────────────────────────────────────┘
```

Color rules: green <15%, yellow 15–35%, red >35%. Conservative thresholds; tighten after a year of out-of-sample data.

### 9.2 Detail page sections

**§1 Current Regime Snapshot**
- Composite probability (large gauge)
- 12 sparklines: 4 horizons × 3 targets, last 24 months
- "Disagreement" panel: max(P) − min(P) across the 4 models per target. High disagreement = model uncertainty.

**§2 Model Comparison**
- Side-by-side: M1 / M2 / M3 / M4 / C1 / C2 probabilities for the 4 horizons × 3 targets.
- Toggleable overlay vs published external models (NY Fed probit, Chauvet-Piger, Sahm).
- Download as CSV.

**§3 Feature Contribution**
- For M1/M2: coefficient bar chart with confidence intervals.
- For M4: SHAP summary plot.
- For M3: smoothed regime probability + transition probabilities over time.
- "What's driving the current reading" — top 3 features by contribution this month.

**§4 Trigger Monitoring (AI Kill-Switch)**
- 5 triggers as traffic lights (green / yellow / red).
- Current value of each underlying indicator with threshold marker.
- Historical trigger time series.

**§5 Backtest Performance**
- The 96-row metrics table from §7.5 (filterable).
- Calibration plots per target × horizon (toggleable).
- Diebold-Mariano test results: which models statistically beat which.
- Sub-sample comparison: pre-1990 / 1990-2019 / 2020+ stability.

**§6 External Benchmarks**
- NY Fed yield curve probability (live)
- Chauvet-Piger smoothed recession probability (FRED RECPROUSM156N)
- Cleveland Fed yield curve model
- Sahm rule indicator
- Sell-side estimates (Goldman, JPM, etc.) — manual update or scrape

**§7 Methodology**
- Plain-language explanation of each model.
- Caveats: real-time vs revised data, COVID-as-outlier, parameter stability warnings.
- Link to this spec doc.

**§8 Run Log**
- Last fit date, next scheduled refit
- Data freshness per feature
- Manual "🔄 Refresh Now" button (uses your existing `📦 Run Strategy` / `🔄 Refresh Live` cache pattern)

### 9.3 UI integration rules (from prior conversation)

- New file: `ui/pages/14_Recession_Regime.py` — slot the actual number wherever it fits in your page order.
- Reuses your existing Streamlit cache pattern (`st.session_state["auto_load_cache"]` flag).
- Uses your timezone helpers (`utils/timezone.py`).
- Does NOT modify any other dashboard page.
- Reads only from `recession.db`; never touches `accuracy.db`.

---

## 10. Build Sequence

Two-phase rollout. Phase 1 ships standalone; Phase 2 is gated behind your May 1 validation checkpoint.

### Phase 1: Standalone (post-spec lock, target ~2 weeks)

| Step | Deliverable | Estimate |
|---|---|---|
| 1 | `recession.db` schema + migrations | 0.5 day |
| 2 | ALFRED client with caching | 1 day |
| 3 | Feature pipeline (21 features, with availability rules) | 2 days |
| 4 | Target generation (T1, T2, T3) | 1 day |
| 5 | M1 static probit (baseline; reproduce NY Fed numbers) | 1 day |
| 6 | M2 dynamic probit | 1 day |
| 7 | M3 Markov-switching with TVTP | 2 days |
| 8 | M4 XGBoost with regularization + Platt calibration | 1 day |
| 9 | C1, C2 forecast combination | 0.5 day |
| 10 | Validation: walk-forward, block bootstrap, DM, RC | 2 days |
| 11 | Streamlit page 14 | 2 days |
| 12 | Monthly cron | 0.5 day |
| 13 | Documentation + smoke tests | 1 day |
| **Total** | | **~15.5 days** |

Parallelizable to ~10 calendar days if blocks 5–8 are tackled concurrently.

### Phase 2: Optional integration (post May 1 SELL-signal validation)

After the May 1 SELL-signal validation gate (separate item on your roadmap), test whether adding `recession_prob_12m` (the C1 composite) as a feature in `prediction_features` improves the equity model. Strict gate:

- Hold out 6 months of equity data.
- Train two ensemble variants: with and without `recession_prob_12m` feature.
- Measure Δ hit rate on BUY signals at 3d and 5d.
- Merge only if Δ ≥ +1.0pp on held-out data, with no regression on existing horizons.

If gate fails, the recession dashboard still ships; the feature merge does not. Standalone value remains.

---

## 11. Risks and Open Questions

### 11.1 Known limitations

- **Sample size.** ~10 NBER recessions, ~12 drawdowns, ~4–8 AI kill-switch events. Any model with > ~5 effective parameters risks overfitting. Mitigated by regularization, PCA, forecast combination, and aggressive sub-sample testing.
- **Regime change risk.** The post-2020 macro regime (fiscal dominance, near-ZIRP exit, AI capex super-cycle) may make 1960–2019 patterns non-applicable. Mitigated by sub-sample stability reporting, but not eliminated.
- **COVID outlier.** 2020 recession was 2 months long, exogenously triggered. Mitigated by COVID dummy and a "drop COVID" robustness run.
- **Real-time data degradation.** ALFRED coverage is sparse pre-1990. Backtests over that period overstate accuracy.
- **Causality vs forecasting.** This is a forecasting model, not a structural one. Predictive relationships (e.g., yield-curve → recession) reflect equilibrium expectations, not causal mechanisms. Lucas critique applies to any policy-relevant interpretation.
- **AI kill-switch novelty.** No historical record of "AI capex cycle break"; we proxy with semi industry sales. The target is partially synthetic for pre-2010 history.

### 11.2 Decisions deferred

These are not blocking Phase 1 but should be revisited at first quarterly retrain:

- **Should T2 use 15% or 20% drawdown threshold?** Locked at 15% for v1; revisit if the model's drawdown signal is too noisy in production.
- **Is real-time HLW r* worth the implementation cost?** Yes, but using current-vintage HLW for v1 is acceptable if it saves a week. Tag in code as `TODO: switch to real-time HLW`.
- **Should we add ECRI Weekly Leading Index as a benchmark?** Paywalled; defer unless free access available.
- **Should the AI kill-switch composite be backtested with all 5 triggers as a binary rule (no ML), and reported alongside?** Yes — adds 1 day, useful sanity check. Add to Phase 1 step 4.

### 11.3 Things I committed to and will not change without re-spec

- The 4 targets, 4 horizons, 4 models, 2 combinations.
- ALFRED for revised series; current FRED for market data.
- Hamilton (2018) detrending, not HP.
- Equal-weighted as the headline composite.
- 3 sample splits, side-by-side reporting.
- Recursive expanding-window + block bootstrap + DM + Reality Check validation.
- AUC primary, hit rate secondary, Sharpe tertiary.
- Strict isolation from ML Quant Fund production.

---

## 12. Future Work (post-v1)

Backlog of model extensions, ranked by expected value.

### 12.1 SPX 6m Forward Return Regression (priority 1)
A continuous-target sibling to the recession classifiers.

- **Target:** SPX 6-month forward log return (continuous).
- **Features:** same 21-feature set as the recession model.
- **Model:** ridge regression baseline + XGBoost regression + ensemble.
- **Output:** point forecast + 80% / 95% prediction intervals (via quantile regression or bootstrap).
- **Use cases:**
  - **Position sizing input.** Binary regime models say "risk on/off"; this says *how much* to risk-on.
  - **Cross-validation of the regime models.** If T1/T2/T3 all say "no recession" but this regression says E[R_6m] = −8%, that's a useful disagreement signal.
  - **Higher-power validation.** Regressions use all observations' return information, not just binary labels — much tighter parameter estimates than the classifiers.
  - **Dashboard view.** Add as 5th view on the recession page: "Expected 6m SPX return: −4%, 80% CI: [−15%, +6%]."
- **Why deferred:** different loss function (MSE vs cross-entropy), different evaluation (R², MAE, directional hit rate vs AUC, Brier), different feature treatment. Bundling would compromise both.

### 12.2 Volatility regime model
Realized vol forecast at the 1m and 3m horizon, conditioned on macro state. Combines with §12.1 to produce expected Sharpe.

### 12.3 Severity / depth model
Conditional on a recession occurring, predict severity (peak-to-trough GDP decline, max unemployment rate). Likely unbuildable with 10 events but worth scoping.

### 12.4 Real-time HLW r* estimation
Currently using current-vintage HLW. Re-implement with vintage-aware r* estimates per Lubik-Matthes / Holston-Laubach-Williams real-time series.

### 12.5 Mixed-frequency MIDAS extension
Use daily yield-curve and credit-spread data directly (not month-end snapshots) via MIDAS regression. Captures intra-month variation.

### 12.6 International extension
Repeat the framework for Eurozone, UK, Japan, China. Multi-country panel adds sample size and tests for global vs national factors.

### 12.7 Causal layer
Identify Romer-Romer monetary shocks, Kilian oil shocks, Bloom uncertainty shocks. Trace impulse responses on recession probability. Useful for stress-testing.

---

## 13. Definition of "Done" for v1

Before declaring Phase 1 complete:

- [ ] All 4 targets, 4 horizons, 4 models, 2 combinations producing predictions monthly.
- [ ] `recession.db` populated with 60+ years of feature history (where ALFRED supports).
- [ ] All validation tests run and reported.
- [ ] Dashboard page 14 live and reading from `recession.db`.
- [ ] One full out-of-sample month: predict using only data available at month-end M, evaluate against actuals at M+12 (not possible for 12-month horizon at launch — use shorter horizons for initial validation).
- [ ] Smoke test: NY Fed probit reproduction. Our M1 with just `T10Y3M` should match the published NY Fed series within ~2pp at every historical date. If not, something is broken.
- [ ] Run log shows successful monthly cron for 2 consecutive months.
- [ ] Spec doc updated with any deviations.

---

## 14. Versioning

- **v1.0 (this doc):** Initial locked spec, April 27, 2026.
- Subsequent versions require: revised spec doc with diff, written justification for each change, re-run of any affected validation.
- v1.0 supersedes all prior conversation about the build. Where this doc disagrees with anything I said in the conversation, this doc wins.

---

*End of Recession Model Spec v1.0.*

*Next action: confirm spec, then begin Phase 1 step 1 (`recession.db` schema).*
