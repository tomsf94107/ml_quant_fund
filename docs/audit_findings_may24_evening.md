# Audit Findings — Sunday May 24, 2026 (Evening)

Continuation of `audit_findings_may24_morning.md`. Today we ran systematic
experiments on target engineering after discovering that cross-sectional
pooling alone did NOT unlock 8-K/rev/inst features.

## Tier 1 Experiments

| ID | Target | OOS AUC | rev imp | 8-K imp | inst imp | macro | vol |
|---|---|---|---|---|---|---|---|
| A1_pct0 | any positive | 0.633 | 0.000 | 0.000 | 0.000 | 0.699 | 0.003 |
| A1_pct2 | >= +2% in 5d | 0.669 | 0.000 | 0.002 | 0.000 | 0.511 | 0.202 |
| A1_pct3 | >= +3% in 5d | 0.668 | 0.010 | 0.000 | 0.000 | 0.363 | 0.327 |
| A1_pct5 | >= +5% in 5d | 0.677 | 0.007 | 0.000 | 0.000 | 0.249 | 0.348 |
| **A1_pct7** | **>= +7% in 5d** | **0.684** | 0.002 | 0.000 | 0.000 | 0.221 | 0.379 |
| A5_excess_spy_1pct | beats SPY +1% | 0.630 | **0.050** | **0.015** | 0.000 | 0.321 | 0.153 |
| A6_excess_sector_1pct | beats sector +1% | 0.585 | **0.051** | 0.005 | 0.000 | 0.241 | 0.154 |

All 7 models saved to models/research/ with .meta.json metadata.

## Two distinct alpha sources discovered

### Alpha source 1: Big-move detection (A1 family)
- Highest AUC at +7% threshold (0.684)
- Driven by volatility features (volatility_10d, bb_width, volatility_5d)
- Macro features supplement
- **Cross-sectional features (rev/8K/inst) do NOT contribute**
- Predicts: "will this stock have a big up-move in 5 days"
- Use case: position sizing, options trading, finding volatile names

### Alpha source 2: Stock-picking (A5, A6 family)
- AUC lower (0.58-0.63) but engages cross-sectional features
- rev_growth importance 0.050 (vs 0.000 elsewhere)
- 8-K importance 0.005-0.015
- macro 0.24-0.32 (still important but not dominant)
- volatility 0.15
- Predicts: "will this stock beat market/sector"
- Use case: long-short portfolio construction

### Why these are different alphas
- A1 = predict regime/volatility (more market-aware)
- A5/A6 = predict relative performance (true cross-sectional)
- A1 has higher AUC but produces correlated bets (all big movers fire on volatile days)
- A5/A6 has lower AUC but produces decorrelated bets (different stocks each day)
- For Sharpe-style portfolio, A5/A6 likely better despite lower AUC

## Key observations

1. **Threshold sweep monotonic:** AUC increases from 0% to +7% target. The "harder" the event, the cleaner the prediction.

2. **Base rate trade-off:** +7% has only 19% base rate vs 50.7% for "any positive". Fewer opportunities but higher accuracy.

3. **Macro features dominate "any positive" (0.699 of importance)** because predicting market direction IS macro prediction. Once you fix the question to "this stock specifically", macro share drops.

4. **inst features still useless** in all experiments. 97% NaN is too much even with native NaN handling. Wait until ~Sept 2026 for inst feature accumulation to ~6 months.

5. **rev_growth + 8-K finally matter** when target is excess return. The Polygon backfill (2009-2026, 4762 rows) was essential — but only with the right target.

6. **A6 (vs sector) underperformed A5 (vs market)** by 4.5pp AUC. Sector is too tight a benchmark; SPY is the right one.

## Production implications

| Question | Answer |
|---|---|
| Should we change production target? | **Not unilaterally.** A1_pct7 has higher AUC but predicts different signal. Discuss before changing. |
| Should we ship Path A as-is? | Original Path A is A1_pct0 effectively (any-positive target). Not the best. |
| Best target for "stock-picking" goal | A5_excess_spy_1pct |
| Best target for "big-move" goal | A1_pct7 |
| Combined approach? | Could use both: A1_pct7 for sizing + A5 for direction |

## Sequence remaining

Tier 1 not done. Still:
- **A8** — top-decile cross-sectional target (cleanest cross-sectional test)
- **C1** — LightGBM only (model architecture test)

After Tier 1 complete, end-of-day decision on what to productionize.

## Files referenced
- models/research/*.joblib (7 models)
- models/research/*.meta.json (7 metadata)
- data/sector_etf_map.json (125 ticker → SPDR ETF)
- scripts/save_experiment_artifact.py (artifact helper)

## Engineering fixes also committed today
- a56ed40: VIX leak in risk_next_* + audit findings
- 1ebdbb4: Polygon rev backfill (2009-2026) + inst NaN preservation



## Addendum: May 25 2026 (Monday) — Phase epsilon + Phase 1 D shipped

### Phase epsilon (shadow logging) — DEPLOYED

Implemented A/B shadow logging of A1_pct7 model alongside production.
- accuracy/sink.py: added prob_pct7 column + auto-migration
- signals/generator.py: loads PCT7 model, computes today_prob_pct7
- scripts/daily_runner.py: threads prob_pct7 through log_prediction_to_db()

End-to-end verified. Today's Pipeline C run logged 127 prob_pct7 values
(one per h=5 prediction).

### Phase 1 D (interaction features) — DEPLOYED

5 features added to FEATURE_COLUMNS:
- vol_x_short (raw)
- rev_x_low52w (raw)
- vol_10d_self_rank (rolling self-rank 252d)
- vol_zscore_60d (rolling z-score 60d)
- is_squeeze_setup (binary threshold)

2 features dropped during testing:
- short_self_rank, short_zscore_60d — short_pct_float is constant per ticker
  (single yfinance broadcast), rolling stats degenerate.

AUC impact:
- +7% target: 0.7135 -> 0.7192 (+0.57pp)
- A8 target: 0.687 -> 0.684 (slight noise)

### PCT7 v2 deployment

Initial deploy failed: LGBOnlyResult class defined inline in __main__,
joblib pickled the class reference but pickle couldn't find it on load.
Production broke briefly; reverted within minutes.

Fixed with models/wrappers.py — proper importable class. Retrained PCT7
v2 with 95 features (90 original + 5 interactions). Deployed.

OOS AUC: 0.7192 (vs yesterday's 0.6842) = +3.5pp lift on production.

### Architectural finding

Initially planned a polymorphic refactor of predict_proba_ensemble.
After code audit: it already works via duck-typing (only calls
.predict_proba on the loaded object). EnsembleResult.load doesn't
check returned type. So a proper importable class alone was sufficient.

Less code, same outcome. Pragmatism > over-engineering.

### Files added today

- models/wrappers.py
- scripts/monitor_pct7_ab.py
- docs/phase_2H_overlay_spec.md
- docs/monitoring_phase_epsilon.md

### Commits today

a56ed40 -> 1ebdbb4 -> 7385bb1 -> 135af9a -> 79d79cc -> 7fd0110
-> 89b85fe -> 695ab41 -> 3aa166f

9 commits, all on research-track.

### Status of original 4-phase A8 plan

- Phase epsilon (shadow log PCT7): DEPLOYED
- Phase 1 D (interaction features): DEPLOYED in source, awaits Pipeline B retrain
- Phase 1 E (A/B tracking): SUPERSEDED by Phase epsilon (does same thing)
- Phase 2 A (A8 prob as feature): NEXT — needs design
- Phase 2 H (overlay scoring): spec written, implementation pending
