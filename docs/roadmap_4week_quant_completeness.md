# 4-Week Roadmap — Closing the Quant Fund Capability Gap

**Created:** May 23, 2026
**Context:** After Path A architectural work, identified that accuracy improvements alone won't deliver fund-grade performance. The other 4 multiplicative factors — position sizing, risk management, execution quality, portfolio construction — are largely missing.

**Scope:** Bring ML Quant Fund to operational parity with a top-quartile systematic equity fund's day-1 infrastructure.

---

## What we HAVE (assets to build on)

- Cross-sectional ML model (GLOBAL) — Path A validated OOS
- Per-ticker fallback model
- Multi-horizon predictions (h=1, h=3, h=5)
- 125-ticker universe + 2 watchlist
- Daily prediction logging + reconciliation (accuracy.db)
- Walk-forward validation
- Risk regime classifier (BULL/BEAR/VOLATILE)
- 92+ features including 8-K events, inst flow, FinBERT sentiment, options data
- Sprint W1 cost model (10bp flat) on fitness scorer
- WAL-mode DB, DuckDB inst features, UW + Massive caching
- A/B testing infrastructure (Path A)

## What we DON'T have (gaps to close)

1. Position sizing logic (flat $ amounts, throws away conviction-based alpha)
2. Portfolio construction (no concentration / sector / volatility caps)
3. Live P&L tracking dashboard
4. Stop-loss / exit signal logic (only entry signals exist)
5. Drawdown circuit breakers
6. Realistic slippage + spread model (10bp flat is too crude)
7. Multi-horizon signal aggregation (each horizon scored independently)
8. Daily P&L attribution by feature
9. Regression test suite for hit rate (predict_proba=0 bug went uncaught yesterday)
10. Model rollback infrastructure
11. Pair trades / sector-neutral strategies (all long-only)
12. Options strategies (equity only)
13. Earnings drift (PEAD) strategy
14. Live paper trading vs benchmark
15. Macro overlay on signal generation
16. GLOBAL retraining cadence in Pipeline B

---

## Week 1: Highest-Impact Foundations

### Item 1.1: Position sizing by prob_eff bucket
**Impact:** 9/10 (single biggest missing piece)
**Effort:** 4-6 hours

Replace flat-$ allocation with conviction-scaled sizing.

| prob_eff bucket | Position weight |
|---|---|
| >= 0.80 | 4x baseline |
| 0.70-0.80 | 2.5x baseline |
| 0.60-0.70 | 1.5x baseline |
| 0.55-0.60 | 1x baseline |
| < 0.55 | 0 (HOLD anyway) |

Implementation:
- New module: `signals/position_sizer.py`
- Reads `prob_eff` from SignalResult
- Outputs `target_weight` field
- Add `position_weight` column to predictions table

Risk-1 audit checklist:
- (a) Audit how signal_df currently flows to any execution layer
- (b) Cap total weight at 1.0 (no leverage by accident)
- (e) Verify expected_return calculation accounts for sizing

### Item 1.2: Stop-loss + exit rules
**Impact:** 9/10 (risk asymmetry)
**Effort:** 6-8 hours

Today the model says BUY but never says when to EXIT. Add:

1. Time-based exit: close position at horizon (h=3 → 3 days)
2. Stop-loss: -2 × ATR from entry
3. Profit target: +2 × ATR from entry
4. Trailing stop: ratchet up at +1 × ATR moves

Implementation:
- New module: `signals/exit_logic.py`
- For each open position, daily check if stop/target/horizon hit
- Log exits to new table `position_exits`

Risk-1 audit:
- (g) Gap-check: ATR-based stops use historical vol; sudden vol spike could blow through
- (h) Verify chain: entry signal → position → exit signal → P&L

### Item 1.3: Live performance dashboard
**Impact:** 9/10 (can't manage what you don't see)
**Effort:** 1 day

New Streamlit page or extension to existing dashboard:
- Today's BUYs with prob_eff and target weight
- Open positions (entry date, P&L %, days to exit)
- Realized P&L this week / month
- Hit rate this week vs 30-day average
- Drawdown from peak
- Sharpe rolling 30 days
- Per-ticker P&L heatmap

Implementation:
- Reads from predictions + outcomes + new position_exits table
- Single page: `ui/pages/2_Live_Performance.py`

Risk-1 audit:
- Data integrity: cross-reference outcomes ↔ predictions before display

### Item 1.4: Hit-rate regression test suite
**Impact:** 7/10 (yesterday's bug)
**Effort:** 4 hours

After every Pipeline B retrain, run automated check:
- Predict on hold-out date (most recent)
- Hit rate must be > 50% (sanity)
- Prob distribution must look non-degenerate (std > 0.05)
- No prediction can be 0.0 or 1.0 exactly
- Fail loud + email alert if any check fires

Implementation:
- `scripts/regression_test_model.py`
- Add to Pipeline B Stage 3

---

## Week 2: Risk Management

### Item 2.1: Portfolio-level risk limits
**Impact:** 9/10
**Effort:** 1 day

Hard rules:
- Max position size: 10% of portfolio
- Max sector exposure: 25%
- Max correlated cluster: 30%
- Max total exposure: 100% (no leverage by default)

Implementation: 
- `signals/portfolio_constructor.py`
- Takes today's BUYs + sizing → applies caps → outputs final orders

### Item 2.2: Volatility-scaled position sizing
**Impact:** 8/10
**Effort:** 4 hours

Modify position sizer:
- target_weight × (target_vol / asset_vol)
- target_vol = 1% daily (≈16% annualized)
- More aggressive on low-vol stocks, less on high-vol

### Item 2.3: Daily P&L attribution
**Impact:** 8/10
**Effort:** 1 day

For each day's realized P&L, attribute to:
- Which BUYs contributed how much
- Which features predicted correctly (high-confidence and right vs high-confidence and wrong)
- Sector / horizon / prob bucket decomposition

Implementation:
- New table `pnl_attribution`
- Daily script after EOD reconciliation
- Dashboard view

### Item 2.4: Drawdown monitor + circuit breaker
**Impact:** 8/10
**Effort:** 4 hours

If portfolio drawdown > X%:
- Reduce all position sizes by 50%
- If > 2X%: stop opening new positions entirely
- Alert via email/notification

### Item 2.5: GLOBAL retraining in Pipeline B
**Impact:** 8/10 (currently manual / will go stale)
**Effort:** 2 hours

Add to `scripts/pipeline_B_train_predict.sh`:
- Stage 3: `python -m models.train_cross_sectional`
- After per-ticker retrain completes
- Verify both per-ticker AND GLOBAL models present before Pipeline C

---

## Week 3: Alpha Research

### Item 3.1: Multi-horizon signal aggregation
**Impact:** 8/10
**Effort:** 3-5 hours

Today h=1, h=3, h=5 are independent. Combine:
- Weighted ensemble: w_1 * prob_1 + w_3 * prob_3 + w_5 * prob_5
- Disagreement penalty (down-weight if horizons disagree)
- Or: only BUY if ≥2 horizons agree

Implementation: 
- Test ensemble weights on Path A validation data
- Add to signals/generator.py

### Item 3.2: Slippage + spread cost model
**Impact:** 8/10
**Effort:** 1 day

Replace 10bp flat with:
- Bid-ask spread per ticker (from UW or historical data)
- Slippage = f(volume, position size, volatility)
- Per-ticker, time-varying costs

Implementation:
- Update `analysis/fitness_scorer.py`
- Verify Path A Sharpe survives realistic costs

### Item 3.3: Earnings drift (PEAD) strategy
**Impact:** 8/10
**Effort:** 1 week (full implementation)
**Mini version effort:** 1 day

Post-earnings announcement drift is one of the most-replicated anomalies. We have post_earnings_1d/3d/5d features already — never built a strategy around them.

Quick version:
- After earnings, if eps_surprise > 1σ AND first-day reaction muted: BUY
- Held for 5 days
- Compare to model BUYs in same period

### Item 3.4: Path A A/B decision review
**Date:** Tue Jun 23, 2026
**Action:** Apply decision matrix from docs/path_a_ab_test_plan.md

---

## Week 4: Operational Maturity

### Item 4.1: Model versioning + rollback
**Impact:** 8/10
**Effort:** 1 day

Auto-snapshot models before each Pipeline B retrain:
- Save to `models/saved/snapshots/{date}/`
- Keep last 7 snapshots
- One-command rollback: `python -m models.rollback --date YYYY-MM-DD`

### Item 4.2: Live paper trading + Sharpe vs benchmark
**Impact:** 9/10 (the real validation)
**Effort:** 1 week

Track hypothetical $100k portfolio:
- Use real BUYs from Pipeline C
- Real fills at next-day open
- Cost model applied
- Daily mark-to-market
- Compare Sharpe to SPY hold

### Item 4.3: Feature lineage tracking
**Impact:** 7/10
**Effort:** 1 day

For each prediction, log:
- Top 5 contributing features (SHAP values)
- Feature staleness (when was each last updated)
- Audit trail: data → feature → model → signal

### Item 4.4: Macro overlay
**Impact:** 7/10
**Effort:** 1 week

Filter or scale signals based on macro regime:
- VIX > 30 → reduce sizing
- Yield curve inversion → tilt defensive
- Fed cycle phase → adjust risk

### Item 4.5: Cost-attribution dashboard
**Impact:** 6/10
**Effort:** 1 day

Track:
- Daily API costs (UW, Polygon, Anthropic)
- Compute costs (CPU-hours)
- Cost per prediction
- Cost per BUY signal
- ROI per dollar spent on data

---

## Deferred (after week 4)

| Item | Why deferred |
|---|---|
| Bayesian/probabilistic models | Big architectural change, after fundamentals |
| Deep learning experimentation | Lower priority than ops gaps |
| Alternative data (satellite, etc.) | Cost-effective only at larger AUM |
| Multi-asset (futures, FX, crypto) | Equity model not yet validated live |
| Options strategies | Need equity validation first |
| Risk-parity portfolio construction | Need vol-scaled sizing first |
| Statistical arbitrage on pairs | Cross-sectional global model is first step |
| Pair trades / sector-neutral | Same as above |
| Phase 4 LLM extraction ($9 budget, 4-5h) | Existing alpha not yet exploited |
| Sessions B/C/D (FinBERT 10-Q, 10-K) | Existing FinBERT not yet driving signals |
| Foreign filer FinBERT 6-K | 10 tickers, marginal |
| Massive options historical IV | 6-8h, low priority |

---

## Honest assessment

**Today's state:** ML model with validated OOS edge but no production-grade trading layer.

**End-of-week-4 target:** Complete trading system that:
- Predicts (DONE — improving via Path A)
- Sizes positions by conviction (week 1)
- Manages risk per position (week 1) and portfolio-wide (week 2)
- Exits systematically (week 1)
- Tracks performance live (week 1)
- Attributes P&L (week 2)
- Retrains automatically (week 2)
- Aggregates multi-horizon (week 3)
- Costs realistically (week 3)
- Versions models (week 4)
- Paper-trades against benchmark (week 4)

**What this roadmap gets us:** Operational parity with mid-tier systematic funds. Not Renaissance, but a serious systematic shop.

**What this roadmap doesn't get us:** Truly novel alpha. The cross-sectional model and 8-K features are good but not unique. Real alpha edge requires alternative data, deeper research, or non-equity strategies.

---

## Sequenced summary

| Week | Items | Cumulative gain |
|---|---|---|
| 1 | Position sizing, stop-loss, dashboard, regression tests | ~30% Sharpe improvement |
| 2 | Risk limits, vol-scaled sizing, P&L attribution, drawdown brake, GLOBAL cron | ~20% Sharpe improvement |
| 3 | Multi-horizon aggregation, slippage model, PEAD, A/B decision | ~15% Sharpe improvement |
| 4 | Versioning, paper trading, lineage, macro overlay, cost dashboard | Validation + operational maturity |

**Cumulative expected Sharpe improvement if everything works: 2-3x current**
(from "model with OOS edge" to "deployed system with realistic risk management")

Bear case: A/B kills Path A; we still have position sizing + risk management which alone improve per-ticker baseline by 20-30%.
