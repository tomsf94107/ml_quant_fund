# Intraday Model Gap Analysis

**Date:** May 4, 2026  
**Trigger:** Memory #5 NEUTRAL=0% review · Validated against 30 trading days, n=3,879 predictions, 128 tickers  
**Status:** Model is broken — not just under-trained, structurally flawed

---

## Bottom line

The current intraday signal is a **dressed-up momentum follower**. It has no predictive power over and above the broad market direction. When the market trends, it works. When the market chops or reverses, it actively loses.

The "model" isn't ML at all. It's a hand-tuned formula combining return, VWAP deviation, and RSI into a "momentum score," then mapped to probability via sigmoid. There is no learned weighting, no out-of-sample validation, no feature selection.

**Do not trade off these signals.**

---

## Performance summary (n=3,879 predictions over 30 trading days)

| Horizon | Signal | Sample | Accuracy | Status |
|---|---|---|---|---|
| 1h | UP | 378 | 52.4% | Coin flip |
| 1h | DOWN | 439 | **39.0%** | **Statistically inverted (4-5σ)** |
| 1h | NEUTRAL | 406 | 53.2% UP rate | Uninformative |
| 2h | (any) | 8-35 | — | Sample too small |
| 4h | UP | 356 | **44.1%** | **Statistically inverted (2-3σ)** |
| 4h | DOWN | 408 | 48.8% | Coin flip |
| 4h | NEUTRAL | 459 | 47.7% UP rate | Uninformative |

---

## What's actually wrong (in priority order)

### 1. The "model" is just current momentum, mapped to a probability

Code (features/intraday_builder.py):


This is a hand-coded formula. No learning. The weights (0.4, 0.3, 0.3) were chosen by eye, not fit to data. There is nothing predictive about it — it just describes what the price did in the past N minutes.

When that information continues forward (trending day), the formula is right. When it reverses (choppy day), it's wrong. Long-run that averages to ~50%.

### 2. Strong correlation with broad market direction = no idiosyncratic edge

DOWN h=1 accuracy by day, sorted by market direction:
- Down days (avg ret < 0): 50-100% accuracy ✓
- Up days (avg ret > 0): 0-25% accuracy ✗

The signal isn't predicting individual ticker moves. It's predicting "the market is down today" → all stocks tagged DOWN → mostly right when market is actually down. Zero alpha over `signal = market_direction_today`.

### 3. Time-of-day clustering at 11 AM ET (89% of predictions)

The intraday cron logs predictions at 11 AM ET (per `daily_runner.log_intraday_snapshot`). 11 AM is structurally a momentum-exhaustion / mean-reversion zone:
- Morning catalyst-driven moves complete by ~10:30
- Lunchtime liquidity drop-off begins ~11:30
- Afternoon drift lacks the conviction of opening hour

A momentum signal fired at 11 AM is fighting against intraday market structure. Predicted moves are forecast through 12 PM (1h), 1 PM (2h), 3 PM (4h) — all in or after the lunchtime mean-reversion window.

### 4. No regime awareness

Performance shifts violently between trending and choppy regimes. The system doesn't know it's in a regime where its signals invert. There is no:
- VIX gate (volatile day → suppress signals)
- Trend filter (already extended morning move → distrust momentum continuation)
- Reversal flag (RSI > 70 with positive momentum → suspect reversal)

### 5. Independent horizon scoring with shared error mode

p1 is built from `mom`. p4 is built from `mom * (1 - late_day_factor*0.4) + vdev * late_day_factor * -2`. Different formulas but **all share the same input variables**. When mom is wrong (wrong regime), all three horizons are wrong simultaneously. They don't disagree because they don't have independent information.

### 6. NEUTRAL band is fine, but the function it filters is broken

The 0.40/0.60 NEUTRAL band is reasonable. But filtering a broken signal doesn't fix it — NEUTRAL just hides cases where the model is unsure of nothing useful.

### 7. No XGBoost/LGB model trained for intraday

EOD models use XGBoost+LightGBM ensemble with 79 features and isotonic calibration. The intraday "model" uses 3 features and zero learning. Per project memory, proper XGBoost retrain was scheduled for May 13.

---

## Why this matters for the day-trading layer

The roadmap (memory #7) calls for intraday → alerts → entry/exit → Alpaca. Building any of that on top of a 49% accuracy "model" with negative edge in inverted regimes loses money.

Before any day-trading layer ships, the underlying intraday model has to be replaced with something with verifiable edge.

---

## Fix plan

### Phase 1 — Stop using broken signals immediately (today)
- Add visible warning banner on `12_Intraday.py` page
- Memory captures: do not trust UP/DOWN/NEUTRAL outputs for trading
- Keep technical indicators (RSI, VWAP, vol surge) — these are factual computations, useful for human review

### Phase 2 — Replace with proper ML (3-5 weekends)
- Train XGBoost on intraday snapshots (already collecting via cron)
- Features:
  - Same as EOD where applicable (vix_close, sentiment, options flow lag)
  - Time-of-day (morning vs afternoon vs lunch)
  - Volatility regime (VIX bucket)
  - Trend strength (ATR-normalized)
  - Recent reversals (5-min return autocorrelation)
- Target: forward 1h / 2h / 4h close-to-close return direction
- Strict walk-forward backtest with embargo
- Calibration via isotonic regression
- Hysteresis bands (entry vs exit)

### Phase 3 — Add regime gating
- VIX > 25 → suppress all directional signals (regime is too noisy)
- Trend strength < threshold → favor mean-reversion features
- Trend strength > threshold → favor momentum features
- Time-of-day weighting (don't fire at 11 AM unless edge demonstrated by hour)

### Phase 4 — Validate before trading
- 60+ trading days of out-of-sample data
- Sample size threshold (>100 per signal type)
- Accuracy threshold (>55% on UP and DOWN)
- Stable across regimes (CI doesn't include 50% in any market state)

---

## Immediate action

1. Add warning banner to intraday page (5 min)
2. Update memory with corrected diagnosis (done)
3. Skip the day-trading layer plan until intraday model is fixed
4. Schedule Phase 2 for after walk-forward + fitness rerun (May–June)

