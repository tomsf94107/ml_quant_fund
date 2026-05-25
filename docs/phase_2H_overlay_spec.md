# Phase 2 H — Overlay Scoring Filter Spec

## Goal

Use PCT7 prob (logged via Phase ε) as an overlay filter on production BUYs.
If a BUY signal has very low prob_pct7, downgrade it.

## Premise

prob_pct7 estimates probability of +7% move in 5 days. Even if per-ticker model
says "BUY" (prob_up > 0.55), if PCT7 says "almost no chance of +7%" (prob_pct7 < X),
the trade is likely a tight/marginal win at best.

This combines two independent models:
- per-ticker model (any-positive direction)
- PCT7 (big-move likelihood)

Both must agree for a confident BUY.

## Rule
if signal == “BUY” and prob_pct7 is not None:
if prob_pct7 < OVERLAY_THRESHOLD:
signal = “HOLD”
overlay_downgraded = 1
overlay_reason = “low prob_pct7”

## Threshold selection

| Threshold | Implication |
|---|---|
| 0.05 | Extremely lenient — only filters ETFs/super-stable names |
| 0.10 | Lenient — base-rate ≈ 13%, threshold at 10% means "PCT7 is non-bearish" |
| 0.13 | Match training base rate |
| 0.15 | Modestly strict — requires above-average +7% likelihood |
| 0.20 | Strict — filters all but high-confidence big-move candidates |

**Recommend: 0.10** (lenient first, tune up after seeing data).

## Implementation

### File: signals/generator.py
Add overlay logic near end of generate_signals(), AFTER hysteresis check but BEFORE returning result.

### Schema
Add column to predictions table:
- `overlay_downgraded INTEGER` (0 = passed, 1 = downgraded by overlay)
- `overlay_reason TEXT` (free-text reason for downgrade)

### Threading
- generator.py computes overlay decision
- daily_runner.py passes to log_prediction
- log_prediction stores in DB

## Rollout

### Phase H.1 — Shadow mode (no actual downgrade)
- Compute overlay decision
- Log to predictions table
- DO NOT change signal
- Monitor for 1 week
- If overlay would have prevented N% of bad BUYs, promote

### Phase H.2 — Active downgrade
- After H.1 validates the rule
- Actually downgrade BUYs that fail overlay
- Position sizer respects HOLD output

## Rule #1 audit

(a) Audit: predictions table needs schema migration
(b) No silent error: if PCT7 model not loaded, prob_pct7 is None → SKIP overlay (don't fail)
(c) Flag flip: OVERLAY_ENABLED env var, default OFF until validated
(d) Verify: live A/B before any active downgrade
(e) Built not known: don't know what threshold is optimal yet
(f) Test: backtest on past predictions
(g) Gap-check: what about h=1 and h=3 predictions? PCT7 only trained on h=5
   -> only apply overlay to h=5
(h) Verify chain: generator -> log -> DB -> query back -> validate

## Validation metric

After H.1 shadow mode for 1 week:
- N BUYs total
- N would have been downgraded
- Of those downgraded, what fraction actually FAILED (return < 0% in 5d)?
- If downgrade-precision > base-fail-rate, overlay is adding value

## Estimated effort

- Schema migration: 15 min
- Generator.py overlay logic: 30 min
- daily_runner pass-through: 15 min
- Testing: 30 min
- Shadow week: passive wait
- Promotion to active: 15 min

Total active engineering: ~2 hours over Tuesday-Wednesday.

## NOT in scope for this phase

- Filtering based on prob_up_global (Path A) — separate decision
- Position sizing changes (that is Phase 3 B)
- Multi-horizon overlay (only h=5)
