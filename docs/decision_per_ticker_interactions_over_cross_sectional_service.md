# Decision: Why we picked per-ticker interactions (A+C+D+E) over
# a proper cross-sectional service (Option G) — May 25, 2026

## Context

After the A8 finding (top-decile cross-sectional target unlocks 8-K/rev
features), we needed to engineer the corresponding features for production.
Investigated 7 options:

| Option | Description | Effort | Risk |
|---|---|---|---|
| A | Raw-value interactions (vol*short, rev*low52w) | 1-2 hr | Medium (NaN handling) |
| B | Cross-sectional rank in training only | 3-4 hr | HIGH (train/serve mismatch) |
| C | Per-ticker rolling self-rank | 1-2 hr | Low |
| D | Per-ticker z-score (rolling 60d) | 1-2 hr | Low |
| E | Binary squeeze-setup indicator | 30 min | Low |
| F | Deferred to training only | 1 hr | None (but useless) |
| G | Proper cross-sectional service (new schema, cron, integration) | 1-2 days | Medium |

## Decision: A + C + D + E (rejected B, F, G for now)

### Why reject B (cross-sectional rank in training only)

B introduces train/serve mismatch — at training, we have all 125 tickers
for cross-sectional ranking. At prediction, we score one ticker at a time.
Rank values would be missing or wrong in production.

This is the exact bug we spent yesterday fixing (VIX leak in risk_next_*).
We learned: don't repeat the same mistake pattern.

VERDICT: Hard reject. Never ship features with train/serve asymmetry.

### Why reject F (deferred to training only)

If features only exist during training but not at inference, model learns
to depend on signals that aren't there in production. Actively harmful.

VERDICT: Hard reject.

### Why DEFER G (proper cross-sectional service)

G is the architecturally correct solution. Pros:
- Eliminates train/serve mismatch (symmetric data access)
- Reusable: future cross-sectional features can plug in
- Captures true peer-relative signals

Cons:
- 1-2 days of work (new schema, new cron, integration testing)
- Race conditions: what if prediction runs before service updates?
- Schema design needs care (caching, staleness, atomicity)
- Not battle-tested yet

REASONING:
- Phase 1 D is a quick-win iteration. We want to ship today.
- The per-ticker interactions (A+C+D+E) deliver +0.57pp AUC at 2 hr cost.
- G's expected lift is similar (maybe +1-2pp if we get the peer ranks right),
  but at 5x-10x the engineering cost.
- Marginal-value-per-hour favors per-ticker interactions today.

DEFER PLAN:
Q3 2026 sprint when:
- We have more evidence cross-sectional alpha is real (Phase 2 A validation)
- Inst data has 6 months depth (Sept 2026)
- Other phase 2/3 work is done

G becomes the architectural upgrade that unlocks BOTH true peer ranks AND
the inst features that are useless today.

## Why we picked A + C + D + E together

Each captures a different signal:
- A (raw interactions): captures absolute interaction effects
- C (rolling self-rank): captures within-ticker regime shifts
- D (z-score): captures statistical deviation from ticker's baseline
- E (binary indicator): captures specific squeeze setup patterns

Each is per-ticker only. No train/serve mismatch. All compute inside
build_feature_dataframe() so both training and prediction see identical values.

## Audit hits (Rule #1)

- (a) Audit subsystem: confirmed OUTPUT_COLUMNS + FEATURE_COLUMNS dual update needed.
- (b) Silent error: NaN handling in interactions; logged to error if compute fails.
- (c) Flag flip: ML_QUANT_INST_FEATURES NOT needed; these are additive.
- (d) Verify script: built smoke test on AAPL + 4 edge-case tickers.
- (e) Built-not-known: we ran the 30-ticker test BEFORE shipping to production.
- (f) Test patches with real data: SMCI, QURE, BYND, NVDA, AAPL all tested.
- (g) Gap-check: caught short_pct_float constant-per-ticker bug, dropped 2 features.
- (h) Verify chain: code -> build_feature -> train -> predict -> verify.
- (i) Compiled OK != verified: AST checked AND real-data verified.

## Honest assessment

The A+C+D+E approach is tactical, not strategic. Real cross-sectional alpha
extraction (Option G) is the long-term answer. We chose A+C+D+E to ship
incremental value today while planning Option G as the Q3 architectural
upgrade.

If Phase 2 A/H validate the A8 prob_top_decile signal, Option G becomes
high-priority. Until then, today's interactions are sufficient.

## Open risk

These per-ticker interactions are not TRUE cross-sectional. The model still
doesn't see peer-relative signals. If we find that the model underperforms
specifically when peer rank would have mattered, we'll know Option G should
be prioritized.

Monitor: in monitor_pct7_ab.py results, look for systematic bias toward or
away from certain sectors. Sector clustering would suggest peer-rank signal
is missing.
