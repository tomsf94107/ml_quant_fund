# 8-K Gating Backtest — Underpowered, NO Rule Shipped

**Date:** May 23, 2026
**Status:** Investigation complete. No production gating rule ships.
**Companion:** docs/8k_alpha_finding_and_gating_plan.md (prior context)

---

## TL;DR

Backtested Path B (8-K post-prediction gating overlay) on 885 closed
BUYs + 12,858 HOLDs. **At current sample sizes, no rule shows statistically
significant lift.** Interesting non-significant patterns warrant a re-test
in 4-8 weeks once sample doubles.

## Methodology bugs caught (Rule #1 hits)

1. **Silent error in first script:** when joining e_df (unique-date indexed) 
   to df (multi-horizon, duplicate dates), `e_df.loc[d, col]` returned a 
   Series for duplicate dates, silently failing to assign except for h=1.
   First-pass results showed n=0 for h=3 and h=5 exec_change events — a 
   ghost finding. Fixed by switching to dict-based lookup.

2. **Sample sizing not considered before drawing conclusions.** Initial 
   backtest reported "rules don't help" at n=5-28 — utterly underpowered.
   Wilson CIs revealed no claim could be supported.

3. **Selection mismatch:** pooled cross-ticker analysis showed exec_change 
   = -5.4pp on all-rows. I assumed this would survive on BUY-only subset.
   The BUY filter already pre-selects positively; the cross-ticker edge 
   does not extrapolate to BUY-conditional rates.

## Real findings after methodology fixes (Wilson 95% CI)

| Cell | Hit (e=1) | Baseline (e=0) | Edge | CI overlap? |
|---|---|---|---|---|
| h=1 BUY + exec_change | 47.1% (n=17) | 67.5% (n=80) | **-20.4pp** | YES (overlap) |
| h=1 HOLD + exec_change | 51.9% (n=855) | 51.4% (n=4459) | +0.5pp | overlap |
| h=3 BUY + exec_change | 54.9% (n=71) | 54.6% (n=295) | +0.4pp | overlap |
| h=3 HOLD + exec_change | 56.8% (n=621) | 56.7% (n=3306) | +0.2pp | overlap |
| **h=5 BUY + exec_change** | **63.6%** (n=88) | **53.9%** (n=334) | **+9.7pp** | overlap |
| h=5 HOLD + exec_change | 59.6% (n=564) | 57.3% (n=3053) | +2.3pp | overlap |
| h=3 BUY + other_events | 57.8% (n=45) | 54.2% (n=321) | +3.6pp | overlap |
| **h=5 BUY + other_events** | **61.2%** (n=67) | **54.9%** (n=355) | **+6.3pp** | overlap |

**No cell achieves statistical significance.** Point estimates suggest:

- exec_change at h=1 BUY: large bearish effect (n=17, suggestive but unproven)
- exec_change at h=5 BUY: large BULLISH effect (sign FLIPS vs h=1!)
- other_events at h=5 BUY: bullish effect consistent with pooled finding

The horizon-dependent sign flip on exec_change is unexpected. Possible
explanations:

1. Short-term: exec change announcement creates uncertainty → down move
2. 5-day window: market digests, leadership clarity → up move
3. Or just noise at the sample sizes available

## What does NOT ship

- No suppression rule based on eightk_exec_change_30d
- No boosting rule based on eightk_other_events_30d
- No combined inst_flow × earnings × 8-K gating

## What ships (already)

- Polygon-revenue → rev_growth_yoy/qoq features (now populated for 115/125
  tickers after this session's fix). Per-ticker models will use them when
  re-trained, even though models ignored them previously (cross-sectional
  signal, same as 8-K).

## Re-test plan (revised May 23 evening)

The earlier "8 weeks" estimate was optimistic. CI tightens by sqrt(N), so
doubling sample only narrows CIs by 1.4x. To move from "edges ~6-10pp with
overlapping CIs" to "edges with non-overlapping CIs", need ~4x sample.

Schedule:

| Milestone | Date | Decision |
|---|---|---|
| Quick check | **Mon Jul 20, 2026** (8 weeks) | Have point estimates stayed in same direction? |
| Real stat-sig check | **Mon Sep 21, 2026** (17 weeks) | Wilson CIs should start separating if effect is real |
| Final verdict | **Mon Nov 23, 2026** (6 months) | If still ns at this point, declare feature noise |

For each milestone, re-run scripts/backtest_8k_inst_gating.py with proper
Wilson CIs and report:
- Whether sign of edges held
- Whether CIs of suppressed/boosted cells separated from baseline
- Whether to ship any production gating rule

## Implication for Path A (cross-sectional model)

The fact that exec_change behaves DIFFERENTLY at h=1 vs h=5 (in opposite
directions) suggests the signal is not horizon-invariant. A cross-sectional
model would naturally separate horizons into different model heads and
could pick up the per-horizon nuance per-ticker models miss.

This makes Path A MORE valuable, not less. Underway as separate project.
