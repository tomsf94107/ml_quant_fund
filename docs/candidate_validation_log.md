
## C1 Stream Validation — 2026-06-19

Method: weekday-robustness → residual-vs-momentum + residual-vs-admitted-streams
(short_pct_float, pc_ratio_snap) → daily-IC t-stat (groupby date, ≥10-day floor).

| candidate | status | basis |
|---|---|---|
| pc_ratio_snap | ADMIT | residual IC t=6.6/8.3, mom corr 0.11, full 597-day history |
| short_pct_float | ADMIT | residual IC t=-5.1/-6.2, mom corr 0.12, full history |
| monday_sentiment | DROP | residual IC ~0 (t=-1.25/-0.14) on cs_rank; signed_power_p05 degenerate. Earlier t=6.85 was transform-selection artifact. |
| intraday_ret | DROP | raw IC ~0 (t=-0.99); corr to return_1d 0.765 (redundant). LPS intraday leg no signal on this universe. premarket_gap already = overnight leg. |
| skew_change | DEFER | decorrelated (max corr 0.053, incl vs iv_skew_snap level) but only 42 usable days. NO signal yet — insufficient history, not failed. RE-TEST ~Sept 2026 when options_skew_history has ~150+ dates. IV-spread/VRP also wait on put_iv/call_iv null fill. |

Admitted streams toward C1 combiner: 2 (pc_ratio_snap, short_pct_float).
Still need ~6 decorrelated for combiner. Remaining gates for the 2 admitted:
both-eras stability + net-of-cost (deferred to post-July battery).
Scratch validators: candidate_validation.py, residual_validation.py,
validate_intraday_ret.py, validate_skew_change.py (keep for re-runs).

Pending re-tests (accrual-gated):
- skew_change: re-run validate_skew_change.py when options_skew_history >= 150 dates (~Sept)
- analyst revisions: testable ~Sept-Dec (analyst_snapshots accruing weekly)

## Tier-1 new-axis sweep — 2026-06-19 (continued)

Built/tested 3 new candidate axes from owned data, full validation gauntlet:

| candidate | status | basis |
|---|---|---|
| intraday_ret (LPS intraday leg, close/open-1) | DROP | raw IC ~0 (t=-0.99); corr to return_1d 0.765 — redundant. premarket_gap already = overnight leg (the leg that matters). Reverted builder edit. |
| skew_change (Δ skew_25d) | DEFER | decorrelated (max corr 0.053 incl vs iv_skew_snap) but only 42 usable days. No signal yet — insufficient history. Re-test ~Sept (accrual cron watches options_skew_history>=150). |
| pead_drift (SUE x 60d linear decay) | DROP | raw IC real (t=3.43/3.75) but DISTRIBUTED OVERLAP — residual vs momentum-only t=1.55/2.43, vs short_float-only t=1.66/2.60, vs BOTH t=0.70/1.36. No single absorber; momentum+short_float jointly leave <t1.4 residual. Not incrementally decorrelated. |

STRUCTURAL FINDING: PEAD earnings-drift edge on this universe is partially
captured by BOTH momentum and short_pct_float (not one alone). Useful for
combiner interpretation — short_pct_float is partly proxying earnings drift.

Tier-1 sweep outcome: 0 new admits. Confirmed pattern — most candidates fail
residualization (McLean-Pontiff decay + crowding). Still 2 admitted streams
total (pc_ratio_snap, short_pct_float). Next: residual momentum (#5, expect
high momentum-family correlation), Lazy Prices 10-K text (#4, deferred — needs
EDGAR text ingestion from scratch).

Scratch validators added: validate_intraday_ret.py, validate_skew_change.py,
validate_pead.py, validate_pead_diag.py
