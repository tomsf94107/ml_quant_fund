
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
