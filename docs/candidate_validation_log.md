
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

## MAJOR FIX: dead macro/sector feature layer — 2026-06-19

DISCOVERED while testing residual momentum (#5): the entire market/sector/macro-ETF
feature layer was 100% NaN in the panel back to 2024. Root cause: massive_client
routed sector/macro ETFs (SPY, XLK, all sectors, USO, credit ETFs) to Massive,
which (options-focused tier) returns NO ETF data. Failure cached + circuit-broken
-> 0/394 non-null for spy_ret, xlk_ret, all sectors, oil_ret, sector_rel_ret.
vix_close + dxy_ret survived (FRED path). Model trained WITHOUT market/sector
context despite sector-beta thesis.

FIX (commit 62f6cb1): added MACRO_ETF_SYMBOLS set to massive_client _is_index()
routing -> ETFs now fetch via yfinance. Verified: 0/1100 -> 1100/1100 non-null
for xlk_ret/spy_ret/sector_rel_ret/oil_ret at builder level.

DEPLOYMENT: needs panel rebuild (D) + retrain (B) to take effect. Material change —
8 features the model never trained on. Watch post-retrain validation.

BASELINE (pre-fix, directional acc last 60d, to compare after retrain):
  h=1: 60.9% (n=6470)
  h=3: 53.9% (n=6205)   <- weakest, most room for macro/sector context to help
  h=5: 55.6% (n=5395)
  alpha_fitness h=1: avg ic_t 0.35, max 4.23

Interpretation guide: expect biggest lift at h=3 if macro helps. If h=1 drops
and h=3/h=5 dont improve -> added noise, consider regularization or revert.
Also unblocks residual momentum (#5) testing once sector_rel_ret is live in panel.

## Both-eras + cost gates — 2026-06-20

Both admitted streams cleared ALL FOUR gates (weekday → residual-vs-momentum → both-eras → cost):

| stream | h3 era1/era2 t | h5 era1/era2 t | turnover | verdict |
|---|---|---|---|---|
| pc_ratio_snap | 4.88 / 6.65 | 5.02 / 8.70 | 0.0016 | PASS (strengthening era1→era2) |
| short_pct_float | -4.16 / -2.87 | -4.76 / -2.88 | 0.0016 | PASS (mild decay era1→era2, watch) |

Turnover ~0.0016 = near-zero churn (slow snapshots) -> costs negligible, IC≈realized edge.
pc_ratio = stronger/strengthening. short_pct_float = weaker/mildly decaying (short-interest
crowding, but still |t|>2 recent era). BOTH combiner-ready. 2 validated decorrelated streams.

## Residual momentum (#5) — 2026-06-20 — DROP

Now testable (sector_rel_ret live across all 599 panel files post macro-fix rebuild).
Windowed sector_rel_ret (ts_mean/ts_decay w10/w20) tested as residual momentum:
- raw IC ~0 everywhere (t=0.03 to 1.84)
- corr to return_20d 0.79-0.91, rsi_14 0.67-0.79 — momentum-family by construction
- verdict REDUNDANT. (Some residual t=2.67/3.59 are spurious — residualizing a
  near-zero-raw signal against a 0.91-corr feature leaves noise, not edge.)
NOTE: sector_rel_ret as a BASE feature has 7.5 model importance (valuable). Its
windowed momentum just doesn't add a SEPARATE decorrelated stream. Both true.

## "Do all" batch complete — 2026-06-20
- #2 both-eras+cost: pc_ratio + short_float PASS all 4 gates. 2 streams combiner-ready.
- #3/#5 residual momentum: DROP (momentum-family).
- #4 Lazy Prices: separate multi-session build (EDGAR 10-K text from scratch). Not started.
Net: 2 validated decorrelated streams (pc_ratio_snap, short_pct_float). skew_change +
analyst-revisions deferred (accrual). Macro layer resurrected + used (importance 7-11).

## "Combiner" investigation — RESOLVED via Rule-1 audit — 2026-06-20

QUESTION: build a combiner for the 2 validated streams (pc_ratio_snap, short_pct_float)?
AUDIT FINDING: there is no missing combiner. The models (per-ticker classifier,
cross-sectional ensemble, global ranker) ARE combiners. The real issue is the
validated signals are architecturally STRANDED:

1. pc_ratio_snap / iv_skew_snap are COMMENTED OUT of classifier.FEATURE_COLUMNS
   (line 98, "dropped 2026-05-21 train/serve mismatch"). All 3 models import this
   same list, so excluded everywhere.
2. short_ratio IS in the list but 0 importance — SAME root cause: only 62 days in
   prediction_features (2026-03-24+), and per-ticker it's a flat constant.
3. The builder computes pc_ratio via get_pc_ratio_uw() = TODAY'S SNAPSHOT broadcast
   across all dates (builder line 1502-1530). UW gives only current snapshot.
4. The global ranker builds via fresh build_feature_dataframe() calls -> gets
   today's snapshot stamped on history, NOT real history. Uncommenting wouldn't help.
5. The REAL historical signal exists ONLY in the alpha panel parquets (captured
   live day-by-day, 597 days) — which is what validation ran on. NO model trains
   on the panel directly.

CONCLUSION: validated cross-sectional signals (pc_ratio t=8, short_float t=-5) are
REAL but architecturally stranded — their true history lives only in the alpha
panel, and no model trains on the panel. To use them, need a model that trains on
the panel directly (a real new training path), OR a forward-accruing prediction_features
backfill (slow, the 62 days will grow). Per feature_improvement_plan.md item 4, this
was DEFERRED INDEFINITELY ("revisit only if items 2+3 deliver gains & still need edge").

This explains the whole chain: signals validate cross-sectionally on the panel, but
production models (per-ticker + ranker-on-fresh-builder) can't access the historical
cross-section. NOT a tonight-fix. A scoped architecture decision: "train a model on
the alpha panel where the validated signals actually live."
