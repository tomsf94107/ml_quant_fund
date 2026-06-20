
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

## Panel ranker — RULE-1 AUDIT CORRECTION — 2026-06-20

Pre-audit claims were OPTIMISTIC. Corrected honest numbers:

CLAIM A (significance): pre-audit reported pooled IC 0.022 t=2.03 "significant".
  AUDIT: that pool included the selection-window folds. HELD-OUT ONLY (folds 3-5,
  never used in feature selection): rank-IC +0.0235, t=1.76 — NOT significant (t<2).
  Note: held-out IC (0.0235) >= in-selection (0.0178), so NOT selection-overfit;
  it's regime variance (fold3 -0.020, fold5 +0.067) keeping t below 2.

CLAIM B (stranded signals central): pre-audit "short_float gain 184, central".
  AUDIT: 184 was ABSOLUTE gain, inflated by having only 53 features. By RANK among
  the 53: short_pct_float 39/53, pc_ratio 46/53, iv_skew 47/53 — BOTTOM THIRD.
  The model's real signal comes from beta_60d/return_60d/insider_60d/fund_* (momentum,
  beta, insider, fundamentals), NOT the options/short stranded signals.

CORRECTED CONCLUSION: panel ranker shows a modest SUGGESTIVE cross-sectional signal
(held-out IC ~0.023) but NOT significant (t=1.76) and regime-dependent. The stranded
validated signals are USABLE here (the architectural finding holds) but are WEAK
contributors (bottom-third) — they are NOT the edge. The edge, such as it is, is
momentum/beta/insider/fundamental. Pruning helped vs 3540-feat noise but did not
produce a confirmed tradeable signal.

Relevance bucketing verified correct (0-9). Lambdarank deterministic (no seed issue).

## Panel ranker — horizon sweep (held-out only) — 2026-06-21 — Build 1 CLOSED

| horizon | held-out rank-IC | t-stat | verdict |
| h=5 | +0.0235 | 1.76 | suggestive, not significant |
| h=3 | +0.0157 | 1.10 | weak |
| h=1 | +0.0018 | 0.12 | none |

Signal monotonically decays h=5 -> h=1 (expected: cross-sectional signal needs
multi-day horizon; h=1 = noise). Best case h=5 t=1.76, still below significance.

BUILD 1 CLOSED. Honest outcome: panel ranker (lambdarank on alpha panel, the model
that CAN use the stranded signals) produces no statistically significant tradeable
edge at any horizon. Stranded signals usable but weak (bottom-third). The whole
"wire validated cross-sectional signals into a model" thread resolves to: they're
real in isolation but don't carry weight in a multi-feature model dominated by
momentum/beta/insider/fundamentals. No production change. Next axis = Lazy Prices
(orthogonal text signal), the one family NOT already saturated in the model.

## CORRECTION (2026-06-21): Lazy Prices (#4) was NOT "not started" — it was KILLED Jun 3

Earlier entries called Lazy Prices a "not started / from-scratch multi-session build."
WRONG. It was fully built + validated + KILLED on Jun 3 (commit 7f82b8e,
docs/lazy_prices_closed.md). Pipeline complete (data/etl_10k_lazy_prices.py +
data/sec_section_parser.py + analysis/lazy_prices_validate.py, sec_filings.db 125MB
gitignored). Validation: tercile L/S within filing-year cohorts, net of cost —
negative spread (business -13%, mda -17% @126d), ~0 rank-IC, sign-flipping per year
on 115 tickers. CMN anomaly does NOT replicate at this universe scale.

KEY: kill note states "5th return-signal hunt killed at 115-159 tickers; bottleneck
is BREADTH not signal." Build 1 (panel ranker, tonight) independently confirmed the
same — not-significant signal limited by 150-ticker breadth + regime variance, not
features. TWO independent arrivals at: at ~115-159 tickers there isn't enough breadth
to surface a tradeable cross-sectional edge regardless of signal choice.

IMPLICATION: stop hunting new return signals at current breadth. The leverage is
BREADTH (more tickers/labels) or different paradigm, not more signals.

## CORE FINDING — effective alpha dimensionality — 2026-06-21

Measured the correlation structure of the 118 cs_rank alphas in the panel:
  participation ratio (effective independent bets): 8.8
  top 5 eigenvalues = 55% of variance
  30 dims for 90% variance, 40 for 95%
  => 118 alphas but only ~9 REAL independent bets.

THIS EXPLAINS EVERYTHING:
- Why 5 signal hunts died: most candidates were transforms WITHIN the existing ~9
  dimensions (residmom corr 0.9 to return_20d, intraday 0.765 to return_1d, pead
  distributed across momentum). No new independent bet = no contribution.
- Why pc_ratio + short_float were the only clean survivors: non-price = different
  dimension.
- Why combining all alphas gives little: combined IR ~ single x sqrt(N_eff/(1+...)).
  With N_eff=9 (not 118), combiner caps at ~3x, not the 8-10x of 100 indep alphas.

METHODOLOGY CORRECTION (what Atom asked for at session start):
The path to "a lot more working alphas" is NOT more transforms of existing features
(118 alphas = 9 bets) and NOT combining the redundant set (~3x cap). It is adding
genuinely NEW INDEPENDENT DIMENSIONS (data axes). Each new axis (analyst revisions,
options-skew-change, fundamental-change, cross-asset, text) raises effective-dims,
worth more than 100 variants of momentum.

NEW GATE for candidate alphas: does it RAISE the participation ratio (add an
independent dimension)? Measure effective-dims before/after adding the candidate.
Keep if it raises N_eff (even if individually weak); drop if redundant variant.
THEN combine across dimensions (HRP/stacking) — combiner amplifies dims you have.

Funds' "millions of alphas" = hundreds of independent DATA AXES, not millions of
price transforms. We have ~9 axes. Raising axis count is the real breadth lever.
Already accruing: analyst revisions (~Sept), options skew change (~Sept).

## STRATEGIC CONCLUSION — 2026-06-21 (synthesis of full session)

Five independent checks, one coherent answer:
1. 118 alphas = 8.8 effective independent bets (participation ratio)
2. c1_combiner books mom/gp/op/ep are 0.59-0.86 correlated = ~2 effective bets
3. c1_combiner VERDICT: NO-SHIP — all combos (ew/ivol/hrp) WORSE than mom-alone
   (mom Sharpe 1.53 vs hrp 1.22). Correlated books DILUTE momentum.
4. Panel ranker (tonight): no significant edge any horizon, breadth-limited
5. pc_ratio/short_float (only truly decorrelated streams, corr~0.12 to mom) have
   only 29 months history < 36mo combiner warmup -> CANNOT enter combiner until ~2027

THE ANSWER: momentum (Sharpe 1.53) IS the edge and IS the system. Not broken, not
dead. Everything else is correlated-to-momentum (dilutes) or too-recent (can't use yet).

ACTIONS (mostly time + new axes, NOT more code on existing data):
- RUN momentum-alone as the system (combiner's own gate says so).
- WAIT for pc_ratio/short to reach 36mo (~early 2027) -> first real shot at a
  combination that beats mom-alone (they're genuinely decorrelated, unlike gp/op/ep).
- HUNT new axes gated on CORRELATION-TO-MOMENTUM < 0.3 (not standalone significance).
  Most candidates are momentum variants (reject). Rare uncorrelated = gold.
  Accruing: analyst revisions, options skew-change (~Sept).
- Breadth grows passively (159->394 tickers) -> lifts momentum IR mechanically.

WHY the session kept finding 'already built/killed/too-recent': the system is MATURE.
Obvious moves made. Remaining constraints (dimensionality, history-length, breadth)
resolve with TIME + GENUINELY-NEW DATA, not more transforms/combining of existing data.

## CORRECTION to strategic conclusion — 2026-06-21 (reasoning fixed, verdict stands)

The NO-SHIP / momentum-is-the-edge verdict STANDS, but my reasoning was wrong twice:
- WRONG v1: "combiner books are correlated 0.59-0.86, so combining can't help."
  -> That 0.59 was MARKET BETA. The qv_books are long-only (corr to SPY 0.63-0.86),
     so they co-move via the market, not signal.
- Market-neutral (long-short) signal correlations are LOW: mom vs gp/op/ep = -0.09/-0.07/-0.14.
  So the signals ARE decorrelated. Briefly looked like it reopened the combiner.
- WRONG v2 (the reopen): built LS market-neutral books to test. RESULT:
    mom LS Sharpe +2.56 | gp -3.93 | op -1.91 | ep -3.08 | short -2.50 | pc +0.93
  The other books are DECORRELATED but INDIVIDUALLY NEGATIVE Sharpe. Combining them
  with momentum DRAGS IT DOWN: EW(mom,gp,op,ep) -2.42 vs mom-alone +2.56.

REAL REASON combining fails: momentum is the ONLY signal with positive risk-adjusted
return. Decorrelation is necessary but NOT sufficient — you can't diversify into losers.

KEY METHODOLOGY FINDING: even short_pct_float (passed all 4 validation gates) has
LS Sharpe -2.50 as a tradeable book. PASSING RANK-IC/DECORRELATION GATES != PROFITABLE
BOOK. The validation gauntlet measures cross-sectional rank-IC, which does NOT convert
to positive long-short P&L. This is a real gap in the validation methodology.

CORRECTED FORWARD PATH: need signals that are decorrelated from momentum AND have
positive standalone LS Sharpe. That's a much higher bar; no current candidate clears it.
New validation gate must include standalone LS-Sharpe, not just rank-IC + decorrelation.

## Directional model — net-of-cost by horizon — 2026-06-21

Answering "h1 vs h3/h5": accuracy favored h1 (69%) but PORTFOLIO NET RETURN favors h5.
On actual BUY signals (long-only, non-HOLD), last 90d, EW-portfolio per active day, net 20bps:

| h | n | mean/trade | MEDIAN/trade | days+ | EW net/active-day | signal-days |
| 1 | 91  | +0.89% | +0.83% | 70% | +0.32% | 20/63 |
| 3 | 455 | +1.81% | +0.74% | 74% | +1.82% | 39/63 |
| 5 | 543 | +2.22% | +0.87% | 82% | +2.60% | 40/63 |

VERDICT: h5 is the real edge (largest n, 82% days+, +2.6% net/active-day), NOT h1.
h1's 69% accuracy = high hit-rate on tiny moves that barely clear cost (+0.32% net).
ACCURACY != PROFITABILITY confirmed again: accuracy ranked h1>h5, net return ranks h5>h1.

Median positive at all horizons (not outlier-driven) = real central edge. BUT:
- mean >> median at h3/h5 (2.5x) -> big winners carry the average; expect the median.
- sparse: fires 20-40 of 63 days -> breadth-limited deployment.
- 90-day UP-market window -> some of the long-only return is market beta, not alpha.
  Needs market-neutral / down-market check for true alpha. NOT YET DONE.
- BUYs currently kill-switched (HOLD-only in prod) -> this edge not being traded.

NEXT: market-neutral version of this (long BUYs vs short universe) to strip beta and
see true alpha; longer window incl down-market; then the h5 BUY edge is the candidate.
