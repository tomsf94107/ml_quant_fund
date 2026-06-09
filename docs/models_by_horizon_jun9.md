# Models by Horizon — What Top Funds Actually Run, and What’s Viable at Your Scale

*June 9 2026. Companion to strategic_review_jun9.md and QUANT_BUILD_MENU. Organized by the
three horizons you asked for: days, months, 6-12 months.*

-----

## PART 0 — Read this first: the momentum “unreliability” question

Momentum picks being down “the past few days” is NOT evidence the model is unreliable:

- It is a ~20-TRADING-DAY hold, BASKET-level signal. Its unit of success is the basket
  return over a month, not any ticker over any week. Days-scale drawdown inside the hold
  window is expected behavior, present in every momentum backtest that ever worked.
- The validated number (purged-WF net Sharpe +1.24) INCLUDES weeks like this one.
- The pre-committed live verdict is ~Jun 29 (first cohort completes its hold). Judging
  before then = the exact “emotional override” your own playbook’s Process Rules ban.
- Momentum’s known weakness is real but different: MOMENTUM CRASHES (sharp reversals,
  Daniel-Moskowitz) — it loses in violent regime turns, not in ordinary red weeks.
  **Discipline: let Jun 29 decide. Strategy-hopping on a week of noise is how retail dies.**

-----

## PART 1 — The honest horizon map (where edge is even possible)

|Horizon        |Who wins there                                                   |How they win                                                                                                                         |Viable solo at ~150 names?                                                                                                                                 |
|---------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
|**Intraday-5d**|Renaissance Medallion, HFT/market makers, Citadel Securities     |NOT prediction — execution: market-making spreads, microstructure, rebates, thousands of tiny signals, massive leverage, sub-ms infra|**NO.** Your 5 kills + the 0.51-0.55 AUC ceiling are the proof. Funds here win on plumbing, not models.                                                    |
|**2 wks-3 mo** |WorldQuant, AQR, Two Sigma, GKX-style ML books                   |Cross-sectional anomalies (momentum, quality, value, insider, revisions), ranking ML, many weak alphas combined                      |**YES — this is YOUR zone.** Momentum lives here. Insider/quality/value belong here.                                                                       |
|**6-12 mo**    |AQR, managed futures (Man AHL, Winton), factor shops, Dimensional|Trend following (TSMOM), factor premia harvesting, fundamental quant                                                                 |**YES, partially.** Slower feedback (fewer independent bets/yr = slower validation), but real. Your AI-thesis sleeve already operates here discretionarily.|

Key fact from the research: the big quants “rarely rely on directional bets — they thrive as
market makers, arbitrageurs, and volatility traders.” The fantasy of “a model that predicts
next week’s price” is not what even Renaissance does at scale; Medallion is short-term
holdings + enormous turnover + cost control on thousands of instruments — an
infrastructure business. Nobody sells what you’re missing at 1-5d; it doesn’t exist solo.

-----

## PART 2 — THE MODEL CATALOG (how it works / who uses / pros / cons / verdict for you)

### Horizon: DAYS (1-5d) — listed for completeness; the verdict is uniform

**M1. Microstructure / market-making / order-flow models**
How: predict order-flow imbalance, earn spread, manage inventory. Who: Citadel Securities,
Virtu, RenTec. Pros: the only consistently profitable days-scale paradigm. Cons: needs
colocation, tick data, rebate economics — capital + infra you cannot have. **Verdict: NO.**

**M2. Short-horizon direction classifiers (your XGBoost)** — KILLED, at industry ceiling
(0.535 vs 0.51-0.55 honest range). Re-attempting with LSTM/Transformer changes nothing:
sequence models on low-SNR daily data overfit worse. **Verdict: CLOSED — your own 5 kills.**

**M3. Short-term reversal / stat-arb (pairs, PCA residual)** — KILLED on your universe
(regime-dependent, net-negative current regime; pairs -0.82). **Verdict: CLOSED.**

**M4. Event-window models (earnings, FDA, M&A)** — the ONLY defensible days-scale activity
for you, and you ALREADY run it: the pre-earnings monitoring layer (dark pool + options
flow + insider before catalysts). It is discretionary, event-gated, not a continuous model.
Pros: events concentrate information; your UW data is built for this. Cons: doesn’t scale,
sparse. **Verdict: keep as-is (discretionary layer), don’t systematize at 150 names.**

### Horizon: WEEKS-MONTHS (your viable zone)

**M5. Cross-sectional momentum (yours)** — the validated survivor. Pros: most documented
anomaly in finance, decades/countries/asset classes. Cons: momentum crashes; decay.
**Verdict: LIVE candidate, verdict Jun 29. Possible robustness upgrades (same family, not
new signals): residual momentum (Blitz — momentum on factor-residuals, lower vol),
52-week-high anchor, industry momentum. These can harden the ONE signal.**

**M6. Insider opportunistic/routine (Cohen-Malloy-Pomorski)** — the #1 NEW candidate
(strategic review Track B): 82bp/mo VW documented, replications hold, free EDGAR data,
infra half-built, monthly horizon, orthogonal axis. Cons: event sparsity at 150 names.
**Verdict: BUILD NEXT.**

**M7. Quality + Value factor signals (Novy-Marx GP, B/M, E/P)** — data-gated on free XBRL
ingestion (Track C). Monthly rebalance, durable anomalies, low correlation to momentum.
**Verdict: build the XBRL feed, then test.**

**M8. Cross-sectional ranking ML (GKX-style: pooled panel, trees/shallow NN, monthly)**
How: one pooled model across all stocks, predict next-month relative return, trade decile
spread. Who: Two Sigma-adjacent, the academic benchmark. Pros: the documented right way to
do equity ML (R² ~0.4%/mo but decile Sharpe ~1+). Cons: needs breadth — at 149 names the
cross-section is too thin (your IPCA overfit for exactly this reason). **Verdict: AFTER
universe expansion (400+). At current scale: NO.**

**M9. IPCA / Conditional Autoencoder (Kelly-Pruitt-Su, GKX 2021)** — the state-of-the-art
cross-sectional models (autoencoder decile Sharpe 2.16 headline, ~half on clean universes).
Cons: data-hungry; you already tried IPCA → overfit at 149×T. **Verdict: the model to
build WHEN the universe is 400+. Not before. The model isn’t the constraint — breadth is.**

**M10. Learning-to-rank (LightGBM lambdarank)** — tried: @5d leaked, @20d 2/4 folds. Dead
at current scale; possibly revisit at expanded universe alongside M8. **Verdict: CLOSED now.**

**M11. Analyst-revision momentum (Chan-Jegadeesh-Lakonishok; ~7.5%/6mo documented)**
How: rank by recent consensus-EPS revision direction/magnitude. Pros: durable, different
axis (analyst behavior). Cons: needs a revisions feed — you have eps_surprise +
earnings_cache (partial); full revisions history needs a data source (check what UW/
Massive expose before pricing IBES-style feeds). **Verdict: AUDIT data availability;
mid-tier candidate behind insider + quality.**

**M12. Hierarchical Bayesian pooling** — fix for per-ticker small samples (pool each name
toward sector/market). Pros: principled, rescues thin data. Cons: a method, not a signal —
needs an edge to pool. **Verdict: methods layer, apply to survivors only.**

### Horizon: 6-12 MONTHS

**M13. Time-series momentum / trend following (Moskowitz-Ooi-Pedersen; Man AHL, Winton)**
How: own assets whose own 6-12mo return is positive, short/avoid negatives; vol-scaled;
classically on futures, runnable on your index ETFs (SPY/QQQ/sector ETFs/GLD/SLV).
Pros: ~century of evidence across asset classes; crisis-alpha profile (made money 2008,
2022); simple, near-unkillable; LOW correlation to cross-sectional momentum. Cons: modest
standalone Sharpe (~0.4-0.8); long flat stretches; on ETFs only (your single names already
feed M5). **Verdict: ★ BUILDABLE NOW — cheap (same panel machinery), genuinely different
paradigm (absolute vs relative momentum), and a candidate 2nd “signal” for a portfolio-
level overlay. Test through the same purged-WF gate.**

**M14. Factor premia harvesting (AQR style: value/quality/low-vol/carry tilts, annual-ish)**
How: persistent long tilts toward documented premia. Pros: the institutional core; slow,
low-cost. Cons: this is beta-plus, not alpha — and at 12mo horizon you get ~1 independent
bet per year per factor: validation takes years. **Verdict: this is effectively what your
AI-thesis sleeve already is (thematic 12-18mo tilts with rules). Keep it discretionary
with the playbook; don’t pretend it’s a “model.”**

**M15. Fundamental quant / earnings-power models (Dimensional, GMO)** — long-horizon
valuation reversion. Needs the XBRL feed + years of patience. **Verdict: subsumed into
M7 (quality/value); no separate build.**

**M16. Macro regime allocation (HMM/Kalman over rates/credit/vol → risk-on/off)**
You already have the components: vol regime classifier + recession model (M1/M2 probits)

- VIX/credit features. Pros: regime gates demonstrably help sizing/exposure. Cons: as
  RETURN predictors regimes are weak; as GATES they’re useful. **Verdict: wire what exists
  (recession dashboard + vol classifier) into a simple exposure gate for the momentum book —
  audit first, most parts are built.**

### Cross-cutting (any horizon)

**M17. Volatility forecasting (HAR-RV, GARCH) + VRP harvesting** — funds’ bread and butter
(“they thrive as vol traders”). Your VRP test ~Aug is exactly this family. HAR-RV on
realized vol is the standard, easy, high-R² model — but you oracle-proved vol-sizing
doesn’t help the momentum book, so its use would be VRP signal + options strategies only.
**Verdict: Aug as planned; no new build now.**

**M18. Meta-labeling + fractional Kelly (Lopez de Prado)** — secondary model predicts when
the primary is right; size accordingly. **Verdict: apply to momentum AFTER Jun 29 if it
goes live; untested C2 item.**

**M19. Reinforcement learning for execution/allocation** — frontier at funds; data-hungry,
fragile, unverifiable solo. **Verdict: NO.**

**M20. LLM alpha generation behind your gate (AlphaForge/AlphaAgent lineage)** — generate
candidate formulaic alphas → HLZ t>3 + PBO gate → survivors. The gate is what makes it
honest; expect ~1-in-20 survival. **Verdict: cheap weekend loop AFTER insider/TSMOM tests;
a generator, not a model.**

-----

## PART 3 — THE DECISION TABLE (your three asks, answered)

|You asked for                |The honest answer                                                                                                                                                                                                                                |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|“Predict price next few DAYS”|**Does not exist at your scale.** 5 kills + industry ceiling. The only days-scale edge you have is the event-driven pre-earnings layer — keep it discretionary. Anyone selling a days-scale prediction model to a solo trader is selling fiction.|
|“Next few MONTHS”            |**Your real zone.** Momentum (verdict Jun 29) + build NEXT: insider (M6), TSMOM on ETFs (M13), quality/value after XBRL (M7). At 400+ names: GKX-ranking (M8) then autoencoder (M9).                                                             |
|“6-12 MONTHS”                |TSMOM/trend (M13) is the buildable systematic piece. Factor tilts (M14) = your AI-thesis sleeve, already running with a playbook — leave it discretionary. Macro regime gate (M16) from parts you already built.                                 |

## PART 4 — What to build, in order (supersedes nothing; extends strategic_review Part 4)

1. **Hold momentum to its Jun 29 verdict.** No strategy-hopping on a red week.
1. **M6 Insider** (Track B) — the new-axis candidate, free data, infra half-built.
1. **M13 TSMOM on your ETF set** — NEW from this review: cheap, different paradigm
   (absolute trend vs relative rank), century of evidence, same harness. A real candidate
   2nd signal that needs NO new data. Run it through the standard kill-gate.
1. **M16 regime gate** — wire existing recession+vol pieces into a momentum exposure gate (audit first).
1. **Track A universe expansion** in parallel → unlocks M8/M9 properly.
1. **M7 quality/value** after XBRL feed. **M17 VRP** in Aug. **M18/M20** methods after a 2nd survivor.

*One line: at days-scale nobody honest can help you; at months-scale you already own the
right signal and the next three builds (insider, TSMOM, quality) are mapped; at 6-12mo
your thesis sleeve already does the job. The constraint is still breadth — expansion
multiplies everything else.*