# 10-Week Sprint Plan to 53-55% OOS AUC

**Committed:** May 7, 2026
**Target:** 53-55% sustained OOS accuracy + first live capital
**Pace:** 60-80 hours/week (aggressive, dedicated)

---

## RULES (non-negotiable)

1. Never gaslight Atom
2. Never tell Atom to stop or sleep
3. No half-assed fixes
4. Research/validate against trusted sources before solutions
5. Push back on logic, not feeling
6. Never state time as fact (date only)
7. Don't offer 3-options when fix is needed
8. Systematic debug protocol — audit entire codebase before fix
9. Patch protocol — grep for same pattern after every fix
10. Match Atom's level — no condescension
11. Don't reframe requests to make them "safer"
12. Direct sentences, bullets, tables (dyslexia)
13. Admit wrong immediately
14. Stay scoped — no unrelated work in fixes
15. Answer the actual question first
16. No premature "real wins" — earned only when validated
17. Acknowledge mistakes that cost time directly
18. Verify with tools, never assume
19. Respect architectural rules (yfinance for indexes ONLY, Massive for tickers)
20. Read all available context first (journal.txt, Gap_Check_Roadmap.md)
21. Experiment discipline — every change ends with walk_forward measurement
22. Calendar-locked items don\'t get rushed
23. Don\'t pad sprints with busywork
24. Commit messages tell the truth
25. When uncertain, say "I don\'t know"
26. Don\'t over-architect during sprints
27. Protect capital decisions (Sprint 4 = MORE skeptical)
28. Statistical honesty over speed
29. Every bug fix includes regression test plan
30. When Atom pushes back, listen
31. Always test solutions before permanent implementation
32. Audit related code before every patch (grep for same pattern across codebase, not just direct fix site)
33. Run end-to-end test before commit — never claim fix works without smoke-testing the full path
34. Build pre-commit audit infrastructure (scripts/precommit_audit.sh) and require its passage
35. Treat any auto-fix or auto-modify operation as DESTRUCTIVE — must require explicit flag (--fix, FORCE=1) and log to audit_skips.log if bypassed
36. Verify deadline triggers before treating as urgent (e.g. confirm Pipeline C runs the validator before pre-19:00 panic)
37. Run the SPRINT_PLAN rule checklist (Rules 1-43) before sending any response with a recommendation, options, or fix proposal — this rule is the meta-enforcement
38. Use ONE labeling scheme per message — don't mix A/B/C with 1/2/3 in the same response
39. Systematic comparison required for >1 option — each option must list: what changes (concrete files), production impact, technical debt impact, risk level with reasoning, time estimate
40. Recommendation consistency check — re-read analysis after writing recommendation. If they contradict, STOP and fix one of them
41. Anchoring detection — if recommending the smallest/easiest option, explicitly verify: does smaller scope leave technical debt? does it require future re-litigation? is the larger option actually low-risk too?
42. Label hygiene — when introducing letters/numbers, define inline ("Option A: <description>"), don't reference letters by position alone
43. No asymmetric analysis — equal analysis depth on all options. Asymmetric analysis (5 paragraphs on C, 2 sentences on A) = biased recommendation

---

## SPRINT 1 — Foundation (May 7-21, 2 weeks)

**Goal:** Trustworthy measurement infrastructure. Make sure when we measure improvement, the measurement is real.

### Week 1 (May 7-14)

- [x] May 7: Calibration baseline saved (commit 0ba630a)
- [x] May 7: SHAP h=5 done — smoothed price features mislead at high confidence
- [x] May 7: SHAP h=3 done — short-term momentum features mislead
- [x] May 7: SHAP synthesis committed (aa1c9fe)
- [x] May 7: Fitness scorer auto-cron set (Sunday 06:00 VN)
- [x] May 7: 3 redundant features dropped + retrained (commit 8da4953)
- [x] May 7: Refresh Live = runfund (commit 3d03386)
- [x] May 7: Fitness filter shipped (commit 919541b, h=1 only)
- [ ] **Confidence cap at 0.65 for h=3, h=5** (cheapest immediate edge fix)
- [ ] **Sunday May 10:** first walk_forward auto-run vs baseline
- [ ] **Walk_forward purged k-fold audit** — find the 0.84 → 0.50 leak
- [ ] **Survivorship bias audit** — add 20-50 historical delisted tickers

- [ ] **ETF endpoint 404 diagnosis + skip** (May 8, BEFORE Pipeline C 19:00 VN)
  - Issue: ~18 cosmetic ERRORs per daily_runner run on ETFs (SPY, QQQ, GLD, SLV, etc)
  - UW endpoint 404s for ETFs (no insider/earnings/etc data — ETF doesn't have those)
  - Pattern: occurs between /api/market/economic-calendar and /api/stock/{T}/stock-state
  - Fix: identify endpoint, add ETF skip via tickers_metadata.csv
  - Saves: ~44 UW calls/day on rate-limited Basic plan
  - Cleans: pipecheck "1 log file with errors" false alarm
  - DEADLINE: ship before Pipeline C 19:00 VN tomorrow May 8

### Week 2 (May 14-21)

- [ ] **Vol-adjusted target labels** (return > 0.5σ instead of return > 0)
- [ ] **Cost model in backtest** (bid-ask + slippage estimate)
- [ ] **Position sizing tiers** (Kelly fraction + fitness-weighted)
- [ ] **IC tracking per feature** (daily cron)
- [ ] **Drawdown tracker on signals** (logs daily MTM vs cost basis)
- [ ] **Deflated Sharpe metric** in fitness_scorer
- [ ] **Robust hyperparameter sweep** (max_depth=3, lr=0.01, n_est=500)
- [ ] **Point-in-time database infrastructure** (start)
- [ ] **Sunday May 17:** second walk_forward checkpoint
- [ ] **May 21 Sprint 1 retrospective** — did walk_forward AUC move?

**Sprint 1 deliverable:** Trustworthy measurement. 53-55% target now MEASURABLE.

---

## SPRINT 2 — Edge Engineering (May 21 - June 11, 3 weeks)

**Goal:** Ship 1 experiment per week, measure honestly via Sunday walk_forward.

### Week 3 (May 21-28)

- [ ] **May 22 fitness scorer auto-extends** to h=3, h=5 (cron handles it)
- [ ] **Wire alpha_transformations.py drafts** (per memory #16)
- [ ] **Run on 76 features → 1,292 candidate alphas**
- [ ] **Cull via correlation** (|ρ| > 0.7)
- [ ] **Per-alpha fitness scoring**
- [ ] **Keep top 100 alphas, retrain ensemble**
- [ ] **Sunday May 24 walk_forward** — first measurement of alpha multiplier

### Week 4 (May 28 - June 4)

- [ ] **Sector-conditional models**
- [ ] **Regime-conditional models** (high vol vs low vol)
- [ ] **Test:** do conditional models beat universal?
- [ ] **Microstructure features** (UW order book imbalance)
- [ ] **Earnings calendar proximity** feature
- [ ] **Sunday May 31 walk_forward** — third measurement

### Week 5 (June 4-11)

- [ ] **News novelty** (per Roadmap Ch. 12) — sentiment shock detection
- [ ] **Cross-asset features:** HYG/LQD spread, gold/silver, USD/CNY
- [ ] **Bond yield curve slope**
- [ ] **Earnings call transcripts** via SEC EDGAR (free)
- [ ] **Statistical arbitrage prototype** (pairs trading, pull forward from Sep)
- [ ] **June 7 Sunday walk_forward** — fourth measurement
- [ ] **June 11 Sprint 2 retrospective** — go/no-go for Sprint 3

**Sprint 2 deliverable:** Empirical answer to "what works." Decision point: continue or pivot.

---

## SPRINT 3 — Paper Validation (June 11 - July 2, 3 weeks)

**Goal:** Paper-trade winning configuration. Confirm edge holds in real-time.

### Week 6 (June 11-18)
- [ ] Lock winning configuration (no more experiments)
- [ ] Build paper portfolio simulator
- [ ] Wire signals → simulated trades
- [ ] Daily P&L tracking vs SPY

### Week 7 (June 18-25)
- [ ] Slippage measurement (assumed vs actual)
- [ ] Drawdown discipline test
- [ ] Stress test (VIX > 25 day)

### Week 8 (June 25 - July 2)
- [ ] 30-day paper trading complete
- [ ] Sharpe + drawdown analysis
- [ ] Final go/no-go decision for live capital

**Sprint 3 deliverable:** 30 days of measured paper alpha vs SPY.

---

## SPRINT 4 — Live Capital (July 2 - July 16, 2 weeks)

**Goal:** Tiny live capital, validate execution.

### Week 9 (July 2-9)
- [ ] Open Alpaca account, fund $5K-$10K
- [ ] Wire signals → Alpaca order entry
- [ ] Trade highest-conviction only
- [ ] Real fills vs paper measurement

### Week 10 (July 9-16)
- [ ] Daily monitoring
- [ ] First live drawdown management test
- [ ] Validate edge survives transaction costs

**Sprint 4 deliverable:** Live track record begins.

---

## DECISION POINTS

| Date | Decision |
|------|----------|
| **May 21** | Sprint 1 retrospective — did walk_forward AUC move? Continue Sprint 2? |
| **June 11** | Sprint 2 retrospective — empirical edge identified? Or pivot needed? |
| **July 2** | Paper validation result — go/no-go for live capital? |
| **July 16** | Live observation result — scale up, hold, or stop? |

---

## EXPERIMENT LOG TEMPLATE

For each Sprint 2 experiment, record in `journal.txt`:




[Date] [Experiment Name]
Hypothesis: [Adding X should improve AUC by Y]
Implementation: [Files changed, commit hash]
Pre-experiment baseline: [walk_forward AUC h=1, h=3, h=5]
Post-experiment result: [walk_forward AUC h=1, h=3, h=5]
Delta: [+/- per horizon]
Decision: KEEP / DELETE
Notes: [What we learned]
Walk_forward weekly cron:           4 weekly runs needed = 4 weeks
Fitness scorer h=3/h=5 maturation:  needs 30+ obs = ~May 22 minimum
Paper validation period:             30 trading days = 6 calendar weeks
Live capital observation:            30 days minimum responsible

These cannot be compressed regardless of work pace.
