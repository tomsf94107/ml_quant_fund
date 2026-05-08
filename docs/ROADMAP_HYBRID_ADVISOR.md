# Roadmap: Hybrid Advisor System

**Status:** Design document, not implementation spec
**Created:** May 8, 2026 (Friday evening)
**Scope:** Q1-Q4 vision (hybrid dashboard, continuous sizing, tooltips, manual trade log)
**Target delivery:** 4 weeks (May 8 - June 5, 2026)
**Owner:** Atom

---

## 0. Why This Document Exists

This Friday session generated a series of questions about what a sophisticated quant
system should look like — Bloomberg/WorldQuant/Renaissance-style — vs what ML Quant
Fund currently is. The questions were thoughtful enough that answering them ad-hoc
in chat would lose the vision. This doc captures it before it dissipates.

This is **not a build spec**. It is a vision + sequencing document. Detailed designs
for each component come in their respective Sprint weeks.

The original intent was "ship Q1-Q4 in 3 days." After honest analysis (see Decision
Log §12), the realistic delivery window is **4 weeks**, with a 2-week MVP path
explicitly rejected as too aggressive given today's rule-violation track record.

---

## 1. The Problem

### 1.1 What ML Quant Fund is today (May 8, 2026)

- 125 tickers, 3 horizons (1d/3d/5d), XGBoost+LightGBM ensemble + FinBERT sentiment
- Daily Pipeline B retrains all models, generates signals
- Live ROC-AUC ~0.510 (per Apr 29 Gap_Check) — barely above coin flip
- Production threshold: 0.70 (gate at daily_runner.py L61)
- Confidence labels: HIGH/MEDIUM/LOW (3-tier with downstream consumers)
- Position sizing: tier-based multipliers (HIGH=1.0x, MEDIUM=0.6x, LOW=0.0x)
- No live capital, no real broker connection
- Streamlit Cloud deployment, manual broker execution

### 1.2 Limitations the operator hits in practice

1. **Hard to make quick decisions.** Dashboard shows numbers but not actions.
   "Should I buy NVDA today?" requires reading 4 tabs and interpreting 8 metrics.

2. **Hard to learn the system.** No inline explanations. Each metric requires
   external research to interpret. Atom has dyslexia (per project memory) —
   walls of numbers without context cost real time.

3. **No record of what was actually traded.** System suggests, Atom executes
   manually in Schwab/etc, but there's no closed loop. Can't measure personal
   slippage. Can't compare "system's predicted P&L" vs "Atom's actual P&L."

4. **Tier-based sizing is a 1990s pattern.** Industry standard (per Finding Alphas
   Ch 6-7) is continuous signal → continuous weight. The 3 hard tiers throw away
   information at every step.

5. **Auto-execution security risk.** Alpaca-style integration is intentionally
   not happening. Atom wants advisor mode (system suggests, human decides).
   Current dashboard isn't optimized for that workflow.

### 1.3 The four questions this doc answers

- **Q1** — Hybrid dashboard: quick decision labels + drill-down details + inline learning
- **Q2** — Continuous Kelly-based sizing replacing tier multipliers
- **Q3** — Simplify Portfolio filter (remove redundant `(HIGH,MEDIUM)` check)
- **Q4** — Advisor mode: trade log + P&L tracking without auto-execution

---

## 2. Vision: Hybrid Advisor System

### 2.1 Core principle

**Quick decisions on the surface, math underneath, learning embedded.**

A retail operator (Atom) needs:
- 60-second morning brief: "what do I do today?"
- 5-minute drill-down: "why does the system think this?"
- 10-minute deep dive: "what does each metric mean and is it good or bad?"

The system serves all three needs from the same data.

### 2.2 Three view modes

| Mode   | Purpose         | Reading time | Used when           |
|--------|-----------------|--------------|---------------------|
| Brief  | What to do      | 60 seconds   | Morning, premarket  |
| Detail | Why             | 3-5 minutes  | Before placing trade|
| Learn  | How to read it  | 10+ minutes  | Building knowledge  |

### 2.3 What the system explicitly does NOT do

- Auto-execute trades. Ever. Atom decides + executes manually.
- Connect to brokers via API. Manual entry only.
- Hide complexity. Math is always available, just not always visible.
- Trust itself blindly. Every claim shows confidence + uncertainty.

---

## 3. Mockups

### 3.1 Brief mode (morning glance)

```
┌────────────────────────────────────────────────────────────┐
│ MORNING BRIEF — May 9, 2026                                │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ SUGGESTED ENTRIES (3)                                      │
│                                                             │
│ NVDA  prob 0.81  rank 3/125   weight 13.5%  =$13,500       │
│   Why: strong signal, no earnings risk, good regime        │
│                                                             │
│ MSFT  prob 0.78  rank 8/125   weight 11.2%  =$11,200       │
│   Why: top decile, sector cap not reached                  │
│                                                             │
│ AVGO  prob 0.72  rank 14/125  weight 8.4%   =$ 8,400       │
│   Why: momentum + custom-silicon thesis intact             │
│                                                             │
│ TRIM SUGGESTIONS (1)                                       │
│                                                             │
│ TSLA  signal weakening 0.62 to 0.41 in 2 days              │
│   Why: model shifting bearish, novelty=high                │
│                                                             │
│ HOLD ALL OTHERS (122)                                      │
│                                                             │
│ --- PORTFOLIO HEALTH ---                                   │
│ Total value:        $100,000                                │
│ Sector exposure:    Tech 32% (cap 35%)  near limit         │
│ Drawdown from peak: -2.1% (within limits)                   │
│ VIX regime:         18.5 (normal)                           │
│                                                             │
│ [Mark NVDA executed]  [Mark TSLA executed]  [Snooze]       │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Detail mode (per-ticker drill-down)

```
┌────────────────────────────────────────────────────────────┐
│ NVDA — Decision Detail                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ ACTION: ENTER  $13,500 (5.0% of portfolio)                 │
│                                                             │
│ --- PROBABILITY ---                                        │
│ prob_eff:   0.81   model's UP probability after adj        │
│ prob_up:    0.78   raw model output                        │
│ rank:       3/125  today's strength vs all tickers         │
│   GOOD: Top 10 = strong signal                              │
│                                                             │
│ --- SIZING MATH ---                                        │
│ Edge (prob - 0.5):   +0.31                                  │
│ Vol (60-day daily):  2.4% ($6 swing on $250)                │
│ Kelly raw:           54%                                    │
│ Kelly x 0.25 cap:    13.5%                                  │
│ Vol-target cap:      18.3% (not binding)                    │
│ Final weight:        13.5%                                  │
│                                                             │
│ --- RISK CHECKS ---                                        │
│ Earnings <=5d:       No  GOOD                               │
│ Sector cap (Tech):   18.3% of book (cap 25%)  GOOD          │
│ VIX regime:          18.5 (normal)  GOOD                    │
│ Recent drawdown:     -2.1% past 7d  CAUTION minor           │
│                                                             │
│ --- HOW TO READ THIS ---                                   │
│ 3 GOOD checks, 1 CAUTION minor = strong actionable signal   │
│ Position risk: $13,500 x 2.4% vol = +/-$324 daily move      │
└────────────────────────────────────────────────────────────┘
```

### 3.3 Learn mode (tooltip example)

Each metric gets a 3-line tooltip when clicked:

```
prob_eff: 0.81
  WHAT: Effective probability the model thinks NVDA goes UP tomorrow,
        AFTER adjustments for sentiment, options flow, regime, etc.
  GOOD: > 0.65 (strong directional signal)
  BAD:  < 0.50 (model thinks DOWN is more likely)
  NOTE: This is the production gate. BUYs only fire above 0.70.
```

```
rank: 3/125
  WHAT: Where NVDA sits among today's signals across all 125 tickers.
        Lower number = stronger signal relative to the universe.
  GOOD: Top 10 (top 8% of universe)
  BAD:  Bottom 50 (likely below average alpha)
  NOTE: Cross-sectional rank often matters more than absolute prob.
        A 0.65 prob in a weak day may rank #1; 0.65 in a strong day
        may rank #40.
```

---

## 4. Continuous Position Sizing (Q2)

### 4.1 What changes

Replace `position_sizer.py` TIER_MULTIPLIERS dict with continuous Kelly + vol-cap.

**Current (tier-based):**
```python
TIER_MULTIPLIERS = {"HIGH": 1.0, "MEDIUM": 0.60, "LOW": 0.00}
size = base_size * TIER_MULTIPLIERS[confidence]
```

**Target (continuous):**
```python
def get_position_weight(prob_eff, vol_60d_daily,
                        max_kelly=0.25, vol_cap=0.20):
    # Edge = how much above coin flip
    edge = prob_eff - 0.50

    if edge <= 0:
        return 0.0

    # Simplified Kelly (assumes ~symmetric payoff)
    kelly_raw = edge * 2.0
    kelly_capped = kelly_raw * max_kelly  # fractional Kelly

    # Vol-target scaling
    annualized_vol = vol_60d_daily * (252 ** 0.5)
    vol_target = vol_cap
    vol_mult = min(vol_target / max(annualized_vol, vol_target), 1.0)

    return kelly_capped * vol_mult
```

### 4.2 Why this is better

- **No information loss at tier boundaries.** A 0.71 BUY and 0.85 BUY
  no longer get identical sizing (tier system gave both 1.0x).
- **Vol-aware.** High-volatility tickers get smaller weight automatically.
- **Industry standard.** WorldQuant / Renaissance / Citadel / Two Sigma
  all use continuous sizing at the strategy level.

### 4.3 Why we don't ship this in Week 1

- Math change to live system. Untested = real money risk later.
- Needs walk-forward validation (Sprint 1 Week 1 prerequisite — Item 1.1).
- Per project memory: walk_forward currently hangs in setup. Must fix first.
- Position_sizer has downstream consumers (Portfolio filter, signal_alerter).

### 4.4 Migration path

1. Build continuous function ALONGSIDE existing tier function (Week 3).
2. Add feature flag: `USE_CONTINUOUS_SIZING = False` (default).
3. Run both in parallel for 1 week — log both outputs to compare (Week 3-4).
4. Validate via walk-forward + fitness scoring (Week 4).
5. Switch flag to True in Sprint 2 if validated.
6. Keep tier function as fallback for 1 month.

### 4.5 Confidence label becomes display-only

After continuous sizing ships:
- HIGH/MEDIUM/LOW labels remain in dashboards (humans read them)
- Production logic (position sizing, alerter) reads `prob_eff` directly
- Tier labels are decorative, not load-bearing

---

## 5. Tooltip Teaching System (Q2 part 2)

### 5.1 Why this matters

Atom has dyslexia. Walls of numbers without context cost real time. Every
metric needs:
- **WHAT it is** (plain English, no jargon)
- **GOOD range** (typical "this is a positive sign")
- **BAD range** (when to worry)
- **NOTE** (caveats, edge cases)

This is what BlackRock Aladdin and Bloomberg do for retail products. Pros
don't need explanations; tools FOR humans embed them inline.

### 5.2 Implementation pattern

```python
# In each Streamlit page
def metric_with_tooltip(label, value, tooltip_md):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label, value)
    with col2:
        with st.expander("?"):
            st.markdown(tooltip_md)
```

### 5.3 Coverage by Sprint week

- **Week 2 (May 15-22):** Existing metrics (prob_up, prob_eff, signal,
  confidence) get tooltips on Dashboard + Portfolio + Accuracy pages.
- **Week 3 (May 23-29):** New metrics (rank, weight, Kelly, vol) get
  tooltips as they're added.
- **Week 4 (May 30 - Jun 5):** Cross-page consistency audit. Same metric
  uses same tooltip wording everywhere.

### 5.4 docs/glossary.md

Single markdown file. Every metric used in the system gets one entry.
Format:

```
## prob_eff
**Type:** Probability (0.0 to 1.0)
**What it is:** Effective probability the model thinks the ticker goes
                UP tomorrow, after multiplier adjustments.
**Good range:** > 0.65 (strong directional signal)
**Bad range:**  < 0.50 (model thinks DOWN)
**Source:** signals/generator.py
**Used by:** Dashboard, Portfolio, position_sizer, signal_alerter
**Caveats:** This is the production BUY gate at 0.70. Below 0.70, no BUY.
```

---

## 6. Advisor Mode Trade Log (Q4)

### 6.1 The workflow

System produces morning brief at 7:30 AM Vietnam (after Pipeline B).
Atom reads brief, decides what to trade, executes manually in
Schwab/Fidelity/whatever. Then logs back into the system:

```
[Mark Executed]
  Ticker:  NVDA
  Side:    BUY (default from brief)
  Shares:  53
  Price:   $254.85 (filled at)
  Time:    2026-05-09 09:32 ET
  Notes:   "Slightly above brief's $255 estimate"
  [Save]
```

System stores this in `manual_trade_log` table:

```sql
CREATE TABLE manual_trade_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    side         TEXT NOT NULL,           -- BUY | SELL | TRIM
    shares       REAL NOT NULL,
    price        REAL NOT NULL,
    fill_time    TEXT NOT NULL,           -- ISO 8601
    notes        TEXT,
    signal_id    INTEGER,                 -- FK to signals table
    suggested_price REAL,                 -- price brief showed
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 What the system learns from trade log

**Slippage:** difference between brief's `suggested_price` and actual `fill_price`.
Builds personal slippage profile over time.

**Realized P&L:** when SELL/TRIM logged for ticker, compute return:
`(sell_price - buy_price) * shares - costs`

**Personal alpha:** compare YOUR realized P&L vs system's predicted P&L.
Shows whether YOU execute as well as the system suggests.

### 6.3 Why this is better than auto-execution

- No API integration with brokers (security)
- No risk of runaway algo trades
- Atom retains full discretion
- Forces deliberate review of each suggestion
- Builds operator skill alongside system improvement

### 6.4 What this enables in Sprint 2+

- Slippage-aware sizing (smaller positions in stocks where YOUR fills are bad)
- Personal alpha vs market alpha attribution
- Honest live-capital decision in Sprint 4 (do you actually beat SPY when
  executing manually?)

---

## 7. 4-Week Tier-Prioritized Plan

### 7.1 Tier rationale

Items are ranked by:
- **Financial impact** (1-10): Will this directly improve $ returns?
- **Tech impact** (1-10): Is it foundation work that unblocks future improvements?
- **Risk reduction** (1-10): Catches bugs before $ loss?
- **Speed-to-value** (1-10): How quickly does the win materialize?
- **Effort** (Low/Med/High): Hours required

ROI scores are **rough estimates**, not measurements. Bias high on
"foundation work" because debt compounds.

### 7.2 Week 1 (May 8-14) — TIER 1: Foundation

**Item 1.1 — Walk-forward measurement validation**
- Financial: 9 | Tech: 10 | Risk: 10 | Speed: 8 | Effort: Low (8-12 hrs)
- Why: Without honest measurement, every future change is guesswork.
  Walk_forward currently hangs in setup (per project memory).
- Deliverable: walk_forward producing trustworthy AUC numbers
- Critical path: blocks Items 3.1, 3.3

**Item 1.2 — Pipeline B Schema v2 + SHAP validation (Saturday May 9)**
- Financial: 6 | Tech: 9 | Risk: 10 | Speed: 10 | Effort: Low (1 hr)
- Why: Already shipped today (commits 7951556 + a2f9401), needs validation
  tomorrow when cron runs.
- Deliverable: Confirmed clean Pipeline B run with v2 columns + SHAP rows

**Item 1.3 — Manual trade log MVP**
- Financial: 8 | Tech: 7 | Risk: 6 | Speed: 7 | Effort: Med (12-16 hrs)
- Why: Without trade log, system can't learn from YOUR fills. You'll
  guess whether to trust signals.
- Deliverable: manual_trade_log table + Mark Executed UI + basic P&L view

### 7.3 Week 2 (May 15-22) — TIER 2: Cleanup + UX

**Item 2.1 — γ: 2-tier confidence cleanup (deferred from May 8)**
- Financial: 3 | Tech: 7 | Risk: 5 | Speed: 8 | Effort: Med (8-12 hrs)
- Why: Foundation for continuous sizing. Cleans dead code that blocked
  Item 3.1 from shipping cleanly.
- Deliverable: 2-tier system (HIGH/LOW only), position_sizer simplified

**Item 2.2 — Tooltip teaching system on existing metrics**
- Financial: 4 | Tech: 5 | Risk: 3 | Speed: 9 | Effort: Med (8-12 hrs)
- Why: Unblocks operator's ability to debug system independently.
- Deliverable: Dashboard + Portfolio + Accuracy pages have inline tooltips

**Item 2.3 — SELL signal validation (overdue per project memory)**
- Financial: 5 | Tech: 4 | Risk: 7 | Speed: 10 | Effort: Low (1-2 hrs)
- Why: Already overdue from May 1 milestone. Quick to resolve.
- Deliverable: SELL signal lives OR formally deferred with reason logged

### 7.4 Week 3 (May 23-29) — TIER 3: Sophisticated math

**Item 3.1 — Continuous Kelly sizing**
- Financial: 7 | Tech: 9 | Risk: 6 | Speed: 5 | Effort: High (20-30 hrs)
- Why: Real prediction improvement. Industry-standard foundation.
- Deliverable: Continuous sizing live behind feature flag, validated via
  walk-forward
- Prerequisite: Item 1.1 walk-forward must be working

**Item 3.2 — Hybrid dashboard MVP (Q1)**
- Financial: 5 | Tech: 6 | Risk: 4 | Speed: 6 | Effort: High (25-35 hrs)
- Why: Better decisions from clearer info. Operator usability.
- Deliverable: Dashboard + Portfolio + Accuracy show hybrid view (brief +
  detail + tooltips)

**Item 3.3 — SHAP-driven feature culling**
- Financial: 6 | Tech: 7 | Risk: 5 | Speed: 7 | Effort: Med (10-15 hrs)
- Why: SHAP fix shipped today (a2f9401). By Week 3, ~14 days of clean
  SHAP data accumulated.
- Deliverable: Feature set culled to top 60-65 from current 79

### 7.5 Week 4 (May 30 - Jun 5) — TIER 4: Polish + risk

**Item 4.1 — Realized P&L slippage tracker (advisor v2)**
- Financial: 6 | Tech: 4 | Risk: 3 | Speed: 4 | Effort: Med (10-15 hrs)
- Why: Builds on Item 1.3. Measures execution quality.
- Deliverable: Slippage analysis page

**Item 4.2 — Sector concentration limits**
- Financial: 5 | Tech: 5 | Risk: 8 | Speed: 7 | Effort: Med (8-12 hrs)
- Why: Per AI Investment Thesis playbook — sector caps designed but not
  enforced. Real risk management for live capital.
- Deliverable: Sector exposure tracker + kill-switch alerts

**Item 4.3 — Validator hysteresis-aware logic**
- Financial: 2 | Tech: 4 | Risk: 5 | Speed: 6 | Effort: Low (4-6 hrs)
- Why: Validator's BUY threshold check should match generator's hysteresis
  (entry 0.70 / exit 0.50).
- Deliverable: Validator imports hysteresis logic from generator

**Item 4.4 — Centralize ETF list (today's deferred Sprint task)**
- Financial: 1 | Tech: 6 | Risk: 2 | Speed: 10 | Effort: Low (3-5 hrs)
- Why: Three places (signals/generator.py, scripts/daily_uw_snapshot.py,
  features/builder.py) have different ETF lists. Config drift risk.
- Deliverable: Single source of truth for ETF list

### 7.6 Deferred to Sprint 2 (June 11+)

- Hardcoded "accuracy.db" path centralization (16 files)
- Intraday model rebuild (broken since May 4)
- Alpha multiplier (1,292 candidates per project memory)
- Dollar-neutral portfolio mode (neutralizer.py exists, integration deferred)
- News novelty / cross-asset / stat arb prototypes
- Per-tier model classification (4-tier A/B/C/D Wilson CI)
- Polished hybrid dashboard with animations + mobile
- Multi-broker integration

---

## 8. Risks + Mitigations

### Risk 1: Walk-forward debug takes longer than estimated
- **Probability:** Medium-High (per memory: "still hangs in setup")
- **Impact:** -1 to -3 days for debugging, blocks Item 3.1
- **Mitigation:** ChatGPT debug plan exists in memory. Step 1: faulthandler
  + SIGUSR1 hook. Limit debug effort to 16 hrs before escalating.

### Risk 2: Continuous sizing math errors
- **Probability:** Medium (math change to live system)
- **Impact:** -1 day for revert + redesign
- **Mitigation:** Feature flag default OFF. Log both tier and continuous
  outputs in parallel for 1 week before activating.

### Risk 3: Operator burnout at 60-80 hr/week
- **Probability:** HIGH (per project memory)
- **Impact:** Catastrophic — broken system + tired operator
- **Mitigation:** Hard stop at 4-5 hr/day. Full days off when needed.
  This roadmap targets 20-25 hrs/week, not 60-80.

### Risk 4: Pipeline B fails Saturday May 9
- **Probability:** Medium (Schema v2 untested in cron)
- **Impact:** -1 to -2 days for fix
- **Mitigation:** Backup at accuracy.db.bak.before_v2_20260508_1355.
  Migration script idempotent. Revert path documented.

### Risk 5: Claude rule violations cost session time
- **Probability:** HIGH (today's track record: 6+ violations)
- **Impact:** 30-60 min per incident
- **Mitigation:** precommit_audit.sh runs on every commit.
  Rules 32-43 explicit in SPRINT_PLAN.md. Atom continues to enforce.

### Risk 6: Scope creep within 4 weeks
- **Probability:** Medium
- **Impact:** Items slip into Sprint 2
- **Mitigation:** Tier structure makes cuts mechanical. If a week's work
  isn't done, Tier N+1 items defer first, not Tier 1.

---

## 9. Sprint Integration

### 9.1 How this fits SPRINT_PLAN.md

This roadmap maps onto Sprint 1 + Sprint 2 weeks already in SPRINT_PLAN.md:

| This roadmap | SPRINT_PLAN.md | Calendar |
|--------------|----------------|----------|
| Week 1 (Tier 1) | Sprint 1 Week 1 | May 7-14 |
| Week 2 (Tier 2) | Sprint 1 Week 2 | May 14-21 |
| Week 3 (Tier 3) | Sprint 2 Week 3 | May 21-28 |
| Week 4 (Tier 4) | Sprint 2 Week 4 | May 28 - Jun 4 |
| Deferred | Sprint 2 Week 5 + Sprint 3 | Jun 4+ |

### 9.2 SPRINT_PLAN.md updates needed

After this doc is committed:
- Add reference to docs/ROADMAP_HYBRID_ADVISOR.md in Sprint 1 Week 2
- Add Item 1.3 (manual trade log) to Sprint 1 Week 1 if not already there
- Add Item 3.1 (continuous sizing) explicitly to Sprint 2 Week 3
- Confirm Item 4.x items align with existing Sprint 2 scope

### 9.3 Decision points

- **End Sprint 1 (May 21):** Walk-forward AUC measured. If still ~0.510,
  decide whether to continue Sprint 2 OR pivot.
- **End Sprint 2 (June 11):** Continuous sizing validated. Decide whether
  to advance to Sprint 3 (paper trading).
- **End Sprint 3 (July 2):** Paper trading results. Decide whether to
  fund live capital.

---

## 10. What I Don't Know

Honest uncertainty list:

1. **Whether continuous sizing actually improves accuracy.** Math is
   defensible; empirical impact unknown until walk-forward + fitness
   scoring runs.

2. **Whether tooltips will reduce decision time meaningfully.** They
   should help dyslexia-friendly reading, but adoption depends on actual
   use patterns.

3. **Whether manual trade log will get used consistently.** Discipline
   matters. If logs are sporadic, slippage analysis is meaningless.

4. **Whether 4 weeks is enough for MVP.** 4-week estimate assumes:
   - Walk-forward debug succeeds in <16 hrs
   - No major Pipeline B regression
   - Claude rule violations stay <2/day on average
   - Operator hits 20-25 hrs/week sustainably

5. **Whether the tier system has hidden coupling I haven't found.**
   Today's audit found dependencies in position_sizer, Portfolio, and
   signal_alerter. There may be more.

6. **Whether the "advisor mode" UX is right.** Mark Executed button vs
   form vs CSV import — operator preference unknown until tested.

7. **Whether Kelly fraction of 0.25 is the right starting point.**
   Industry typical is 0.20-0.50. May need tuning per ticker volatility.

8. **Whether SHAP feature culling will help or hurt.** Removing
   bottom-importance features sometimes removes signal in interaction terms.

---

## 11. Success Criteria

### 11.1 By end of Week 1 (May 14)

- [x] Schema v2 migration complete (shipped May 8 — commit 7951556)
- [x] SHAP fix shipped (May 8 — commit a2f9401)
- [x] Pre-commit audit infrastructure (May 8 — commit d64b286)
- [x] Threshold default aligned (May 8 — commit fe840e9)
- [ ] Pipeline B Saturday run validates Schema v2 + SHAP
- [ ] Walk-forward producing trustworthy AUC
- [ ] Manual trade log table + Mark Executed UI
- [ ] Initial P&L tracker view

### 11.2 By end of Week 2 (May 21)

- [ ] 2-tier system clean (tier multipliers simplified)
- [ ] Tooltips on Dashboard + Portfolio + Accuracy
- [ ] SELL signal decision logged (live OR formally deferred)
- [ ] Sprint 1 retrospective written

### 11.3 By end of Week 3 (May 28)

- [ ] Continuous Kelly sizing live behind feature flag
- [ ] Hybrid dashboard MVP on 3 main pages
- [ ] SHAP feature culling complete
- [ ] Walk-forward measurement of impact

### 11.4 By end of Week 4 (June 4)

- [ ] Realized P&L slippage tracker
- [ ] Sector concentration limits + alerts
- [ ] Validator hysteresis-aware
- [ ] ETF list centralized
- [ ] Sprint 2 retrospective

### 11.5 KPIs

- **Walk-forward AUC:** target ≥ 0.52 by end of Week 4 (vs 0.510 today)
- **BUY accuracy:** target ≥ 51% by end of Week 4 (vs 49.3% today)
- **Operator decision time:** < 60 sec for morning brief, < 5 min for
  per-ticker drill-down (subjective, self-reported)
- **Trade log usage:** ≥ 80% of executed trades logged within 1 day
- **Rule violations:** ≤ 2 per session by Week 4

---

## 12. Decision Log (May 8 Session)

This section captures the reasoning that led to this doc. Future-you
will read this in 3 weeks and need context.

### 12.1 What we shipped today (6 commits)

1. **7951556** — Schema v2: 9 multiplier columns + non-destructive validator
2. **a2f9401** — SHAP fix: missing kwarg silently dropping importance data
3. **d64b286** — precommit_audit.sh: enforces Rules 32-35 mechanically
4. **c1e66db** — ETF 404 cleanup: XLI/XLU/XLV added to skip set
5. **8116fa7** — Rules 32-43 in SPRINT_PLAN.md (12 new rules)
6. **fe840e9** — Threshold alignment: generator default 0.55 → 0.70

### 12.2 What we considered but didn't ship

**Option α (rollback all changes):** Considered for safety. Rejected because
generator alignment was a real win.

**Option β (partial — keep generator, revert UI labels):** SHIPPED. This
is fe840e9.

**Option γ (full 2-tier conversion):** Discussed extensively. Audit found
position_sizer dependency means γ changes production sizing behavior.
Cannot ship without walk-forward validation. Deferred to Week 2.

**Option δ (continuous Kelly sizing):** This roadmap's Item 3.1. Industry
standard. Real fix. Requires Week 1 walk-forward debug as prerequisite.

### 12.3 Why we rejected the 3-day timeline

Atom asked: "Can we crunch Q1-Q4 in 3 days?"

Honest answer: No. Realistic MVP scope is 12-16 days of focused work.
3 days = ~15 hours = roughly 1/5 of needed effort.

What ACTUALLY fits in 3 days:
- Schema v2 validation (Saturday)
- Walk-forward weekly run (Sunday)
- ~6-8 hours of new code

The 4-week plan in §7 honestly fits the scope.

### 12.4 Why we rejected the 2-week MVP timeline

Atom asked: "Can we crunch it in 2 weeks?"

Honest answer: Possible but high-risk. 2-week MVP requires:
- Pipeline B works first try
- Walk-forward debug succeeds in <8 hrs
- No major rule violations
- Operator at 30+ hrs/week

Today's session demonstrated 6+ rule violations, walk-forward still
broken, 60-80 hr/week pace unsustainable. 2-week timeline assumes
none of this is true.

4-week timeline: same scope, more buffer for reality.

### 12.5 Lessons from today's session

- Audit BEFORE patching. Caught 3 partial-fix attempts today.
- Smaller scope ≠ safer. Sometimes leaves dead code that creates worse bugs.
- Dead code compounds. classifier.py predict_today, daily_runner MEDIUM
  bucket, Portfolio L86 — all tier-related dead code from prior incomplete
  fixes.
- The 4-tier model classification (A/B/C/D Wilson CI) in Sprint 1 Week 2
  is the right venue for tier system rebuild, NOT a 4 PM Friday cleanup.

---

## 13. Glossary (preview)

Full version lands in `docs/glossary.md` Sprint 1 Week 2.

| Term | Plain English |
|------|---------------|
| **prob_up** | Raw model probability of UP move |
| **prob_eff** | Adjusted probability after sentiment, options, regime |
| **edge** | prob - 0.50 (how much above coin flip) |
| **Kelly fraction** | Position size that maximizes long-run growth |
| **Vol-target** | Cap weight so portfolio targets X% annual volatility |
| **rank** | Cross-sectional position (e.g. 3rd of 125 today) |
| **alpha** | Return above benchmark (SPY) |
| **fitness** | Per-model historical accuracy + sample size signal |
| **hysteresis** | Different thresholds for entry vs exit |
| **regime** | Macro state (VIX, rates, sentiment) |

---

## 14. Open Questions for Atom

These need decisions before Week 2 work starts:

1. **Trade log entry format:** Quick form vs full form vs CSV import?
2. **Feature flag mechanism:** env var, config file, or DB toggle?
3. **Glossary location:** docs/glossary.md or in-app help page?
4. **Walk-forward debug priority:** Spend up to 16 hrs OR escalate sooner?
5. **Sector cap values:** Use AI thesis playbook caps (25%) or different?
6. **Slippage threshold for alerts:** > 10 bps? > 25 bps? configurable?

Decide before May 14 (Week 1 closeout).

---

## 15. Versioning

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-08 | Atom + Claude | Initial draft, Q1-Q4 vision |

---

*End of roadmap. Revisit weekly during Sprint 1, monthly thereafter.*
