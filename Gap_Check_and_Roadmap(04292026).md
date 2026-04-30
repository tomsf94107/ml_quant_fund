# ML Quant Fund — Gap Check & Accuracy Roadmap

*Prepared April 29, 2026 — after the Accuracy Explorer dashboard went live.*

This document does three things:
1. Reconciles what's been built across the WorldQuant + sector-accuracy work into a single inventory.
2. Confronts the realistic AUC (0.510) vs the in-context AUC (0.967) — and what it means.
3. Lays out a prioritized roadmap for actually moving accuracy.

---

## Part 1 — The numbers that just changed the conversation

Tonight's Accuracy dashboard showed these realized numbers:

| Metric | Value | Reading |
|---|---|---|
| Live ROC-AUC (avg across 130 tickers) | **0.510** | 1 percentage point above coin flip |
| Live BUY accuracy | **49.3%** (n=3,483) | Below coin flip; CI likely ~[47%, 51%] |
| Avg daily alpha vs SPY | **-0.08%** (38 days) | Slightly losing to SPY |
| Days beating SPY | **16/38 (42%)** | Should be 50%+ to claim alpha |
| Avg tickers beating SPY | 48% | Coin flip |
| Avg Brier score | 0.257 | 0.25 = random; you are at random |

**What this tells us:**

- **The AUC-0.967 figure in your project memory is training AUC**, not realized AUC. The realized number is 0.510. That ~0.46 gap is a leak — the model fit the training data nearly perfectly but the pattern doesn't generalize.
- **You have a marginally negative system right now.** Not a disaster, but not making money either. Slightly losing to SPY over 38 trading days.
- **This is normal for a well-designed retail quant.** *Finding Alphas* Ch. 8 explicitly predicts this: any out-of-sample equity AUC above 0.6 is suspicious; 0.51–0.55 is the realistic range. You're hitting the lower end of realistic.
- **The fix is not "find a stronger signal."** At 0.510, no individual signal is strong. The fix is the WorldQuant playbook: many weak decorrelated alphas + tight execution discipline + neutralization.

This finding makes the WorldQuant work I've been building **more important**, not less. With a 0.96 AUC system you'd already be done. With a 0.51 AUC system, the alpha-multiplier + fitness scoring + neutralization stack is exactly the toolkit you need.

---

## Part 2 — Inventory: what's actually been built

Three categories: deployed and running, built but not wired in, documentation only.

### 2.1 Deployed and live (in production code, running today)

| File | Path | Status |
|---|---|---|
| Unified Accuracy dashboard with Explorer tab | `ui/2_Accuracy.py` | ✅ **Live** — 4 tabs working, Wilson CIs + Bayesian shrinkage + multi-dim grouping all operational |
| Sector metadata source of truth | `tickers_metadata.csv` | ✅ Live — read by Explorer tab |
| Per-ticker fitness scoring (committed) | `analysis/fitness_scorer.py` | ✅ Pushed but not yet *run* against your data |

### 2.2 Built and in the project, NOT wired in yet

| File | Path | Why it's not running |
|---|---|---|
| Sector / dollar-neutral portfolio construction | `portfolio/neutralizer.py` | No Streamlit page calls it yet; no production signal uses it |
| Alpha transformation operators (17 ops) | `features/alpha_transformations.py` | Not called from `features/builder.py`; opt-in flag not flipped |
| Sector accuracy v2 with Wilson + shrinkage (CLI) | `scripts/sector_accuracy_v2.py` | Standalone CLI; logic now lives inside Explorer tab too, so this is mostly redundant — keep for command-line use |

### 2.3 Documentation only (no executable code)

| File | Purpose |
|---|---|
| `Finding_Alphas_Summary.md` | The book mapped to your stack |
| `BRAIN_replication.md` | 8-step plan for solo dev to replicate WorldQuant's alpha factory |
| `Sector_Integration_Guide.md` | How sector work threads into the WQ-style stack |
| `RECONCILIATION_NOTE.md` | Which older files needed updates |

### 2.4 The Accuracy Explorer screenshots already revealed something useful

From the per-sector results visible in your screenshots:

**Sectors with statistically real edge (CI lower bound > 50% by my read of the visible columns):**
- Core Silicon: n=319, raw acc 0.624, CI [0.569, 0.675]
- Financials: n=128, raw acc 0.641, CI [0.555, 0.719]
- Market ETF: n=117, raw acc 0.641, CI [0.551, 0.722]
- Server Hardware: n=39, raw acc 0.692, CI [0.536, 0.814]
- Materials: n=53, raw acc 0.660, CI [0.526, 0.773]

**Sectors that look strong but are statistically borderline (CI lower bound just below 50%):**
- Custom Silicon, Crypto, Infrastructure, Consumer Tech, Biotech, Fintech

**Sectors clearly at coin flip (CI straddles 50%):**
- Hyperscaler, Defense, Networking, Energy Storage, Memory

This is your real edge map. Not what you would have guessed from raw point estimates. The system has 5-ish sectors with statistically defensible edge, ~6 borderline, and the rest at noise.

---

## Part 3 — Gap analysis: what's missing

Organized by impact on accuracy, highest first.

### CRITICAL — find the AUC 0.967 → 0.510 leak

This is upstream of everything else. A 46-point AUC gap between training and live cannot be ignored.

**What's missing:**
- A strict walk-forward backtest harness (`analysis/walk_forward.py`)
- Purged k-fold cross-validation
- Feature/target audit for look-ahead leakage
- Embargo period (5 trading days) between train and test
- Deflated Sharpe ratio reporting

**Common leak sources to audit:**
1. **Target/feature contamination.** Is `actual_return` computed using close-of-prediction-day data that's also in features? (This is how most leaks happen.)
2. **Restated fundamentals.** Are you using as-of-today fundamentals when training on past dates? They'd contain post-period restatements.
3. **Look-ahead in normalization.** If you z-score features using stats computed across the entire dataset (including future), you've leaked future info.
4. **Survivorship bias.** Are inactive tickers (BNED, BYND, CRCL, RZLV, SENS, WCC) in training but not test? Or vice versa?
5. **Hyperparameter selection on the validation set that's later treated as test.**

**Action:** build `analysis/walk_forward.py` with strict embargo. Run on the 303-model ensemble. Report realistic AUC. If it stays at 0.510 — at least you've confirmed the live number is real. If it pops to 0.55-0.6 — you have a more useful baseline than you thought.

### HIGH PRIORITY — turnover discipline

At 49.3% BUY accuracy, transaction costs eat any tiny edge. Every avoided trade saves money.

**What's missing:**
- Hysteresis bands in `signals/generator.py` (BUY at >0.55, exit at <0.45 instead of flip-on-cross at 0.50)
- Per-ticker turnover measurement (currently only per-model aggregate via fitness_scorer)
- Cost model in backtest (fitness_scorer assumes turnover = cost, but real-world cost depends on bid-ask spread + market impact)

**Action:** add hysteresis. 5-line change. Will cut turnover 30-50% per *Finding Alphas* Ch. 6. Should help live alpha-vs-SPY immediately.

### HIGH PRIORITY — calibration audit

The Calibration tab in your new dashboard exists. Look at it. If the chart shows the predicted probability bucket aligning with actual win rate (i.e., 70% confidence ≈ 70% win rate), the model is calibrated and high-confidence trades are trustworthy. If the chart is *flat* (all buckets show ~50% actual win rate regardless of predicted confidence), the model output isn't even ordinally meaningful.

**Action:** open Calibration tab. Look at the dots vs the dashed diagonal line. If they sit on the diagonal, calibration is fine. If they're flat at 50%, you have a calibration problem and need Platt scaling or isotonic regression.

### HIGH PRIORITY — feature-feature correlation cull

You have 79 features. Almost certainly 30-40 are redundant.

**What's missing:**
- A cull script (described in `BRAIN_replication.md` step 5 but not built)
- Correlation matrix run against `prediction_features` table
- Drop or merge anything ρ > 0.7

**Action:**
```bash
python -c "
import sqlite3, pandas as pd, numpy as np
conn = sqlite3.connect('accuracy.db')
df = pd.read_sql('SELECT * FROM prediction_features LIMIT 50000', conn)
feat_cols = [c for c in df.columns if c not in ('ticker', 'prediction_date', 'horizon')]
corr = df[feat_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [(col, upper[col].idxmax(), upper[col].max())
           for col in upper.columns if upper[col].max() > 0.7]
for col, with_col, c in sorted(to_drop, key=lambda x: -x[2])[:30]:
    print(f'{col:30s} corr={c:.2f} with {with_col}')
"
```

This will print the worst redundancy offenders. Then you decide which to drop.

### MEDIUM — alpha multiplier (the WorldQuant Ch. 10 lever)

Even after culling, you'd have ~40-50 unique features. Apply 17 transformations → ~700 candidate alphas. Cull those by correlation → ~150-300 surviving alphas. Combine via neutralizer → portfolio with diversified weak edges.

**What's missing:**
- `alpha_transformations.py` not yet wired into `features/builder.py`
- Per-alpha fitness scoring (currently per-(ticker, horizon) only)
- A proper alpha registry as described in BRAIN_replication.md
- The correlation cull on transformed alphas

**Action:** sequence over 2-3 weekends:
1. Wire `explode_panel()` in `features/builder.py` behind a config flag
2. Run the correlation cull on the exploded panel
3. Compute per-alpha fitness via a modified fitness_scorer
4. Drop low-fitness alphas
5. Use survivors via the neutralizer

### MEDIUM — research portfolio page using neutralizer

Right now your live signals are long-only BUY/HOLD per ticker. The neutralizer can show you what a sector-neutral or dollar-neutral portfolio would look like with the same `prob_up` inputs. It's a research view, not a production change.

**What's missing:**
- A Streamlit page `ui/8_Research_Portfolio.py` that:
  - Pulls today's `prob_up` from latest predictions
  - Runs `neutralize(..., mode='sector', long_only=True)` for production-comparable view
  - Runs `neutralize(..., mode='sector', long_only=False)` for dollar-neutral view
  - Shows weights, sector exposure, and a backtest of the neutralized portfolio over the last 30/60/90 days
  - Compares neutralized portfolio's Sharpe vs your existing long-only Sharpe

This is one of the highest-value experiments you can run because it directly tests whether sector-neutralization adds Sharpe. If the neutralized portfolio backtests to higher Sharpe than your current selection, you have evidence to switch production logic.

### MEDIUM — SELL signal validation (your existing May 1 milestone)

This is in your project plan already.

**What's missing:** the validation query and ship/no-ship decision.

**Action:** May 1, run the existing query you have:
```sql
SELECT COUNT(*), 
       ROUND(100.0*SUM(CASE WHEN actual_return<0 THEN 1 ELSE 0 END)/COUNT(*),1) as acc,
       ROUND(AVG(actual_return)*100,3) as avg_ret 
FROM predictions p
JOIN outcomes o ON p.ticker=o.ticker
  AND p.prediction_date=o.prediction_date
  AND p.horizon=o.horizon 
WHERE p.prob_up<0.30;
```
If: 50+ samples, >60% accuracy, negative avg_ret → ship SELL.
If not: keep waiting.

### LONG-TERM — data-axis expansion

*Finding Alphas* Ch. 4: the highest-yield search is across data axes, not model architectures.

**What you have:** OHLCV, sentiment, options flow, dark pool, fear/greed, oil, vix, dxy, market regime.

**What you're queuing per project memory:**
- Polygon (real-time) — pending API key
- Unusual Whales ($125/mo) — pending key
- Massive Stocks Starter ($29/mo) — active

**What's still missing as data axes:**
- **Credit spreads** (HYG/LQD ratio as cheap proxy, real CDS via Markit if budget allows). Per your AI thesis playbook, this is a major regime indicator.
- **Cross-asset signals** (bonds, gold/silver ratio, USD/CNY)
- **Sector rotation indicators** (sector relative-strength vs SPY)
- **News novelty** (your sentiment scorer treats all news linearly; novelty-weighted decay would be sharper per Ch. 12)
- **Earnings calendar proximity** (alpha behaves very differently 1 week before vs after earnings)
- **Analyst revision data** (changes in consensus EPS — well-known WorldQuant alpha)

### LONG-TERM — alpha registry + Claude-as-consultant loop

The BRAIN replication plan, end state. Not for tonight.

**What's missing:**
- `alphas/` directory with standardized alpha format
- `scripts/run_alpha_library.py` runner
- `scripts/discover_new_alphas.py` weekly Claude prompt loop
- `scripts/kill_decayed_alphas.py` daily decay tracker

---

## Part 4 — Five accuracy-improvement levers, ranked

If realistic AUC is 0.510 and BUY accuracy is 49.3%, the math of what helps changes. Here's what *actually* moves the needle, in order:

### Lever 1: Find the leak
- **Potential improvement:** AUC could be anywhere from 0.50 to 0.70 depending on what's leaking.
- **Cost:** 1-2 days to build walk_forward.py, plus 1-3 days of detective work on features.
- **Risk:** Worst case the live AUC was always 0.51 and there's no leak — you've spent 3 days confirming what you already see. Best case you find a feature with future-info contamination and AUC pops to 0.58.

### Lever 2: Diversify into many weak decorrelated alphas
- **Potential improvement:** Per Ch. 10 math, 100 alphas at Sharpe 0.4 with ρ=0.15 → portfolio Sharpe ~1.5. From -0.08% daily alpha to +0.15-0.30% daily alpha is plausible.
- **Cost:** 2-4 weekends. Feature explode (1 day) + correlation cull (1 day) + fitness scoring (1 day) + neutralizer integration (1 day) + validation (1 day).
- **Risk:** If live AUC is genuinely 0.51 because there's no signal, multiplying it doesn't help. Won't know until you try.

### Lever 3: Cut turnover via hysteresis + longer holds
- **Potential improvement:** At 49.3% accuracy, every avoided losing trade saves money. Could push effective accuracy to 51-52% even if model accuracy stays at 49.3%, because hysteresis filters out the lowest-conviction trades.
- **Cost:** 1 hour. 5-line change in `signals/generator.py`.
- **Risk:** None. If hysteresis doesn't help, you remove it.

### Lever 4: Calibrate the model output
- **Potential improvement:** If model is uncalibrated, "70% confidence" trades are no better than "55% confidence" trades. Platt scaling makes prob_up actually predictive of win rate. Doesn't raise AUC but raises trade-selection quality at high confidence.
- **Cost:** 4 hours. Add calibration step in scoring pipeline.
- **Risk:** Low. Worst case calibration is already fine and the change is a no-op.

### Lever 5: Add new data axes (Polygon + UW + missing data)
- **Potential improvement:** Genuinely uncertain. Some data axes will help, most won't. Industry-standard win rate for new alpha-data feeds is 1-in-5.
- **Cost:** $125-250/mo subscriptions + 2-3 weekends per integration.
- **Risk:** Spending money on data that doesn't help. Mitigate by validating each new feature via fitness_scorer before going live.

---

## Part 5 — Prioritized roadmap

### THIS WEEKEND (now → Sunday)

1. **Open Calibration tab on the dashboard** (5 min). Look at the diagonal. Tells you whether calibration is part of the problem.
2. **Run fitness_scorer on existing models** (15 min):
   ```bash
   python -m analysis.fitness_scorer --db accuracy.db --write-table --csv fitness.csv
   ```
   Compare fitness ranking to AUC ranking. Look for high-AUC / low-fitness models — those are your overfitting victims.
3. **Add hysteresis to `signals/generator.py`** (30 min). 5-line change. Cheapest win available.
4. **Run feature correlation cull script** above (15 min). See how many of your 79 features are redundant.

### NEXT WEEK (May 1-8, includes the SELL milestone)

5. **May 1: SELL validation query.** Run the standard validation. Ship or don't.
6. **Build `analysis/walk_forward.py`** (1-2 days). Strict purged k-fold + 5-day embargo. This is the most important thing on the list.
7. **Run walk_forward on the 303-model ensemble.** Report AUC honestly. Decide whether to keep all 303 or cull.

### MID-MAY (May 8-22)

8. **Calibration fix** — if Calibration tab showed flat dots, add Platt scaling.
9. **Wire `alpha_transformations.py` into `features/builder.py`** behind a config flag. Don't enable by default until validated.
10. **Fitness-score the exploded alpha set.** Cull aggressively (keep top 20%).

### LATE MAY → JUNE

11. **Build `ui/8_Research_Portfolio.py`** using neutralizer.
12. **Backtest neutralized portfolio.** Compare Sharpe vs current long-only.
13. **If neutralized backtests better → switch production signal logic.**
14. **Begin alpha registry** (`alphas/` directory + standardized format).

### AFTER VIX < 20 (per your existing plan)

15. Vol-adjusted position sizing (already in your AI playbook).

### SEPTEMBER 2026

16. Statistical arbitrage pairs trading + sector limits + stop-loss (already in your plan).

---

## Part 6 — One uncomfortable truth

The AI Investment Thesis you wrote (in `/mnt/project/AI_Investment_Thesis.md`) is built on a base case of **mid-teens to low-30s percent returns** for the AI sleeve, with bull case 40-70%. That's a directional thesis on AI infrastructure.

Your **ML Quant Fund as it sits today does -0.08% daily alpha**, i.e., slightly underperforming SPY. The two are not competing — the thesis is your thinking framework for picking AI names; the quant fund is a different beast that's supposed to generate alpha through systematic prediction.

**The honest read:** the quant fund hasn't found edge yet. It might never find edge — most retail systematic systems don't. But the work I've helped you build (Wilson CIs in dashboard, fitness scoring, neutralization, alpha multiplier framework) gives you the infrastructure to either *find* edge or *prove honestly that there isn't any* without burning capital chasing it.

Both outcomes are valuable. The worst outcome — running production on a 0.967 AUC fiction without ever verifying the live number — is the one you've now avoided.

---

## Part 7 — Update your project memory

Your saved context still has "AUC ~0.967". This is misleading future-you and any AI you talk to. Suggest updating to something like:

> **Model performance (live, OOS, as of Apr 29 2026):** ROC-AUC ~0.510, BUY accuracy 49.3% on n=3,483 BUY signals, daily alpha vs SPY -0.08% over 38 days. Training AUC was ~0.967 — gap of ~0.46 indicates significant overfitting or feature/target leakage. Walk-forward backtest pending. The 0.510 is the number to optimize against.

That single edit changes how every future conversation (with AI or yourself) frames the work.

---

*Refer back to this document weekly. Update it as gaps close. The goal is to drive each "missing" item to "deployed and live" — and to be honest about what works and what doesn't along the way.*
