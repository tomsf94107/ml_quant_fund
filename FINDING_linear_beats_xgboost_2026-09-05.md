# FINDING — the direction model uses the wrong estimator

**2026-09-05.** XGBoost overfits this feature set. A heavily-regularised linear
model beats it on AUC universe-wide and on day-weighted return at every position
cap.

**Status: not yet actionable.** The multi-seed check that killed an earlier
top-N result is still running. This records what is established and what is not.

---

## The diagnosis was written in June and had never run

`models/linear_baseline.py`, dated 2026-06-01, exists to answer one question,
in its own words:

> *"is the per-ticker direction model's ~0.50 test AUC from the TREE
> OVERFITTING, or from the SIGNAL being genuinely absent at this target? Linear
> (L2-logistic) cannot memorize noise like XGBoost."*

It crashed on every ticker with `Input X contains NaN`. `LogisticRegression`
raises on NaN; XGBoost handles it natively, so the harness that works for the
tree fails for the linear model. Three months, no answer.

The NaNs are real data states, not corruption: `short_pct_float` is 100% NaN
universe-wide, the four `inst_*` are ~94% NaN because their source only begins
2026-03-19, several `fund_*` run 21–40%. Dropping rows with any NaN would delete
the entire panel.

**Fix:** a `Pipeline` of `SimpleImputer(median) → StandardScaler →
LogisticRegression`, so imputer and scaler are fit on the training fold only and
merely applied to test. Median rather than mean because the panel is
heavy-tailed. A column that is 100% NaN in the training fold is dropped by the
imputer, which is correct.

## The regularisation sweep

10 tickers, h=5, expanding walk-forward:

| C | train AUC | test AUC | gap |
|---|---|---|---|
| 0.1 | 0.757 | 0.559 | +0.198 |
| 0.01 | — | 0.562 | +0.159 |
| 0.001 | 0.662 | 0.564 | +0.098 |
| 1e-05 | 0.609 | 0.574 | +0.034 |
| 1e-06 | 0.608 | 0.575 | +0.033 |

Train falls, test **rises**, gap collapses — monotone over five steps, then
plateaus. That is regularisation stripping memorised noise while leaving signal
intact. If the 0.559 were noise, tightening C by four orders of magnitude would
have destroyed it.

**At the plateau the coefficients are shrunk nearly to zero, so the model
approaches a plain equal-weighted composite of standardised features.** The
reading is that the signal is broad and diffuse across many weak features, and
that fitting weights to them destroys it. That is consistent with everything
else measured this session: six insider constructions null individually, four
label definitions null, top-decile lift 2–4pp. Nothing is individually strong; a
flat combination of everything beats trying to learn which matter.

## It holds universe-wide, and at both horizons

| run | tickers | train | test | gap | >0.52 |
|---|---|---|---|---|---|
| h=5, C=0.001 | 246 | 0.6703 | **0.5756** | +0.095 | **219/246 (89%)** |
| h=5, C=1e-05 | 131 | 0.6256 | **0.5797** | +0.046 | 110/131 (84%) |
| h=3, C=1e-05 | 131 | 0.6069 | **0.5592** | +0.048 | 106/131 (81%) |

**XGBoost on identical features, folds and target: train 0.66–0.79, test ~0.50,
gap ~0.25.**

Two things make this more than a single number. **89% of 246 tickers clear
0.52** — breadth, not a handful of names carrying a mean. And **h=3 works too**,
so the overfit is a property of the estimator on this feature set, not something
specific to the 5-day target.

## The AUC converts to return — on one seed

AUC has failed to convert repeatedly today. The `top_decile` target posted AUC
0.7316 against production's 0.5100 and delivered **+0.01pp**. PCT7 had a real
within-date edge and −1.97% day-weighted. So the economic test matters more than
the AUC.

Day-weighted excess over the same day's universe, 120 tickers, seed 5, C=1e-05:

| cap | linear | xgboost | linear turnover |
|---|---|---|---|
| 1 | **+1.283pp** | −0.660pp | 49% |
| 3 | **+0.645pp** (t +1.48) | −0.467pp (t −2.01) | 42% |
| 5 | +0.357pp | −0.311pp | 36% |
| 10 | +0.210pp | −0.122pp | 29% |
| decile | +0.180pp | −0.105pp | 29% |

**Linear is monotone at the head** — better as the book concentrates, which is
the signature of a real ranking and the exact place PCT7 failed (+1.46% over all
its selections, −2.38% at the top 3). Turnover is also lower than XGBoost's
70% at cap 3.

**And XGBoost's top-3 is significantly NEGATIVE at t=−2.01.** The production
model's highest-conviction picks underperform the universe. That is an
independent measurement of what the 2026-05-31 kill switch asserted:
*"near-coin-flip AND inverted at the extremes."*

## What is NOT established

- **Significance.** Linear cap-3 is t=+1.48 across 339 overlapping days, which
  is roughly 68 independent 5-day periods. It clears no bar.
- **Seed stability.** One seed, one split. Earlier today a top-N book measured
  +1.17pp on seed 5 and +0.14 / −0.32 / −0.81pp on seeds 1–3. The multi-seed
  walk-forward is running; until it returns, this is a draw, not a finding.
- **How much is linear being good versus XGBoost being bad.** XGBoost's −0.467pp
  at t=−2.01 is the strongest single number in the table, and it is negative.
  Part of the gap is the tree actively hurting rather than the linear model
  helping, and only the seed spread separates them.
- **Costs.** +0.645pp gross at 42% turnover. At 10bps a leg that is roughly
  −0.08pp, leaving +0.56pp — but that arithmetic rests on an unverified
  execution cost.

## If it survives the seeds

The change is to `models/classifier.py`: a regularised linear estimator, or a
linear-plus-tree blend, in place of XGBoost alone. Ensembling XGB with LightGBM
was measured today at **+0.002 AUC (t=+0.63)** — two gradient-boosted tree
libraries on one feature set produce correlated errors and cancel nothing. A
linear model is a genuinely different model class and is what ensembling theory
actually requires.

**The kill switch stays on regardless.** Nothing here clears an economic bar
yet, and re-enabling BUYs is a deliberate act after Step 4 validation.
