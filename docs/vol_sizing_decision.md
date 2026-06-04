# Vol-Prediction → Momentum Sizing: DECISION (June 3 2026)

## Verdict: DO NOT build a vol-level forecaster. Keep naive trailing-vol sizing.

## How we decided (oracle pre-test, before building anything)

Tested 3 sizings of the long-only top-decile momentum book on the SAME purged
folds (analysis: vol_sizing_ceiling.py). The “oracle” uses PERFECT future vol
(look-ahead, deliberately, to find the CEILING of what any forecaster could do):

|sizing                               |mom_6_1 net Sh|mom_12_1 net Sh|
|-------------------------------------|--------------|---------------|
|equal-weight                         |+1.225        |+0.770         |
|naive trailing-40d-vol (CURRENT LIVE)|**+1.437**    |**+0.942**     |
|PERFECT future-vol (ORACLE ceiling)  |+1.097        |+0.635         |

**Room a forecaster could fill = oracle − naive = −0.34 / −0.31 (NEGATIVE).**

## Interpretation

Even PERFECT knowledge of future volatility makes sizing WORSE than the naive
trailing-vol already shipped. A better vol forecast would actively hurt.

Why: inverse-vol sizing up-weights low-vol names. For a MOMENTUM book, the names
that keep running are often higher-vol; sizing toward future-calm names tilts into
the momentum underperformers. Naive trailing-vol works better precisely because it
is laggier/gentler and does not over-commit to future-calm names. Vol and
momentum-return are entangled such that better vol forecasting is counterproductive
for sizing here.

## Consequences

- Keep naive trailing-40d-vol inverse sizing (best of the three, beats equal-weight).
- Do NOT build the Ridge vol-LEVEL forecaster. The oracle test proves no forecaster
  can help (ceiling is below the incumbent).
- vol_prediction.py status RESOLVED: it is a validated VOL-REGIME classifier
  (forward vol predictable, AUC>0.58) — usable for regime GATING if ever wanted,
  but PROVEN not useful for position sizing. Not orphaned, not pending — tested
  and shelved-for-sizing with evidence.

## Rule #1 note

Commit 3a7e8d5 described “Ridge vol LEVEL, rank-IC +0.10, feeds C2 sizing.” The
SHIPPED file is actually a LogisticRegression classifier scored by AUC (regime,
not level). The level-forecaster the commit implied was never built. The oracle
test makes building it moot. Saved a full build by testing the ceiling first.