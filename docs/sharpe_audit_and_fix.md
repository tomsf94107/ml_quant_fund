# Dashboard Data Audit + Sharpe Fix (June 3 2026)

## What triggered this

Kill switch OFF; dashboard showed SMMT prob_eff 91.4% (h3), 94.0% (h5) and per-ticker
Sharpe 2.5-4.1. Looked unreliable. Full audit below.

## Audit findings (what is and is NOT a bug)

1. **No display bug.** The dashboard faithfully renders signals_cache.json. (An early
   “display bug” suspicion was a horizon mismatch during the audit — the cache holds
   1/3/5d and the screen showed 3d; values match the cache exactly.)
1. **No prob_eff inflation.** prob_eff == base prob (ratio 1.0 at every horizon); the
   7 overlay multipliers (risk/sent/regime/options/squeeze/intraday/fg) were neutral/off.
   prob_eff = generator.py:854, capped 0.95 at :855.
1. **prob distributions are HEALTHY, not overconfident.** Across 149 tickers:
- h1: mean 0.496, median 0.505, ZERO >=0.80
- h3: mean 0.498, median 0.522, 3 >=0.80, 1 >=0.90
- h5: mean 0.510, median 0.513, 1 >=0.80, 1 >=0.90
  The model is appropriately uncertain (centered ~0.50) for a 0.535-AUC system.
  SMMT is a 1-of-149 OUTLIER, not systemic. The screen looked alarming only because
  it was SORTED by prob desc — you saw the handful of extreme tails at the top.
1. **SMMT specifically: distrust it.** 0.91/0.94 prob on a small, volatile, low-float
   name ($15.71, ATR 1.39) is a classic per-ticker OVERFIT signature. Not a system bug.
1. **Sharpe IS inflated — confirmed in code (generator.py:449).** Formula is standard
   `sqrt(252) * mean/std` but the CONSTRUCTION inflates it:
- **In-sample**: ret_strat = signal * return over the same window the model saw.
  In-sample Sharpe is the same illusion as the 0.967 in-sample AUC.
- **Small-sample / low-trade**: volatile names with few long-days get a small, fragile
  std; sqrt(252) magnifies the fragile ratio. SMMT 4.07, DNA 4.09 = exactly this.
- **Gross of costs**: no slippage/turnover deduction (fitness_scorer uses 10bps/turn;
  this Sharpe does not). Worst on high-turnover names.

## Verdict

The dashboard is NOT miscalculating. It is faithfully showing an AT-CEILING model
(most probs ~0.50 = no edge), a few overfit per-ticker outliers (SMMT), and an
in-sample/gross/short-sample-inflated Sharpe. The unreliability is in the MODEL and
the SHARPE METRIC, not the rendering. With kill switch OFF you would be trading a
no-edge model — recommend kill switch back ON unless this was a deliberate test.

## Sharpe fix — researched solution (Bailey & Lopez de Prado)

Literature (already cited in your research files) identifies 3 inflation sources:
short samples, non-normal returns, and selection bias. Established corrections,
in increasing effort:

### Fix tier 1 (DO THIS NOW — cheap, high value): honesty guardrails

- **Relabel** the column: it is an IN-SAMPLE, GROSS Sharpe — not tradeable. Tooltip
  must say so. (Current tooltip already hints “suspiciously high can mean overfit”.)
- **Grey/flag** any per-ticker Sharpe with n_trades < ~30 (small-sample = unreliable),
  same n-guard pattern we shipped for accuracy.
- **Cap the display** or show “n/a (insufficient trades)” below a trade threshold.

### Fix tier 2 (BETTER — moderate): Probabilistic Sharpe Ratio (PSR)

Replace/augment the raw SR with PSR = probability the true SR > 0, adjusting for
sample length + skew + kurtosis. Standard estimator (Bailey-Lopez de Prado 2012):

```
SR_std = sqrt( (1 - skew*SR + ((kurt-1)/4)*SR^2) / (n-1) )
PSR(0) = Phi( SR / SR_std )    # Phi = standard normal CDF
```

PSR in [0,1] is far harder to game than a raw 4.07 — a high SR on n=12 noisy trades
yields a low PSR. Show PSR instead of (or beside) raw SR. Also report Minimum Track
Record Length (minTRL): how many trades needed before the SR is trustworthy.

### Fix tier 3 (BEST — bigger): out-of-sample, net-of-cost Sharpe

The real fix for the in-sample problem: compute Sharpe on PURGED walk-forward returns
(you already have analysis/walk_forward.py + the 10bps cost model in fitness_scorer).
Report the OOS net Sharpe, not the in-sample gross one. This is the only Sharpe that
predicts live performance.

## Recommendation

Ship Tier 1 now (relabel + n_trades guard + cap) so the column stops inviting false
confidence. Add Tier 2 (PSR) as the displayed metric — it is a ~20-line function and
directly answers “is this SR real given the sample.” Defer Tier 3 to when you wire
the WF harness into the per-ticker metrics (it is the correct long-term answer).

## TESTED — what actually works (June 3, verified in sandbox)

Built + tested the PSR helper (out/sharpe_psr.py). Test result corrected my own plan:

- **The n_trades < 30 GUARD is the PRIMARY fix.** Tested on an SMMT-like case
  (8 lucky up-trades, raw SR 6.36): the guard correctly flags reliable=False.
  This alone kills the inflated per-ticker Sharpe problem.
- **PSR(0) is NOT sufficient alone** — “P(true SR > 0)” is ~1.0 even on 8 lucky
  trades. PSR must be benchmarked at a MEANINGFUL SR (e.g. > 1.0 annual), and even
  then is a SECONDARY signal. A real-edge n=120 SR 1.28 gave PSR(>1)=0.65; pure
  noise gave 0.00. So PSR(>1) is a good “is this real” probability for names that
  already pass the trade-count guard.

## FINAL shippable fix (in order)

1. **n_trades guard (primary):** per-ticker Sharpe with n_trades < 30 -> display
   “n/a (insufficient trades)” or grey it. Kills SMMT 4.07 (n=8). PROVEN.
1. **Relabel:** “in-sample, gross-of-cost Sharpe — not tradeable.”
1. **PSR(>1) secondary column:** shown only where n_trades >= 30; the probability
   the true SR beats 1.0 given sample/skew/kurtosis. Code in sharpe_psr.py.
1. **Tier 3 (defer):** replace with OOS net-of-cost Sharpe from the WF harness — the
   only truly tradeable number, but a bigger build.