# FINDING — the horizon was wrong, not the model

**2026-09-05.** XGBoost on the existing feature set produces a consistent,
seed-stable, low-turnover edge at **h=40**. At h=3/5 — the horizon the fund has
been modelling — it is below random.

Reproducible via `analysis/horizon_sweep_test.py`.

---

## The result

Three seeds × 80 tickers, quarterly refits, day-weighted excess return over the
same day's universe. `+` counts how many of three seeds were positive.

| model | h | AUC | cap-3 | + | cap-5 | + | cap-10 | + |
|---|---|---|---|---|---|---|---|---|
| xgboost | 3 | 0.4796 | +0.069pp | 2/3 | +0.049pp | 3/3 | −0.011pp | 2/3 |
| xgboost | 5 | 0.4778 | +0.041pp | 1/3 | +0.110pp | 2/3 | +0.041pp | 1/3 |
| **xgboost** | **20** | **0.5153** | **+1.413pp** | **3/3** | **+1.195pp** | **3/3** | **+1.049pp** | **3/3** |
| **xgboost** | **40** | **0.5772** | **+5.194pp** | **3/3** | **+4.476pp** | **3/3** | **+3.724pp** | **3/3** |
| linear | 3 | 0.5129 | −0.096pp | 1/3 | | | | |
| linear | 20 | 0.5197 | −1.165pp | 0/3 | | | | |
| linear | 40 | 0.5452 | +0.342pp | 2/3 | +0.507pp | 2/3 | +0.434pp | 2/3 |

**Normalised per day, because a 40-day return is not comparable to a 3-day one:**

| horizon | cap-3 per day | turnover |
|---|---|---|
| 3 | 0.023pp | 61% |
| 5 | 0.008pp | 51% |
| 20 | 0.071pp | 29% |
| **40** | **0.130pp** | **27%** |

Roughly **6× the per-day excess at less than half the turnover.**

## Why this passes where twenty other configurations failed

The conditions were written into the script before it ran: excess return rising
with horizon, positive in EVERY seed, and turnover low enough that the gross
number survives cost. All three hold.

That matters because three single-seed results reversed on replication earlier
the same day — a top-N book at +1.17pp went to −0.81pp across seeds, a linear
economic edge at +0.645pp went to −0.021pp, and 3-of-3 model consensus came in
at −0.173pp against 1-of-3. Multi-seed was the filter that caught all three, and
this result passes it at 3/3 on both long horizons and all three caps.

## The model split is the interesting part

**XGBoost is below random at h=3/5 (AUC 0.478) and best at h=40 (0.577).
Linear is the reverse** — 0.513 at h=3 and negative economics at h=20.

That is consistent with the mechanism. Gu, Kelly & Xiu (2020) trace the gains of
trees and neural nets to nonlinear predictor interactions. Interactions need
signal to estimate; at daily SNR (~0.8%, return sd 3.49% against a mean 0.028%)
there is not enough, which is why the L2 sweep earlier that day improved
monotonically as coefficients were shrunk toward zero. At h=40 the SNR is high
enough for the tree's flexibility to pay.

So the earlier finding — "XGBoost overfits this feature set, linear generalises
better" — was **horizon-specific and stated too broadly.** It holds at h=3/5.
It inverts at h=40.

## What is NOT established

- **~14 independent periods.** 564 dates at h=40 with overlapping windows.
  Seed-consistency is meaningful; the confidence interval is still wide.
- **Costs are unmodelled.** +5.194pp gross at 27% turnover over 40 days. At
  10bps a leg that is roughly −0.05pp per rebalance, comfortably survivable
  unlike every h=5 result — but it needs the same day-weighted cost ladder the
  SI brick was given, not an assumed figure.
- **Capacity, concentration, drawdown path.** None measured. PCT7 looked
  validated on statistics and failed on exactly these.
- **Overlap with the SI brick.** The fund's one validated edge also runs at
  h=40. These may be picking up the same slow effect rather than two. Until
  their selections are compared, treat them as possibly one signal.
- **Survivorship.** `prices.db` is survivor-tilted. The SI leg decomposition
  sized this as near-moot for a low-DTC long leg; it has not been sized here.

## What this changes

**The scope document's Phase 4 was "target h=20/40". It should now be h=40
first**, and the firm-characteristics build should be designed for that horizon
from the start — which suits it anyway, since annual accounting characteristics
are near-constant within a year and were always a poor fit for a 5-day target.

**And it reframes the day's negatives.** Six insider constructions, four label
definitions, two estimator classes, ensembles and consensus voting all failed at
h=3/5. This suggests they failed because of the horizon, not because the
features are empty. Several deserve a rerun at h=40 before being called closed —
in particular the four label definitions, which were only ever tested at h=5.

**The kill switch stays on.** Gross excess return in a backtest is not a
validated brick, and the four unmeasured items above are exactly what turned
PCT7 from "brick #2" into a retraction earlier the same day.
