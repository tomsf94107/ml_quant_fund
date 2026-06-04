# Short-Term Reversal: CLOSED — do not revisit (June 3 2026)

## Verdict: tested 3 ways, all KILLED under full-history regime validation. Not tradeable.

Reversal keeps looking attractive (strong academic literature: Jegadeesh 1990,
Lehmann 1990, ~2%/month historically; negatively correlated with momentum = a
natural hedge). It nearly got rebuilt June 3 2026. DO NOT. It was already tested
and rejected. Evidence:

## What was tested (commits d3a14fc, 56bb2c2)

1. **cs-demean reversion** (rank by -trailing-5d-relative-return, long oversold):
   2-month backtest showed +0.55 net Sharpe, BUT full-history non-overlap net-of-
   cost = NEGATIVE (-0.43 at 5d, -0.35 at 10d). The +0.55 was a single-regime artifact.
1. **PCA residual reversal** (Avellaneda-Lee — residualize vs top-K principal
   components, the sophisticated version): full-history best config net +0.46, BUT
   per-regime: +2.27 (2020), +0.64 (2021), -0.49 (2022), -0.15 (2023-24),
   **-0.92 (2025-26, current regime)**. The positive full-history number is a
   2020-2021 COVID-era artifact. Param sweep spiky, no stable basin = not robust.
1. **Horizon sweep:** weekly reversion DEAD; edge only at h=5, and even that fails
   net-of-cost in current regime.

## Why it fails for US (not a bug — a real finding)

- Academic reversal works on AVERAGE over DECADES on BROAD universes.
- Our universe is 149 concentrated names; McLean-Pontiff decay (~26% OOS) + small
  universe + current regime (2022-2026) = reversal is net-NEGATIVE now.
- Full-history + per-regime + non-overlap validation (Rule #1) caught the false
  positive that the 2-month backtest showed.

## The -0.22 momentum/direction correlation does NOT change this

Momentum and direction are negatively correlated (~-0.22, 3-day prelim) because
direction partly captures the reversal effect. But that reversal effect is itself
net-negative after costs. So the negative correlation is a STRUCTURAL fact, not a
tradeable opportunity. Do not extrapolate “they’re decorrelated” into “build reversal.”

## The conclusion (from commit 56bb2c2, verbatim intent)

“FOUR signals now killed under full-history+regime+non-overlap validation: per-ticker
dir, global dir, cs-demean reversion, PCA residual reversion. Short-horizon 1-5d
reversion at 149-name scale NOT tradeable in current regime. Next axes: longer
horizon / options-VRP / PEAD / combination – NOT more reversion.”

## Standing instruction

Reversal is CLOSED. If it comes up again: point here. The next return-alpha axes are
options/VRP (~Aug, data accruing), PEAD, or longer-horizon — a DIFFERENT information
axis, not another price-history reversion variant.