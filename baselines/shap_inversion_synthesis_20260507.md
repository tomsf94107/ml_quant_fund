# SHAP Inversion Synthesis — May 7, 2026

## Setup
- 100 samples per horizon (50 high-conf wins + 50 high-conf losses)
- prob_up >= 0.70 threshold
- 76-feature model post-cull
- KernelExplainer with 50-sample background

## Key findings

### h=5 misleading features (smoothed price levels)
| Feature | mag_diff | Mechanism |
|---|---|---|
| ma_50 | +0.0070 | Long-term trend at extreme = mean reversion |
| high_52w_ratio | +0.0057 | Near 52w high = profit-taking territory |
| vwap | +0.0051 | Price far from VWAP = exhaustion |
| bb_width | +0.0039 | Wide BB = volatility regime change |
| bb_lower | +0.0031 | BB position at extreme |

### h=3 misleading features (short-term momentum)
| Feature | mag_diff | Mechanism |
|---|---|---|
| return_5d | +0.0045 | Strong 5d rally = short-term reversal |
| obv_trend | +0.0032 | Volume-confirmed trend at peak |
| intraday_momentum | +0.0025 | Today's strength = tomorrow's reversal |
| return_3d | +0.0023 | 3d momentum at extreme |

### Universal pattern
Both horizons: **trend-following features at extremes lead to confident wrong predictions.**

The model learned "trend continues" but didn't learn "trend exhausts at extremes."

## Counter-signals model ignores in losses

### h=3 counter-signals (sign disagreement >0.005, model ignores them on losses)
- vwap_dev_eod: 0.0106 (wins -0.0098, losses +0.0007)
- low_52w_ratio: 0.0087 (wins +0.0100, losses +0.0013)

### h=5 counter-signals
- ma_50: 0.0065 (wins -0.0028, losses +0.0037) — sign FLIP
- bb_width: 0.0065 (wins +0.0021, losses -0.0044) — sign FLIP

## Remediation strategy

**Phase 1 (Sprint 1 Day 1 — immediate):** Confidence cap at 0.65 for h=3, h=5
- Refuses to claim >0.65 confidence in measurably-inverted regions
- 1-line patch in signals/generator.py
- Preserves all features
- Lossless to non-extreme predictions

**Phase 2 (Sprint 1 Week 2):** Vol-adjusted target labels
- Train on (return > 0.5σ) instead of (return > 0)
- Reduces noise around zero, sharpens model's true signal

**Phase 3 (Sprint 2):** Probability-conditional features
- Use trend features at low/mid confidence ranges
- Weight counter-signals (vwap_dev_eod, low_52w_ratio) more at high confidence
- Or train regime-specific models for high-confidence regime

**NOT recommended:** Drop misleading features.
- They carry valid signal at non-extreme confidence
- Smoke test smoke test confirmed: ma_50 wins SHAP correctly negative
- Dropping would lose more than it would fix
