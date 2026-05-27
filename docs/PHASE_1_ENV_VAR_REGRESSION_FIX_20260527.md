# Phase 1 Regression Fix — May 27 2026 AM VN

## Root cause
Yesterday's commit 2b264e0 made `ML_QUANT_DISABLE_MULTIPLIERS=1` env var-gated.
The env var was set in INTERACTIVE shell (where we tested), but NOT in `.env`
file. Cron-triggered Pipeline B sources `.env` and runs unattended — it had
no flag set, so multipliers re-activated.

## Verification of regression
Overnight Pipeline B (finished 07:57 VN May 27) wrote 16 h=5 BUYs to
prediction_date=2026-05-26 with prob_eff > prob_raw:

| Ticker | prob_eff | prob_raw | Gap |
|---|---|---|---|
| BYND | 0.798 | 0.659 | +14pp |
| MNDY | 0.725 | 0.670 | +6pp  |
| ANET | 0.746 | 0.690 | +6pp  |
| ETN  | 0.670 | 0.651 | +2pp  |

## Fix applied
Added `ML_QUANT_DISABLE_MULTIPLIERS=1` to `.env`.

## Post-fix verification
Manual generate_signals test (Wed May 27, 13:08 VN):
- BYND: eff=0.6719 raw=0.6719 diff=0.0000 ✓
- ANET: eff=0.6506 raw=0.6506 diff=0.0000 ✓
- MNDY: eff=0.6705 raw=0.6705 diff=0.0000 ✓

## Action items (so this never happens again)
1. ALL env-var-gated features MUST be added to .env immediately on ship
2. Pipeline scripts should EXPLICITLY log all env vars at startup
3. Validator (Stage 4) should check for prob_eff vs prob_raw drift in today's batch
   and ALERT if mean diff > 0.001

## Operational impact on May 26 production data
DB has 16 BUYs from overnight with inflated prob_eff. These are stale (Pipeline C
at 19:00 VN today will overwrite with correct values). Honest record-keeping:
The 14pp BYND inflation was a REGRESSION, not the original 0.824 we fixed yesterday.
