# Baselines

Historical snapshots of model calibration and performance metrics.
Used to compare before/after for model changes (feature culls, retrains, walk_forward results).

Captured by `analysis/save_calibration_baseline.py` (Step 2 of May 7 prep work).

## Files

Each baseline pair is timestamped `calibration_baseline_<YYYYMMDD_HHMM>.{json,md}`:
- `.json`: machine-readable for diff tools
- `.md`: human-readable summary table

## When to capture

- Before significant model changes (retrain, feature cull, hyperparameter swap)
- After walk_forward auto-runs (Sunday cron output)
- Monthly review checkpoint
