"""
Recession-prediction model — standalone research module.

This package is fully isolated from the ML Quant Fund production pipeline:
- Reads from FRED/ALFRED/external sources only
- Writes to recession.db only (NEVER accuracy.db)
- Has its own cron schedule (monthly, day 7 Vietnam time)

See Recession_Model_Spec_v1.md for the full spec.
"""
__version__ = "0.1.0"
__spec_version__ = "v1.0"
