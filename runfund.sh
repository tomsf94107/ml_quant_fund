#!/bin/bash
# Manual full daily run — uses batched orchestrator (May 6 2026)
# Spawns 3 fresh subprocesses (~42 tickers each), peaks ~1-2GB per batch.
# Fixes ticker-99 death pattern from single-process daily_runner.
cd ~/Desktop/ML_Quant_Fund
/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python -m scripts.daily_runner_batched
