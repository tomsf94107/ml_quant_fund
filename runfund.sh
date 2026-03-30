#!/bin/bash
cd ~/Desktop/ML_Quant_Fund
/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python -c "
import sys; sys.path.insert(0, '.')
from scripts.daily_runner import run_daily
run_daily(force=True)
"
