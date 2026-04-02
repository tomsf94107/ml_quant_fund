#!/bin/bash
PYTHON=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
ROOT=/Users/atomnguyen/Desktop/ML_Quant_Fund
LOG=$ROOT/logs
cd $ROOT

echo "=== PIPELINE START $(date) ===" >> $LOG/pipeline.log

echo "--- Step 1: Feature Validator ---" >> $LOG/pipeline.log
$PYTHON scripts/feature_validator.py --fix >> $LOG/feature_validator.log 2>&1
echo "--- Step 1 DONE $(date) ---" >> $LOG/pipeline.log

echo "--- Step 2: Retrain ---" >> $LOG/pipeline.log
$PYTHON -m models.train_all >> $LOG/retrain.log 2>&1
echo "--- Step 2 DONE $(date) ---" >> $LOG/pipeline.log

echo "--- Step 3: Daily Runner ---" >> $LOG/pipeline.log
$PYTHON -c "import sys; sys.path.insert(0,'.'); from scripts.daily_runner import run_daily; run_daily()" >> $LOG/daily_runner.log 2>&1
echo "--- Step 3 DONE $(date) ---" >> $LOG/pipeline.log

echo "--- Step 4: Daily Validator ---" >> $LOG/pipeline.log
$PYTHON scripts/daily_validator.py --days 30 --fix >> $LOG/validator.log 2>&1
echo "--- Step 4 DONE $(date) ---" >> $LOG/pipeline.log

echo "=== PIPELINE DONE $(date) ===" >> $LOG/pipeline.log
