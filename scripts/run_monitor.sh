#!/bin/zsh
python3 ~/ML_Quant_Fund/scripts/monitor_earnings.py --tickers "$1"
python3 ~/ML_Quant_Fund/scripts/fundamentals_block.py "$1"
