#!/bin/bash
# Runs all 4 next-step scripts in order. Run from your repo root:
#   cd ~/Desktop/ML_Quant_Fund && bash run_all.sh

echo "########## 1/4: INDEPENDENT TWO-BRICK BOOK ##########"
python two_brick_book.py --root .

echo ""
echo "########## 2/4: COST & HORIZON-ROBUST COMBINATION ##########"
python combine_robust.py --root .

echo ""
echo "########## 3/4: SHORT-INTEREST REFRESH (status) ##########"
python si_refresh.py --status

echo ""
echo "########## 4/4: BRICK #3 OPTIONS VALIDATOR (status) ##########"
python validate_brick3.py --root .

echo ""
echo "########## ALL 4 DONE ##########"
