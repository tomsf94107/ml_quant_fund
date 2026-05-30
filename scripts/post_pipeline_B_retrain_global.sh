#!/bin/bash
# Run AFTER Pipeline B completes to:
#  1. Retrain GLOBAL models with 97 features (matches per-ticker)
#  2. Validate via verify SQL
#  3. Compare signal distribution today vs yesterday

set -e
cd ~/Desktop/ML_Quant_Fund

LOG_DIR="logs/global_retrain_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo "=== GLOBAL retrain — $(date) ==="

for H in 1 3 5; do
    echo ""
    echo "--- Training GLOBAL h=${H} ---"
    ML_QUANT_INST_FEATURES=1 python -m models.train_cross_sectional --horizons $H 2>&1 | tee "$LOG_DIR/global_h${H}.log"
done

echo ""
echo "=== Verify retrained GLOBAL models ==="
PYTHONPATH=. ML_QUANT_INST_FEATURES=1 python3 -c "
from models.ensemble import EnsembleResult
for h in [1, 3, 5]:
    m = EnsembleResult.load('GLOBAL', h)
    has60 = 'insider_60d' in m.feature_cols
    has90 = 'insider_90d' in m.feature_cols
    print(f'GLOBAL h={h}: {len(m.feature_cols)} features  insider_60d={has60}  insider_90d={has90}')
"

echo ""
echo "=== Verification SQL ==="
sqlite3 accuracy.db < scripts/verify_post_retrain_nvda.sql

echo ""
echo "=== Compare today's BUYs vs yesterday's ==="
sqlite3 accuracy.db "
SELECT prediction_date, signal, COUNT(*) AS n, 
       printf('%.3f', AVG(prob_up)) AS avg_eff,
       printf('%.3f', AVG(prob_raw)) AS avg_raw
FROM predictions
WHERE prediction_date >= date('now', '-2 days') AND horizon = 5
GROUP BY prediction_date, signal
ORDER BY prediction_date DESC, signal;
"

echo ""
echo "=== ALL TASKS DONE ==="
