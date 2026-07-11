#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
# Sequential pipeline chain: A (ingest) -> D (alpha panel) -> B (train+predict).
# Replaces fixed-time entries (03:00 A / 04:00 D / 07:00 B) that raced when A
# overran (Jun 12: first 394-name ingest took 4.6h, B fired into missing marker).
# B's own marker guard remains as failure protection; this handles timing.
LOG=/Users/atomnguyen/Desktop/ML_Quant_Fund/logs/pipeline_chain.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === CHAIN START ===" >> "$LOG"
/Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/pipeline_A_ingest.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] A exited $? -> starting D" >> "$LOG"
/Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/pipeline_D_alpha_panel.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] D exited $? -> starting B" >> "$LOG"
/Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/pipeline_B_train_predict.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] B exited $? === CHAIN END ===" >> "$LOG"
