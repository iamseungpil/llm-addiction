#!/bin/bash
# Gemma v4 Role Instruction experiments - GPU 1
# Slot Machine + Coin Flip
cd /home/jovyan/llm-addiction
export CUDA_VISIBLE_DEVICES=1
OUT=/home/jovyan/beomi/llm-addiction-data
LOG=/home/jovyan/beomi/llm-addiction-data/logs
SRC=exploratory_experiments/alternative_paradigms/src

echo "=== Slot Machine === $(date)"
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py \
  --model gemma --gpu 0 \
  2>&1 | tee $LOG/sm_v4_role.log

echo "=== Coin Flip c10 === $(date)"
python $SRC/coin_flip/run_experiment.py \
  --model gemma --gpu 0 --constraint 10 \
  --output-dir $OUT/coin_flip_v2_role/ \
  2>&1 | tee $LOG/cf_v2_role_c10.log

echo "=== GPU 1 ALL DONE === $(date)"
