#!/bin/bash
# Gemma v4 Role Instruction experiments - GPU 0
# Investment Choice (5 constraints) + Mystery Wheel
cd /home/jovyan/llm-addiction
export CUDA_VISIBLE_DEVICES=0
OUT=/home/jovyan/beomi/llm-addiction-data
LOG=/home/jovyan/beomi/llm-addiction-data/logs
SRC=exploratory_experiments/alternative_paradigms/src

for C in 10 30 50 70; do
  echo "=== IC constraint=$C === $(date)"
  python $SRC/investment_choice/run_experiment.py \
    --model gemma --gpu 0 --constraint $C \
    --output-dir $OUT/investment_choice_v2_role/ \
    2>&1 | tee $LOG/ic_v2_role_c${C}.log
done

echo "=== IC unlimited === $(date)"
python $SRC/investment_choice/run_experiment.py \
  --model gemma --gpu 0 --constraint unlimited --bet-type variable \
  --output-dir $OUT/investment_choice_v2_role/ \
  2>&1 | tee $LOG/ic_v2_role_unlimited.log

echo "=== Mystery Wheel c30 === $(date)"
python $SRC/mystery_wheel/run_experiment.py \
  --model gemma --gpu 0 --constraint 30 \
  --output-dir $OUT/mystery_wheel_v2_role/ \
  2>&1 | tee $LOG/mw_v2_role_c30.log

echo "=== GPU 0 ALL DONE === $(date)"
