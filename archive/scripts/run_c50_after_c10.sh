#!/bin/bash
# Wait for c10 process to finish, then run c50 on GPU 0
C10_PID=$1
OUTPUT_DIR=/home/jovyan/beomi/llm-addiction-data/investment_choice/gemma_parser_fixed_v2
LOG_DIR=/home/jovyan/beomi/llm-addiction-data/logs

echo "=== Waiting for c10 (PID $C10_PID) to finish ==="
while kill -0 $C10_PID 2>/dev/null; do
    sleep 30
done
echo "=== c10 finished at $(date), starting c50 ==="

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src
python investment_choice/run_experiment.py \
  --model gemma --gpu 0 --constraint 50 \
  --output-dir $OUTPUT_DIR

echo "=== c50 finished at $(date) ==="
