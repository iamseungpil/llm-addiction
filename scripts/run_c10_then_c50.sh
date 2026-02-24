#!/bin/bash
# Run c10 then c50 sequentially on GPU 0
set -e

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src
OUTPUT_DIR=/home/jovyan/beomi/llm-addiction-data/investment_choice/gemma_parser_fixed_v2
LOG_DIR=/home/jovyan/beomi/llm-addiction-data/logs

echo "=== Waiting for c10 (PID $1) to finish ==="
wait $1 2>/dev/null || true
echo "=== c10 finished, starting c50 ==="

python investment_choice/run_experiment.py \
  --model gemma --gpu 0 --constraint 50 \
  --output-dir $OUTPUT_DIR \
  2>&1 | tee ${LOG_DIR}/gemma_c50_fixed_v2_$(date +%Y%m%d_%H%M%S).log

echo "=== c50 finished ==="
