#!/bin/bash
# V14 Causal Validation - wrapper script
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd /home/v-seungplee/llm-addiction/experiments/07_sae_readout/src

echo "=== V14 Causal Validation ==="
echo "Started: $(date)"

conda run -n llm-addiction python run_v14_experiments.py 2>&1 | tee /home/v-seungplee/llm-addiction/experiments/07_sae_readout/results/v14_log.txt

echo "Finished: $(date)"
