#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-addiction
cd /home/v-seungplee/llm-addiction
export PYTHONUNBUFFERED=1
python experiments/07_sae_readout/src/run_v12_all_steering.py --model gemma --task all --n 100 2>&1 | tee experiments/07_sae_readout/results/gemma_steering_log.txt
