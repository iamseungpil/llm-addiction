#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-addiction
cd /home/v-seungplee/llm-addiction
export PYTHONUNBUFFERED=1
python sae_v3_analysis/src/run_v12_all_steering.py --model llama --task all --n 100 2>&1 | tee sae_v3_analysis/results/llama_ic_mw_log.txt
