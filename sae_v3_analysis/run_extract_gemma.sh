#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-addiction
cd /home/v-seungplee/llm-addiction
export PYTHONUNBUFFERED=1
python sae_v3_analysis/src/extract_all_hidden_states.py --model gemma --device cuda:0 2>&1 | tee sae_v3_analysis/results/extract_gemma_log.txt
