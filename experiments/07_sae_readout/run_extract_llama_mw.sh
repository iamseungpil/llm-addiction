#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-addiction
cd /home/v-seungplee/llm-addiction
export PYTHONUNBUFFERED=1
python experiments/07_sae_readout/src/extract_all_hidden_states.py --model llama --device cuda:0 2>&1 | tee experiments/07_sae_readout/results/extract_llama_mw_log.txt
