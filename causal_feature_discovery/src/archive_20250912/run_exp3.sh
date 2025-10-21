#!/bin/bash
cd /home/ubuntu/llm_addiction/causal_feature_discovery/src
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env
python experiment_3_reward_choice.py 2>&1 | tee ../logs/exp3_all_features_$(date +%Y%m%d_%H%M%S).log