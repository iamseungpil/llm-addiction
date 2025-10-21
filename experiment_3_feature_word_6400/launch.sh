#!/bin/bash
# Launch Exp3: Feature-Word Analysis

cd /home/ubuntu/llm_addiction/experiment_3_feature_word_6400

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env

echo "🚀 Launching Exp3: Feature-Word Analysis"
echo "  GPU 3: 441 features × 6,400 responses"

mkdir -p logs

tmux new-session -d -s exp3_word "CUDA_VISIBLE_DEVICES=3 python experiment_3_feature_word.py --gpu 3 2>&1 | tee logs/exp3_word.log"
echo "✅ Started GPU 3: Exp3 Feature-Word"

echo ""
echo "Monitor with: tmux attach -t exp3_word"
