#!/bin/bash
# Launch Exp1: Layer Pathway Tracking

cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env

echo "🚀 Launching Exp1: Layer Pathway Tracking (L1-31)"
echo "  GPU 3: 50 games, tracking all 31 layers per decision"

mkdir -p logs

tmux new-session -d -s exp1_pathway "CUDA_VISIBLE_DEVICES=3 python experiment_1_pathway.py --gpu 3 2>&1 | tee logs/exp1_pathway.log"
echo "✅ Started GPU 3: Exp1 Pathway"

echo ""
echo "Monitor with: tmux attach -t exp1_pathway"
