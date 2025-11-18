#!/bin/bash
#
# Launch Activation Extraction for Experiment 3 Word Analysis
# Distributes layers across GPUs 5, 6, 7 for parallel processing
#

echo "================================"
echo "Activation Extraction Launch"
echo "Total: 2,787 features across 30 layers"
echo "Strategy: Layer-by-layer parallel processing"
echo "GPUs: 5, 6, 7"
echo "================================"

cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching

# Create logs directory
mkdir -p logs

# Layer distribution for balanced load
# ONLY layers with Exp2 data: 1, 4, 8, 9, 12, 14, 15, 17, 20, 24, 25, 26, 30
# Missing from Exp2: 2, 3, 5, 6, 7, 10, 11, 13, 16, 18, 19, 21, 22, 23, 27, 28, 29

GPU5_LAYERS=(1 4 12 20 25)      # 5 layers
GPU6_LAYERS=(9 14 26)            # 3 layers
GPU7_LAYERS=(8 15 17 24 30)     # 5 layers

# Launch GPU 5
echo ""
echo "Launching GPU 5..."
for layer in ${GPU5_LAYERS[@]}; do
    tmux new-session -d -s extract_L${layer}_gpu5
    tmux send-keys -t extract_L${layer}_gpu5 "conda activate llama_sae_env" C-m
    tmux send-keys -t extract_L${layer}_gpu5 "cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching/src" C-m
    tmux send-keys -t extract_L${layer}_gpu5 "python step1_extract_activations.py --gpu 5 --layer ${layer} 2>&1 | tee ../logs/extract_L${layer}_\$(date +%Y%m%d_%H%M%S).log" C-m
    echo "  Layer ${layer} -> tmux session: extract_L${layer}_gpu5"
    sleep 2  # Stagger launches
done

# Launch GPU 6
echo ""
echo "Launching GPU 6..."
for layer in ${GPU6_LAYERS[@]}; do
    tmux new-session -d -s extract_L${layer}_gpu6
    tmux send-keys -t extract_L${layer}_gpu6 "conda activate llama_sae_env" C-m
    tmux send-keys -t extract_L${layer}_gpu6 "cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching/src" C-m
    tmux send-keys -t extract_L${layer}_gpu6 "python step1_extract_activations.py --gpu 6 --layer ${layer} 2>&1 | tee ../logs/extract_L${layer}_\$(date +%Y%m%d_%H%M%S).log" C-m
    echo "  Layer ${layer} -> tmux session: extract_L${layer}_gpu6"
    sleep 2
done

# Launch GPU 7
echo ""
echo "Launching GPU 7..."
for layer in ${GPU7_LAYERS[@]}; do
    tmux new-session -d -s extract_L${layer}_gpu7
    tmux send-keys -t extract_L${layer}_gpu7 "conda activate llama_sae_env" C-m
    tmux send-keys -t extract_L${layer}_gpu7 "cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching/src" C-m
    tmux send-keys -t extract_L${layer}_gpu7 "python step1_extract_activations.py --gpu 7 --layer ${layer} 2>&1 | tee ../logs/extract_L${layer}_\$(date +%Y%m%d_%H%M%S).log" C-m
    echo "  Layer ${layer} -> tmux session: extract_L${layer}_gpu7"
    sleep 2
done

echo ""
echo "================================"
echo "✅ All 13 available layers launched!"
echo ""
echo "Note: Only 13/30 layers have Exp2 data"
echo "  Available: 1, 4, 8, 9, 12, 14, 15, 17, 20, 24, 25, 26, 30"
echo "  Missing: 2, 3, 5, 6, 7, 10, 11, 13, 16, 18, 19, 21, 22, 23, 27, 28, 29"
echo "  Coverage: 1,431/2,787 features (51.3%)"
echo ""
echo "Monitor progress:"
echo "  tmux list-sessions | grep extract"
echo "  tmux attach -t extract_L9_gpu6  # Example: Layer 9 (largest)"
echo ""
echo "Check output:"
echo "  ls -la /data/llm_addiction/experiment_3_activation_cache/"
echo ""
echo "Expected completion: 3-6 hours (layer-dependent)"
echo "================================"
