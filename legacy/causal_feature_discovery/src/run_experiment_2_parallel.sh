#!/bin/bash

# Multi-Layer Population Mean Patching Experiment - GPU Parallelization
# Designed for 1,340 high-effect features across GPUs 4 and 5

set -e

echo "=================================================="
echo "EXPERIMENT 2: MULTI-LAYER POPULATION MEAN PATCHING"
echo "=================================================="

# Activate environment
conda activate llama_sae_env

# Create tmux sessions for parallel execution
echo "Setting up GPU parallelization..."

# GPU 4: Features 0-669 (670 features)
echo "Starting GPU 4 process..."
tmux new-session -d -s exp2_gpu4 "cd /home/ubuntu/llm_addiction/causal_feature_discovery/src && python experiment_2_multilayer_population_mean.py --gpu 4 --start_idx 0 --end_idx 670 --process_id gpu4 2>&1 | tee exp2_gpu4_$(date +%Y%m%d_%H%M%S).log"

# GPU 5: Features 670-1339 (670 features)  
echo "Starting GPU 5 process..."
tmux new-session -d -s exp2_gpu5 "cd /home/ubuntu/llm_addiction/causal_feature_discovery/src && python experiment_2_multilayer_population_mean.py --gpu 5 --start_idx 670 --end_idx 1340 --process_id gpu5 2>&1 | tee exp2_gpu5_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Experiment 2 started on both GPUs!"
echo ""
echo "Monitor progress with:"
echo "  GPU 4: tmux attach -t exp2_gpu4"
echo "  GPU 5: tmux attach -t exp2_gpu5"
echo ""
echo "Check logs:"
echo "  GPU 4: tail -f exp2_gpu4_*.log"
echo "  GPU 5: tail -f exp2_gpu5_*.log"
echo ""

# Estimated runtime calculation
total_features=1340
features_per_gpu=670
conditions=3
prompts=2
trials_per_condition=50
time_per_trial=0.7  # seconds

total_trials_per_gpu=$((features_per_gpu * conditions * prompts * trials_per_condition))
estimated_hours=$(echo "scale=1; $total_trials_per_gpu * $time_per_trial / 3600" | bc)

echo "Runtime Estimates:"
echo "  Features per GPU: $features_per_gpu"
echo "  Total trials per GPU: $total_trials_per_gpu"
echo "  Estimated time per GPU: $estimated_hours hours"
echo "  Expected completion: ~$(date -d "+$estimated_hours hours" +"%Y-%m-%d %H:%M")"
echo ""

# Display monitoring commands
echo "Monitoring Commands:"
echo "  ps aux | grep experiment_2_multilayer"
echo "  nvidia-smi"
echo "  ls -la /data/llm_addiction/results/exp2_multilayer_*"