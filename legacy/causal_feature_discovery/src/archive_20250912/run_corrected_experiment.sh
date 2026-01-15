#!/bin/bash

# Run corrected population mean patching experiment on both GPUs

echo "🚀 Starting CORRECTED Population Mean Patching Experiment"
echo "Date: $(date)"
echo "="*80

# Create tmux sessions for parallel execution
echo "Creating tmux sessions..."

# GPU 4: First half of validated features
tmux new-session -d -s corrected_gpu4 "cd /home/ubuntu/llm_addiction/causal_feature_discovery/src && \
conda activate llama_sae_env && \
python experiment_corrected_population_patching.py --gpu 4 --start_idx 0 --end_idx 50 2>&1 | \
tee corrected_exp_gpu4_$(date +%Y%m%d_%H%M%S).log"

# GPU 5: Second half of validated features  
tmux new-session -d -s corrected_gpu5 "cd /home/ubuntu/llm_addiction/causal_feature_discovery/src && \
conda activate llama_sae_env && \
python experiment_corrected_population_patching.py --gpu 5 --start_idx 50 --end_idx 100 2>&1 | \
tee corrected_exp_gpu5_$(date +%Y%m%d_%H%M%S).log"

echo "✅ Experiments started!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t corrected_gpu4"
echo "  tmux attach -t corrected_gpu5"
echo ""
echo "Check progress with:"
echo "  tmux capture-pane -t corrected_gpu4 -p | tail -10"
echo "  tmux capture-pane -t corrected_gpu5 -p | tail -10"
echo ""
echo "Expected completion time: 4-6 hours per GPU"