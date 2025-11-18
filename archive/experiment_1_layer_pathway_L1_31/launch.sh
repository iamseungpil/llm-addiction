#!/bin/bash
#
# Launch Gradient Pathway Tracking
# Feature-centric backward Jacobian (Anthropic 2025 method)
#

GPU=${1:-4}

echo "================================"
echo "Gradient Pathway Tracking"
echo "GPU: $GPU"
echo "Features: 2,787 (640 safe + 2,147 risky)"
echo "Expected time: ~1.5 hours"
echo "================================"

cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31

# Create tmux session
tmux new-session -d -s pathway_gradient_gpu${GPU}

# Run tracker
tmux send-keys -t pathway_gradient_gpu${GPU} "conda activate llama_sae_env" C-m
tmux send-keys -t pathway_gradient_gpu${GPU} "cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/src" C-m
tmux send-keys -t pathway_gradient_gpu${GPU} "python gradient_pathway_tracker.py --gpu ${GPU} 2>&1 | tee ../logs/pathway_\$(date +%Y%m%d_%H%M%S).log" C-m

echo ""
echo "✅ Launched in tmux session: pathway_gradient_gpu${GPU}"
echo ""
echo "Monitor with:"
echo "  tmux attach -t pathway_gradient_gpu${GPU}"
echo ""
echo "Expected completion: $(date -d '+90 minutes' '+%Y-%m-%d %H:%M')"
echo ""
