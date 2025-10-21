#!/bin/bash
# Restart Exp5: Multi-Round Patching (continues from checkpoint)

cd /home/ubuntu/llm_addiction/experiment_5_multiround_patching

echo "🚀 Restarting Exp5: Multi-Round Patching"
echo "  GPU 3: 441 causal features"
echo "  Progress: 100/441 features completed (22.7%)"
echo "  Will resume from checkpoint"

# Check if already running
if tmux has-session -t exp5_multiround 2>/dev/null; then
    echo "⚠️  Exp5 tmux session already exists"
    echo "   Attach with: tmux attach -t exp5_multiround"
else
    tmux new-session -d -s exp5_multiround "
      source ~/.bashrc
      conda activate llama_sae_env
      export CUDA_VISIBLE_DEVICES=4
      python multiround_patching.py 2>&1 | tee logs/multiround_patching_441.log
    "
    echo "✅ Started GPU 4: Exp5 Multi-Round Patching"
    echo "   Monitor with: tmux attach -t exp5_multiround"
fi
