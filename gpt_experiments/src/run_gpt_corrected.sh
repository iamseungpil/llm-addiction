#!/bin/bash

# GPT-4o-mini Corrected Experiment Runner
# Fixed parsing issues + English prompts + Separate log directory

set -e

echo "============================================================"
echo "GPT-4O-MINI CORRECTED MULTIROUND EXPERIMENT"
echo "Fixed parsing + English prompts + /data storage"
echo "============================================================"

# Check if tmux session exists
SESSION_NAME="gpt_corrected_experiment"
CONDA_ENV="llm_env3"
SCRIPT_PATH="/home/ubuntu/llm_addiction/gpt_experiments/src/gpt_corrected_multiround_experiment.py"

echo "Session: $SESSION_NAME"
echo "Environment: $CONDA_ENV" 
echo "Script: $SCRIPT_PATH"
echo "Results: /data/llm_addiction/gpt_results_corrected/"
echo ""

# Kill existing session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

# Create results directory
mkdir -p /data/llm_addiction/gpt_results_corrected/logs

# Create new tmux session
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Activate conda environment and run experiment
tmux send-keys -t $SESSION_NAME "cd /home/ubuntu/llm_addiction/gpt_experiments/src" Enter
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" Enter

# Install required packages if not available
tmux send-keys -t $SESSION_NAME "pip install openai tqdm" Enter

# Run the corrected experiment
tmux send-keys -t $SESSION_NAME "python $SCRIPT_PATH" Enter

echo "✅ GPT Corrected experiment started!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Check results in:"
echo "  /data/llm_addiction/gpt_results_corrected/"
echo ""
echo "Key improvements:"
echo "  - English prompts (matching LLaMA)"
echo "  - Last $amount parsing (fixed)"
echo "  - Separate log directory"
echo "  - Detailed parsing logs"
echo ""
echo "Expected runtime: ~4-5 hours (1,280 experiments)"
echo "============================================================"