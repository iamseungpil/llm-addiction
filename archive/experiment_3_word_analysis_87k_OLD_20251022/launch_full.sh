#!/bin/bash

# Full analysis script - analyze ALL L1-31 layers
# This will analyze all 87,012 features

# Split into multiple GPU jobs for faster processing:
# GPU 0: Layers 1-8   (~15,000 features)
# GPU 1: Layers 9-16  (~20,000 features)
# GPU 2: Layers 17-24 (~25,000 features)
# GPU 3: Layers 25-31 (~27,000 features)

cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis

# Create logs directory
mkdir -p logs

echo "Starting L1-31 Feature-Word Analysis across 4 GPUs..."

# Launch each GPU job in tmux
tmux new-session -d -s exp3_L1_8 "cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis && python3 experiment_3_L1_31_extended.py --gpu 0 --layer-start 1 --layer-end 8 2>&1 | tee logs/L1_8_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s exp3_L9_16 "cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis && python3 experiment_3_L1_31_extended.py --gpu 1 --layer-start 9 --layer-end 16 2>&1 | tee logs/L9_16_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s exp3_L17_24 "cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis && python3 experiment_3_L1_31_extended.py --gpu 2 --layer-start 17 --layer-end 24 2>&1 | tee logs/L17_24_\$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s exp3_L25_31 "cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis && python3 experiment_3_L1_31_extended.py --gpu 3 --layer-start 25 --layer-end 31 2>&1 | tee logs/L25_31_\$(date +%Y%m%d_%H%M%S).log"

echo "All jobs launched!"
echo "Monitor with:"
echo "  tmux attach -t exp3_L1_8"
echo "  tmux attach -t exp3_L9_16"
echo "  tmux attach -t exp3_L17_24"
echo "  tmux attach -t exp3_L25_31"
