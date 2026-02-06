#!/bin/bash
# Launch corrected experiment_2_L1_31 on GPU 4,5,6,7 with 1 process per GPU
# FIXED: Environment variable INSIDE tmux command quotes

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31
PYTHON=/data/miniforge3/envs/llama_sae_env/bin/python

# GPU 4: L1-8 (8 layers total)
tmux new-session -d -s exp2_gpu4 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 8 --process_id gpu4_L1_8 --trials 30 2>&1 | tee logs/exp2_gpu4_L1_8.log"

# GPU 5: L9-16 (8 layers total)
tmux new-session -d -s exp2_gpu5 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 9 --layer_end 16 --process_id gpu5_L9_16 --trials 30 2>&1 | tee logs/exp2_gpu5_L9_16.log"

# GPU 6: L17-24 (8 layers total)
tmux new-session -d -s exp2_gpu6 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 24 --process_id gpu6_L17_24 --trials 30 2>&1 | tee logs/exp2_gpu6_L17_24.log"

# GPU 7: L25-31 (7 layers total)
tmux new-session -d -s exp2_gpu7 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 31 --process_id gpu7_L25_31 --trials 30 2>&1 | tee logs/exp2_gpu7_L25_31.log"

echo "✅ Launched 4 processes (1 per GPU) across GPU 4,5,6,7"
echo ""
echo "Monitor sessions:"
echo "  tmux attach -t exp2_gpu4  # GPU 4 L1-8"
echo "  tmux attach -t exp2_gpu5  # GPU 5 L9-16"
echo "  tmux attach -t exp2_gpu6  # GPU 6 L17-24"
echo "  tmux attach -t exp2_gpu7  # GPU 7 L25-31"
echo ""
echo "Check all sessions: tmux ls | grep exp2"
