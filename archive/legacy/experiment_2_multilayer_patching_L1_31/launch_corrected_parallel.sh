#!/bin/bash
# Launch corrected experiment_2_L1_31 on GPU 4,5,6,7 with 4 processes each

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31
PYTHON=/data/miniforge3/envs/llama_sae_env/bin/python

# GPU 4: L1-8 (4 processes)
CUDA_VISIBLE_DEVICES=4 tmux new-session -d -s exp2_gpu4_p1 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 2 --process_id gpu4_L1_2 --trials 30 2>&1 | tee logs/exp2_gpu4_L1_2.log"
CUDA_VISIBLE_DEVICES=4 tmux new-session -d -s exp2_gpu4_p2 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 3 --layer_end 4 --process_id gpu4_L3_4 --trials 30 2>&1 | tee logs/exp2_gpu4_L3_4.log"
CUDA_VISIBLE_DEVICES=4 tmux new-session -d -s exp2_gpu4_p3 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 5 --layer_end 6 --process_id gpu4_L5_6 --trials 30 2>&1 | tee logs/exp2_gpu4_L5_6.log"
CUDA_VISIBLE_DEVICES=4 tmux new-session -d -s exp2_gpu4_p4 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 7 --layer_end 8 --process_id gpu4_L7_8 --trials 30 2>&1 | tee logs/exp2_gpu4_L7_8.log"

# GPU 5: L9-16 (4 processes)
CUDA_VISIBLE_DEVICES=5 tmux new-session -d -s exp2_gpu5_p1 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 9 --layer_end 10 --process_id gpu5_L9_10 --trials 30 2>&1 | tee logs/exp2_gpu5_L9_10.log"
CUDA_VISIBLE_DEVICES=5 tmux new-session -d -s exp2_gpu5_p2 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 12 --process_id gpu5_L11_12 --trials 30 2>&1 | tee logs/exp2_gpu5_L11_12.log"
CUDA_VISIBLE_DEVICES=5 tmux new-session -d -s exp2_gpu5_p3 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 13 --layer_end 14 --process_id gpu5_L13_14 --trials 30 2>&1 | tee logs/exp2_gpu5_L13_14.log"
CUDA_VISIBLE_DEVICES=5 tmux new-session -d -s exp2_gpu5_p4 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 15 --layer_end 16 --process_id gpu5_L15_16 --trials 30 2>&1 | tee logs/exp2_gpu5_L15_16.log"

# GPU 6: L17-24 (4 processes)
CUDA_VISIBLE_DEVICES=6 tmux new-session -d -s exp2_gpu6_p1 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 18 --process_id gpu6_L17_18 --trials 30 2>&1 | tee logs/exp2_gpu6_L17_18.log"
CUDA_VISIBLE_DEVICES=6 tmux new-session -d -s exp2_gpu6_p2 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 20 --process_id gpu6_L19_20 --trials 30 2>&1 | tee logs/exp2_gpu6_L19_20.log"
CUDA_VISIBLE_DEVICES=6 tmux new-session -d -s exp2_gpu6_p3 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 21 --layer_end 22 --process_id gpu6_L21_22 --trials 30 2>&1 | tee logs/exp2_gpu6_L21_22.log"
CUDA_VISIBLE_DEVICES=6 tmux new-session -d -s exp2_gpu6_p4 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 23 --layer_end 24 --process_id gpu6_L23_24 --trials 30 2>&1 | tee logs/exp2_gpu6_L23_24.log"

# GPU 7: L25-31 (4 processes)
CUDA_VISIBLE_DEVICES=7 tmux new-session -d -s exp2_gpu7_p1 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 26 --process_id gpu7_L25_26 --trials 30 2>&1 | tee logs/exp2_gpu7_L25_26.log"
CUDA_VISIBLE_DEVICES=7 tmux new-session -d -s exp2_gpu7_p2 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 27 --layer_end 28 --process_id gpu7_L27_28 --trials 30 2>&1 | tee logs/exp2_gpu7_L27_28.log"
CUDA_VISIBLE_DEVICES=7 tmux new-session -d -s exp2_gpu7_p3 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 30 --process_id gpu7_L29_30 --trials 30 2>&1 | tee logs/exp2_gpu7_L29_30.log"
CUDA_VISIBLE_DEVICES=7 tmux new-session -d -s exp2_gpu7_p4 "$PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 31 --layer_end 31 --process_id gpu7_L31 --trials 30 2>&1 | tee logs/exp2_gpu7_L31.log"

echo "✅ Launched 16 processes across GPU 4,5,6,7"
echo ""
echo "Monitor sessions:"
echo "  tmux attach -t exp2_gpu4_p1  # GPU 4 L1-2"
echo "  tmux attach -t exp2_gpu5_p1  # GPU 5 L9-10"
echo "  tmux attach -t exp2_gpu6_p1  # GPU 6 L17-18"
echo "  tmux attach -t exp2_gpu7_p1  # GPU 7 L25-26"
echo ""
echo "Check all sessions: tmux ls | grep exp2"
