#!/bin/bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31
PYTHON=/data/miniforge3/envs/llama_sae_env/bin/python

# GPU 4 (3 processes): L1-3, L4-5, L6-8
tmux new-session -d -s exp2_gpu4_1 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 3 --process_id gpu4_L1_3 --trials 30 2>&1 | tee logs/exp2_gpu4_L1_3.log"
tmux new-session -d -s exp2_gpu4_2 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 4 --layer_end 5 --process_id gpu4_L4_5 --trials 30 2>&1 | tee logs/exp2_gpu4_L4_5.log"
tmux new-session -d -s exp2_gpu4_3 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 6 --layer_end 8 --process_id gpu4_L6_8 --trials 30 2>&1 | tee logs/exp2_gpu4_L6_8.log"

# GPU 5 (3 processes): L9-11, L12-13, L14-16
tmux new-session -d -s exp2_gpu5_1 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 9 --layer_end 11 --process_id gpu5_L9_11 --trials 30 2>&1 | tee logs/exp2_gpu5_L9_11.log"
tmux new-session -d -s exp2_gpu5_2 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 12 --layer_end 13 --process_id gpu5_L12_13 --trials 30 2>&1 | tee logs/exp2_gpu5_L12_13.log"
tmux new-session -d -s exp2_gpu5_3 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 14 --layer_end 16 --process_id gpu5_L14_16 --trials 30 2>&1 | tee logs/exp2_gpu5_L14_16.log"

# GPU 6 (3 processes): L17-19, L20-21, L22-24
tmux new-session -d -s exp2_gpu6_1 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 19 --process_id gpu6_L17_19 --trials 30 2>&1 | tee logs/exp2_gpu6_L17_19.log"
tmux new-session -d -s exp2_gpu6_2 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 20 --layer_end 21 --process_id gpu6_L20_21 --trials 30 2>&1 | tee logs/exp2_gpu6_L20_21.log"
tmux new-session -d -s exp2_gpu6_3 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 22 --layer_end 24 --process_id gpu6_L22_24 --trials 30 2>&1 | tee logs/exp2_gpu6_L22_24.log"

# GPU 7 (3 processes): L25-27, L28-29, L30-31
tmux new-session -d -s exp2_gpu7_1 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 27 --process_id gpu7_L25_27 --trials 30 2>&1 | tee logs/exp2_gpu7_L25_27.log"
tmux new-session -d -s exp2_gpu7_2 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 28 --layer_end 29 --process_id gpu7_L28_29 --trials 30 2>&1 | tee logs/exp2_gpu7_L28_29.log"
tmux new-session -d -s exp2_gpu7_3 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 30 --layer_end 31 --process_id gpu7_L30_31 --trials 30 2>&1 | tee logs/exp2_gpu7_L30_31.log"

echo "✅ Launched 12 processes across 4 GPUs"
echo "GPU 4: 3 processes (L1-3, L4-5, L6-8)"
echo "GPU 5: 3 processes (L9-11, L12-13, L14-16)"
echo "GPU 6: 3 processes (L17-19, L20-21, L22-24)"
echo "GPU 7: 3 processes (L25-27, L28-29, L30-31)"
