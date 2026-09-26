#!/bin/bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31
PYTHON=/data/miniforge3/envs/llama_sae_env/bin/python

# GPU 4 (5 processes, 8 layers): L1-2, L3-4, L5-6, L7, L8
tmux new-session -d -s exp2_gpu4_1 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 2 --process_id gpu4_L1_2 --trials 30 2>&1 | tee logs/exp2_gpu4_L1_2.log"
tmux new-session -d -s exp2_gpu4_2 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 3 --layer_end 4 --process_id gpu4_L3_4 --trials 30 2>&1 | tee logs/exp2_gpu4_L3_4.log"
tmux new-session -d -s exp2_gpu4_3 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 5 --layer_end 6 --process_id gpu4_L5_6 --trials 30 2>&1 | tee logs/exp2_gpu4_L5_6.log"
tmux new-session -d -s exp2_gpu4_4 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 7 --layer_end 7 --process_id gpu4_L7 --trials 30 2>&1 | tee logs/exp2_gpu4_L7.log"
tmux new-session -d -s exp2_gpu4_5 "export CUDA_VISIBLE_DEVICES=4 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 8 --layer_end 8 --process_id gpu4_L8 --trials 30 2>&1 | tee logs/exp2_gpu4_L8.log"

# GPU 5 (5 processes, 8 layers): L9-10, L11-12, L13-14, L15, L16
tmux new-session -d -s exp2_gpu5_1 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 9 --layer_end 10 --process_id gpu5_L9_10 --trials 30 2>&1 | tee logs/exp2_gpu5_L9_10.log"
tmux new-session -d -s exp2_gpu5_2 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 12 --process_id gpu5_L11_12 --trials 30 2>&1 | tee logs/exp2_gpu5_L11_12.log"
tmux new-session -d -s exp2_gpu5_3 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 13 --layer_end 14 --process_id gpu5_L13_14 --trials 30 2>&1 | tee logs/exp2_gpu5_L13_14.log"
tmux new-session -d -s exp2_gpu5_4 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 15 --layer_end 15 --process_id gpu5_L15 --trials 30 2>&1 | tee logs/exp2_gpu5_L15.log"
tmux new-session -d -s exp2_gpu5_5 "export CUDA_VISIBLE_DEVICES=5 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 16 --layer_end 16 --process_id gpu5_L16 --trials 30 2>&1 | tee logs/exp2_gpu5_L16.log"

# GPU 6 (5 processes, 8 layers): L17-18, L19-20, L21-22, L23, L24
tmux new-session -d -s exp2_gpu6_1 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 18 --process_id gpu6_L17_18 --trials 30 2>&1 | tee logs/exp2_gpu6_L17_18.log"
tmux new-session -d -s exp2_gpu6_2 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 20 --process_id gpu6_L19_20 --trials 30 2>&1 | tee logs/exp2_gpu6_L19_20.log"
tmux new-session -d -s exp2_gpu6_3 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 21 --layer_end 22 --process_id gpu6_L21_22 --trials 30 2>&1 | tee logs/exp2_gpu6_L21_22.log"
tmux new-session -d -s exp2_gpu6_4 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 23 --layer_end 23 --process_id gpu6_L23 --trials 30 2>&1 | tee logs/exp2_gpu6_L23.log"
tmux new-session -d -s exp2_gpu6_5 "export CUDA_VISIBLE_DEVICES=6 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 24 --layer_end 24 --process_id gpu6_L24 --trials 30 2>&1 | tee logs/exp2_gpu6_L24.log"

# GPU 7 (5 processes, 7 layers): L25-26, L27-28, L29, L30, L31
tmux new-session -d -s exp2_gpu7_1 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 26 --process_id gpu7_L25_26 --trials 30 2>&1 | tee logs/exp2_gpu7_L25_26.log"
tmux new-session -d -s exp2_gpu7_2 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 27 --layer_end 28 --process_id gpu7_L27_28 --trials 30 2>&1 | tee logs/exp2_gpu7_L27_28.log"
tmux new-session -d -s exp2_gpu7_3 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 29 --process_id gpu7_L29 --trials 30 2>&1 | tee logs/exp2_gpu7_L29.log"
tmux new-session -d -s exp2_gpu7_4 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 30 --layer_end 30 --process_id gpu7_L30 --trials 30 2>&1 | tee logs/exp2_gpu7_L30.log"
tmux new-session -d -s exp2_gpu7_5 "export CUDA_VISIBLE_DEVICES=7 && $PYTHON experiment_2_L1_31_top300.py --gpu 0 --layer_start 31 --layer_end 31 --process_id gpu7_L31 --trials 30 2>&1 | tee logs/exp2_gpu7_L31.log"

echo "✅ Launched 20 processes (5 per GPU)"
echo ""
echo "GPU 4 (8 layers): L1-2, L3-4, L5-6, L7, L8"
echo "GPU 5 (8 layers): L9-10, L11-12, L13-14, L15, L16"
echo "GPU 6 (8 layers): L17-18, L19-20, L21-22, L23, L24"
echo "GPU 7 (7 layers): L25-26, L27-28, L29, L30, L31"
echo ""
echo "To monitor: tmux attach -t exp2_gpu4_1 (or any other session)"
