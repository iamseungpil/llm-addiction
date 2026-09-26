#!/bin/bash
# Launch 16 patching processes (4 per GPU × 4 GPUs: 4,5,6,7)
# Each process: 50 trials per feature (default)

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31

echo "🚀 Launching 16 patching processes across 4 GPUs (4,5,6,7)"
echo "   Trials per feature: 50"

# GPU 4: 4 processes (L1-8)
tmux new-session -d -s exp2_gpu4_p1 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 1 --layer_end 2 --process_id gpu4_L1_2 --trials 50 \
    2>&1 | tee logs/exp2_gpu4_L1_2.log"
sleep 3

tmux new-session -d -s exp2_gpu4_p2 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 3 --layer_end 4 --process_id gpu4_L3_4 --trials 50 \
    2>&1 | tee logs/exp2_gpu4_L3_4.log"
sleep 3

tmux new-session -d -s exp2_gpu4_p3 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 5 --layer_end 6 --process_id gpu4_L5_6 --trials 50 \
    2>&1 | tee logs/exp2_gpu4_L5_6.log"
sleep 3

tmux new-session -d -s exp2_gpu4_p4 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 7 --layer_end 8 --process_id gpu4_L7_8 --trials 50 \
    2>&1 | tee logs/exp2_gpu4_L7_8.log"
sleep 3

# GPU 5: 4 processes (L9-16)
tmux new-session -d -s exp2_gpu5_p1 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 9 --layer_end 10 --process_id gpu5_L9_10 --trials 50 \
    2>&1 | tee logs/exp2_gpu5_L9_10.log"
sleep 3

tmux new-session -d -s exp2_gpu5_p2 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 11 --layer_end 12 --process_id gpu5_L11_12 --trials 50 \
    2>&1 | tee logs/exp2_gpu5_L11_12.log"
sleep 3

tmux new-session -d -s exp2_gpu5_p3 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 13 --layer_end 14 --process_id gpu5_L13_14 --trials 50 \
    2>&1 | tee logs/exp2_gpu5_L13_14.log"
sleep 3

tmux new-session -d -s exp2_gpu5_p4 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 15 --layer_end 16 --process_id gpu5_L15_16 --trials 50 \
    2>&1 | tee logs/exp2_gpu5_L15_16.log"
sleep 3

# GPU 6: 4 processes (L17-24)
tmux new-session -d -s exp2_gpu6_p1 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 17 --layer_end 18 --process_id gpu6_L17_18 --trials 50 \
    2>&1 | tee logs/exp2_gpu6_L17_18.log"
sleep 3

tmux new-session -d -s exp2_gpu6_p2 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 19 --layer_end 20 --process_id gpu6_L19_20 --trials 50 \
    2>&1 | tee logs/exp2_gpu6_L19_20.log"
sleep 3

tmux new-session -d -s exp2_gpu6_p3 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 21 --layer_end 22 --process_id gpu6_L21_22 --trials 50 \
    2>&1 | tee logs/exp2_gpu6_L21_22.log"
sleep 3

tmux new-session -d -s exp2_gpu6_p4 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 23 --layer_end 24 --process_id gpu6_L23_24 --trials 50 \
    2>&1 | tee logs/exp2_gpu6_L23_24.log"
sleep 3

# GPU 7: 4 processes (L25-31)
tmux new-session -d -s exp2_gpu7_p1 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 25 --layer_end 26 --process_id gpu7_L25_26 --trials 50 \
    2>&1 | tee logs/exp2_gpu7_L25_26.log"
sleep 3

tmux new-session -d -s exp2_gpu7_p2 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 27 --layer_end 28 --process_id gpu7_L27_28 --trials 50 \
    2>&1 | tee logs/exp2_gpu7_L27_28.log"
sleep 3

tmux new-session -d -s exp2_gpu7_p3 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 29 --layer_end 30 --process_id gpu7_L29_30 --trials 50 \
    2>&1 | tee logs/exp2_gpu7_L29_30.log"
sleep 3

tmux new-session -d -s exp2_gpu7_p4 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 31 --layer_end 31 --process_id gpu7_L31 --trials 50 \
    2>&1 | tee logs/exp2_gpu7_L31.log"

echo "✅ All 16 processes launched"
echo "GPU 4: 4 processes (L1-8)"
echo "GPU 5: 4 processes (L9-16)"
echo "GPU 6: 4 processes (L17-24)"
echo "GPU 7: 4 processes (L25-31)"
echo "Monitor with: tmux list-sessions | grep exp2"
