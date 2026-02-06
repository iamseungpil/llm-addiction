#!/bin/bash
# Launch 15 patching processes (3 per GPU × 5 GPUs: 2,3,5,6,7)
# Each process handles subset of layers to distribute 9,300 features

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31

# Layer distribution (31 layers total, split into 15 chunks)
# Each chunk: ~2 layers
# L1-2, L3-4, L5-6, L7-8, L9-10, L11-12, L13-14, L15-16, L17-18, L19-20, L21-22, L23-24, L25-26, L27-28, L29-31

echo "🚀 Launching 15 patching processes across 5 GPUs"

# GPU 2: 3 processes
tmux new-session -d -s exp2_gpu2_p1 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 1 --layer_end 2 --process_id gpu2_L1_2 --trials 30 \
    2>&1 | tee logs/exp2_gpu2_L1_2.log"

sleep 3

tmux new-session -d -s exp2_gpu2_p2 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 3 --layer_end 4 --process_id gpu2_L3_4 --trials 30 \
    2>&1 | tee logs/exp2_gpu2_L3_4.log"

sleep 3

tmux new-session -d -s exp2_gpu2_p3 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 5 --layer_end 6 --process_id gpu2_L5_6 --trials 30 \
    2>&1 | tee logs/exp2_gpu2_L5_6.log"

sleep 3

# GPU 3: 3 processes
tmux new-session -d -s exp2_gpu3_p1 \
    "env CUDA_VISIBLE_DEVICES=3 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 7 --layer_end 8 --process_id gpu3_L7_8 --trials 30 \
    2>&1 | tee logs/exp2_gpu3_L7_8.log"

sleep 3

tmux new-session -d -s exp2_gpu3_p2 \
    "env CUDA_VISIBLE_DEVICES=3 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 9 --layer_end 10 --process_id gpu3_L9_10 --trials 30 \
    2>&1 | tee logs/exp2_gpu3_L9_10.log"

sleep 3

tmux new-session -d -s exp2_gpu3_p3 \
    "env CUDA_VISIBLE_DEVICES=3 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 11 --layer_end 12 --process_id gpu3_L11_12 --trials 30 \
    2>&1 | tee logs/exp2_gpu3_L11_12.log"

sleep 3

# GPU 5: 3 processes
tmux new-session -d -s exp2_gpu5_p1 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 13 --layer_end 14 --process_id gpu5_L13_14 --trials 30 \
    2>&1 | tee logs/exp2_gpu5_L13_14.log"

sleep 3

tmux new-session -d -s exp2_gpu5_p2 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 15 --layer_end 16 --process_id gpu5_L15_16 --trials 30 \
    2>&1 | tee logs/exp2_gpu5_L15_16.log"

sleep 3

tmux new-session -d -s exp2_gpu5_p3 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 17 --layer_end 18 --process_id gpu5_L17_18 --trials 30 \
    2>&1 | tee logs/exp2_gpu5_L17_18.log"

sleep 3

# GPU 6: 3 processes
tmux new-session -d -s exp2_gpu6_p1 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 19 --layer_end 20 --process_id gpu6_L19_20 --trials 30 \
    2>&1 | tee logs/exp2_gpu6_L19_20.log"

sleep 3

tmux new-session -d -s exp2_gpu6_p2 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 21 --layer_end 22 --process_id gpu6_L21_22 --trials 30 \
    2>&1 | tee logs/exp2_gpu6_L21_22.log"

sleep 3

tmux new-session -d -s exp2_gpu6_p3 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 23 --layer_end 24 --process_id gpu6_L23_24 --trials 30 \
    2>&1 | tee logs/exp2_gpu6_L23_24.log"

sleep 3

# GPU 7: 3 processes
tmux new-session -d -s exp2_gpu7_p1 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 25 --layer_end 26 --process_id gpu7_L25_26 --trials 30 \
    2>&1 | tee logs/exp2_gpu7_L25_26.log"

sleep 3

tmux new-session -d -s exp2_gpu7_p2 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 27 --layer_end 28 --process_id gpu7_L27_28 --trials 30 \
    2>&1 | tee logs/exp2_gpu7_L27_28.log"

sleep 3

tmux new-session -d -s exp2_gpu7_p3 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 29 --layer_end 31 --process_id gpu7_L29_31 --trials 30 \
    2>&1 | tee logs/exp2_gpu7_L29_31.log"

echo "✅ All 15 processes launched"
echo "Monitor with: tmux list-sessions | grep exp2"
