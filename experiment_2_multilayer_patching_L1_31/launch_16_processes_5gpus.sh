#!/bin/bash
# Launch 16 patching processes across 5 GPUs (3,4,5,6,7)
# Optimized to prevent OOM by reducing processes per GPU

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31

echo "🚀 Launching 16 patching processes across 5 GPUs (3,4,5,6,7)"

# GPU 3: 3 processes (L1-6)
tmux new-session -d -s exp2_gpu3_p1 \
    "env CUDA_VISIBLE_DEVICES=3 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 1 --layer_end 2 --process_id gpu3_L1_2 --trials 30 \
    2>&1 | tee logs/exp2_gpu3_L1_2.log"
sleep 3

tmux new-session -d -s exp2_gpu3_p2 \
    "env CUDA_VISIBLE_DEVICES=3 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 3 --layer_end 4 --process_id gpu3_L3_4 --trials 30 \
    2>&1 | tee logs/exp2_gpu3_L3_4.log"
sleep 3

tmux new-session -d -s exp2_gpu3_p3 \
    "env CUDA_VISIBLE_DEVICES=3 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 5 --layer_end 6 --process_id gpu3_L5_6 --trials 30 \
    2>&1 | tee logs/exp2_gpu3_L5_6.log"
sleep 3

# GPU 4: 4 processes (L7-13)
tmux new-session -d -s exp2_gpu4_p1 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 7 --layer_end 8 --process_id gpu4_L7_8 --trials 30 \
    2>&1 | tee logs/exp2_gpu4_L7_8.log"
sleep 3

tmux new-session -d -s exp2_gpu4_p2 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 9 --layer_end 10 --process_id gpu4_L9_10 --trials 30 \
    2>&1 | tee logs/exp2_gpu4_L9_10.log"
sleep 3

tmux new-session -d -s exp2_gpu4_p3 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 11 --layer_end 12 --process_id gpu4_L11_12 --trials 30 \
    2>&1 | tee logs/exp2_gpu4_L11_12.log"
sleep 3

tmux new-session -d -s exp2_gpu4_p4 \
    "env CUDA_VISIBLE_DEVICES=4 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 13 --layer_end 13 --process_id gpu4_L13 --trials 30 \
    2>&1 | tee logs/exp2_gpu4_L13.log"
sleep 3

# GPU 5: 3 processes (L14-19)
tmux new-session -d -s exp2_gpu5_p1 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 14 --layer_end 15 --process_id gpu5_L14_15 --trials 30 \
    2>&1 | tee logs/exp2_gpu5_L14_15.log"
sleep 3

tmux new-session -d -s exp2_gpu5_p2 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 16 --layer_end 17 --process_id gpu5_L16_17 --trials 30 \
    2>&1 | tee logs/exp2_gpu5_L16_17.log"
sleep 3

tmux new-session -d -s exp2_gpu5_p3 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 18 --layer_end 19 --process_id gpu5_L18_19 --trials 30 \
    2>&1 | tee logs/exp2_gpu5_L18_19.log"
sleep 3

# GPU 6: 3 processes (L20-25)
tmux new-session -d -s exp2_gpu6_p1 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 20 --layer_end 21 --process_id gpu6_L20_21 --trials 30 \
    2>&1 | tee logs/exp2_gpu6_L20_21.log"
sleep 3

tmux new-session -d -s exp2_gpu6_p2 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 22 --layer_end 23 --process_id gpu6_L22_23 --trials 30 \
    2>&1 | tee logs/exp2_gpu6_L22_23.log"
sleep 3

tmux new-session -d -s exp2_gpu6_p3 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 24 --layer_end 25 --process_id gpu6_L24_25 --trials 30 \
    2>&1 | tee logs/exp2_gpu6_L24_25.log"
sleep 3

# GPU 7: 3 processes (L26-31)
tmux new-session -d -s exp2_gpu7_p1 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 26 --layer_end 27 --process_id gpu7_L26_27 --trials 30 \
    2>&1 | tee logs/exp2_gpu7_L26_27.log"
sleep 3

tmux new-session -d -s exp2_gpu7_p2 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 28 --layer_end 29 --process_id gpu7_L28_29 --trials 30 \
    2>&1 | tee logs/exp2_gpu7_L28_29.log"
sleep 3

tmux new-session -d -s exp2_gpu7_p3 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 30 --layer_end 31 --process_id gpu7_L30_31 --trials 30 \
    2>&1 | tee logs/exp2_gpu7_L30_31.log"

echo "✅ All 16 processes launched across 5 GPUs"
echo "GPU 3: 3 processes (L1-6)"
echo "GPU 4: 4 processes (L7-13)"
echo "GPU 5: 3 processes (L14-19)"
echo "GPU 6: 3 processes (L20-25)"
echo "GPU 7: 3 processes (L26-31)"
echo "Monitor with: tmux list-sessions | grep exp2"
