#!/bin/bash
# Sequential launch: Start processes one by one with 30s delay
# Allows GPU memory to stabilize before next process

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill ALL existing exp2 processes
echo "🧹 Cleaning up old processes..."
pkill -9 -f "experiment_2_L1_31_top300.py"
sleep 5

# Kill tmux sessions
for sess in $(tmux list-sessions 2>/dev/null | grep exp2_ | cut -d: -f1); do
    tmux kill-session -t $sess 2>/dev/null
done
sleep 3

# Clear GPU memory
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 5

echo "🚀 Starting 12 processes sequentially..."

# GPU 2: 3 processes
echo "GPU 2 - Process 1/3..."
tmux new-session -d -s exp2_L1_3_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 3 --process_id L1_3_gpu2 2>&1 | tee logs/exp2_L1_3_gpu2.log"
sleep 30

echo "GPU 2 - Process 2/3..."
tmux new-session -d -s exp2_L4_6_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 4 --layer_end 6 --process_id L4_6_gpu2 2>&1 | tee logs/exp2_L4_6_gpu2.log"
sleep 30

echo "GPU 2 - Process 3/3..."
tmux new-session -d -s exp2_L7_10_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 7 --layer_end 10 --process_id L7_10_gpu2 2>&1 | tee logs/exp2_L7_10_gpu2.log"
sleep 30

# GPU 4: 3 processes
echo "GPU 4 - Process 1/3..."
tmux new-session -d -s exp2_L11_13_gpu4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 13 --process_id L11_13_gpu4 2>&1 | tee logs/exp2_L11_13_gpu4.log"
sleep 30

echo "GPU 4 - Process 2/3..."
tmux new-session -d -s exp2_L14_16_gpu4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 14 --layer_end 16 --process_id L14_16_gpu4 2>&1 | tee logs/exp2_L14_16_gpu4.log"
sleep 30

echo "GPU 4 - Process 3/3..."
tmux new-session -d -s exp2_L17_18_gpu4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 18 --process_id L17_18_gpu4 2>&1 | tee logs/exp2_L17_18_gpu4.log"
sleep 30

# GPU 5: 3 processes
echo "GPU 5 - Process 1/3..."
tmux new-session -d -s exp2_L19_21_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 21 --process_id L19_21_gpu5 2>&1 | tee logs/exp2_L19_21_gpu5.log"
sleep 30

echo "GPU 5 - Process 2/3..."
tmux new-session -d -s exp2_L22_24_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 22 --layer_end 24 --process_id L22_24_gpu5 2>&1 | tee logs/exp2_L22_24_gpu5.log"
sleep 30

echo "GPU 5 - Process 3/3..."
tmux new-session -d -s exp2_L25_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 25 --process_id L25_gpu5 2>&1 | tee logs/exp2_L25_gpu5.log"
sleep 30

# GPU 6: 3 processes
echo "GPU 6 - Process 1/3..."
tmux new-session -d -s exp2_L26_28_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 26 --layer_end 28 --process_id L26_28_gpu6 2>&1 | tee logs/exp2_L26_28_gpu6.log"
sleep 30

echo "GPU 6 - Process 2/3..."
tmux new-session -d -s exp2_L29_30_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 30 --process_id L29_30_gpu6 2>&1 | tee logs/exp2_L29_30_gpu6.log"
sleep 30

echo "GPU 6 - Process 3/3..."
tmux new-session -d -s exp2_L31_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 31 --layer_end 31 --process_id L31_gpu6 2>&1 | tee logs/exp2_L31_gpu6.log"

echo ""
echo "✅ All 12 processes started!"
echo "   Total launch time: ~6 minutes"
echo ""
tmux ls | grep exp2_
