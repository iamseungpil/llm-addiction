#!/bin/bash
# Clean restart: Kill ALL, then launch on GPUs 2,5,6,7 (3 per GPU)

echo "🧹 Killing ALL Python processes..."
pkill -9 -f "experiment_2_L1_31_top300.py"
pkill -9 -f "experiment_0_restart.py"
sleep 5

# Kill tmux sessions
for sess in $(tmux list-sessions 2>/dev/null | grep -E "(exp2_|exp0_)" | cut -d: -f1); do
    echo "Killing session: $sess"
    tmux kill-session -t $sess 2>/dev/null
done
sleep 3

echo ""
echo "📊 Current GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting 12 patching processes (3 per GPU on 2,5,6,7)..."
echo ""

# GPU 2: L1-10 (3 processes)
echo "Starting GPU 2 processes..."
tmux new-session -d -s exp2_L1_3_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 3 --process_id L1_3_gpu2 2>&1 | tee logs/exp2_L1_3_gpu2.log"
sleep 3
tmux new-session -d -s exp2_L4_6_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 4 --layer_end 6 --process_id L4_6_gpu2 2>&1 | tee logs/exp2_L4_6_gpu2.log"
sleep 3
tmux new-session -d -s exp2_L7_10_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 7 --layer_end 10 --process_id L7_10_gpu2 2>&1 | tee logs/exp2_L7_10_gpu2.log"
sleep 3

# GPU 5: L11-18 (3 processes)
echo "Starting GPU 5 processes..."
tmux new-session -d -s exp2_L11_13_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 13 --process_id L11_13_gpu5 2>&1 | tee logs/exp2_L11_13_gpu5.log"
sleep 3
tmux new-session -d -s exp2_L14_16_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 14 --layer_end 16 --process_id L14_16_gpu5 2>&1 | tee logs/exp2_L14_16_gpu5.log"
sleep 3
tmux new-session -d -s exp2_L17_18_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 18 --process_id L17_18_gpu5 2>&1 | tee logs/exp2_L17_18_gpu5.log"
sleep 3

# GPU 6: L19-25 (3 processes)
echo "Starting GPU 6 processes..."
tmux new-session -d -s exp2_L19_21_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 21 --process_id L19_21_gpu6 2>&1 | tee logs/exp2_L19_21_gpu6.log"
sleep 3
tmux new-session -d -s exp2_L22_24_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 22 --layer_end 24 --process_id L22_24_gpu6 2>&1 | tee logs/exp2_L22_24_gpu6.log"
sleep 3
tmux new-session -d -s exp2_L25_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 25 --process_id L25_gpu6 2>&1 | tee logs/exp2_L25_gpu6.log"
sleep 3

# GPU 7: L26-31 (3 processes)
echo "Starting GPU 7 processes..."
tmux new-session -d -s exp2_L26_28_gpu7 "CUDA_VISIBLE_DEVICES=7 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 26 --layer_end 28 --process_id L26_28_gpu7 2>&1 | tee logs/exp2_L26_28_gpu7.log"
sleep 3
tmux new-session -d -s exp2_L29_30_gpu7 "CUDA_VISIBLE_DEVICES=7 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 30 --process_id L29_30_gpu7 2>&1 | tee logs/exp2_L29_30_gpu7.log"
sleep 3
tmux new-session -d -s exp2_L31_gpu7 "CUDA_VISIBLE_DEVICES=7 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 31 --layer_end 31 --process_id L31_gpu7 2>&1 | tee logs/exp2_L31_gpu7.log"

echo ""
echo "✅ All 12 patching processes started!"
tmux ls | grep exp2_

echo ""
echo "🤖 Starting Gemma on GPU 1..."
cd /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart
tmux new-session -d -s exp0_gemma "CUDA_VISIBLE_DEVICES=1 python experiment_0_restart.py --model gemma --gpu 0 2>&1 | tee logs/exp0_gemma_gpu1.log"

echo ""
echo "✅ Gemma started on GPU 1"
echo ""
echo "📊 Final GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
