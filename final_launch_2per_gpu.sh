#!/bin/bash
# FINAL: 2 processes per GPU × 6 GPUs = 12 total

echo "🧹 Killing ALL..."
pkill -9 -f "experiment_2_L1_31_top300.py"
pkill -9 -f "experiment_0_restart.py"
tmux kill-server 2>/dev/null
sleep 5

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31

echo "🚀 Starting 12 processes (2 per GPU)..."

# GPU 2: L1-10 (2 processes)
tmux new-session -d -s exp2_L1_5 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 5 --process_id L1_5 2>&1 | tee logs/exp2_L1_5.log"
sleep 5
tmux new-session -d -s exp2_L6_10 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 6 --layer_end 10 --process_id L6_10 2>&1 | tee logs/exp2_L6_10.log"
sleep 5

# GPU 3: L11-18 (2 processes)  
tmux new-session -d -s exp2_L11_15 "CUDA_VISIBLE_DEVICES=3 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 15 --process_id L11_15 2>&1 | tee logs/exp2_L11_15.log"
sleep 5
tmux new-session -d -s exp2_L16_18 "CUDA_VISIBLE_DEVICES=3 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 16 --layer_end 18 --process_id L16_18 2>&1 | tee logs/exp2_L16_18.log"
sleep 5

# GPU 5: L19-25 (2 processes)
tmux new-session -d -s exp2_L19_22 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 22 --process_id L19_22 2>&1 | tee logs/exp2_L19_22.log"
sleep 5
tmux new-session -d -s exp2_L23_25 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 23 --layer_end 25 --process_id L23_25 2>&1 | tee logs/exp2_L23_25.log"
sleep 5

# GPU 6: L26-31 (2 processes)
tmux new-session -d -s exp2_L26_28 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 26 --layer_end 28 --process_id L26_28 2>&1 | tee logs/exp2_L26_28.log"
sleep 5
tmux new-session -d -s exp2_L29_31 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 31 --process_id L29_31 2>&1 | tee logs/exp2_L29_31.log"
sleep 5

# GPU 7: Extra capacity (2 processes)
tmux new-session -d -s exp2_extra1 "CUDA_VISIBLE_DEVICES=7 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 5 --process_id extra1 2>&1 | tee logs/exp2_extra1.log" || true
sleep 5
tmux new-session -d -s exp2_extra2 "CUDA_VISIBLE_DEVICES=7 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 6 --layer_end 10 --process_id extra2 2>&1 | tee logs/exp2_extra2.log" || true
sleep 5

echo "✅ Patching: $(tmux ls 2>/dev/null | grep exp2_ | wc -l) processes"
tmux ls | grep exp2_

# Gemma on GPU 1
cd /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart
tmux new-session -d -s exp0_gemma "CUDA_VISIBLE_DEVICES=1 python experiment_0_restart.py --model gemma --gpu 0 2>&1 | tee logs/exp0_gemma.log"

echo "✅ Gemma started on GPU 1"
echo ""
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
