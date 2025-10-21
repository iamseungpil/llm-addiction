#!/bin/bash
# Launch 12 parallel patching processes (3 per GPU on GPUs 2,4,5,6)
# Each process handles more layers to reduce memory pressure

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill existing processes
echo "Killing existing experiment_2 processes..."
pkill -f "experiment_2_L1_31_top300.py"
sleep 3

# Kill tmux sessions
for sess in $(tmux list-sessions 2>/dev/null | grep exp2_ | cut -d: -f1); do
    tmux kill-session -t $sess 2>/dev/null
done
sleep 2

echo "Starting 12 processes..."

# GPU 2: L1-10 (3 processes, ~1000 features each)
tmux new-session -d -s exp2_L1_3_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 3 --process_id L1_3_gpu2 2>&1 | tee logs/exp2_L1_3_gpu2.log"
tmux new-session -d -s exp2_L4_6_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 4 --layer_end 6 --process_id L4_6_gpu2 2>&1 | tee logs/exp2_L4_6_gpu2.log"
tmux new-session -d -s exp2_L7_10_gpu2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 7 --layer_end 10 --process_id L7_10_gpu2 2>&1 | tee logs/exp2_L7_10_gpu2.log"

# GPU 4: L11-18 (3 processes)
tmux new-session -d -s exp2_L11_13_gpu4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 13 --process_id L11_13_gpu4 2>&1 | tee logs/exp2_L11_13_gpu4.log"
tmux new-session -d -s exp2_L14_16_gpu4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 14 --layer_end 16 --process_id L14_16_gpu4 2>&1 | tee logs/exp2_L14_16_gpu4.log"
tmux new-session -d -s exp2_L17_18_gpu4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 18 --process_id L17_18_gpu4 2>&1 | tee logs/exp2_L17_18_gpu4.log"

# GPU 5: L19-25 (3 processes)
tmux new-session -d -s exp2_L19_21_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 21 --process_id L19_21_gpu5 2>&1 | tee logs/exp2_L19_21_gpu5.log"
tmux new-session -d -s exp2_L22_24_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 22 --layer_end 24 --process_id L22_24_gpu5 2>&1 | tee logs/exp2_L22_24_gpu5.log"
tmux new-session -d -s exp2_L25_gpu5 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 25 --process_id L25_gpu5 2>&1 | tee logs/exp2_L25_gpu5.log"

# GPU 6: L26-31 (3 processes)
tmux new-session -d -s exp2_L26_28_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 26 --layer_end 28 --process_id L26_28_gpu6 2>&1 | tee logs/exp2_L26_28_gpu6.log"
tmux new-session -d -s exp2_L29_30_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 30 --process_id L29_30_gpu6 2>&1 | tee logs/exp2_L29_30_gpu6.log"
tmux new-session -d -s exp2_L31_gpu6 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 31 --layer_end 31 --process_id L31_gpu6 2>&1 | tee logs/exp2_L31_gpu6.log"

sleep 5

echo ""
echo "✅ Launched 12 parallel processes:"
echo "   GPU 2: L1-3, L4-6, L7-10"
echo "   GPU 4: L11-13, L14-16, L17-18"
echo "   GPU 5: L19-21, L22-24, L25"
echo "   GPU 6: L26-28, L29-30, L31"
echo ""
echo "Monitor with:"
echo "  tmux ls | grep exp2_"
echo "  tmux attach -t exp2_L1_3_gpu2"
