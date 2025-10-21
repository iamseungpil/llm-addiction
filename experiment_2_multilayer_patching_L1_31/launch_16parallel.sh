#!/bin/bash
# Launch 16 parallel patching processes (4 per GPU on GPUs 2,4,5,6)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill existing processes
echo "Killing existing experiment_2 processes..."
pkill -f "experiment_2_L1_31_top300.py"
sleep 2

# GPU 2: L1-8 split into 4 parts (each 600 features)
# Layer 1-2: 600 features
tmux new-session -d -s exp2_L1_2_p1 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 1 --layer_end 2 --process_id L1_2_p1 2>&1 | tee logs/exp2_L1_2_p1.log"

# Layer 3-4: 600 features
tmux new-session -d -s exp2_L3_4_p2 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 3 --layer_end 4 --process_id L3_4_p2 2>&1 | tee logs/exp2_L3_4_p2.log"

# Layer 5-6: 600 features
tmux new-session -d -s exp2_L5_6_p3 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 5 --layer_end 6 --process_id L5_6_p3 2>&1 | tee logs/exp2_L5_6_p3.log"

# Layer 7-8: 600 features
tmux new-session -d -s exp2_L7_8_p4 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 7 --layer_end 8 --process_id L7_8_p4 2>&1 | tee logs/exp2_L7_8_p4.log"

# GPU 4: L9-16 split into 4 parts
# Layer 9-10: 600 features
tmux new-session -d -s exp2_L9_10_p1 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 9 --layer_end 10 --process_id L9_10_p1 2>&1 | tee logs/exp2_L9_10_p1.log"

# Layer 11-12: 600 features
tmux new-session -d -s exp2_L11_12_p2 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 11 --layer_end 12 --process_id L11_12_p2 2>&1 | tee logs/exp2_L11_12_p2.log"

# Layer 13-14: 600 features
tmux new-session -d -s exp2_L13_14_p3 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 13 --layer_end 14 --process_id L13_14_p3 2>&1 | tee logs/exp2_L13_14_p3.log"

# Layer 15-16: 600 features
tmux new-session -d -s exp2_L15_16_p4 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 15 --layer_end 16 --process_id L15_16_p4 2>&1 | tee logs/exp2_L15_16_p4.log"

# GPU 5: L17-24 split into 4 parts
# Layer 17-18: 600 features
tmux new-session -d -s exp2_L17_18_p1 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 17 --layer_end 18 --process_id L17_18_p1 2>&1 | tee logs/exp2_L17_18_p1.log"

# Layer 19-20: 600 features
tmux new-session -d -s exp2_L19_20_p2 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 19 --layer_end 20 --process_id L19_20_p2 2>&1 | tee logs/exp2_L19_20_p2.log"

# Layer 21-22: 600 features
tmux new-session -d -s exp2_L21_22_p3 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 21 --layer_end 22 --process_id L21_22_p3 2>&1 | tee logs/exp2_L21_22_p3.log"

# Layer 23-24: 600 features
tmux new-session -d -s exp2_L23_24_p4 "CUDA_VISIBLE_DEVICES=5 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 23 --layer_end 24 --process_id L23_24_p4 2>&1 | tee logs/exp2_L23_24_p4.log"

# GPU 6: L25-31 split into 4 parts
# Layer 25-26: 600 features
tmux new-session -d -s exp2_L25_26_p1 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 25 --layer_end 26 --process_id L25_26_p1 2>&1 | tee logs/exp2_L25_26_p1.log"

# Layer 27-28: 600 features
tmux new-session -d -s exp2_L27_28_p2 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 27 --layer_end 28 --process_id L27_28_p2 2>&1 | tee logs/exp2_L27_28_p2.log"

# Layer 29-30: 600 features
tmux new-session -d -s exp2_L29_30_p3 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 29 --layer_end 30 --process_id L29_30_p3 2>&1 | tee logs/exp2_L29_30_p3.log"

# Layer 31: 300 features
tmux new-session -d -s exp2_L31_p4 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 0 --layer_start 31 --layer_end 31 --process_id L31_p4 2>&1 | tee logs/exp2_L31_p4.log"

echo ""
echo "✅ Launched 16 parallel processes:"
echo "   GPU 2: L1-2, L3-4, L5-6, L7-8"
echo "   GPU 4: L9-10, L11-12, L13-14, L15-16"
echo "   GPU 5: L17-18, L19-20, L21-22, L23-24"
echo "   GPU 6: L25-26, L27-28, L29-30, L31"
echo ""
echo "Monitor with:"
echo "  tmux ls"
echo "  tmux attach -t exp2_L1_2_p1  (etc.)"
