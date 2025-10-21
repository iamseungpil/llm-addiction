#!/bin/bash
# Launch only patching experiments (12 processes on GPUs 2,5,6,7)

echo "=================================================="
echo "🚀 Launching Patching Experiments"
echo "=================================================="

# Kill existing patching sessions
for i in {1..12}; do
    tmux kill-session -t exp2_p${i} 2>/dev/null
done

echo ""
echo "📊 Launching 12 processes:"
echo "  GPU 2: L1-3, L4-6, L7-9"
echo "  GPU 5: L10-12, L13-15, L16-17"
echo "  GPU 6: L18-20, L21-23, L24-25"
echo "  GPU 7: L26-27, L28-29, L30-31"
echo ""

sleep 1

# GPU 2: 3 processes (L1-9)
tmux new-session -d -s exp2_p1 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 1 --layer_end 3 --process_id L1_3_gpu2 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L1_3_gpu2.log"

sleep 0.5

tmux new-session -d -s exp2_p2 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 4 --layer_end 6 --process_id L4_6_gpu2 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L4_6_gpu2.log"

sleep 0.5

tmux new-session -d -s exp2_p3 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 7 --layer_end 9 --process_id L7_9_gpu2 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L7_9_gpu2.log"

sleep 0.5

# GPU 5: 3 processes (L10-17)
tmux new-session -d -s exp2_p4 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 10 --layer_end 12 --process_id L10_12_gpu5 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L10_12_gpu5.log"

sleep 0.5

tmux new-session -d -s exp2_p5 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 13 --layer_end 15 --process_id L13_15_gpu5 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L13_15_gpu5.log"

sleep 0.5

tmux new-session -d -s exp2_p6 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 16 --layer_end 17 --process_id L16_17_gpu5 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L16_17_gpu5.log"

sleep 0.5

# GPU 6: 3 processes (L18-25)
tmux new-session -d -s exp2_p7 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 18 --layer_end 20 --process_id L18_20_gpu6 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L18_20_gpu6.log"

sleep 0.5

tmux new-session -d -s exp2_p8 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 21 --layer_end 23 --process_id L21_23_gpu6 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L21_23_gpu6.log"

sleep 0.5

tmux new-session -d -s exp2_p9 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 24 --layer_end 25 --process_id L24_25_gpu6 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L24_25_gpu6.log"

sleep 0.5

# GPU 7: 3 processes (L26-31)
tmux new-session -d -s exp2_p10 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 26 --layer_end 27 --process_id L26_27_gpu7 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L26_27_gpu7.log"

sleep 0.5

tmux new-session -d -s exp2_p11 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 28 --layer_end 29 --process_id L28_29_gpu7 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L28_29_gpu7.log"

sleep 0.5

tmux new-session -d -s exp2_p12 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 30 --layer_end 31 --process_id L30_31_gpu7 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L30_31_gpu7.log"

echo ""
echo "=================================================="
echo "✅ All 12 patching processes launched!"
echo "=================================================="
echo ""
echo "📋 Monitor: tmux attach -t exp2_p1 (or p2...p12)"
echo "🔍 GPU usage: nvidia-smi"
echo ""
