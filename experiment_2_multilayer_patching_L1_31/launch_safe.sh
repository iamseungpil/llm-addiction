#!/bin/bash
# SAFE launch: 1 process per GPU, 30s delays

echo "=================================================="
echo "🚀 SAFE Launch - Patching Only (4 processes)"
echo "=================================================="
echo ""
echo "🛠️  Strategy:"
echo "  - 1 process per GPU (GPUs 2,5,6,7)"
echo "  - 30-second delays between launches"
echo "  - Modified SAE loader (.to() instead of .float())"
echo ""

# Kill existing patching sessions
for i in {1..12}; do
    tmux kill-session -t exp2_p${i} 2>/dev/null
done

sleep 2

echo "▶️  [1/4] GPU 2: L1-8 (2,400 features)"
tmux new-session -d -s exp2_p1 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 1 --layer_end 8 --process_id L1_8_gpu2 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L1_8_gpu2.log"
echo "   Waiting 30s for safe SAE loading..."
sleep 30

echo "▶️  [2/4] GPU 5: L9-16 (2,400 features)"
tmux new-session -d -s exp2_p2 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 9 --layer_end 16 --process_id L9_16_gpu5 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L9_16_gpu5.log"
echo "   Waiting 30s for safe SAE loading..."
sleep 30

echo "▶️  [3/4] GPU 6: L17-24 (2,400 features)"
tmux new-session -d -s exp2_p3 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 17 --layer_end 24 --process_id L17_24_gpu6 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L17_24_gpu6.log"
echo "   Waiting 30s for safe SAE loading..."
sleep 30

echo "▶️  [4/4] GPU 7: L25-31 (2,100 features)"
tmux new-session -d -s exp2_p4 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 25 --layer_end 31 --process_id L25_31_gpu7 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L25_31_gpu7.log"

echo ""
echo "=================================================="
echo "✅ All 4 patching processes launched!"
echo "=================================================="
echo ""
echo "📋 Monitor: tmux attach -t exp2_p1 (or p2, p3, p4)"
echo "🔍 GPU usage: nvidia-smi"
echo ""
echo "Total features: 9,300"
echo "  GPU 2: 2,400 features (L1-8)"
echo "  GPU 5: 2,400 features (L9-16)"
echo "  GPU 6: 2,400 features (L17-24)"
echo "  GPU 7: 2,100 features (L25-31)"
echo ""
