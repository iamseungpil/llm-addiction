#!/bin/bash
# Staggered launch to prevent SAE loading deadlock
# Start processes with 15-second delays to avoid simultaneous file access

echo "=================================================="
echo "🚀 Staggered Launch - All Experiments"
echo "=================================================="
echo ""
echo "🔍 Root causes identified:"
echo "  1. Gemma: JAX dependency issue (FIXED: removed jax/jaxlib)"
echo "  2. Patching: 12 processes loading SAE simultaneously → deadlock"
echo ""
echo "🛠️  Solution: Staggered launch with 15s delays"
echo ""
echo "📊 Launch plan:"
echo "  - Exp0 LLaMA: GPU 0"
echo "  - Exp0 Gemma: GPU 1"
echo "  - Exp2 Patching: 2 per GPU on GPUs 2,5,6,7 (8 processes total)"
echo "    Each process starts 15s apart to avoid SAE loading conflicts"
echo ""

# Kill existing sessions
tmux kill-session -t exp0_llama 2>/dev/null
tmux kill-session -t exp0_gemma 2>/dev/null
for i in {1..12}; do
    tmux kill-session -t exp2_p${i} 2>/dev/null
done

sleep 2

# ======================
# Experiment 0: LLaMA (GPU 0)
# ======================
echo "▶️  [0/10] Starting Exp0 LLaMA on GPU 0..."
tmux new-session -d -s exp0_llama \
    "env CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE=0 \
    python /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/experiment_0_restart.py \
    --model llama --gpu 0 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/logs/exp0_llama.log"

echo "   Waiting 5s..."
sleep 5

# ======================
# Experiment 0: Gemma (GPU 1)
# ======================
echo "▶️  [1/10] Starting Exp0 Gemma on GPU 1 (JAX removed)..."
tmux new-session -d -s exp0_gemma \
    "env CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE=0 \
    python /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/experiment_0_restart.py \
    --model gemma --gpu 1 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/logs/exp0_gemma.log"

echo "   Waiting 5s..."
sleep 5

# ======================
# Experiment 2: Patching (Staggered, 2 per GPU)
# ======================
echo ""
echo "▶️  Starting Exp2 Patching (8 processes, staggered)..."
echo ""

# GPU 2: 2 processes (L1-6, L7-12)
echo "▶️  [2/10] GPU 2: L1-6"
tmux new-session -d -s exp2_p1 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 1 --layer_end 6 --process_id L1_6_gpu2 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L1_6_gpu2.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

echo "▶️  [3/10] GPU 2: L7-12"
tmux new-session -d -s exp2_p2 \
    "env CUDA_VISIBLE_DEVICES=2 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 7 --layer_end 12 --process_id L7_12_gpu2 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L7_12_gpu2.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

# GPU 5: 2 processes (L13-18, L19-23)
echo "▶️  [4/10] GPU 5: L13-18"
tmux new-session -d -s exp2_p3 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 13 --layer_end 18 --process_id L13_18_gpu5 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L13_18_gpu5.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

echo "▶️  [5/10] GPU 5: L19-23"
tmux new-session -d -s exp2_p4 \
    "env CUDA_VISIBLE_DEVICES=5 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 19 --layer_end 23 --process_id L19_23_gpu5 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L19_23_gpu5.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

# GPU 6: 2 processes (L24-26, L27-28)
echo "▶️  [6/10] GPU 6: L24-26"
tmux new-session -d -s exp2_p5 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 24 --layer_end 26 --process_id L24_26_gpu6 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L24_26_gpu6.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

echo "▶️  [7/10] GPU 6: L27-28"
tmux new-session -d -s exp2_p6 \
    "env CUDA_VISIBLE_DEVICES=6 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 27 --layer_end 28 --process_id L27_28_gpu6 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L27_28_gpu6.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

# GPU 7: 2 processes (L29-30, L31)
echo "▶️  [8/10] GPU 7: L29-30"
tmux new-session -d -s exp2_p7 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 29 --layer_end 30 --process_id L29_30_gpu7 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L29_30_gpu7.log"
echo "   Waiting 15s for SAE loading..."
sleep 15

echo "▶️  [9/10] GPU 7: L31"
tmux new-session -d -s exp2_p8 \
    "env CUDA_VISIBLE_DEVICES=7 \
    python /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py \
    --gpu 0 --layer_start 31 --layer_end 31 --process_id L31_gpu7 --trials 30 \
    2>&1 | tee /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L31_gpu7.log"

echo ""
echo "=================================================="
echo "✅ All 10 experiments launched!"
echo "=================================================="
echo ""
echo "📋 Running experiments:"
echo "  • Exp0 LLaMA  (GPU 0): 3,200 games"
echo "  • Exp0 Gemma  (GPU 1): 3,200 games"
echo "  • Exp2 Patch  (GPUs 2,5,6,7): 9,300 features × 30 trials"
echo "  • Exp1 Pathway: Already completed ✅"
echo "  • Exp3 Word:    Already completed ✅"
echo ""
echo "📋 Monitor commands:"
echo "  tmux list-sessions"
echo "  tmux attach -t exp0_llama"
echo "  tmux attach -t exp2_p1"
echo ""
echo "🔍 Check status:"
echo "  nvidia-smi"
echo "  tail -f /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/logs/exp0_llama.log"
echo ""
echo "⏱️  Total launch time: ~2 minutes (staggered start)"
echo ""
