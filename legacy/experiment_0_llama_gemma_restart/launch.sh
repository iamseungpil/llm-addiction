#!/bin/bash
# Launch Exp0: LLaMA/Gemma Restart Experiment

cd /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart

echo "🚀 Launching Exp0: LLaMA/Gemma Restart"
echo "  GPU 0: LLaMA (3,200 games)"
echo "  GPU 1: Gemma (3,200 games)"
echo "  Infinite retry on empty responses"
echo "  Using English prompts (matching Gemini experiment)"

# GPU 0: LLaMA
tmux new-session -d -s exp0_llama "
  source ~/.bashrc
  conda activate llama_sae_env
  export CUDA_VISIBLE_DEVICES=0
  export TORCH_COMPILE=0
  python -u experiment_0_restart.py --model llama --gpu 0 2>&1 | tee logs/exp0_llama.log
"
echo "✅ Started GPU 0: LLaMA"

# GPU 1: Gemma
tmux new-session -d -s exp0_gemma "
  source ~/.bashrc
  conda activate llama_sae_env
  export CUDA_VISIBLE_DEVICES=1
  export TORCH_COMPILE=0
  python -u experiment_0_restart.py --model gemma --gpu 1 2>&1 | tee logs/exp0_gemma.log
"
echo "✅ Started GPU 1: Gemma"

echo ""
echo "Monitor with:"
echo "  tmux attach -t exp0_llama"
echo "  tmux attach -t exp0_gemma"
