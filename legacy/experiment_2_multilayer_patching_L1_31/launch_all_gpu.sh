#!/bin/bash
# Launch Exp2 Multilayer Patching across 4 GPUs
# Total: 9,300 features (31 layers × 300)

cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31

# Activate environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env

echo "🚀 Launching Exp2 Multilayer Patching L1-31"
echo "=" * 80
echo "GPU allocation:"
echo "  GPU 2: L1-8 (2,400 features)"
echo "  GPU 4: L9-16 (2,400 features, shares with Exp5)"
echo "  GPU 6: L17-24 (2,400 features)"
echo "  GPU 7: L25-31 (2,100 features)"
echo "=" * 80

# GPU 2: L1-8
tmux new-session -d -s exp2_L1_8 "CUDA_VISIBLE_DEVICES=2 python experiment_2_L1_31_top300.py --gpu 2 --layer_start 1 --layer_end 8 --process_id L1_8 --trials 20 2>&1 | tee logs/exp2_L1_8.log"
echo "✅ Started GPU 2: L1-8"

# GPU 4: L9-16 (shares with ongoing Exp5)
tmux new-session -d -s exp2_L9_16 "CUDA_VISIBLE_DEVICES=4 python experiment_2_L1_31_top300.py --gpu 4 --layer_start 9 --layer_end 16 --process_id L9_16 --trials 20 2>&1 | tee logs/exp2_L9_16.log"
echo "✅ Started GPU 4: L9-16"

# GPU 6: L17-24
tmux new-session -d -s exp2_L17_24 "CUDA_VISIBLE_DEVICES=6 python experiment_2_L1_31_top300.py --gpu 6 --layer_start 17 --layer_end 24 --process_id L17_24 --trials 20 2>&1 | tee logs/exp2_L17_24.log"
echo "✅ Started GPU 6: L17-24"

# GPU 7: L25-31
tmux new-session -d -s exp2_L25_31 "CUDA_VISIBLE_DEVICES=7 python experiment_2_L1_31_top300.py --gpu 7 --layer_start 25 --layer_end 31 --process_id L25_31 --trials 20 2>&1 | tee logs/exp2_L25_31.log"
echo "✅ Started GPU 7: L25-31"

echo ""
echo "🔄 All Exp2 processes launched!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t exp2_L1_8"
echo "  tmux attach -t exp2_L9_16"
echo "  tmux attach -t exp2_L17_24"
echo "  tmux attach -t exp2_L25_31"
