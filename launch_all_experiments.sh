#!/bin/bash
# Master launcher for all 5 experiments

echo "=" * 80
echo "🚀 LAUNCHING ALL 5 EXPERIMENTS"
echo "=" * 80
echo ""
echo "GPU Allocation:"
echo "  GPU 0: Exp0 - LLaMA restart (3,200 games)"
echo "  GPU 1: Exp0 - Gemma restart (3,200 games)"
echo "  GPU 2: Exp2 - L1-8 multilayer patching (2,400 features)"
echo "  GPU 3: Exp1 - Layer pathway (50 games) + Exp3 - Feature-word (441 features)"
echo "  GPU 4: Exp2 - L9-16 (2,400 features) + Exp5 - Multiround patching (341 remaining)"
echo "  GPU 6: Exp2 - L17-24 (2,400 features)"
echo "  GPU 7: Exp2 - L25-31 (2,100 features)"
echo ""
echo "=" * 80
echo ""

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env

# Launch Exp0: LLaMA/Gemma restart
echo "🔵 Launching Exp0 (LLaMA/Gemma)..."
cd /home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart
bash launch.sh
sleep 2

# Launch Exp1: Layer Pathway
echo ""
echo "🔵 Launching Exp1 (Layer Pathway)..."
cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31
bash launch.sh
sleep 2

# Launch Exp2: Multilayer Patching L1-31
echo ""
echo "🔵 Launching Exp2 (Multilayer Patching L1-31)..."
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31
bash launch_all_gpu.sh
sleep 2

# Launch Exp3: Feature-Word Analysis
echo ""
echo "🔵 Launching Exp3 (Feature-Word Analysis)..."
cd /home/ubuntu/llm_addiction/experiment_3_feature_word_6400
bash launch.sh
sleep 2

# Launch Exp5: Multi-Round Patching (restart)
echo ""
echo "🔵 Restarting Exp5 (Multi-Round Patching)..."
cd /home/ubuntu/llm_addiction/experiment_5_multiround_patching
bash restart_exp5.sh
sleep 2

echo ""
echo "=" * 80
echo "✅ ALL EXPERIMENTS LAUNCHED!"
echo "=" * 80
echo ""
echo "Active tmux sessions:"
tmux ls
echo ""
echo "Monitor individual experiments:"
echo "  Exp0: tmux attach -t exp0_llama"
echo "  Exp0: tmux attach -t exp0_gemma"
echo "  Exp1: tmux attach -t exp1_pathway"
echo "  Exp2: tmux attach -t exp2_L1_8"
echo "  Exp2: tmux attach -t exp2_L9_16"
echo "  Exp2: tmux attach -t exp2_L17_24"
echo "  Exp2: tmux attach -t exp2_L25_31"
echo "  Exp3: tmux attach -t exp3_word"
echo "  Exp5: tmux attach -t exp5_multiround"
echo ""
echo "Check errors with:"
echo "  python /home/ubuntu/llm_addiction/monitor_all_experiments.py"
