#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=inv_llama_c50_new
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=40G
# [SLURM-DISABLED] #SBATCH --time=05:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/inv_llama_c50_new_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/inv_llama_c50_new_%j.err

# IMPORTANT: Only submit this AFTER verifying new prompt test succeeds!

# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

echo "================================"
echo "LLaMA Investment Choice - c50"
echo "WITH V3 RETRY MECHANISM"
echo "================================"
echo "V3 Improvements (builds on V2):"
echo "  - V2: Anti-hallucination (0% halluc achieved)"
echo "  - V3: Retry with hints for Variable betting"
echo "  - No immediate fallback → Model learns format"
echo "  - Retry prompt: 'Specify both option and amount: Option X, \$Y'"
echo "  - Final fallback after 5 retries (logged as ERROR)"
echo "  - Parsing bug fixed: choice != bet_amount"
echo "================================"
echo "Expected results:"
echo "  - Hallucination: 0% (maintained)"
echo "  - Variable betting: Actual amounts generated"
echo "  - Choice-bet correlation: Removed (bug fixed)"
echo "================================"

# Full experiment: 2 bet types × 4 conditions × 50 reps = 400 games
# Using v2 anti-hallucination prompt (no examples, minimal tokens, low temp)

python investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint 50

echo "LLaMA c50 (v2 anti-hallucination) completed!"
