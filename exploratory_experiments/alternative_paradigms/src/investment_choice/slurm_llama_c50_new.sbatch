#!/bin/bash
#SBATCH --job-name=inv_llama_c50_new
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --comment=python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/inv_llama_c50_new_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/inv_llama_c50_new_%j.err

# IMPORTANT: Only submit this AFTER verifying new prompt test succeeds!

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src

echo "================================"
echo "LLaMA Investment Choice - c50"
echo "WITH V2 ANTI-HALLUCINATION"
echo "================================"
echo "V2 Improvements applied:"
echo "  - max_new_tokens: 250 → 20 (75% reduction)"
echo "  - temperature: 0.7 → 0.3 (more deterministic)"
echo "  - Few-shot examples removed (key fix!)"
echo "  - History format simplified (no forum-style)"
echo "  - Prompt prefix changed (no meta-confusion)"
echo "  - More permissive parsing (first digit)"
echo "================================"
echo "Expected results:"
echo "  - Hallucination: <5% (was 33.7%)"
echo "  - Valid parsing: >95% (was ~35%)"
echo "================================"

# Full experiment: 2 bet types × 4 conditions × 50 reps = 400 games
# Using v2 anti-hallucination prompt (no examples, minimal tokens, low temp)

python investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint 50

echo "LLaMA c50 (v2 anti-hallucination) completed!"
