#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=investment-c70
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:2
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=16
# [SLURM-DISABLED] #SBATCH --mem=80G
# [SLURM-DISABLED] #SBATCH --time=20:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/investment_c70_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/investment_c70_%j.err

# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

cd /home/jovyan/llm-addiction

echo "======================================================="
echo "INVESTMENT CHOICE - c70 (LLaMA + Gemma)"
echo "======================================================="
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# GPU 0: LLaMA c70 (400 games, ~8h)
(
    echo "=== GPU 0: LLaMA c70 ==="
    CUDA_VISIBLE_DEVICES=0 python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
        --model llama --gpu 0 --constraint 70 \
        2>&1
    if [ $? -eq 0 ]; then echo "✓ Completed: llama_c70"
    else echo "✗ FAILED: llama_c70"; fi
) &

# GPU 1: Gemma c70 (400 games, ~8h)
(
    echo "=== GPU 1: Gemma c70 ==="
    CUDA_VISIBLE_DEVICES=1 python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
        --model gemma --gpu 0 --constraint 70 \
        2>&1
    if [ $? -eq 0 ]; then echo "✓ Completed: gemma_c70"
    else echo "✗ FAILED: gemma_c70"; fi
) &

wait

echo "======================================================="
echo "ALL DONE | End: $(date)"
echo "======================================================="
