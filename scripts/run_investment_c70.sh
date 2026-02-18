#!/bin/bash
#SBATCH --job-name=investment-c70
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/investment_c70_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/investment_c70_%j.err

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction

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
