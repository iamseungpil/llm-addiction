#!/bin/bash
# [SLURM-DISABLED] #SBATCH -J investment_constraints
# [SLURM-DISABLED] #SBATCH -p cas_v100_4
# [SLURM-DISABLED] #SBATCH --comment pytorch
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=12:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/investment_constraints_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/investment_constraints_%j.err

echo "========================================================================"
echo "Investment Choice Experiment - Constraint Levels (10, 30, 50, 70)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Activate conda environment
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# Navigate to project directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi
echo ""

# Constraint levels to test
CONSTRAINTS=(10 30 50 70)

# Run experiments for each constraint level
for CONSTRAINT in "${CONSTRAINTS[@]}"; do
    echo "======================================================================"
    echo "CONSTRAINT LEVEL: \$${CONSTRAINT}"
    echo "======================================================================"

    # LLaMA experiment
    echo ""
    echo "Starting LLaMA-3.1-8B with constraint \$${CONSTRAINT}..."
    echo "Start time: $(date)"

    python src/investment_choice/run_experiment.py \
      --model llama \
      --gpu 0 \
      --constraint ${CONSTRAINT} \
      --quick

    echo "LLaMA experiment completed at: $(date)"

    # Clear GPU memory
    sleep 30

    # Gemma experiment
    echo ""
    echo "Starting Gemma-2-9B with constraint \$${CONSTRAINT}..."
    echo "Start time: $(date)"

    python src/investment_choice/run_experiment.py \
      --model gemma \
      --gpu 0 \
      --constraint ${CONSTRAINT} \
      --quick

    echo "Gemma experiment completed at: $(date)"

    # Clear GPU memory between constraint levels
    sleep 30

    # Check GPU status
    echo ""
    echo "GPU Status after constraint \$${CONSTRAINT}:"
    nvidia-smi
    echo ""
done

# List all output files
echo "======================================================================"
echo "Output Files:"
echo "======================================================================"
ls -lh /home/jovyan/beomi/llm-addiction-data/investment_choice/*$(date +%Y%m%d)*.json

echo ""
echo "======================================================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  - 4 constraint levels: 10, 30, 50, 70"
echo "  - 2 models: LLaMA-3.1-8B, Gemma-2-9B"
echo "  - Each constraint runs both Variable and Fixed betting"
echo "  - Total: 4 × 2 × 2 = 16 experiment files"
echo "  - Each file contains: 4 prompt conditions × 20 reps × 2 bet types = 160 games"
