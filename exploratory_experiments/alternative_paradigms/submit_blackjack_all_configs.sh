#!/bin/bash
#
# Submit all 8 blackjack configurations as separate SLURM jobs
# Each job runs 400 games (8 components × 50 reps)
#

MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Usage: bash submit_blackjack_all_configs.sh <model>"
    echo "  model: llama, gemma, qwen"
    exit 1
fi

echo "========================================="
echo "Submitting Blackjack Experiments"
echo "========================================="
echo "Model: $MODEL"
echo "Total configs: 8"
echo "Games per config: 400"
echo "Total games: 3,200"
echo "========================================="

# Submit all 8 configurations
CONFIGS=(
    "fixed:10"
    "fixed:30"
    "fixed:50"
    "fixed:70"
    "variable:10"
    "variable:30"
    "variable:50"
    "variable:70"
)

for config in "${CONFIGS[@]}"; do
    BET_TYPE="${config%%:*}"
    CONSTRAINT="${config##*:}"

    echo ""
    echo "Submitting: ${BET_TYPE}_${CONSTRAINT}"

    sbatch run_blackjack_single_config.sh $MODEL $BET_TYPE $CONSTRAINT

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Check status with: squeue -u \$USER"
echo "Monitor progress: watch -n 10 'squeue -u \$USER'"
