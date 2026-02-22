#!/bin/bash
# Quick Start - Submit All Priority 1 Experiments
# Run this script to start all critical missing experiments

set -e  # Exit on error

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

echo "============================================"
echo "SUBMITTING PRIORITY 1 EXPERIMENTS"
echo "============================================"

# Priority 1A: Investment Choice - New Prompt Test
echo ""
echo "📊 [1/2] Submitting: Investment Choice New Prompt Test (LLaMA)"
JOB1=$(sbatch investment_choice/slurm_prompt_test.sbatch | awk '{print $4}')
echo "  ✅ Job ID: $JOB1"
echo "  Purpose: Validate new prompt fixes hallucination"
echo "  Time: ~1-2 hours"

# Priority 1B: Blackjack - Gemma Variable
echo ""
echo "🎰 [2/2] Submitting: Blackjack Gemma Variable Betting"
JOB2=$(sbatch blackjack/slurm_gemma_variable.sbatch | awk '{print $4}')
echo "  ✅ Job ID: $JOB2"
echo "  Purpose: Complete Gemma experiments"
echo "  Time: ~2-3 hours"

echo ""
echo "============================================"
echo "SUBMISSION COMPLETE"
echo "============================================"
echo ""
echo "Submitted jobs:"
echo "  - $JOB1 (Investment Choice Prompt Test)"
echo "  - $JOB2 (Blackjack Gemma Variable)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f /home/jovyan/beomi/llm-addiction-data/logs/inv_prompt_test_${JOB1}.out"
echo "  tail -f /home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_var_${JOB2}.out"
echo ""
echo "⏰ Estimated completion: ~3 hours from now"
echo ""
echo "❗ NEXT STEPS:"
echo "  1. Wait for job $JOB1 to complete"
echo "  2. Check results to verify new prompt works"
echo "  3. If successful, submit Priority 2 (LLaMA c30, c50 re-run)"
echo ""
