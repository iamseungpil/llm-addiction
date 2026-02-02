#!/bin/bash
# Complete Prompt Analysis Pipeline
# Runs all three analyses sequentially

set -e  # Exit on error

MODEL=$1
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model>"
    echo "  model: llama or gemma"
    exit 1
fi

if [ "$MODEL" != "llama" ] && [ "$MODEL" != "gemma" ]; then
    echo "Error: model must be 'llama' or 'gemma'"
    exit 1
fi

echo "========================================================================"
echo "Running Complete Prompt Analysis Pipeline for ${MODEL^^}"
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

# Analysis 1: Component Analysis
echo "========== Analysis 1: Component Analysis (G/M/R/W/P) =========="
echo "Sample size: 1,600 vs 1,600 per component"
echo "Expected time: ~15min (LLaMA) or ~25min (Gemma)"
echo ""

python3 -m src.prompt_component_analysis --model $MODEL
COMPONENT_EXIT=$?

if [ $COMPONENT_EXIT -ne 0 ]; then
    echo "❌ Component analysis failed"
    exit 1
fi

echo "✅ Component analysis complete"
echo ""

# Analysis 2: Complexity Analysis
echo "========== Analysis 2: Complexity Analysis (0-5 levels) =========="
echo "Sample sizes: 100-1,000 games per level"
echo "Expected time: ~3min"
echo ""

python3 -m src.prompt_complexity_analysis --model $MODEL
COMPLEXITY_EXIT=$?

if [ $COMPLEXITY_EXIT -ne 0 ]; then
    echo "❌ Complexity analysis failed"
    exit 1
fi

echo "✅ Complexity analysis complete"
echo ""

# Analysis 3: Individual Combo Analysis
echo "========== Analysis 3: Individual Combo Analysis (32 combos) =========="
echo "WARNING: Small sample sizes (50 per combo) - exploratory only!"
echo "Expected time: ~10min"
echo ""

python3 -m src.prompt_combo_explorer --model $MODEL --top-n 10
COMBO_EXIT=$?

if [ $COMBO_EXIT -ne 0 ]; then
    echo "❌ Combo analysis failed"
    exit 1
fi

echo "✅ Combo analysis complete"
echo ""

# Calculate total time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================================================"
echo "ALL ANALYSES COMPLETE for ${MODEL^^}"
echo "========================================================================"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to:"
echo "  - results/prompt_component/"
echo "  - results/prompt_complexity/"
echo "  - results/prompt_combo/"
echo ""
echo "Next step: Generate visualizations"
echo "  python3 scripts/visualize_all_results.py --model $MODEL"
echo ""
