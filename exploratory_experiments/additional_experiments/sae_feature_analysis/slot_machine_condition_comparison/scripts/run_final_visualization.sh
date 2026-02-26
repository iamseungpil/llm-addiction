#!/bin/bash
# Final visualization script - run after both LLaMA and Gemma analyses complete

echo "=================================================================="
echo "Generating Final Visualizations for Prompt Component Analysis"
echo "=================================================================="
echo ""

# Check if Gemma results exist
GEMMA_RESULTS_DIR="results/prompt_component"
GEMMA_FILES=$(ls ${GEMMA_RESULTS_DIR}/*_gemma_*.json 2>/dev/null | wc -l)

echo "Checking analysis completion status..."
echo "  LLaMA results: $(ls ${GEMMA_RESULTS_DIR}/*_llama_*.json 2>/dev/null | wc -l) files"
echo "  Gemma results: ${GEMMA_FILES} files"
echo ""

if [ "${GEMMA_FILES}" -lt 5 ]; then
    echo "⚠️  WARNING: Gemma analysis not complete yet (${GEMMA_FILES}/5 components)"
    echo "   Expected files: G, M, R, W, P (all _gemma_*.json)"
    echo ""
    echo "   Current Gemma files:"
    ls ${GEMMA_RESULTS_DIR}/*_gemma_*.json 2>/dev/null | xargs -n 1 basename
    echo ""
    echo "   You can either:"
    echo "   1. Wait for Gemma analysis to complete and re-run this script"
    echo "   2. Generate LLaMA-only visualizations with: python3 scripts/visualize_prompt_results.py --model llama"
    echo ""
    read -p "Continue with LLaMA-only visualizations? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Re-run this script when Gemma analysis is complete."
        exit 0
    fi
    MODEL_FLAG="llama"
else
    echo "✅ Both analyses complete! Generating full comparison..."
    MODEL_FLAG="both"
fi

echo ""
echo "Generating visualizations for: ${MODEL_FLAG}"
echo ""

# Run visualization
python3 scripts/visualize_prompt_results.py --model ${MODEL_FLAG}

echo ""
echo "=================================================================="
echo "Visualization Complete"
echo "=================================================================="
echo ""
echo "Generated files in: results/figures/"
ls -lh results/figures/*.png results/figures/*.csv 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Review figures: results/figures/"
echo "  2. Read interpretation guide: PROMPT_ANALYSIS_GUIDE.md"
echo "  3. Check summary CSV: results/figures/component_summary_*.csv"
echo ""
