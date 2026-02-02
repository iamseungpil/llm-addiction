#!/bin/bash
# SAE Condition Comparison Analysis Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default model
MODEL=${1:-llama}

# Validate model argument
if [[ "$MODEL" != "llama" && "$MODEL" != "gemma" ]]; then
    echo "Usage: $0 [llama|gemma]"
    echo "  Default: llama"
    exit 1
fi

echo "=========================================="
echo "SAE Condition Comparison Analysis"
echo "Model: $MODEL"
echo "=========================================="

# Change to project directory
cd "$PROJECT_DIR"

# Run analysis
python -m src.condition_comparison --model "$MODEL" --config configs/analysis_config.yaml

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Results saved to: $PROJECT_DIR/results/"
echo "=========================================="
