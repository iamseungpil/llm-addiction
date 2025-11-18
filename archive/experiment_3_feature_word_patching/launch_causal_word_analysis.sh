#!/bin/bash
#
# Launch Causal Word Patching Analysis
# Direct comparison of responses before/after feature patching
#

echo "================================"
echo "Causal Word Patching Analysis"
echo "Features: 2,787 (100% coverage)"
echo "Method: Before/After Patching Comparison"
echo "Expected time: ~30 minutes"
echo "================================"

cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching

# Create results and logs directories
mkdir -p results logs

# Launch analysis
tmux new-session -d -s causal_word_analysis
tmux send-keys -t causal_word_analysis "conda activate llama_sae_env" C-m
tmux send-keys -t causal_word_analysis "cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching/src" C-m
tmux send-keys -t causal_word_analysis "python causal_word_patching_analyzer.py 2>&1 | tee ../logs/causal_word_\$(date +%Y%m%d_%H%M%S).log" C-m

echo ""
echo "✅ Launched in tmux session: causal_word_analysis"
echo ""
echo "Monitor with:"
echo "  tmux attach -t causal_word_analysis"
echo ""
echo "Expected completion: $(date -d '+30 minutes' '+%Y-%m-%d %H:%M' 2>/dev/null || date -v +30M '+%Y-%m-%d %H:%M' 2>/dev/null || echo 'in ~30 minutes')"
echo ""
echo "Key insights:"
echo "  • Words ADDED by risky features"
echo "  • Words REMOVED by safe features"
echo "  • Direct causal effects (no correlation!)"
echo "================================"
