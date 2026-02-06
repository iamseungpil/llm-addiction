#!/bin/bash
# Phase 4 실행 스크립트 (6 conditions × 50 trials 데이터용)

set -e

PHASE1_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_6cond_50trials"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_6cond_50trials"
SCRIPT="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase4_word_feature_correlation.py"

mkdir -p $OUTPUT_DIR

echo "=== Phase 4: Word-Feature Correlation (6 cond × 50 trials) ==="
echo ""

# Phase 1 결과 병합
echo "1. Phase 1 결과 병합..."
cat $PHASE1_DIR/phase1_6cond_gpu*.jsonl > $PHASE1_DIR/phase1_6cond_combined.jsonl
TOTAL=$(wc -l < $PHASE1_DIR/phase1_6cond_combined.jsonl)
echo "   총 records: $TOTAL (예상: 79,500)"
echo ""

# Phase 4 실행
echo "2. Phase 4 실행..."
python3 $SCRIPT \
    --patching-file $PHASE1_DIR/phase1_6cond_combined.jsonl \
    --output-file $OUTPUT_DIR/phase4_word_feature_6cond.json \
    --min-word-count 5

echo ""
echo "=== Phase 4 완료 ==="
echo "결과: $OUTPUT_DIR/phase4_word_feature_6cond.json"
