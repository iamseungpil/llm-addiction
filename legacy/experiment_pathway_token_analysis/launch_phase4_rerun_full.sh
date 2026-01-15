#!/bin/bash

# Phase 4 재실행: 전체 features 커버 (top 10,000 제한 제거됨)
# 새로운 디렉토리에 결과 저장

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_FULL"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================================================="
echo "Phase 4 FULL Re-run (전체 2,787 features 커버)"
echo "========================================================================="
echo ""
echo "변경사항: top 10,000 제한 제거 → 모든 word-feature correlations 저장"
echo "Output: $OUTPUT_DIR"
echo ""

# GPU 4-7 병렬 실행
for gpu in 4 5 6 7; do
    echo "Starting GPU $gpu..."
    
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 src/phase4_word_feature_correlation.py \
        --patching-file /data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu${gpu}.jsonl \
        --output "$OUTPUT_DIR/word_feature_correlation_gpu${gpu}.json" \
        > "${LOG_DIR}/phase4_full_gpu${gpu}.log" 2>&1 &
    
    echo "  GPU $gpu started (PID: $!)"
done

echo ""
echo "모든 GPU 실행 완료!"
echo ""
echo "진행 상황 확인:"
echo "  tail -f ${LOG_DIR}/phase4_full_gpu4.log"
echo "  tail -f ${LOG_DIR}/phase4_full_gpu5.log"
echo "  tail -f ${LOG_DIR}/phase4_full_gpu6.log"
echo "  tail -f ${LOG_DIR}/phase4_full_gpu7.log"
echo ""
echo "완료 후 결과:"
echo "  ls -lh $OUTPUT_DIR/"

