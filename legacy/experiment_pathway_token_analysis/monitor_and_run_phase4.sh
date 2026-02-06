#!/bin/bash
# Phase 1 완료 모니터링 및 Phase 4 자동 실행

PHASE1_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_6cond_50trials"
EXPECTED=79500
CHECK_INTERVAL=300  # 5분마다 체크

echo "=== Phase 1 모니터링 시작 ==="
echo "예상 총 records: $EXPECTED"
echo "체크 간격: ${CHECK_INTERVAL}초"
echo ""

while true; do
    # 현재 진행률 확인
    if ls $PHASE1_DIR/phase1_6cond_gpu*.jsonl 1> /dev/null 2>&1; then
        CURRENT=$(cat $PHASE1_DIR/phase1_6cond_gpu*.jsonl | wc -l)
    else
        CURRENT=0
    fi

    PERCENT=$(echo "scale=2; $CURRENT * 100 / $EXPECTED" | bc)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 진행: $CURRENT / $EXPECTED ($PERCENT%)"

    # 완료 확인
    if [ $CURRENT -ge $EXPECTED ]; then
        echo ""
        echo "=== Phase 1 완료! ==="
        echo ""

        # Phase 4 실행
        echo "Phase 4 실행 중..."
        /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/run_phase4_6cond.sh

        echo ""
        echo "=== 모든 작업 완료 ==="
        break
    fi

    sleep $CHECK_INTERVAL
done
