#!/bin/bash
# 현재 실험 모니터링 후 자동으로 남은 실험 시작
# 사용법: ./monitor_and_continue.sh

SESSIONS="exp_claude_10v exp_gemini_10v exp_gpt41_10v exp_gpt4o_10v exp_claude_30f exp_gemini_30f"

echo "=========================================="
echo "실험 모니터링 시작"
echo "현재 실행 중인 세션: $SESSIONS"
echo "=========================================="

check_sessions() {
    active=0
    for sess in $SESSIONS; do
        if tmux has-session -t $sess 2>/dev/null; then
            # 세션이 여전히 실행 중인지 확인
            pane_pid=$(tmux list-panes -t $sess -F "#{pane_pid}" 2>/dev/null)
            if [ -n "$pane_pid" ]; then
                children=$(pgrep -P $pane_pid 2>/dev/null | wc -l)
                if [ "$children" -gt 0 ]; then
                    active=$((active + 1))
                fi
            fi
        fi
    done
    echo $active
}

while true; do
    active=$(check_sessions)
    echo "[$(date '+%H:%M:%S')] 활성 실험 세션: $active 개"

    if [ "$active" -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "모든 현재 실험 완료!"
        echo "남은 실험 시작합니다..."
        echo "=========================================="

        # 남은 실험 실행
        cd /home/ubuntu/llm_addiction/investment_choice_extended_cot
        ./run_remaining_sequential.sh

        echo "모든 실험 완료!"
        exit 0
    fi

    # 5분마다 체크
    sleep 300
done
