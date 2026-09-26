#!/bin/bash
# 현재 실험 모니터링 후 자동으로 다음 wave 시작

CURRENT_SESSIONS="exp_claude_10v exp_gemini_10v exp_gpt41_10v exp_gpt4o_10v exp_gemini_30f"

echo "=========================================="
echo "자동 모니터링 시작"
echo "현재 세션: $CURRENT_SESSIONS"
echo "=========================================="

check_active() {
    active=0
    for sess in $CURRENT_SESSIONS; do
        if tmux has-session -t $sess 2>/dev/null; then
            # 프로세스가 실행 중인지 확인
            if tmux list-panes -t $sess -F "#{pane_pid}" 2>/dev/null | xargs -I{} ps -p {} >/dev/null 2>&1; then
                active=$((active + 1))
            fi
        fi
    done
    echo $active
}

while true; do
    active=$(check_active)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 실행 중인 실험: $active 개"

    # 각 세션 진행률 표시
    for sess in $CURRENT_SESSIONS; do
        if tmux has-session -t $sess 2>/dev/null; then
            progress=$(tmux capture-pane -t $sess -p 2>/dev/null | grep -E "Game [0-9]+|%" | tail -1)
            if [ -n "$progress" ]; then
                echo "  $sess: $progress"
            fi
        fi
    done

    if [ "$active" -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ 현재 모든 실험 완료!"
        echo "🚀 다음 wave 시작합니다..."
        echo "=========================================="

        # 다음 wave 시작
        chmod +x /home/ubuntu/llm_addiction/investment_choice_extended_cot/start_next_wave.sh
        /home/ubuntu/llm_addiction/investment_choice_extended_cot/start_next_wave.sh

        echo ""
        echo "다음 wave 시작 완료!"
        echo "모니터링 종료"
        exit 0
    fi

    # 3분마다 체크
    sleep 180
done
