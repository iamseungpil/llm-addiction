#!/bin/bash
# 다음 8개 실험 시작: $30 variable (4) + $50 fixed (4)

cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src

echo "=========================================="
echo "다음 Wave 시작: $30 variable + $50 fixed"
echo "시작 시간: $(date)"
echo "=========================================="

# $30 variable (4개) - 병렬
echo "Starting $30 variable experiments..."
tmux new-session -d -s exp_gpt4o_30v "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model gpt4o --constraint 30 --bet_type variable 2>&1 | tee ../logs/gpt4o_mini_30_variable_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s exp_gpt41_30v "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model gpt41 --constraint 30 --bet_type variable 2>&1 | tee ../logs/gpt41_mini_30_variable_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s exp_claude_30v "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model claude --constraint 30 --bet_type variable 2>&1 | tee ../logs/claude_haiku_30_variable_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s exp_gemini_30v "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model gemini --constraint 30 --bet_type variable 2>&1 | tee ../logs/gemini_flash_30_variable_\$(date +%Y%m%d_%H%M%S).log"

sleep 2

# $50 fixed (4개) - 병렬
echo "Starting $50 fixed experiments..."
tmux new-session -d -s exp_gpt4o_50f "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model gpt4o --constraint 50 --bet_type fixed 2>&1 | tee ../logs/gpt4o_mini_50_fixed_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s exp_gpt41_50f "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model gpt41 --constraint 50 --bet_type fixed 2>&1 | tee ../logs/gpt41_mini_50_fixed_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s exp_claude_50f "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model claude --constraint 50 --bet_type fixed 2>&1 | tee ../logs/claude_haiku_50_fixed_\$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s exp_gemini_50f "cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src && python run_all_experiments.py --model gemini --constraint 50 --bet_type fixed 2>&1 | tee ../logs/gemini_flash_50_fixed_\$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "8개 실험 시작됨!"
echo "확인: tmux ls | grep exp_"
tmux ls | grep exp_
