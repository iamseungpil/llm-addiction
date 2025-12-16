#!/bin/bash
# Launch all conditions for extended investment choice experiment
# Usage: ./launch_all_conditions.sh <model> [tmux_session_name]
# Example: ./launch_all_conditions.sh claude claude_extended

MODEL=$1
SESSION_NAME=${2:-"${MODEL}_extended"}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> [tmux_session_name]"
    echo "  model: claude, gpt4o, gpt41, gemini"
    exit 1
fi

cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src

# Constraints to test
CONSTRAINTS=("10" "30" "50" "70" "unlimited")
BET_TYPES=("fixed" "variable")

echo "=========================================="
echo "Extended Investment Choice Experiment"
echo "Model: $MODEL"
echo "Constraints: ${CONSTRAINTS[*]}"
echo "Bet Types: ${BET_TYPES[*]}"
echo "=========================================="

# Run each condition sequentially
for constraint in "${CONSTRAINTS[@]}"; do
    for bet_type in "${BET_TYPES[@]}"; do
        echo ""
        echo ">>> Running: $MODEL, constraint=$constraint, bet_type=$bet_type"
        echo ""
        python run_experiment.py --model $MODEL --constraint $constraint --bet_type $bet_type --trials 50

        if [ $? -ne 0 ]; then
            echo "❌ Error in $MODEL $constraint $bet_type"
        else
            echo "✅ Completed: $MODEL $constraint $bet_type"
        fi
    done
done

echo ""
echo "=========================================="
echo "All conditions completed for $MODEL"
echo "=========================================="
