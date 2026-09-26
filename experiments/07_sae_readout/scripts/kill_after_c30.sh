#!/bin/bash
# Monitor for c30 completion and kill the original shell script
# to prevent it from starting c50 (which GPU 1 is handling)
SHELL_PID=2602520
MAIN_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice_v2_role_llama"

echo "Watching for c30 completion (shell PID=$SHELL_PID)..."

while true; do
    # Check if c30 final file exists
    if ls ${MAIN_DIR}/llama_investment_c30_*.json 1>/dev/null 2>&1; then
        echo "c30 complete! Killing shell script PID=$SHELL_PID to prevent c50 start..."
        # Kill only the shell script, not the python process (which already finished)
        kill $SHELL_PID 2>/dev/null || echo "Shell script already exited"
        echo "Done. GPU 0 is now free."
        exit 0
    fi
    sleep 60
done
