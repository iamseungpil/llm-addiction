#!/bin/bash
# Monitor all running experiments

while true; do
    clear
    echo "============================================"
    echo "EXPERIMENT MONITORING DASHBOARD"
    echo "Time: $(date)"
    echo "============================================"
    echo ""
    
    # Check Experiment 2 restart
    echo "EXPERIMENT 2 (ACTIVATION PATCHING RESTART):"
    if tmux has-session -t exp2_restart 2>/dev/null; then
        progress=$(tmux capture-pane -t exp2_restart -p | grep -E "Testing features:" | tail -1)
        if [ -n "$progress" ]; then
            echo "  Status: RUNNING"
            echo "  Progress: $progress"
        else
            echo "  Status: INITIALIZING..."
        fi
        
        # Check for causal features
        causal=$(tmux capture-pane -t exp2_restart -p | grep -E "CAUSAL" | tail -2)
        if [ -n "$causal" ]; then
            echo "  Recent: $causal"
        fi
    else
        echo "  Status: NOT RUNNING"
    fi
    
    echo ""
    echo "--------------------------------------------"
    echo ""
    
    # Check Experiment 3
    echo "EXPERIMENT 3 (RISK PREFERENCE TEST):"
    if tmux has-session -t exp3_risk_v2 2>/dev/null; then
        progress=$(tmux capture-pane -t exp3_risk_v2 -p | grep -E "Testing features:" | tail -1)
        if [ -n "$progress" ]; then
            echo "  Status: RUNNING"
            echo "  Progress: $progress"
        else
            echo "  Status: INITIALIZING..."
        fi
        
        # Check for effects
        effects=$(tmux capture-pane -t exp3_risk_v2 -p | grep -E "Shows effect|No significant" | tail -3)
        if [ -n "$effects" ]; then
            echo "  Recent results:"
            echo "$effects" | sed 's/^/    /'
        fi
    else
        echo "  Status: NOT RUNNING"
    fi
    
    echo ""
    echo "--------------------------------------------"
    echo ""
    
    # Check GPU usage
    echo "GPU USAGE:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while IFS=, read -r gpu name mem_used mem_total util; do
        echo "  GPU $gpu: $mem_used/$mem_total, Util: $util"
    done
    
    echo ""
    echo "============================================"
    echo "Press Ctrl+C to exit monitoring"
    
    sleep 30
done