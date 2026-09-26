#!/bin/bash

echo "=== Phase 1 Patching Experiment Monitor ==="
echo "Date: $(date)"
echo ""

# Check running processes
echo "📊 Running Processes:"
ps aux | grep "phase1_patching_multifeature.py" | grep -v grep | awk '{print "  GPU "$17": PID "$2", CPU "$3"%, MEM "$4"%"}'
echo ""

# Check GPU memory
echo "💾 GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read line; do
    echo "  GPU $line"
done
echo ""

# Check output files
echo "📁 Output Files:"
for gpu in 4 5 6 7; do
    file="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl"
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  GPU $gpu: $lines records, $size"
    else
        echo "  GPU $gpu: No output yet"
    fi
done
echo ""

# Check log files for errors
echo "⚠️  Recent Errors (last 5):"
for gpu in 4 5 6 7; do
    logfile="logs/phase1_patching_gpu${gpu}.log"
    if [ -f "$logfile" ]; then
        errors=$(grep -i "error\|exception\|traceback" "$logfile" 2>/dev/null | tail -5)
        if [ -n "$errors" ]; then
            echo "  GPU $gpu:"
            echo "$errors" | sed 's/^/    /'
        fi
    fi
done

# Estimate progress (assuming 50 features * 4 conditions * 30 trials = 6000 total per GPU)
echo ""
echo "📈 Estimated Progress:"
for gpu in 4 5 6 7; do
    file="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl"
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        progress=$(echo "scale=1; $lines / 6000 * 100" | bc)
        echo "  GPU $gpu: ${progress}% ($lines / 6000 trials)"
    else
        echo "  GPU $gpu: 0.0% (0 / 6000 trials)"
    fi
done

echo ""
echo "==============================================="
