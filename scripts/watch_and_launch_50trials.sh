#!/bin/bash
# Monitor current 25-trial experiments, launch 50-trial when GPU frees up
# GPU 0: gemma fixed 25t → gemma 50t (c10, c30)
# GPU 1: gemma variable 25t → gemma 50t (c50, c70)

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable}"

SCRIPT_DIR="/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src"
OUTPUT_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice/gemma_50trials"
LOG_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# PIDs of current 25-trial experiments
GPU0_PID=3956501
GPU1_PID=3956660

echo "$(date): Watching for 25-trial experiment completion..."
echo "  GPU 0 PID: $GPU0_PID"
echo "  GPU 1 PID: $GPU1_PID"

# Wait for GPU 0 to finish, then launch c10+c30 on GPU 0
(
    while kill -0 $GPU0_PID 2>/dev/null; do
        sleep 60
    done
    echo "$(date): GPU 0 free! Starting 50-trial c10, c30"

    cd "$SCRIPT_DIR"
    for CONSTRAINT in 10 30; do
        echo "$(date): GPU 0 - Constraint $CONSTRAINT"
        python investment_choice/run_experiment.py \
            --model gemma --gpu 0 --constraint "$CONSTRAINT" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee "$LOG_DIR/gemma_50t_c${CONSTRAINT}_gpu0.log"
    done
    echo "$(date): GPU 0 - ALL DONE"
) &
WATCHER0_PID=$!

# Wait for GPU 1 to finish, then launch c50+c70 on GPU 1
(
    while kill -0 $GPU1_PID 2>/dev/null; do
        sleep 60
    done
    echo "$(date): GPU 1 free! Starting 50-trial c50, c70"

    cd "$SCRIPT_DIR"
    for CONSTRAINT in 50 70; do
        echo "$(date): GPU 1 - Constraint $CONSTRAINT"
        python investment_choice/run_experiment.py \
            --model gemma --gpu 1 --constraint "$CONSTRAINT" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee "$LOG_DIR/gemma_50t_c${CONSTRAINT}_gpu1.log"
    done
    echo "$(date): GPU 1 - ALL DONE"
) &
WATCHER1_PID=$!

echo "Watchers launched: GPU0=$WATCHER0_PID, GPU1=$WATCHER1_PID"
echo "Logs will be in: $LOG_DIR/gemma_50t_c*"

wait $WATCHER0_PID $WATCHER1_PID
echo "$(date): ALL 50-TRIAL EXPERIMENTS COMPLETED"
