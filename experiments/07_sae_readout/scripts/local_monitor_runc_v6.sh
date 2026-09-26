#!/bin/bash
# Local monitor for run_c v6 — called every 15 min by cron
# 1. pull HF ckpts + JSONs + logs
# 2. critic on combined JSONs → report tier counts
# 3. check Wave 1 IC finished → auto-dispatch Wave 2
# 4. check full completion → exit signal

set -uo pipefail
REPO=/home/v-seungplee/llm-addiction
RESULTS=$REPO/sae_v3_analysis/results
LOG=$RESULTS/logs/monitor_runc_v6.log
PYTHON=python3

ts() { date -u +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

mkdir -p $RESULTS/logs

log "=== monitor tick ==="

# 1. Check run_c status
STATUS=$(amlt status node-recovery-h200-0415 2>&1 | awk '/metacognition_run_c/{print $3}' | head -1)
log "run_c: ${STATUS:-unknown}"
if [ "$STATUS" != "running" ]; then
    log "WARN: run_c not running"
    exit 0
fi

# 2. Pull HF
log "HF pull..."
$PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='iamseungpil/llm-addiction', repo_type='dataset',
    allow_patterns=['sae_v3_analysis/results/checkpoints/**', 'sae_v3_analysis/results/json/**', 'sae_v3_analysis/results/logs/**'],
    local_dir='/tmp/hf_pull_runc',
    token='$$HF_TOKEN',
    max_workers=6,
) " 2>&1 | tail -2

# Sync to repo
rsync -a /tmp/hf_pull_runc/sae_v3_analysis/results/ $RESULTS/ 2>/dev/null || true

# 3. Run critic
log "Running critic..."
cd $REPO
ALIGNED=$($PYTHON sae_v3_analysis/scripts/critic_v7_results.py 'aligned_steering_combined_*.json' 2>&1)
SHARED=$($PYTHON sae_v3_analysis/scripts/critic_v7_results.py 'shared_axis_*_202*.json' 2>&1)
echo "$ALIGNED" | head -2 | tee -a "$LOG"
echo "$SHARED" | head -2 | tee -a "$LOG"

# 4. Commit + push if new results
if ! git diff --quiet HEAD -- sae_v3_analysis/results/checkpoints sae_v3_analysis/results/json 2>/dev/null; then
    git add sae_v3_analysis/results/checkpoints sae_v3_analysis/results/json 2>/dev/null
    git commit -m "monitor auto-commit $(ts)" --quiet 2>/dev/null
    git push origin main 2>&1 | tail -1 | tee -a "$LOG"
fi

# 5. Wave 2 dispatch check
if echo "$ALIGNED" | grep -q '\[canonical\].*experiment_a/llama/ic' && \
   echo "$ALIGNED" | grep -q '\[canonical\].*experiment_b/gemma/ic'; then
    log "Exp A IC + Exp B IC canonical — checking Wave 2 status..."
    WAVE2_RUNNING=$(timeout 30 amlt ssh node-recovery-h200-0415 :metacognition_run_c -c "pgrep -f 'run_shared_axis.*task mw' | wc -l" 2>/dev/null | tail -1)
    if [ "${WAVE2_RUNNING:-0}" -lt 1 ]; then
        log "Dispatching Wave 2..."
        timeout 45 amlt ssh node-recovery-h200-0415 :metacognition_run_c -c "bash /scratch/llm_addiction/sae_v3_analysis/scripts/launch_runc_v6_wave2.sh" 2>&1 | tail -5 | tee -a "$LOG"
    else
        log "Wave 2 already running ($WAVE2_RUNNING procs)"
    fi
fi

# 6. Completion check
ALIGNED_CANON=$(echo "$ALIGNED" | grep -c "canonical")
SHARED_CANON=$(echo "$SHARED" | grep -c "canonical")
log "tier status: aligned=$ALIGNED_CANON canonical, shared_axis=$SHARED_CANON canonical"
if [ "$ALIGNED_CANON" -ge 7 ] && [ "$SHARED_CANON" -ge 6 ]; then
    log "🎉 v6 COMPLETE — all 7 aligned + 6 shared_axis canonical"
    touch $RESULTS/.v6_complete
fi

log "=== tick done ==="
